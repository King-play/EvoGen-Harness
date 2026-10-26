from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..backends import build_backend
from ..executor import HarnessExecutor
from ..fast_loop import require_full_llm_fast_loop
from ..inspection import build_inspector
from ..io import append_jsonl, write_json, write_jsonl
from ..prompt_optimizer import build_prompt_optimizer
from ..repository import HarnessRepository
from ..schema import VisualExperience
from ..patcher import build_harness_localizer, build_patch_proposer, HarnessLocalizer, PatchProposer
from ..search import build_search
from ..tasks import load_tasks


class SuiteEvaluationExperiment:
    """Run Gen-Harness on generic tasks and summarize internal inspections."""

    def __init__(
        self,
        harness: str | Path,
        backend_config: str | Path,
        inspection_config: str | Path,
        search_config: str | Path | None = None,
        prompt_optimizer_config: str | Path | None = None,
        *,
        fast_loop_attempts: int = 5,
        fast_loop_stochastic_k: int = 4,
        harness_localizer_config: str | Path | None = None,
        patch_proposer_config: str | Path | None = None,
        harness_localizer: HarnessLocalizer | None = None,
        patch_proposer: PatchProposer | None = None,
        policy_extractor_config: str | Path | None = None,
    ):
        if policy_extractor_config is None:
            raise ValueError("policy_extractor_config is required for formal suite evaluation")
        self.repo = HarnessRepository(
            harness,
            policy_extractor_config=policy_extractor_config,
        )
        self.backend = build_backend(backend_config)
        self.inspector = build_inspector(inspection_config)
        self.search = build_search(search_config, required=False) if search_config else None
        self.prompt_optimizer = build_prompt_optimizer(prompt_optimizer_config)
        self.fast_loop_attempts = int(fast_loop_attempts)
        self.fast_loop_stochastic_k = int(fast_loop_stochastic_k)
        self.harness_localizer = harness_localizer or build_harness_localizer(harness_localizer_config, repo=self.repo)
        self.patch_proposer = patch_proposer or build_patch_proposer(
            patch_proposer_config,
            memory=self.repo.build_memory(),
            repo=self.repo,
        )
        require_full_llm_fast_loop(
            inspector=self.inspector,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            max_attempts=self.fast_loop_attempts,
            stochastic_evidence_k=self.fast_loop_stochastic_k,
            context="formal suite evaluation",
        )

    def run(
        self,
        tasks_path: str | Path,
        output_dir: str | Path,
        *,
        seeds: Sequence[int],
        limit: int | None = None,
        continue_on_error: bool = False,
    ) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        tasks = load_tasks(tasks_path)
        if limit is not None:
            tasks = tasks[:limit]
        seed_list = list(seeds) or [0]
        experiences_path = output / "experiences.jsonl"
        errors_path = output / "errors.jsonl"
        write_jsonl(experiences_path, [])
        write_jsonl(errors_path, [])
        executor = HarnessExecutor(
            self.repo,
            self.backend,
            self.inspector,
            self.search,
            prompt_optimizer=self.prompt_optimizer,
            fast_loop_attempts=self.fast_loop_attempts,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            fast_loop_stochastic_k=self.fast_loop_stochastic_k,
        )
        experiences: List[VisualExperience] = []
        errors: List[Dict[str, Any]] = []
        for seed in seed_list:
            for task in tasks:
                try:
                    task_experiences = executor.run_task(task, seed=seed, write_memory=False)
                    experiences.extend(task_experiences)
                    append_jsonl(experiences_path, [e.to_dict() for e in task_experiences])
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    error = {"task_id": task.task_id, "seed": seed, "error": f"{exc.__class__.__name__}: {exc}"}
                    errors.append(error)
                    append_jsonl(errors_path, [error])
        summary = {
            "suite": "generic_task_internal_inspection",
            "tasks_path": str(tasks_path),
            "num_tasks": len(tasks),
            "seeds": seed_list,
            "num_experiences": len(experiences),
            "num_errors": len(errors),
            "experiences_path": str(experiences_path),
            "errors_path": str(errors_path),
            "fast_loop_attempts": self.fast_loop_attempts,
            "policy_extractor_manifest": self.repo.policy_extractor_manifest(),
            "overall": self._summarize(experiences),
            "by_split": self._group_summary(experiences, "split"),
            "by_family": self._group_summary(experiences, "family"),
            "by_constraint_type": self._group_summary(experiences, "constraint_type"),
            "notes": [
                "Scores are internal Gen-Harness inspection pass rates.",
                "No official benchmark evaluator is called.",
            ],
        }
        write_json(output / "suite_evaluation_summary.json", summary)
        return summary

    def _group_summary(self, experiences: Sequence[VisualExperience], key: str) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, List[VisualExperience]] = {}
        for exp in experiences:
            if key == "constraint_type":
                value = exp.constraint_type
            else:
                value = str(exp.metadata.get(key, "unknown"))
            groups.setdefault(value, []).append(exp)
        return {name: self._summarize(rows) for name, rows in sorted(groups.items())}

    def _summarize(self, experiences: Sequence[VisualExperience]) -> Dict[str, Any]:
        total = len(experiences)
        passed = sum(1 for exp in experiences if bool(exp.verification_result.get("passed", False)))
        attempts = []
        selected = []
        for exp in experiences:
            fast_loop = exp.metadata.get("fast_loop") if isinstance(exp.metadata, dict) else None
            if not isinstance(fast_loop, dict):
                continue
            attempt_rows = fast_loop.get("attempts", [])
            if isinstance(attempt_rows, list):
                attempts.append(len(attempt_rows))
            if isinstance(fast_loop.get("selected_attempt"), int):
                selected.append(int(fast_loop["selected_attempt"]))
        return {
            "num_experiences": total,
            "num_passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "mean_fast_loop_attempts": sum(attempts) / len(attempts) if attempts else 0.0,
            "max_selected_attempt": max(selected) if selected else 0,
        }
