from __future__ import annotations

import os
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..backend import VisualBackend
from ..debugger import FailureDebugger
from ..executor import HarnessExecutor
from ..fast_loop import require_full_llm_fast_loop
from ..inspection import InspectionRunner
from ..io import append_jsonl, write_json, write_jsonl
from ..miner import WeaknessMiner
from ..patcher import EDITABLE_RESPONSIBILITIES, HarnessLocalizer, PatchProposer, build_harness_localizer, build_patch_proposer
from ..prompt_optimizer import PromptOptimizer
from ..repository import HarnessRepository
from ..schema import GenerationTask
from ..search import SearchBackend
from ..self_evolve import build_self_evolve_manifest
from ..tasks import load_tasks, split_tasks
from ..telemetry import counting_backend, counting_inspector
from ..trace_protocol import (
    TRACE_DEFAULT_K,
    TRACE_PROTOCOL_SCHEMA,
    validate_paper_dataset,
    validate_trace_seeds,
)
from ..validator import PatchValidator, ValidationReport
from .collector import NaturalFailureCollector


REQUIRED_EVOLUTION_SPLITS = ("target", "heldout", "preservation")


def trace_search_strategy(frontier_size: int) -> str:
    size = min(len(EDITABLE_RESPONSIBILITIES), max(1, int(frontier_size)))
    if size == 1:
        return "hard_top_1"
    if size >= 5:
        return "full_space"
    return "trace_top_r"


def _memory_file_fingerprints(memory_dir: Path) -> Dict[str, str]:
    """Capture the exact persistent-memory inputs visible at round start."""
    result: Dict[str, str] = {}
    if not memory_dir.exists():
        return result
    for path in sorted(memory_dir.glob("*.jsonl")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[path.name] = digest
    return result


def estimate_slow_loop_budget(
    tasks_path: str | Path,
    *,
    seeds: Sequence[int],
    fast_loop_attempts: int = TRACE_DEFAULT_K + 1,
    allow_benchmark_tasks: bool = False,
    trace_frontier_size: int = 3,
    trace_candidate_budget: int = 3,
    strict_trace_protocol: bool = False,
    dataset_protocol: str = "development",
    allow_trace_seed_ablation: bool = False,
) -> Dict[str, Any]:
    tasks = load_tasks(tasks_path)
    _guard_clean_tasks(tasks, allow_benchmark_tasks=allow_benchmark_tasks)
    _guard_required_splits(tasks)
    seed_list = validate_trace_seeds(
        list(seeds) or [0],
        expected_k=TRACE_DEFAULT_K,
        allow_ablation=allow_trace_seed_ablation or not strict_trace_protocol,
    )
    dataset_validation = validate_paper_dataset(tasks, protocol=dataset_protocol)
    exploration_multiplier = max(1, int(fast_loop_attempts))
    splits = split_tasks(tasks)
    validation_budget = _validation_budget_config()
    patch_runtime = _patch_validation_runtime_config()
    validation_multiplier = max(1, patch_runtime["fast_loop_attempts"])
    effective_frontier_size = min(len(EDITABLE_RESPONSIBILITIES), max(1, int(trace_frontier_size)))
    max_candidate_edits = effective_frontier_size * max(1, int(trace_candidate_budget))
    validation_task_upper_bound = (
        min(len(splits.get("target", [])), validation_budget["target_max_tasks"])
        + min(len(splits.get("heldout", [])), validation_budget["heldout_max_tasks"])
        + min(len(splits.get("preservation", [])), validation_budget["preservation_max_tasks"])
    )
    return {
        "mode": "slow_loop_evolution",
        "num_tasks": len(tasks),
        "splits": {name: len(rows) for name, rows in splits.items()},
        "required_splits": list(REQUIRED_EVOLUTION_SPLITS),
        "num_seeds": len(seed_list),
        "seeds": seed_list,
        "exploration_generation_calls": len(tasks) * len(seed_list) * exploration_multiplier,
        "exploration_internal_inspection_calls": len(tasks) * len(seed_list) * exploration_multiplier,
        "validation_generation_calls_per_patch": validation_task_upper_bound * len(seed_list) * validation_multiplier * 2,
        "validation_internal_inspection_calls_per_patch": validation_task_upper_bound * len(seed_list) * validation_multiplier * 2,
        "trace_frontier_size": effective_frontier_size,
        "trace_search_strategy": trace_search_strategy(trace_frontier_size),
        "trace_candidate_budget_per_responsibility": max(1, int(trace_candidate_budget)),
        "max_candidate_edits_per_step": max_candidate_edits,
        "max_validation_generation_calls_per_step": validation_task_upper_bound * len(seed_list) * validation_multiplier * 2 * max_candidate_edits,
        "max_validation_internal_inspection_calls_per_step": validation_task_upper_bound * len(seed_list) * validation_multiplier * 2 * max_candidate_edits,
        "fast_loop_attempts": max(1, int(fast_loop_attempts)),
        "patch_validation_budget": validation_budget,
        "patch_validation_runtime": patch_runtime,
        "allow_benchmark_tasks": allow_benchmark_tasks,
        "trace_protocol": TRACE_PROTOCOL_SCHEMA,
        "strict_trace_protocol": strict_trace_protocol,
        "dataset_protocol": dataset_protocol,
        "dataset_validation": dataset_validation,
        "note": "Exploration and patch validation use the same serial LLM Fast Loop; validation remains bounded to target/heldout/preservation splits.",
    }


class SlowLoopEvolutionRunner:
    """Run real harness evolution from exploration through validated memory.

    This is intentionally separate from the fast loop. The fast loop repairs a
    single generation attempt from internal inspection. The slow loop consumes
    accumulated VisualExperience records, proposes file-backed harness patches,
    validates them against target/heldout/preservation splits, and records only
    validation results in memory.
    """

    def __init__(
        self,
        repo: HarnessRepository,
        backend: VisualBackend,
        inspector: InspectionRunner,
        search_backend: SearchBackend | None = None,
        prompt_optimizer: PromptOptimizer | None = None,
        *,
        validation_metric_mode: str = "pass_rate",
        fast_loop_attempts: int = TRACE_DEFAULT_K + 1,
        regression_epsilon: float = 0.01,
        harness_localizer_config: str | Path | None = None,
        harness_localizer: HarnessLocalizer | None = None,
        trace_frontier_size: int = 3,
        trace_candidate_budget: int = 3,
        trace_preservation_weight: float = 0.5,
        trace_execution_cost_weight: float = 0.05,
        patch_proposer_config: str | Path | None = None,
        patch_proposer: PatchProposer | None = None,
        fast_loop_stochastic_k: int = TRACE_DEFAULT_K,
        strict_trace_protocol: bool = False,
        dataset_protocol: str = "development",
        allow_trace_seed_ablation: bool = False,
    ):
        self.repo = repo
        self.backend = counting_backend(backend)
        self.inspector = counting_inspector(inspector)
        self.search_backend = search_backend
        self.prompt_optimizer = prompt_optimizer
        self.validation_metric_mode = validation_metric_mode
        self.fast_loop_attempts = int(fast_loop_attempts)
        self.regression_epsilon = float(regression_epsilon)
        self.fast_loop_stochastic_k = max(1, int(fast_loop_stochastic_k))
        self.strict_trace_protocol = bool(strict_trace_protocol)
        self.dataset_protocol = str(dataset_protocol)
        self.allow_trace_seed_ablation = bool(allow_trace_seed_ablation)
        self.harness_localizer_config = harness_localizer_config
        self.harness_localizer = harness_localizer or (
            build_harness_localizer(
                harness_localizer_config,
                repo=self.repo,
                strict_protocol=self.strict_trace_protocol,
            )
            if harness_localizer_config
            else None
        )
        self.trace_frontier_size = min(len(EDITABLE_RESPONSIBILITIES), max(1, int(trace_frontier_size)))
        self.trace_candidate_budget = max(1, int(trace_candidate_budget))
        self.trace_preservation_weight = float(trace_preservation_weight)
        self.trace_execution_cost_weight = float(trace_execution_cost_weight)
        self.patch_proposer_config = patch_proposer_config
        self.patch_proposer = patch_proposer or (
            build_patch_proposer(
                patch_proposer_config,
                memory=self.repo.build_memory(),
                repo=self.repo,
                strict_protocol=self.strict_trace_protocol,
            )
            if patch_proposer_config
            else None
        )
        require_full_llm_fast_loop(
            inspector=self.inspector,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            max_attempts=self.fast_loop_attempts,
            stochastic_evidence_k=self.fast_loop_stochastic_k,
            context="SlowLoopEvolutionRunner",
        )
        if self.strict_trace_protocol:
            if self.harness_localizer is not None and not bool(getattr(self.harness_localizer, "strict_protocol", False)):
                raise ValueError("paper TRACE requires a strict HarnessLocalizer instance")
            if self.patch_proposer is not None and not bool(getattr(self.patch_proposer, "strict_protocol", False)):
                raise ValueError("paper TRACE requires a strict PatchProposer instance")

    def run(
        self,
        tasks_path: str | Path,
        output_dir: str | Path,
        *,
        seeds: Sequence[int],
        min_support: int = 2,
        promote: bool = False,
        write_memory: bool = True,
        allow_benchmark_tasks: bool = False,
    ) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        run_started = time.perf_counter()
        generator_calls_before = self.backend.call_count
        verifier_calls_before = self.inspector.call_count
        localizer_llm_calls = 0
        proposer_llm_calls = 0
        tasks = load_tasks(tasks_path)
        self._guard_clean_tasks(tasks, allow_benchmark_tasks=allow_benchmark_tasks)
        self._guard_required_splits(tasks)
        seed_list = validate_trace_seeds(
            list(seeds) or [0],
            expected_k=TRACE_DEFAULT_K,
            allow_ablation=self.allow_trace_seed_ablation or not self.strict_trace_protocol,
        )
        dataset_validation = validate_paper_dataset(tasks, protocol=self.dataset_protocol)

        memory = self.repo.build_memory()
        policy_extractor_manifest = self.repo.policy_extractor_manifest()
        harness_before = self.repo.snapshot()
        splits = split_tasks(tasks)
        round_input_snapshot = {
            "schema": "gen_harness.round_input_snapshot.v1",
            "tasks_path": str(tasks_path),
            "task_ids_by_split": {name: [task.task_id for task in rows] for name, rows in splits.items()},
            "seeds": seed_list,
            "fast_loop_attempts": self.fast_loop_attempts,
            "validation_metric_mode": self.validation_metric_mode,
            "regression_epsilon": self.regression_epsilon,
            "policy_extractor_manifest": policy_extractor_manifest,
            "memory_files": _memory_file_fingerprints(memory.memory_dir),
            "memory_write_deferred_until_round_end": True,
            "trace_protocol": TRACE_PROTOCOL_SCHEMA,
            "strict_trace_protocol": self.strict_trace_protocol,
            "dataset_protocol": self.dataset_protocol,
            "dataset_validation": dataset_validation,
        }
        write_json(output / "round_input_snapshot.json", round_input_snapshot)
        experiences_path = output / "experiences.jsonl"
        write_jsonl(experiences_path, [])
        # A round reads a frozen memory view.  New failures/weaknesses/patch
        # results are committed only after validation, so they cannot influence
        # later proposals in the same round.
        fast_loop_localizer = self.harness_localizer
        fast_loop_proposer = self.patch_proposer
        experiences = self._explore(
            tasks,
            seed_list,
            write_memory=False,
            experiences_path=experiences_path,
            harness_localizer=fast_loop_localizer,
            patch_proposer=fast_loop_proposer,
        )
        self._attach_failure_diagnoses(experiences)
        write_jsonl(experiences_path, [e.to_compact_dict() for e in experiences])
        natural_summary = NaturalFailureCollector().collect(experiences, output)

        target_experiences = [e for e in experiences if e.metadata.get("split") == "target"]
        weaknesses = WeaknessMiner(
            min_support=min_support,
            expected_stochastic_conditions=len(seed_list),
        ).mine(target_experiences)
        write_jsonl(output / "weaknesses.jsonl", [w.to_dict() for w in weaknesses])

        localizations = []
        # One SlowLoopEvolutionRunner invocation is one TRACE depth step.
        # Localize every residual weakness from this step; the proposer still
        # enforces the global r*M candidate budget across the full frontier.
        localization_weaknesses = sorted(weaknesses, key=lambda row: (-row.support_count, row.weakness_id))
        if localization_weaknesses:
            localizer = self.harness_localizer or build_harness_localizer(
                self.harness_localizer_config,
                repo=self.repo,
                strict_protocol=self.strict_trace_protocol,
            )
            localizer_requests_before = int(getattr(localizer, "request_count", 0) or 0)
            localizations = [
                localizer.localize(weakness, frontier_size=self.trace_frontier_size)
                for weakness in localization_weaknesses
            ]
            localizer_llm_calls = max(
                0,
                int(getattr(localizer, "request_count", localizer_requests_before) or 0) - localizer_requests_before,
            )
        write_jsonl(output / "harness_localizations.jsonl", [row.to_dict() for row in localizations])
        frontier_by_weakness = {
            row.weakness_id: row.frontier
            for row in localizations
            if not row.no_patch_preferred and row.frontier
        }
        active_weaknesses = [
            weakness
            for weakness in localization_weaknesses
            if weakness.weakness_id in frontier_by_weakness
        ]

        patches = []
        if active_weaknesses:
            proposer = self.patch_proposer or build_patch_proposer(
                self.patch_proposer_config,
                memory=memory,
                repo=self.repo,
                strict_protocol=self.strict_trace_protocol,
            )
            proposer_requests_before = int(getattr(proposer, "request_count", 0) or 0)
            patches = [
                patch
                for patch in proposer.propose_many(
                    active_weaknesses,
                    frontier_by_weakness=frontier_by_weakness,
                    max_candidates_per_responsibility=self.trace_candidate_budget,
                )
                if _patch_is_complete(patch)
            ]
            proposer_llm_calls = max(
                0,
                int(getattr(proposer, "request_count", proposer_requests_before) or 0) - proposer_requests_before,
            )
        write_jsonl(output / "patches.jsonl", [p.to_dict() for p in patches])

        patch_validation_runtime = _patch_validation_runtime_config()
        patch_validation_budget = _validation_budget_config()
        validator = PatchValidator(
            self.repo,
            self.backend,
            self.inspector,
            self.search_backend,
            regression_epsilon=self.regression_epsilon,
            prompt_optimizer=self.prompt_optimizer,
            metric_mode=self.validation_metric_mode,
            fast_loop_attempts=patch_validation_runtime["fast_loop_attempts"],
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            fast_loop_stochastic_k=self.fast_loop_stochastic_k,
            preservation_weight=self.trace_preservation_weight,
            execution_cost_weight=self.trace_execution_cost_weight,
        )

        reports = []
        validated = []
        evolution_steps = []
        write_jsonl(output / "validation_reports.jsonl", [])
        validation_seeds = _ensure_multiple_validation_seeds(seed_list)
        for patch in patches:
            validation_tasks = _validation_tasks_for_patch(patch, splits)
            try:
                report = validator.validate(
                    patch,
                    target_tasks=validation_tasks["target"],
                    heldout_tasks=validation_tasks["heldout"],
                    preservation_tasks=validation_tasks["preservation"],
                    seeds=validation_seeds,
                )
            except (KeyError, TypeError, ValueError) as exc:
                report = _invalid_patch_report(patch, validation_seeds, exc)
            report_row = report.to_dict()
            reports.append(report_row)
            validated.append((patch, report))
            append_jsonl(output / "validation_reports.jsonl", [report_row])
            evolution_steps.append(
                {
                    "patch_id": patch.patch_id,
                    "accepted": report.accepted,
                    "promoted_to_harness": False,
                    "target_component": patch.target_component,
                    "changed_artifact": patch.changed_artifact,
                    "changed_fields": patch.changed_fields,
                    "operation": patch.operation,
                    "payload": patch.payload,
                    "supporting_experience": patch.supporting_experience,
                    "evidence_summary": patch.evidence_summary,
                    "regression_risk": patch.regression_risk,
                    "validation_budget": validation_tasks["budget"],
                    "validation_runtime": patch_validation_runtime,
                    "validation_report": report_row,
                }
            )
        # Compare every candidate against the same parent harness under the
        # same stochastic conditions.  Equation (7) utility is the primary
        # ranking signal; visual probabilities only break exact ties.
        accepted = [(patch, report) for patch, report in validated if report.accepted]
        winner = max(
            accepted,
            key=lambda item: (
                item[1].utility,
                float((item[1].ranking_signal or {}).get("target_nvila_probability_delta", 0.0)),
                item[1].target_delta,
                item[1].heldout_delta,
            ),
            default=None,
        )
        if promote and winner is not None:
            validator.promote_if_accepted(*winner)
            for step in evolution_steps:
                if step["patch_id"] == winner[0].patch_id:
                    step["promoted_to_harness"] = True
                    step["selected_candidate"] = True
        if write_memory:
            memory.write_experiences(experiences)
            memory.write_weaknesses(weaknesses)
            for patch, report in zip(patches, reports):
                memory.write_patch_result(patch, report)
        trace_certificate = _trace_certificate(
            localizations=localizations,
            active_weaknesses=active_weaknesses,
            patches=patches,
            reports=reports,
            evolution_steps=evolution_steps,
            validation_seeds=validation_seeds,
            output=output,
            preservation_weight=self.trace_preservation_weight,
            execution_cost_weight=self.trace_execution_cost_weight,
        )
        write_json(output / "trace_certificate.json", trace_certificate)
        harness_after = self.repo.snapshot()
        evolution_efficiency = {
            "schema": "gen_harness.trace_evolution_efficiency.v1",
            "wall_clock_seconds": round(time.perf_counter() - run_started, 6),
            "generator_calls": self.backend.call_count - generator_calls_before,
            "visual_verifier_calls": self.inspector.call_count - verifier_calls_before,
            "localizer_llm_calls": localizer_llm_calls,
            "proposer_llm_calls": proposer_llm_calls,
            "llm_vlm_calls": localizer_llm_calls + proposer_llm_calls + (self.inspector.call_count - verifier_calls_before),
            "candidate_edits_proposed": len(patches),
            "candidate_edits_evaluated": len(reports),
            "evolution_trigger_count": len(active_weaknesses),
            "candidate_edits_per_evolution_task": len(reports) / max(1, len(active_weaknesses)),
            "call_count_source": "runtime_wrappers_and_llm_request_counters",
            "estimated": False,
        }
        evolution_log = {
            "schema": "gen_harness.slow_loop_evolution.v1",
            "harness_root": str(self.repo.root),
            "tasks_path": str(tasks_path),
            "allow_benchmark_tasks": allow_benchmark_tasks,
            "write_memory": write_memory,
            "promote": promote,
            "seeds": seed_list,
            "fast_loop_attempts": self.fast_loop_attempts,
            "policy_extractor_manifest": policy_extractor_manifest,
            "trace": {
                "protocol_schema": TRACE_PROTOCOL_SCHEMA,
                "strict_trace_protocol": self.strict_trace_protocol,
                "dataset_protocol": self.dataset_protocol,
                "expected_k": len(seed_list),
                "depth_semantics": "one SlowLoopEvolutionRunner invocation equals one TRACE depth",
                "search_strategy": trace_search_strategy(self.trace_frontier_size),
                "frontier_size": self.trace_frontier_size,
                "candidate_budget_per_responsibility": self.trace_candidate_budget,
                "preservation_weight": self.trace_preservation_weight,
                "execution_cost_weight": self.trace_execution_cost_weight,
                "num_localized_weaknesses": len(localizations),
                "num_no_patch": sum(1 for row in localizations if row.no_patch_preferred),
            },
            "patch_validation_runtime": patch_validation_runtime,
            "patch_validation_budget": patch_validation_budget,
            "validation_metric_mode": self.validation_metric_mode,
            "regression_epsilon": self.regression_epsilon,
            "trace_certificate": trace_certificate,
            "evolution_efficiency": evolution_efficiency,
            "harness_before": harness_before,
            "round_input_snapshot": round_input_snapshot,
            "failure_summary": {
                "num_experiences": len(experiences),
                "num_failed_experiences": sum(1 for e in experiences if not e.verification_result.get("passed")),
                "failed_experience_ids": [e.experience_id for e in experiences if not e.verification_result.get("passed")],
            },
            "weaknesses": [w.to_dict() for w in weaknesses],
            "patches": [p.to_dict() for p in patches],
            "evolution_steps": evolution_steps,
            "harness_after": harness_after,
        }
        write_json(output / "harness_evolution_log.json", evolution_log)
        self_evolve_manifest = build_self_evolve_manifest(output, write_path=output / "self_evolve_manifest.json")

        summary = {
            "schema": "gen_harness.slow_loop_evolution_summary.v1",
            "tasks_path": str(tasks_path),
            "num_tasks": len(tasks),
            "seeds": seed_list,
            "num_experiences": len(experiences),
            "natural_failures": natural_summary,
            "num_weaknesses": len(weaknesses),
            "num_localizations": len(localizations),
            "num_no_patch": sum(1 for row in localizations if row.no_patch_preferred),
            "num_patches": len(patches),
            "num_reports": len(reports),
            "num_accepted_patches": sum(1 for report in reports if report.get("accepted")),
            "num_promoted_patches": sum(1 for step in evolution_steps if step.get("promoted_to_harness")),
            "write_memory": write_memory,
            "promote": promote,
            "allow_benchmark_tasks": allow_benchmark_tasks,
            "validation_metric_mode": self.validation_metric_mode,
            "trace_protocol": TRACE_PROTOCOL_SCHEMA,
            "strict_trace_protocol": self.strict_trace_protocol,
            "dataset_protocol": self.dataset_protocol,
            "dataset_validation": dataset_validation,
            "fast_loop_attempts": self.fast_loop_attempts,
            "policy_extractor_manifest": policy_extractor_manifest,
            "trace": {
                "protocol_schema": TRACE_PROTOCOL_SCHEMA,
                "strict_trace_protocol": self.strict_trace_protocol,
                "dataset_protocol": self.dataset_protocol,
                "expected_k": len(seed_list),
                "depth_semantics": "one SlowLoopEvolutionRunner invocation equals one TRACE depth",
                "search_strategy": trace_search_strategy(self.trace_frontier_size),
                "frontier_size": self.trace_frontier_size,
                "candidate_budget_per_responsibility": self.trace_candidate_budget,
                "preservation_weight": self.trace_preservation_weight,
                "execution_cost_weight": self.trace_execution_cost_weight,
                "num_localized_weaknesses": len(localizations),
                "num_no_patch": sum(1 for row in localizations if row.no_patch_preferred),
            },
            "patch_validation_runtime": patch_validation_runtime,
            "patch_validation_budget": patch_validation_budget,
            "artifacts": {
                "experiences": str(output / "experiences.jsonl"),
                "natural_failures": str(output / "natural_failures.jsonl"),
                "failure_diagnoses": str(output / "failure_diagnoses.jsonl"),
                "weaknesses": str(output / "weaknesses.jsonl"),
                "harness_localizations": str(output / "harness_localizations.jsonl"),
                "patches": str(output / "patches.jsonl"),
                "validation_reports": str(output / "validation_reports.jsonl"),
                "evolution_log": str(output / "harness_evolution_log.json"),
                "self_evolve_manifest": str(output / "self_evolve_manifest.json"),
                "round_input_snapshot": str(output / "round_input_snapshot.json"),
                "trace_certificate": str(output / "trace_certificate.json"),
            },
            "self_evolve_gate_summary": self_evolve_manifest["gate_summary"],
            "trace_certificate": trace_certificate,
            "evolution_efficiency": evolution_efficiency,
        }
        write_json(output / "slow_loop_summary.json", summary)
        return summary

    def estimate_budget(self, tasks_path: str | Path, *, seeds: Sequence[int], allow_benchmark_tasks: bool = False) -> Dict[str, Any]:
        return estimate_slow_loop_budget(
            tasks_path,
            seeds=seeds,
            fast_loop_attempts=self.fast_loop_attempts,
            allow_benchmark_tasks=allow_benchmark_tasks,
            trace_frontier_size=self.trace_frontier_size,
            trace_candidate_budget=self.trace_candidate_budget,
            strict_trace_protocol=self.strict_trace_protocol,
            dataset_protocol=self.dataset_protocol,
            allow_trace_seed_ablation=self.allow_trace_seed_ablation,
        )

    def _attach_failure_diagnoses(self, experiences: List[Any]) -> None:
        diagnoses = FailureDebugger().diagnose_many(experiences)
        by_experience = {diag.experience_id: diag.to_dict() for diag in diagnoses}
        for exp in experiences:
            diagnosis = by_experience.get(exp.experience_id)
            if not diagnosis:
                continue
            exp.metadata["failure_diagnosis"] = diagnosis
            if not exp.failure_symptom or exp.failure_symptom == "unknown":
                exp.failure_symptom = diagnosis.get("failure_symptom")

    def _explore(
        self,
        tasks: Sequence[GenerationTask],
        seeds: Sequence[int],
        *,
        write_memory: bool,
        experiences_path: Path | None = None,
        harness_localizer: HarnessLocalizer | None = None,
        patch_proposer: PatchProposer | None = None,
    ):
        executor = HarnessExecutor(
            self.repo,
            self.backend,
            self.inspector,
            self.search_backend,
            prompt_optimizer=self.prompt_optimizer,
            fast_loop_attempts=self.fast_loop_attempts,
            harness_localizer=harness_localizer,
            patch_proposer=patch_proposer,
            fast_loop_stochastic_k=self.fast_loop_stochastic_k,
        )
        experiences = []
        for seed in seeds:
            for task in tasks:
                task_experiences = executor.run_task(task, seed=seed, write_memory=write_memory)
                experiences.extend(task_experiences)
                if experiences_path is not None:
                    append_jsonl(experiences_path, [e.to_compact_dict() for e in task_experiences])
        return experiences

    def _guard_clean_tasks(self, tasks: Sequence[GenerationTask], *, allow_benchmark_tasks: bool) -> None:
        _guard_clean_tasks(tasks, allow_benchmark_tasks=allow_benchmark_tasks)

    def _guard_required_splits(self, tasks: Sequence[GenerationTask]) -> None:
        _guard_required_splits(tasks)

    def _is_benchmark_task(self, task: GenerationTask) -> bool:
        return _is_benchmark_task(task)


def _guard_clean_tasks(tasks: Sequence[GenerationTask], *, allow_benchmark_tasks: bool) -> None:
    if allow_benchmark_tasks and str(os.getenv("GENHARNESS_DIAGNOSTIC_ALLOW_BENCHMARK_EVOLVE", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    offenders: List[str] = []
    for task in tasks:
        if _is_benchmark_task(task):
            offenders.append(task.task_id)
    if offenders:
        preview = ", ".join(offenders[:5])
        suffix = "..." if len(offenders) > 5 else ""
        raise ValueError(
            "slow-loop evolution refuses benchmark tasks by default; "
            f"found benchmark metadata in {len(offenders)} task(s): {preview}{suffix}. "
            "Use non-benchmark target/heldout/preservation tasks. Diagnostic benchmark evolution requires both "
            "--allow-benchmark-tasks and GENHARNESS_DIAGNOSTIC_ALLOW_BENCHMARK_EVOLVE=1."
        )


def _guard_required_splits(tasks: Sequence[GenerationTask]) -> None:
    splits = split_tasks(tasks)
    missing = [name for name in REQUIRED_EVOLUTION_SPLITS if not splits.get(name)]
    if missing:
        raise ValueError(
            "slow-loop evolution requires non-empty target, heldout, and preservation splits; "
            f"missing: {', '.join(missing)}"
        )


def _is_benchmark_task(task: GenerationTask) -> bool:
    meta = task.metadata if isinstance(task.metadata, dict) else {}
    benchmark_keys = {"benchmark", "geneval_metadata", "geneval_tag", "geneval_index", "source_benchmark", "source_tag"}
    if benchmark_keys & set(meta):
        return True
    for constraint in task.constraints:
        cmeta = constraint.metadata if isinstance(constraint.metadata, dict) else {}
        if benchmark_keys & set(cmeta):
            return True
    return False


def _validation_tasks_for_patch(patch: Any, splits: Dict[str, List[GenerationTask]]) -> Dict[str, Any]:
    constraint_type = _patch_constraint_type(patch)
    target = _filter_tasks_by_constraint_type(splits.get("target", []), constraint_type)
    heldout = _filter_tasks_by_constraint_type(splits.get("heldout", []), constraint_type)
    preservation = _filter_tasks_by_constraint_type(splits.get("preservation", []), constraint_type)
    if not target:
        target = list(splits.get("target", []))
    if not heldout:
        heldout = list(splits.get("heldout", []))
    if not preservation:
        preservation = list(splits.get("preservation", []))

    config = _validation_budget_config()
    target = target[:config["target_max_tasks"]]
    heldout = heldout[:config["heldout_max_tasks"]]
    preservation = preservation[:config["preservation_max_tasks"]]
    return {
        "target": target,
        "heldout": heldout,
        "preservation": preservation,
        "budget": {
            **config,
            "constraint_type": constraint_type,
            "target_task_ids": [task.task_id for task in target],
            "heldout_task_ids": [task.task_id for task in heldout],
            "preservation_task_ids": [task.task_id for task in preservation],
            "selection_signal": "internal non-benchmark validation tasks matching the patch failure constraint type",
            "official_benchmark_score_used": False,
        },
    }


def _trace_certificate(
    *,
    localizations: Sequence[Any],
    active_weaknesses: Sequence[Any],
    patches: Sequence[Any],
    reports: Sequence[Dict[str, Any]],
    evolution_steps: Sequence[Dict[str, Any]],
    validation_seeds: Sequence[int],
    output: Path,
    preservation_weight: float,
    execution_cost_weight: float,
) -> Dict[str, Any]:
    validation_reports = [row for row in reports if isinstance(row, dict)]
    promoted_steps = [row for row in evolution_steps if isinstance(row, dict) and row.get("promoted_to_harness") is True]
    accepted_patch_ids = {
        str(row.get("patch_id"))
        for row in validation_reports
        if row.get("accepted") is True and row.get("patch_id")
    }
    promoted_patch_ids = {
        str(row.get("patch_id"))
        for row in promoted_steps
        if row.get("patch_id")
    }
    validation_scope_rows = []
    for step in evolution_steps:
        if not isinstance(step, dict):
            continue
        budget = step.get("validation_budget", {})
        report = step.get("validation_report", {})
        report_scope = report.get("validation_scope", {}) if isinstance(report, dict) else {}
        report_guards = report.get("validation_guards", {}) if isinstance(report, dict) else {}
        validation_scope_rows.append({
            "patch_id": step.get("patch_id"),
            "target_task_ids": budget.get("target_task_ids", []) if isinstance(budget, dict) else [],
            "heldout_task_ids": budget.get("heldout_task_ids", []) if isinstance(budget, dict) else [],
            "preservation_task_ids": budget.get("preservation_task_ids", []) if isinstance(budget, dict) else [],
            "seeds": list(report.get("seeds", [])) if isinstance(report, dict) else [],
            "accepted": bool(report.get("accepted", False)) if isinstance(report, dict) else False,
            "target_delta": report.get("target_delta") if isinstance(report, dict) else None,
            "heldout_delta": report.get("heldout_delta") if isinstance(report, dict) else None,
            "regression_delta": report.get("regression_delta") if isinstance(report, dict) else None,
            "execution_cost_delta": report.get("execution_cost_delta") if isinstance(report, dict) else None,
            "utility": report.get("utility") if isinstance(report, dict) else None,
            "utility_terms": report.get("utility_terms", {}) if isinstance(report, dict) else {},
            "report_scope_complete": report_scope.get("target_heldout_preservation_present") if isinstance(report_scope, dict) else None,
            "official_evaluator_used": report_guards.get("official_evaluator_used") if isinstance(report_guards, dict) else None,
            "official_scores_used": report_guards.get("official_scores_used") if isinstance(report_guards, dict) else None,
            "fallback_used": report_guards.get("fallback_used") if isinstance(report_guards, dict) else None,
            "internal_verifier_only": report_guards.get("internal_verifier_only") if isinstance(report_guards, dict) else None,
            "matched_stochastic_conditions": report_guards.get("matched_stochastic_conditions") if isinstance(report_guards, dict) else None,
            "heldout_improved": report_guards.get("heldout_improved") if isinstance(report_guards, dict) else None,
            "positive_utility": report_guards.get("positive_utility") if isinstance(report_guards, dict) else None,
        })
    non_accepted_promotions = sorted(promoted_patch_ids - accepted_patch_ids)
    validation_scope_complete = bool(validation_scope_rows) and all(
        row["target_task_ids"] and row["heldout_task_ids"] and row["preservation_task_ids"] and row["seeds"]
        and row["report_scope_complete"] is True
        for row in validation_scope_rows
    )
    official_evaluator_used = any(row["official_evaluator_used"] is not False for row in validation_scope_rows) if validation_scope_rows else False
    official_scores_used = any(row["official_scores_used"] is not False for row in validation_scope_rows) if validation_scope_rows else False
    fallback_used = any(row["fallback_used"] is not False for row in validation_scope_rows) if validation_scope_rows else False
    no_patch_count = sum(1 for row in localizations if getattr(row, "no_patch_preferred", False))
    active_localization_count = sum(1 for row in localizations if not getattr(row, "no_patch_preferred", False) and getattr(row, "frontier", []))
    return {
        "schema": "gen_harness.trace_certificate.v1",
        "flow": [
            "constraint_indexed_visual_experience",
            "llm_harness_localization_with_no_patch",
            "responsibility_scoped_patch_proposal",
            "target_heldout_preservation_validation",
            "regression_safe_promotion_only",
        ],
        "artifacts": {
            "experiences": str(output / "experiences.jsonl"),
            "failure_diagnoses": str(output / "failure_diagnoses.jsonl"),
            "weaknesses": str(output / "weaknesses.jsonl"),
            "harness_localizations": str(output / "harness_localizations.jsonl"),
            "patches": str(output / "patches.jsonl"),
            "validation_reports": str(output / "validation_reports.jsonl"),
            "harness_evolution_log": str(output / "harness_evolution_log.json"),
        },
        "localization": {
            "num_localizations": len(localizations),
            "num_no_patch": no_patch_count,
            "num_active_frontiers": active_localization_count,
            "no_patch_supported": True,
            "llm_localizer_required_for_weaknesses": bool(localizations or not active_weaknesses),
            "frontier_source": "llm_harness_localizer",
            "responsibilities": ["policy", "tools", "skills", "middleware", "memory"],
        },
        "patch_proposals": {
            "num_patches": len(patches),
            "all_component_scoped": all(
                getattr(patch, "target_component", None) in {"policy", "tools", "skills", "middleware", "memory"}
                and bool(getattr(patch, "changed_artifact", None))
                and bool(getattr(patch, "changed_fields", None))
                for patch in patches
            ),
            "proposal_source": "llm_patch_proposer" if patches else "none",
        },
        "validation": {
            "num_reports": len(validation_reports),
            "validation_seeds": list(validation_seeds),
            "matched_seed_count": len(validation_seeds),
            "utility": {
                "formula": "target_delta - lambda * preservation_penalty - mu * execution_cost_delta",
                "lambda": preservation_weight,
                "mu": execution_cost_weight,
            },
            "scope_rows": validation_scope_rows,
            "target_heldout_preservation_present": validation_scope_complete or not patches,
            "official_evaluator_used": official_evaluator_used,
            "official_scores_used": official_scores_used,
            "fallback_used": fallback_used,
        },
        "promotion": {
            "num_promoted": len(promoted_patch_ids),
            "promoted_patch_ids": sorted(promoted_patch_ids),
            "accepted_patch_ids": sorted(accepted_patch_ids),
            "non_accepted_promotions": non_accepted_promotions,
            "accepted_only": not non_accepted_promotions,
        },
        "leakage_and_fallback_guards": {
            "official_labels_used_for_prompt_compilation": False,
            "official_evaluator_used_for_generation_or_selection": False,
            "fallback_image_substitution_allowed": False,
        },
    }


def _patch_constraint_type(patch: Any) -> str:
    evidence_summary = getattr(patch, "evidence_summary", {}) or {}
    if isinstance(evidence_summary, dict) and evidence_summary.get("constraint_type"):
        return str(evidence_summary["constraint_type"])
    payload = getattr(patch, "payload", {}) or {}
    rationale = payload.get("proposal_rationale", {}) if isinstance(payload, dict) else {}
    if isinstance(rationale, dict) and rationale.get("constraint_type"):
        return str(rationale["constraint_type"])
    return ""


def _filter_tasks_by_constraint_type(tasks: Sequence[GenerationTask], constraint_type: str) -> List[GenerationTask]:
    if not constraint_type:
        return list(tasks)
    out = []
    for task in tasks:
        if any(constraint.constraint_type == constraint_type for constraint in task.constraints):
            out.append(task)
    return out


def _validation_budget_config() -> Dict[str, int]:
    return {
        "target_max_tasks": _positive_env_int("GENHARNESS_SELF_EVOLVE_VALIDATION_TARGET_MAX", 3),
        "heldout_max_tasks": _positive_env_int("GENHARNESS_SELF_EVOLVE_VALIDATION_HELDOUT_MAX", 3),
        "preservation_max_tasks": _positive_env_int("GENHARNESS_SELF_EVOLVE_VALIDATION_PRESERVATION_MAX", 2),
    }


def _patch_validation_runtime_config() -> Dict[str, int]:
    return {
        "fast_loop_attempts": _positive_env_int("GENHARNESS_SELF_EVOLVE_PATCH_VALIDATION_FAST_LOOP_ATTEMPTS", TRACE_DEFAULT_K + 1),
    }


def _ensure_multiple_validation_seeds(seeds: Sequence[int]) -> List[int]:
    """Use at least two deterministic seeds for patch validation.

    Exploration still uses the caller's requested seeds. This only strengthens
    candidate comparison and does not widen the task or benchmark surface.
    """
    unique = list(dict.fromkeys(int(seed) for seed in seeds))
    if not unique:
        unique = [0]
    candidate = unique[0] + 1
    while len(unique) < 2:
        if candidate not in unique:
            unique.append(candidate)
        candidate += 1
    return unique


def _positive_env_int(name: str, default: int) -> int:
    text = os.getenv(name, str(default)).strip()
    try:
        value = int(text)
    except ValueError:
        value = default
    return max(1, value)


def _patch_is_complete(patch: Any) -> bool:
    required = (
        "patch_id",
        "target_component",
        "changed_artifact",
        "changed_fields",
        "operation",
        "payload",
        "supporting_experience",
        "predicted_improvement",
        "possible_regression",
        "target_validation_set",
        "preservation_set",
        "held_out_set",
        "rollback_boundary",
    )
    for field in required:
        value = getattr(patch, field, None)
        if value in (None, "", [], {}):
            return False
    payload = getattr(patch, "payload", {}) or {}
    if getattr(patch, "operation", None) in {"set_path", "append_unique"}:
        return bool(payload.get("path") and "value" in payload)
    if getattr(patch, "operation", None) == "delete_path":
        return bool(payload.get("path"))
    if getattr(patch, "operation", None) == "append_jsonl_unique":
        return bool(payload.get("row") and payload.get("unique_key"))
    return False


def _invalid_patch_report(patch: Any, seeds: Sequence[int], error: Exception) -> ValidationReport:
    scores = {"target": 0.0, "heldout": 0.0, "preservation": 0.0}
    return ValidationReport(
        patch_id=str(getattr(patch, "patch_id", "unknown")),
        accepted=False,
        before=scores,
        after=dict(scores),
        target_delta=0.0,
        heldout_delta=0.0,
        regression_delta=0.0,
        reason=f"rejected: candidate patch could not be applied ({type(error).__name__})",
        seeds=list(seeds),
        validation_scope={"target_heldout_preservation_present": True, "seeds": list(seeds)},
        validation_guards={
            "target_improved": False,
            "heldout_improved": False,
            "heldout_non_regressing": False,
            "preservation_regression_bounded": False,
            "positive_utility": False,
            "matched_stochastic_conditions": True,
            "official_evaluator_used": False,
            "official_scores_used": False,
            "fallback_used": False,
            "internal_verifier_only": True,
        },
    )
