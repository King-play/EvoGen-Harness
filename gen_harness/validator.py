from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .backend import VisualBackend
from .inspection import InspectionRunner
from .search import SearchBackend
from .prompt_optimizer import PromptOptimizer
from .executor import HarnessExecutor
from .fast_loop import require_full_llm_fast_loop
from .patcher import HarnessLocalizer, PatchProposer
from .repository import HarnessRepository
from .schema import GenerationTask, PatchManifest


DEFAULT_IMAGE_METRICS = (
    "overall_quality",
    "aesthetic_quality",
    "prompt_alignment",
    "artifact_free",
    "text_logo_absence",
)


@dataclass
class ValidationReport:
    patch_id: str
    accepted: bool
    before: Dict[str, float]
    after: Dict[str, float]
    target_delta: float
    heldout_delta: float
    regression_delta: float
    reason: str
    seeds: List[int]
    constraint_breakdown: Dict[str, Dict[str, Dict[str, float]]] | None = None
    prompt_effect: Dict[str, object] | None = None
    ranking_signal: Dict[str, float] | None = None
    validation_scope: Dict[str, object] | None = None
    validation_guards: Dict[str, object] | None = None
    execution_cost_delta: float = 0.0
    utility: float = 0.0
    utility_terms: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "patch_id": self.patch_id,
            "accepted": self.accepted,
            "before": self.before,
            "after": self.after,
            "target_delta": self.target_delta,
            "heldout_delta": self.heldout_delta,
            "regression_delta": self.regression_delta,
            "reason": self.reason,
            "seeds": self.seeds,
            "constraint_breakdown": self.constraint_breakdown or {},
            "prompt_effect": self.prompt_effect or {},
            "ranking_signal": self.ranking_signal or {},
            "validation_scope": self.validation_scope or {},
            "validation_guards": self.validation_guards or {},
            "execution_cost_delta": self.execution_cost_delta,
            "utility": self.utility,
            "utility_terms": self.utility_terms or {},
        }


class PatchValidator:
    def __init__(
        self,
        repo: HarnessRepository,
        backend: VisualBackend,
        inspector: InspectionRunner,
        search_backend: SearchBackend | None = None,
        regression_epsilon: float = 0.01,
        prompt_optimizer: PromptOptimizer | None = None,
        metric_mode: str = "pass_rate",
        image_metrics: Sequence[str] | None = None,
        fast_loop_attempts: int = 5,
        harness_localizer: HarnessLocalizer | None = None,
        patch_proposer: PatchProposer | None = None,
        fast_loop_stochastic_k: int = 4,
        preservation_weight: float = 0.5,
        execution_cost_weight: float = 0.05,
    ):
        self.repo = repo
        self.backend = backend
        self.inspector = inspector
        self.search_backend = search_backend
        self.regression_epsilon = regression_epsilon
        self.prompt_optimizer = prompt_optimizer
        self.metric_mode = metric_mode
        self.image_metrics = tuple(image_metrics or DEFAULT_IMAGE_METRICS)
        self.fast_loop_attempts = int(fast_loop_attempts)
        self.harness_localizer = harness_localizer
        self.patch_proposer = patch_proposer
        self.fast_loop_stochastic_k = int(fast_loop_stochastic_k)
        require_full_llm_fast_loop(
            inspector=self.inspector,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            max_attempts=self.fast_loop_attempts,
            stochastic_evidence_k=self.fast_loop_stochastic_k,
            context="PatchValidator",
        )
        self.preservation_weight = float(preservation_weight)
        self.execution_cost_weight = float(execution_cost_weight)
        self._baseline_cache: Dict[tuple, tuple[float, Dict[str, Dict[str, float]], float]] = {}
        if self.metric_mode not in {"pass_rate", "image_metrics"}:
            raise ValueError(f"unsupported validation metric mode: {self.metric_mode}")

    def evaluate_current_harness(
        self,
        *,
        target_tasks: Iterable[GenerationTask],
        heldout_tasks: Iterable[GenerationTask],
        preservation_tasks: Iterable[GenerationTask],
        seeds: Sequence[int],
    ) -> Dict[str, object]:
        """Evaluate a complete harness state under one matched seed set."""
        seed_list = list(seeds)
        if not seed_list:
            raise ValueError("TRACE harness evaluation requires at least one stochastic condition")
        rows: Dict[str, object] = {"seeds": seed_list, "matched_stochastic_conditions": True}
        for name, task_rows in (
            ("target", list(target_tasks)),
            ("heldout", list(heldout_tasks)),
            ("preservation", list(preservation_tasks)),
        ):
            if not task_rows:
                raise ValueError(f"TRACE harness evaluation requires a non-empty {name} split")
            score, breakdown, cost = self._evaluate_tasks(task_rows, seed_list)
            rows[name] = {"score": score, "breakdown": breakdown, "mean_model_calls": cost}
        return rows

    def validate(
        self,
        patch: PatchManifest,
        target_tasks: Iterable[GenerationTask],
        heldout_tasks: Iterable[GenerationTask],
        preservation_tasks: Iterable[GenerationTask],
        seed: int = 0,
        seeds: Sequence[int] | None = None,
    ) -> ValidationReport:
        target_tasks = list(target_tasks)
        heldout_tasks = list(heldout_tasks)
        preservation_tasks = list(preservation_tasks)
        seed_list = list(seeds) if seeds is not None else [seed]
        if not seed_list:
            seed_list = [seed]

        if not target_tasks:
            return self._rejected_empty(patch, "target", seed_list)
        if not heldout_tasks:
            return self._rejected_empty(patch, "heldout", seed_list)
        if not preservation_tasks:
            return self._rejected_empty(patch, "preservation", seed_list)

        before_target, before_target_breakdown, before_target_cost = self._baseline_evaluate("target", target_tasks, seed_list)
        prompt_effect = self._prompt_effect(patch, target_tasks[0]) if patch.target_component == "skills" else {}
        before_heldout, before_heldout_breakdown, _ = self._baseline_evaluate("heldout", heldout_tasks, seed_list)
        before_preservation, before_preservation_breakdown, _ = self._baseline_evaluate("preservation", preservation_tasks, seed_list)
        before = {
            "target": before_target,
            "heldout": before_heldout,
            "preservation": before_preservation,
        }
        before_breakdown = {
            "target": before_target_breakdown,
            "heldout": before_heldout_breakdown,
            "preservation": before_preservation_breakdown,
        }
        with self.repo.temporary_patch(patch):
            after_target, after_target_breakdown, after_target_cost = self._evaluate_tasks(target_tasks, seed_list)
            after_heldout, after_heldout_breakdown, _ = self._evaluate_tasks(heldout_tasks, seed_list)
            after_preservation, after_preservation_breakdown, _ = self._evaluate_tasks(preservation_tasks, seed_list)
            after = {
                "target": after_target,
                "heldout": after_heldout,
                "preservation": after_preservation,
            }
            after_breakdown = {
                "target": after_target_breakdown,
                "heldout": after_heldout_breakdown,
                "preservation": after_preservation_breakdown,
            }

        target_delta = after["target"] - before["target"]
        heldout_delta = after["heldout"] - before["heldout"]
        regression_delta = before["preservation"] - after["preservation"]
        execution_cost_delta = max(0.0, after_target_cost - before_target_cost)
        preservation_penalty = max(0.0, regression_delta)
        utility = (
            target_delta
            - self.preservation_weight * preservation_penalty
            - self.execution_cost_weight * execution_cost_delta
        )
        # TRACE retains a candidate only when it improves both the triggering
        # and held-out cases, remains within the preservation tolerance, and
        # has positive cost-aware utility (paper Eq. 7).
        accepted = (
            target_delta > 0
            and heldout_delta > 0
            and regression_delta <= self.regression_epsilon
            and utility > 0
        )
        if patch.target_component == "skills" and not bool(prompt_effect.get("changed", False)):
            accepted = False
        if accepted:
            reason = "accepted"
        elif patch.target_component == "skills" and not bool(prompt_effect.get("changed", False)):
            reason = "rejected: patch did not change the compiled backend prompt"
        else:
            reason = (
                "rejected: TRACE requires positive target and held-out gains, "
                "bounded preservation regression, and positive cost-aware utility "
                f"under {self.metric_mode}"
            )
        breakdown = {
            "before": before_breakdown,
            "after": after_breakdown,
            "delta": self._breakdown_delta(before_breakdown, after_breakdown),
        }
        ranking_signal = {
            "before_target_nvila_probability": self._nvila_probability(before_target_breakdown),
            "after_target_nvila_probability": self._nvila_probability(after_target_breakdown),
        }
        ranking_signal["target_nvila_probability_delta"] = (
            ranking_signal["after_target_nvila_probability"] - ranking_signal["before_target_nvila_probability"]
        )
        validation_scope = self._validation_scope_record(
            target_tasks=target_tasks,
            heldout_tasks=heldout_tasks,
            preservation_tasks=preservation_tasks,
            seeds=seed_list,
        )
        validation_guards = self._validation_guard_record(
            target_delta=target_delta,
            heldout_delta=heldout_delta,
            regression_delta=regression_delta,
            prompt_effect=prompt_effect,
            patch=patch,
            utility=utility,
        )
        return ValidationReport(
            patch.patch_id,
            accepted,
            before,
            after,
            target_delta,
            heldout_delta,
            regression_delta,
            reason,
            seed_list,
            breakdown,
            prompt_effect,
            ranking_signal,
            validation_scope,
            validation_guards,
            execution_cost_delta,
            utility,
            {
                "target_delta": target_delta,
                "preservation_penalty": preservation_penalty,
                "preservation_weight": self.preservation_weight,
                "execution_cost_delta": execution_cost_delta,
                "execution_cost_weight": self.execution_cost_weight,
            },
        )

    def _baseline_evaluate(
        self,
        split: str,
        tasks: List[GenerationTask],
        seeds: Sequence[int],
    ) -> tuple[float, Dict[str, Dict[str, float]], float]:
        key = (split, tuple(task.task_id for task in tasks), tuple(seeds), self.metric_mode)
        if key not in self._baseline_cache:
            self._baseline_cache[key] = self._evaluate_tasks(tasks, seeds)
        return self._baseline_cache[key]

    def _prompt_effect(self, patch: PatchManifest, task: GenerationTask) -> Dict[str, object]:
        """Prove a skills patch reaches the compiled prompt before generation."""
        def compile_prompt() -> str:
            executor = HarnessExecutor(
                self.repo,
                self.backend,
                self.inspector,
                self.search_backend,
                prompt_optimizer=self.prompt_optimizer,
                fast_loop_attempts=self.fast_loop_attempts,
                harness_localizer=self.harness_localizer,
                patch_proposer=self.patch_proposer,
                fast_loop_stochastic_k=self.fast_loop_stochastic_k,
            )
            program, _ = executor._compile_program(task, seed=0)
            workflow = program.get("workflow", {})
            return str(workflow.get("canonical_prompt") or workflow.get("compiled_prompt") or "")

        before = compile_prompt()
        with self.repo.temporary_patch(patch):
            after = compile_prompt()
        return {
            "checked": True,
            "task_id": task.task_id,
            "changed": before != after,
            "before_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
            "after_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
            "changed_artifact": patch.changed_artifact,
            "changed_fields": patch.changed_fields,
        }

    def promote_if_accepted(self, patch: PatchManifest, report: ValidationReport) -> None:
        if not self._promotion_gate_satisfied(patch, report):
            raise ValueError(f"patch {patch.patch_id} cannot be promoted: incomplete or failed validation gate")
        self.repo.apply_patch(patch)

    def _promotion_gate_satisfied(self, patch: PatchManifest, report: ValidationReport) -> bool:
        if report.patch_id != patch.patch_id or not bool(report.accepted):
            return False
        if not report.seeds:
            return False
        scope = report.validation_scope if isinstance(report.validation_scope, dict) else {}
        guards = report.validation_guards if isinstance(report.validation_guards, dict) else {}
        if scope.get("target_heldout_preservation_present") is not True:
            return False
        if guards.get("target_improved") is not True:
            return False
        if guards.get("heldout_improved") is not True:
            return False
        if guards.get("heldout_non_regressing") is not True:
            return False
        if guards.get("preservation_regression_bounded") is not True:
            return False
        if guards.get("positive_utility") is not True:
            return False
        if guards.get("matched_stochastic_conditions") is not True:
            return False
        if guards.get("official_evaluator_used") is not False:
            return False
        if guards.get("official_scores_used") is not False:
            return False
        if guards.get("fallback_used") is not False:
            return False
        required_splits = {patch.target_validation_set, patch.held_out_set, patch.preservation_set}
        if not required_splits <= set(report.before) or not required_splits <= set(report.after):
            return False
        if report.target_delta <= 0 or report.heldout_delta <= 0 or report.utility <= 0:
            return False
        if report.regression_delta > self.regression_epsilon:
            return False
        if patch.target_component == "skills":
            effect = report.prompt_effect if isinstance(report.prompt_effect, dict) else {}
            if not bool(effect.get("changed", False)):
                return False
        return True

    def _validation_scope_record(
        self,
        *,
        target_tasks: Sequence[GenerationTask],
        heldout_tasks: Sequence[GenerationTask],
        preservation_tasks: Sequence[GenerationTask],
        seeds: Sequence[int],
    ) -> Dict[str, object]:
        split_counts = {
            "target": len(target_tasks),
            "heldout": len(heldout_tasks),
            "preservation": len(preservation_tasks),
        }
        return {
            "schema": "gen_harness.validation_scope.v1",
            "target_heldout_preservation_present": all(value > 0 for value in split_counts.values()),
            "split_counts": split_counts,
            "seeds": list(seeds),
            "metric_mode": self.metric_mode,
            "regression_epsilon": self.regression_epsilon,
        }

    def _validation_guard_record(
        self,
        *,
        target_delta: float,
        heldout_delta: float,
        regression_delta: float,
        prompt_effect: Dict[str, object],
        patch: PatchManifest,
        utility: float,
    ) -> Dict[str, object]:
        prompt_effect_required = patch.target_component == "skills"
        prompt_effect_satisfied = not prompt_effect_required or bool(prompt_effect.get("changed", False))
        return {
            "schema": "gen_harness.validation_guards.v1",
            "target_improved": target_delta > 0,
            "heldout_improved": heldout_delta > 0,
            "heldout_non_regressing": heldout_delta >= 0,
            "preservation_regression_bounded": regression_delta <= self.regression_epsilon,
            "positive_utility": utility > 0,
            "matched_stochastic_conditions": True,
            "skills_prompt_effect_required": prompt_effect_required,
            "skills_prompt_effect_satisfied": prompt_effect_satisfied,
            "official_evaluator_used": False,
            "official_scores_used": False,
            "fallback_used": False,
            "internal_verifier_only": True,
        }

    def _score_tasks(self, tasks: List[GenerationTask], seeds: Sequence[int]) -> float:
        score, _, _ = self._evaluate_tasks(tasks, seeds)
        return score

    def _run_experiences(self, tasks: List[GenerationTask], seeds: Sequence[int]):
        executor = HarnessExecutor(
            self.repo,
            self.backend,
            self.inspector,
            self.search_backend,
            prompt_optimizer=self.prompt_optimizer,
            fast_loop_attempts=self.fast_loop_attempts,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            fast_loop_stochastic_k=self.fast_loop_stochastic_k,
        )
        for seed in seeds:
            for task in tasks:
                yield from executor.run_task(task, seed=seed)

    def _pass_rate(self, tasks: List[GenerationTask], seeds: Sequence[int]) -> float:
        if not tasks:
            return 1.0
        total = 0
        passed = 0
        for exp in self._run_experiences(tasks, seeds):
            total += 1
            passed += int(bool(exp.verification_result.get("passed")))
        return passed / total if total else 1.0

    def _image_metric_score(self, tasks: List[GenerationTask], seeds: Sequence[int]) -> float:
        if not tasks:
            return 1.0
        groups: Dict[tuple[str, int], Dict[str, float]] = {}
        for exp in self._run_experiences(tasks, seeds):
            seed = int(exp.visual_result.get("seed", 0) or 0)
            key = (exp.task_id, seed)
            result = exp.verification_result or {}
            metrics = result.get("image_metrics", {}) if isinstance(result, dict) else {}
            if not isinstance(metrics, dict):
                metrics = {}
            merged = groups.setdefault(key, {})
            for name in self.image_metrics:
                value = metrics.get(name, result.get(name))
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                merged[name] = min(1.0, max(0.0, float(value)))
        image_scores = []
        for metrics in groups.values():
            values = [metrics[name] for name in self.image_metrics if name in metrics]
            if values:
                image_scores.append(sum(values) / len(values))
        return sum(image_scores) / len(image_scores) if image_scores else 0.0


    def _evaluate_tasks(self, tasks: List[GenerationTask], seeds: Sequence[int]) -> tuple[float, Dict[str, Dict[str, float]], float]:
        if not tasks:
            return 1.0, {}, 0.0
        total = 0
        passed = 0
        breakdown_acc: Dict[str, Dict[str, float]] = {}
        image_metric_groups: Dict[tuple[str, int], Dict[str, float]] = {}
        calls_by_execution: Dict[tuple[str, int], float] = {}

        for exp in self._run_experiences(tasks, seeds):
            total += 1
            exp_passed = bool(exp.verification_result.get("passed"))
            passed += 1 if exp_passed else 0
            row = breakdown_acc.setdefault(exp.constraint_type, {"total": 0.0, "passed": 0.0})
            row["total"] += 1.0
            row["passed"] += 1.0 if exp_passed else 0.0
            result = exp.verification_result or {}
            execution_key = (exp.task_id, int(exp.visual_result.get("seed", 0) or 0))
            fast_loop = exp.metadata.get("fast_loop", {}) if isinstance(exp.metadata, dict) else {}
            attempts = fast_loop.get("attempts", []) if isinstance(fast_loop, dict) else []
            # One generation and one internal verification per attempt.  Use
            # the maximum for duplicate constraint-indexed rows belonging to
            # the same task/seed execution.
            attempt_count = max(1, len(attempts) if isinstance(attempts, list) else 1)
            calls_by_execution[execution_key] = max(calls_by_execution.get(execution_key, 0.0), float(2 * attempt_count))
            source = str(result.get("inspection_source", "")).lower()
            if "nvila" in source:
                probability = result.get("yes_probability_min", result.get("selection_score"))
                if isinstance(probability, (int, float)) and not isinstance(probability, bool):
                    row["nvila_probability_sum"] = row.get("nvila_probability_sum", 0.0) + float(probability)
                    row["nvila_probability_n"] = row.get("nvila_probability_n", 0.0) + 1.0

            if self.metric_mode == "image_metrics":
                seed = int(exp.visual_result.get("seed", 0) or 0)
                key = (exp.task_id, seed)
                result = exp.verification_result or {}
                metrics = result.get("image_metrics", {}) if isinstance(result, dict) else {}
                if not isinstance(metrics, dict):
                    metrics = {}
                merged = image_metric_groups.setdefault(key, {})
                for name in self.image_metrics:
                    value = metrics.get(name, result.get(name))
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    merged[name] = min(1.0, max(0.0, float(value)))

        breakdown = {}
        for ctype, row in breakdown_acc.items():
            row_total = row["total"] or 1.0
            breakdown[ctype] = {"pass_rate": row["passed"] / row_total, "n": row["total"]}
            if row.get("nvila_probability_n", 0.0):
                breakdown[ctype]["mean_nvila_probability"] = row["nvila_probability_sum"] / row["nvila_probability_n"]

        if self.metric_mode == "pass_rate":
            runtime_cost = sum(calls_by_execution.values()) / max(1, len(calls_by_execution))
            return (passed / total if total else 1.0), breakdown, runtime_cost

        image_scores = []
        for metrics in image_metric_groups.values():
            values = [metrics[name] for name in self.image_metrics if name in metrics]
            if values:
                image_scores.append(sum(values) / len(values))
        runtime_cost = sum(calls_by_execution.values()) / max(1, len(calls_by_execution))
        return (sum(image_scores) / len(image_scores) if image_scores else 0.0), breakdown, runtime_cost

    def _nvila_probability(self, breakdown: Dict[str, Dict[str, float]]) -> float:
        values = [
            float(row["mean_nvila_probability"])
            for row in breakdown.values()
            if isinstance(row, dict) and isinstance(row.get("mean_nvila_probability"), (int, float))
        ]
        return sum(values) / len(values) if values else 0.0


    def _constraint_breakdown(self, tasks: List[GenerationTask], seeds: Sequence[int]) -> Dict[str, Dict[str, float]]:
        _, breakdown, _ = self._evaluate_tasks(tasks, seeds)
        return breakdown

    def _breakdown_delta(
        self,
        before: Dict[str, Dict[str, Dict[str, float]]],
        after: Dict[str, Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for split in sorted(set(before) | set(after)):
            out[split] = {}
            ctypes = set(before.get(split, {})) | set(after.get(split, {}))
            for ctype in sorted(ctypes):
                b = before.get(split, {}).get(ctype, {}).get("pass_rate", 0.0)
                a = after.get(split, {}).get(ctype, {}).get("pass_rate", 0.0)
                out[split][ctype] = {"pass_rate_delta": a - b}
        return out

    def _rejected_empty(self, patch: PatchManifest, split: str, seeds: List[int]) -> ValidationReport:
        reason = f"rejected: {split} validation split is empty"
        before = {"target": 0.0, "heldout": 0.0, "preservation": 1.0}
        after = dict(before)
        validation_scope = {
            "schema": "gen_harness.validation_scope.v1",
            "target_heldout_preservation_present": False,
            "missing_split": split,
            "seeds": list(seeds),
            "metric_mode": self.metric_mode,
            "regression_epsilon": self.regression_epsilon,
        }
        validation_guards = {
            "schema": "gen_harness.validation_guards.v1",
            "target_improved": False,
            "heldout_improved": False,
            "heldout_non_regressing": False,
            "preservation_regression_bounded": False,
            "positive_utility": False,
            "matched_stochastic_conditions": True,
            "skills_prompt_effect_required": patch.target_component == "skills",
            "skills_prompt_effect_satisfied": False,
            "official_evaluator_used": False,
            "official_scores_used": False,
            "fallback_used": False,
            "internal_verifier_only": True,
        }
        return ValidationReport(
            patch.patch_id,
            False,
            before,
            after,
            0.0,
            0.0,
            0.0,
            reason,
            seeds,
            validation_scope=validation_scope,
            validation_guards=validation_guards,
        )
