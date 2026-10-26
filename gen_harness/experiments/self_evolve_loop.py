from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Dict, Sequence

from ..backend import VisualBackend
from ..component_audit import ComponentBoundaryAuditor
from ..inspection import InspectionRunner
from ..patcher import (
    EDITABLE_RESPONSIBILITIES,
    HarnessLocalizer,
    PatchProposer,
    build_harness_localizer,
    build_patch_proposer,
)
from ..io import write_json
from ..prompt_optimizer import PromptOptimizer
from ..repository import HarnessRepository
from ..tasks import load_tasks, split_tasks
from ..telemetry import counting_backend, counting_inspector
from ..validator import PatchValidator
from ..search import SearchBackend
from .slow_loop import SlowLoopEvolutionRunner, _ensure_multiple_validation_seeds, trace_search_strategy
from ..trace_protocol import TRACE_DEFAULT_K, TRACE_PROTOCOL_SCHEMA, validate_paper_dataset, validate_trace_seeds


SELF_EVOLVE_RUN_SCHEMA = "gen_harness.self_evolve_run.v1"


class SelfEvolveRunner:
    """Iterative Gen-Harness self-evolution loop.

    Each round executes the clean slow-loop pipeline:

    explore with fast-loop enabled -> collect failures -> auto-diagnose -> mine
    weaknesses -> propose component-scoped patches -> validate target/heldout/
    preservation gates -> optionally promote accepted patches.

    The runner is benchmark-leakage guarded by construction because it delegates
    task validation to SlowLoopEvolutionRunner, whose default refuses benchmark
    metadata and requires target/heldout/preservation splits.
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
        fast_loop_attempts: int = 5,
        regression_epsilon: float = 0.01,
        harness_localizer_config: str | Path | None = None,
        harness_localizer: HarnessLocalizer | None = None,
        trace_frontier_size: int = 3,
        trace_candidate_budget: int = 3,
        trace_preservation_weight: float = 0.5,
        trace_execution_cost_weight: float = 0.05,
        patch_proposer_config: str | Path | None = None,
        patch_proposer: PatchProposer | None = None,
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
        self.harness_localizer_config = harness_localizer_config
        self.harness_localizer = harness_localizer
        self.trace_frontier_size = min(len(EDITABLE_RESPONSIBILITIES), max(1, int(trace_frontier_size)))
        self.trace_candidate_budget = max(1, int(trace_candidate_budget))
        self.trace_preservation_weight = float(trace_preservation_weight)
        self.trace_execution_cost_weight = float(trace_execution_cost_weight)
        self.patch_proposer_config = patch_proposer_config
        self.patch_proposer = patch_proposer
        self.strict_trace_protocol = bool(strict_trace_protocol)
        self.dataset_protocol = str(dataset_protocol)
        self.allow_trace_seed_ablation = bool(allow_trace_seed_ablation)

    def run(
        self,
        tasks_path: str | Path,
        output_dir: str | Path,
        *,
        seeds: Sequence[int],
        max_rounds: int = 3,
        min_support: int = 2,
        promote_accepted: bool = True,
        write_memory: bool = True,
    ) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        run_started = time.perf_counter()
        generator_calls_before = self.backend.call_count
        verifier_calls_before = self.inspector.call_count
        seed_list = validate_trace_seeds(
            list(seeds) or [0],
            expected_k=TRACE_DEFAULT_K,
            allow_ablation=self.allow_trace_seed_ablation or not self.strict_trace_protocol,
        )
        validate_paper_dataset(load_tasks(tasks_path), protocol=self.dataset_protocol)
        initial_audit = ComponentBoundaryAuditor().audit(self.repo.root, tasks_path=tasks_path, write_path=output / "component_audit_initial.json")
        if not initial_audit.get("passed"):
            report = self._blocked_report(
                tasks_path,
                output,
                stop_reason="initial_component_audit_failed",
                initial_audit=initial_audit,
                final_audit=initial_audit,
                rounds=[],
                seeds=seed_list,
                max_rounds=max_rounds,
                promote_accepted=promote_accepted,
                write_memory=write_memory,
            )
            write_json(output / "self_evolve_run_summary.json", report)
            raise ValueError(f"self-evolve initial component audit failed; see {output / 'component_audit_initial.json'}")
        initial_harness_state = self.repo.snapshot_persistent_state()
        rounds = []
        stop_reason = "max_rounds_reached"
        for round_index in range(max(1, int(max_rounds))):
            round_dir = output / f"round_{round_index + 1:02d}"
            runner = SlowLoopEvolutionRunner(
                self.repo,
                self.backend,
                self.inspector,
                self.search_backend,
                self.prompt_optimizer,
                validation_metric_mode=self.validation_metric_mode,
                fast_loop_attempts=self.fast_loop_attempts,
                regression_epsilon=self.regression_epsilon,
                harness_localizer_config=self.harness_localizer_config,
                harness_localizer=self.harness_localizer,
                trace_frontier_size=self.trace_frontier_size,
                trace_candidate_budget=self.trace_candidate_budget,
                trace_preservation_weight=self.trace_preservation_weight,
                trace_execution_cost_weight=self.trace_execution_cost_weight,
                patch_proposer_config=self.patch_proposer_config,
                patch_proposer=self.patch_proposer,
                strict_trace_protocol=self.strict_trace_protocol,
                dataset_protocol=self.dataset_protocol,
                allow_trace_seed_ablation=self.allow_trace_seed_ablation,
            )
            summary = runner.run(
                tasks_path,
                round_dir,
                seeds=seed_list,
                min_support=min_support,
                promote=promote_accepted,
                write_memory=write_memory,
                allow_benchmark_tasks=False,
            )
            round_record = {
                "round": round_index + 1,
                "depth": round_index + 1,
                "output_dir": str(round_dir),
                "num_experiences": summary.get("num_experiences"),
                "num_failures": summary.get("natural_failures", {}).get("num_failures"),
                "num_weaknesses": summary.get("num_weaknesses"),
                "num_patches": summary.get("num_patches"),
                "num_accepted_patches": summary.get("num_accepted_patches"),
                "num_promoted_patches": summary.get("num_promoted_patches"),
                "self_evolve_gate_summary": summary.get("self_evolve_gate_summary"),
                "evolution_efficiency": summary.get("evolution_efficiency", {}),
                "artifacts": summary.get("artifacts", {}),
            }
            rounds.append(round_record)
            if int(summary.get("natural_failures", {}).get("num_failures", 0) or 0) == 0:
                stop_reason = "no_internal_failures"
                break
            if int(summary.get("num_patches", 0) or 0) == 0:
                stop_reason = "no_patch_proposals"
                break
            if int(summary.get("num_accepted_patches", 0) or 0) == 0:
                stop_reason = "no_accepted_patches"
                break
            if promote_accepted and int(summary.get("num_promoted_patches", 0) or 0) == 0:
                stop_reason = "accepted_patch_not_promoted"
                break

        promoted_count = sum(int(row.get("num_promoted_patches", 0) or 0) for row in rounds)
        final_path_validation = self._validate_complete_path(
            tasks_path,
            seeds=seed_list,
            initial_harness_state=initial_harness_state,
            promoted_count=promoted_count,
        )
        if promoted_count and not final_path_validation["accepted"]:
            self.repo.restore_persistent_state(initial_harness_state)
            final_path_validation["rolled_back"] = True
            stop_reason = "final_path_validation_failed_rolled_back"

        final_audit = ComponentBoundaryAuditor().audit(self.repo.root, tasks_path=tasks_path, write_path=output / "component_audit_final.json")
        if not final_audit.get("passed") and stop_reason == "max_rounds_reached":
            stop_reason = "final_component_audit_failed"
        report = self._blocked_report(
            tasks_path,
            output,
            stop_reason=stop_reason,
            initial_audit=initial_audit,
            final_audit=final_audit,
            rounds=rounds,
            seeds=seed_list,
            max_rounds=max_rounds,
            promote_accepted=promote_accepted,
            write_memory=write_memory,
        )
        report["final_path_validation"] = final_path_validation
        round_efficiency = [
            row.get("evolution_efficiency", {})
            for row in rounds
            if isinstance(row.get("evolution_efficiency"), dict)
        ]
        trigger_count = sum(int(row.get("evolution_trigger_count", 0) or 0) for row in round_efficiency)
        candidate_edits_evaluated = sum(int(row.get("candidate_edits_evaluated", 0) or 0) for row in round_efficiency)
        report["evolution_efficiency"] = {
            "schema": "gen_harness.trace_evolution_efficiency.v1",
            "wall_clock_seconds": round(time.perf_counter() - run_started, 6),
            "generator_calls": self.backend.call_count - generator_calls_before,
            "visual_verifier_calls": self.inspector.call_count - verifier_calls_before,
            "localizer_llm_calls": sum(int(row.get("localizer_llm_calls", 0) or 0) for row in round_efficiency),
            "proposer_llm_calls": sum(int(row.get("proposer_llm_calls", 0) or 0) for row in round_efficiency),
            "llm_vlm_calls": (
                sum(int(row.get("localizer_llm_calls", 0) or 0) for row in round_efficiency)
                + sum(int(row.get("proposer_llm_calls", 0) or 0) for row in round_efficiency)
                + self.inspector.call_count
                - verifier_calls_before
            ),
            "candidate_edits_proposed": sum(int(row.get("candidate_edits_proposed", 0) or 0) for row in round_efficiency),
            "candidate_edits_evaluated": candidate_edits_evaluated,
            "evolution_trigger_count": trigger_count,
            "candidate_edits_per_evolution_task": candidate_edits_evaluated / max(1, trigger_count),
            "completed_rounds": len(rounds),
            "includes_final_path_validation": bool(final_path_validation.get("applicable")),
            "call_count_source": "runtime_wrappers_and_llm_request_counters",
            "estimated": False,
        }
        write_json(output / "self_evolve_run_summary.json", report)
        return report

    def _validate_complete_path(
        self,
        tasks_path: str | Path,
        *,
        seeds: Sequence[int],
        initial_harness_state: Dict[str, Dict[str, Any]],
        promoted_count: int,
    ) -> Dict[str, Any]:
        """Validate the complete progressive path against H_t and roll back as one unit."""
        validation_seeds = _ensure_multiple_validation_seeds(seeds)
        if promoted_count == 0:
            return {
                "schema": "gen_harness.trace_path_validation.v1",
                "accepted": True,
                "applicable": False,
                "reason": "no promoted path",
                "seeds": validation_seeds,
                "rolled_back": False,
            }

        split_rows = split_tasks(load_tasks(tasks_path))
        target = list(split_rows["target"][:3])
        heldout = list(split_rows["heldout"][:3])
        preservation = list(split_rows["preservation"][:2])

        def make_validator(repo: HarnessRepository) -> PatchValidator:
            localizer = self.harness_localizer or build_harness_localizer(
                self.harness_localizer_config,
                repo=repo,
                strict_protocol=self.strict_trace_protocol,
            )
            proposer = self.patch_proposer or build_patch_proposer(
                self.patch_proposer_config,
                memory=repo.build_memory(),
                repo=repo,
                strict_protocol=self.strict_trace_protocol,
            )
            return PatchValidator(
                repo,
                self.backend,
                self.inspector,
                self.search_backend,
                regression_epsilon=self.regression_epsilon,
                prompt_optimizer=self.prompt_optimizer,
                metric_mode=self.validation_metric_mode,
                fast_loop_attempts=self.fast_loop_attempts,
                harness_localizer=localizer,
                patch_proposer=proposer,
                fast_loop_stochastic_k=4,
                preservation_weight=self.trace_preservation_weight,
                execution_cost_weight=self.trace_execution_cost_weight,
            )

        with TemporaryDirectory(prefix="evogen_trace_initial_") as temp_root:
            initial_repo = HarnessRepository.materialize_persistent_state(
                Path(temp_root) / "harness",
                initial_harness_state,
                policy_extractor_config=self.repo.policy_extractor_config,
            )
            before = make_validator(initial_repo).evaluate_current_harness(
                target_tasks=target,
                heldout_tasks=heldout,
                preservation_tasks=preservation,
                seeds=validation_seeds,
            )
        after = make_validator(self.repo).evaluate_current_harness(
            target_tasks=target,
            heldout_tasks=heldout,
            preservation_tasks=preservation,
            seeds=validation_seeds,
        )
        target_delta = float(after["target"]["score"]) - float(before["target"]["score"])
        heldout_delta = float(after["heldout"]["score"]) - float(before["heldout"]["score"])
        regression_delta = float(before["preservation"]["score"]) - float(after["preservation"]["score"])
        accepted = target_delta > 0 and heldout_delta > 0 and regression_delta <= self.regression_epsilon
        return {
            "schema": "gen_harness.trace_path_validation.v1",
            "applicable": True,
            "accepted": accepted,
            "promoted_steps": promoted_count,
            "seeds": validation_seeds,
            "matched_stochastic_conditions": True,
            "before": before,
            "after": after,
            "target_delta": target_delta,
            "heldout_delta": heldout_delta,
            "preservation_regression": regression_delta,
            "regression_epsilon": self.regression_epsilon,
            "rolled_back": False,
        }

    def _blocked_report(
        self,
        tasks_path: str | Path,
        output: Path,
        *,
        stop_reason: str,
        initial_audit: Dict[str, Any],
        final_audit: Dict[str, Any],
        rounds: list[Dict[str, Any]],
        seeds: Sequence[int],
        max_rounds: int,
        promote_accepted: bool,
        write_memory: bool,
    ) -> Dict[str, Any]:
        return {
            "schema": SELF_EVOLVE_RUN_SCHEMA,
            "harness_root": str(self.repo.root),
            "tasks_path": str(tasks_path),
            "max_rounds": max_rounds,
            "seeds": list(seeds),
            "fast_loop_attempts": self.fast_loop_attempts,
            "trace": {
                "protocol_schema": TRACE_PROTOCOL_SCHEMA,
                "strict_trace_protocol": self.strict_trace_protocol,
                "dataset_protocol": self.dataset_protocol,
                "expected_k": len(seeds),
                "depth_semantics": "one SelfEvolveRunner round equals one TRACE depth",
                "search_strategy": trace_search_strategy(self.trace_frontier_size),
                "frontier_size": self.trace_frontier_size,
                "candidate_budget_per_responsibility": self.trace_candidate_budget,
                "preservation_weight": self.trace_preservation_weight,
                "execution_cost_weight": self.trace_execution_cost_weight,
            },
            "policy_extractor_manifest": self.repo.policy_extractor_manifest(),
            "validation_metric_mode": self.validation_metric_mode,
            "trace_protocol": TRACE_PROTOCOL_SCHEMA,
            "strict_trace_protocol": self.strict_trace_protocol,
            "dataset_protocol": self.dataset_protocol,
            "dataset_validation": validate_paper_dataset(
                load_tasks(tasks_path),
                protocol=self.dataset_protocol,
            ),
            "promote_accepted": promote_accepted,
            "write_memory": write_memory,
            "stop_reason": stop_reason,
            "rounds": rounds,
            "reference_trajectory_policy": {
                "official_benchmark_reference_trajectories_allowed": False,
                "allowed_sources": ["non_benchmark_dev_tasks", "internal_baseline_traces", "human_authored_dev_traces"],
                "official_labels_used_for_generation_or_selection": False,
                "official_evaluator_used_for_generation_or_selection": False,
                "benchmark_tasks_allowed": False,
                "leakage_safe": True,
            },
            "leakage_and_fallback_guards": {
                "allow_benchmark_tasks": False,
                "official_evaluator_used_for_generation_or_selection": False,
                "official_labels_used_for_prompt_compilation": False,
                "fallback_image_substitution_allowed": False,
                "initial_component_audit_passed": bool(initial_audit.get("passed")),
                "final_component_audit_passed": bool(final_audit.get("passed")),
            },
            "artifacts": {
                "initial_component_audit": str(output / "component_audit_initial.json"),
                "final_component_audit": str(output / "component_audit_final.json"),
            },
        }
