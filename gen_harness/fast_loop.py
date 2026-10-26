from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .backend import VisualBackend
from .inspection import InspectionRunner
from .patcher import GenerationPatch, HarnessLocalization, HarnessLocalizer, PatchProposer
from .schema import GenerationTask, Weakness
from .spatial_prompt import prompt_provenance
from .trace_protocol import TRACE_DEFAULT_K, TRACE_EVIDENCE_SCHEMA, validate_trace_evidence


def require_full_llm_fast_loop(
    *,
    inspector: InspectionRunner | None,
    harness_localizer: HarnessLocalizer | None,
    patch_proposer: PatchProposer | None,
    max_attempts: int,
    stochastic_evidence_k: int,
    context: str,
) -> None:
    """Reject every execution path that bypasses the serial LLM fast loop."""
    if inspector is None:
        raise ValueError(f"{context} requires a real internal inspector")
    if harness_localizer is None or patch_proposer is None:
        raise ValueError(f"{context} requires both an LLM HarnessLocalizer and PatchProposer")
    if int(stochastic_evidence_k) < 1:
        raise ValueError(f"{context} requires stochastic_evidence_k >= 1")
    if int(max_attempts) <= int(stochastic_evidence_k):
        raise ValueError(
            f"{context} requires fast_loop_attempts > stochastic_evidence_k "
            "so a post-evidence LLM repair candidate can be generated"
        )


@dataclass
class FastLoopResult:
    generation: Dict[str, Any]
    inspection: Dict[str, Dict[str, Any]]
    attempts: List[Dict[str, Any]]
    passed: bool
    strictly_confirmed: bool
    selected_attempt: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "inspection": self.inspection,
            "attempts": self.attempts,
            "passed": self.passed,
            "strictly_confirmed": self.strictly_confirmed,
            "selected_attempt": self.selected_attempt,
        }


class FastLoopController:
    """Single-task Gen-Harness repair loop driven by internal inspection.

    This controller is intentionally evaluator-agnostic: it never calls a
    benchmark scorer. It retries only from contract-level inspection results
    produced by Gen-Harness inspectors.
    """

    def __init__(
        self,
        backend: VisualBackend,
        inspector: InspectionRunner,
        *,
        max_attempts: int = TRACE_DEFAULT_K + 1,
        harness_localizer: HarnessLocalizer | None = None,
        patch_proposer: PatchProposer | None = None,
        stochastic_evidence_k: int = TRACE_DEFAULT_K,
    ):
        self.backend = backend
        self.inspector = inspector
        self.max_attempts = int(max_attempts)
        self.harness_localizer = harness_localizer
        self.patch_proposer = patch_proposer
        self.stochastic_evidence_k = int(stochastic_evidence_k)
        require_full_llm_fast_loop(
            inspector=inspector,
            harness_localizer=harness_localizer,
            patch_proposer=patch_proposer,
            max_attempts=self.max_attempts,
            stochastic_evidence_k=self.stochastic_evidence_k,
            context="FastLoopController",
        )

    def run(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> FastLoopResult:
        # The bounded fast loop owns transient resampling/repair candidates.
        # The slow loop alone owns persistent patch proposals and promotion.
        return self._run_repair_loop(task, program, seed=seed)

    def _run_repair_loop(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0, attempt_offset: int = 0) -> FastLoopResult:
        attempts: List[Dict[str, Any]] = []
        best_index = -1
        best_score = -1.0
        best_generation: Dict[str, Any] = {}
        best_inspection: Dict[str, Dict[str, Any]] = {}
        active_program = copy.deepcopy(program)
        evidence_attempts: List[Dict[str, Any]] = []

        for local_attempt_index in range(self.max_attempts - attempt_offset):
            attempt_index = attempt_offset + local_attempt_index
            # Retry candidates use deterministic distinct latents; selection is
            # still based only on internal inspection, never official labels.
            attempt_seed = int(seed) + attempt_index
            attempt_started = time.perf_counter()
            generation_latency = 0.0
            inspection_latency = 0.0
            failures: List[Dict[str, Any]] = []
            probe_soft_abstention = False
            try:
                generation_started = time.perf_counter()
                generation = self.backend.generate(task, active_program, seed=attempt_seed)
                generation_latency = time.perf_counter() - generation_started
                inspection_started = time.perf_counter()
                inspection = self.inspector.inspect(task, active_program, generation, seed=attempt_seed)
                inspection_latency = time.perf_counter() - inspection_started
                passed = self._passed(inspection)
                strictly_confirmed = self._strictly_confirmed(inspection)
                score = self._score(inspection)
                failures = self._failures_from_inspection(inspection)
                # Every non-strict result remains in the K-level evidence
                # window. The LLM Localizer is the only attribution path.
                probe_soft_abstention = passed and not strictly_confirmed
                repair_decision = {
                    "action": "collect_stochastic_evidence",
                    "reason": (
                        f"collect constraint-indexed K={self.stochastic_evidence_k} evidence "
                        "before LLM localization"
                    ),
                    "slow_loop_components": [],
                }
                attempt = {
                    "attempt": attempt_index,
                    "seed": attempt_seed,
                    "passed": passed,
                    "strictly_confirmed": strictly_confirmed,
                    "score": score,
                    "prompt": self._prompt_for_program(active_program),
                    "program": copy.deepcopy(active_program),
                    "generation": generation,
                    "inspection": inspection,
                    "latency": self._latency_record(generation_latency, inspection_latency, attempt_started),
                    "repair_action": active_program.get("fast_loop_repair"),
                    "repair_decision": repair_decision,
                    "failure_family": self._failure_family(failures),
                }
            except Exception as exc:
                generation = {}
                inspection = {}
                passed = False
                strictly_confirmed = False
                score = -1.0
                attempt = {
                    "attempt": attempt_index,
                    "seed": attempt_seed,
                    "passed": False,
                    "strictly_confirmed": False,
                    "score": score,
                    "prompt": self._prompt_for_program(active_program),
                    "program": copy.deepcopy(active_program),
                    "generation": {},
                    "inspection": {},
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "latency": self._latency_record(generation_latency, inspection_latency, attempt_started),
                    "repair_action": active_program.get("fast_loop_repair"),
                    "repair_decision": {
                        "action": "generation_error",
                        "reason": "backend_or_inspection_exception",
                    },
                    "failure_family": "tools",
                }
            repair_meta = active_program.get("fast_loop_repair") if isinstance(active_program, dict) else None
            if isinstance(repair_meta, dict):
                attempt["repair_outcome"] = {
                    "status": "applied_and_verified" if passed else "applied_failed",
                    "repair_type": repair_meta.get("repair_type", "prompt_repair"),
                    "failure_family": repair_meta.get("failure_family"),
                    "prompt_changed": bool(repair_meta.get("prompt_changed", False)),
                    "verified_pass": bool(passed),
                }
            elif passed:
                attempt["repair_outcome"] = {"status": "not_needed", "verified_pass": True}
            else:
                attempt["repair_outcome"] = {
                    "status": "repair_pending",
                    "repair_decision": attempt.get("repair_decision", {}),
                    "verified_pass": False,
                }
            workflow = active_program.get("workflow", {}) if isinstance(active_program, dict) else {}
            attempt["prompt_channels"] = {
                "canonical_prompt": workflow.get("canonical_prompt", attempt.get("prompt", "")),
                "clip_prompt": workflow.get("clip_prompt", ""),
                "t5_prompt": workflow.get("t5_prompt", ""),
            }
            attempts.append(attempt)
            evidence_attempts.append(attempt)
            evidence_complete = len(evidence_attempts) >= self.stochastic_evidence_k
            evidence_failures: List[Dict[str, Any]] = []
            for evidence_attempt in evidence_attempts:
                evidence_inspection = evidence_attempt.get("inspection", {})
                if isinstance(evidence_inspection, dict):
                    attempt_failures = self._failures_from_inspection(evidence_inspection)
                    if not attempt_failures and not bool(evidence_attempt.get("strictly_confirmed", False)):
                        attempt_failures = self._strict_failures_from_inspection(evidence_inspection)
                    evidence_failures.extend(attempt_failures)
            if inspection and not evidence_complete:
                attempt["repair_decision"] = {
                    "action": "collect_stochastic_evidence",
                    "reason": (
                        f"collect constraint-indexed K={self.stochastic_evidence_k} evidence "
                        "before LLM localization"
                    ),
                    "slow_loop_components": [],
                }
            elif inspection and evidence_complete and evidence_failures:
                localization = self._localize_fast_loop_failure(
                    task,
                    active_program,
                    evidence_attempts,
                    evidence_failures,
                )
                attempt["llm_localization"] = localization.to_dict()
                if localization.no_patch_preferred or not localization.frontier:
                    attempt["repair_decision"] = {
                        "action": "stop_no_patch",
                        "reason": "TRACE Localizer preferred NO_PATCH for the supplied evidence",
                        "slow_loop_components": [],
                    }
                else:
                    generation_patch = self.patch_proposer.propose_generation_patch(
                        self._fast_loop_weakness(
                            task,
                            active_program,
                            evidence_attempts,
                            evidence_failures,
                        ),
                        localization,
                        active_program,
                    )
                    active_program = self._apply_generation_patch(active_program, generation_patch)
                    evidence_attempts = []
                    attempt["llm_patch"] = generation_patch.to_dict()
                    attempt["repair_action"] = active_program.get("fast_loop_repair")
                    attempt["repair_decision"] = {
                        "action": "retry_with_llm_repair",
                        "reason": "serial Localizer -> Proposer -> Apply transition",
                        "slow_loop_components": list(localization.frontier),
                        "localized_frontier": list(localization.frontier),
                    }
            if self._attempt_is_better(passed, score, inspection, best_score, best_inspection):
                best_index = attempt_index
                best_score = score
                best_generation = generation
                best_inspection = inspection
            if len(evidence_attempts) < self.stochastic_evidence_k:
                continue
            if passed and not probe_soft_abstention and not evidence_failures:
                break
            retry_action = attempt.get("repair_decision", {}).get("action")
            if retry_action not in {"collect_stochastic_evidence", "retry_with_llm_repair"}:
                break
            if attempt_index + 1 < self.max_attempts:
                # The serial LLM path either keeps the current immutable
                # program while collecting K evidence or applies its
                # validated patch above.
                continue

        selected = best_index if best_index >= 0 else len(attempts) - 1
        if best_index < 0:
            errors = [str(attempt.get("error")) for attempt in attempts if attempt.get("error")]
            detail = "; ".join(errors) if errors else "no candidate produced usable generation evidence"
            raise RuntimeError(f"all fast-loop candidates failed for task {task.task_id}: {detail}")
        return FastLoopResult(
            generation=best_generation,
            inspection=best_inspection,
            attempts=attempts,
            passed=self._passed(best_inspection),
            strictly_confirmed=self._strictly_confirmed(best_inspection),
            selected_attempt=selected,
        )

    def _fast_loop_weakness(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        attempts: Sequence[Dict[str, Any]],
        failures: Sequence[Dict[str, Any]],
    ) -> Weakness:
        symptoms = sorted({str(row.get("symptom", "unknown")) for row in failures if isinstance(row, dict)})
        constraint_type = "unknown"
        contract = program.get("contract", {}) if isinstance(program, dict) else {}
        contract_rows = contract.get("constraints", []) if isinstance(contract, dict) else []
        if isinstance(contract_rows, list):
            by_id = {
                str(row.get("constraint_id")): str(row.get("constraint_type", "unknown"))
                for row in contract_rows
                if isinstance(row, dict)
            }
            for failure in failures:
                candidate = by_id.get(str(failure.get("constraint_id")))
                if candidate:
                    constraint_type = candidate
                    break
        trace_evidence = self._trace_evidence(program, attempts)
        return Weakness(
            weakness_id=f"fast_loop_{task.task_id}",
            suspected_component="unknown",
            constraint_type=constraint_type,
            failure_symptom="|".join(symptoms) or "unknown",
            support_count=len(failures),
            support_experience_ids=[],
            hypothesis="Internal verifier evidence requires responsibility-conditioned transient repair.",
            evidence={"trace_evidence": trace_evidence},
        )

    def _localize_fast_loop_failure(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        attempts: Sequence[Dict[str, Any]],
        failures: Sequence[Dict[str, Any]],
    ) -> HarnessLocalization:
        weakness = self._fast_loop_weakness(task, program, attempts, failures)
        return self.harness_localizer.localize(weakness, frontier_size=3)

    def _trace_evidence(
        self,
        program: Dict[str, Any],
        attempts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        contract = program.get("contract", {}) if isinstance(program, dict) else {}
        contract_rows = contract.get("constraints", []) if isinstance(contract, dict) else []
        constraint_rows = {
            str(row.get("constraint_id")): row
            for row in contract_rows
            if isinstance(row, dict) and row.get("constraint_id")
        }
        all_ids = set(constraint_rows)
        for attempt in attempts:
            inspection = attempt.get("inspection", {})
            if isinstance(inspection, dict):
                all_ids.update(str(key) for key in inspection)
        instances = []
        for instance_index, constraint_id in enumerate(sorted(all_ids)):
            source = constraint_rows.get(constraint_id, {})
            executions = []
            for execution_index, attempt in enumerate(attempts):
                inspection = attempt.get("inspection", {})
                row = inspection.get(constraint_id) if isinstance(inspection, dict) else None
                if not isinstance(row, dict):
                    continue
                passed = bool(row.get("passed", False))
                generation = attempt.get("generation", {})
                workflow = attempt.get("program", {}).get("workflow", {})
                tool = workflow.get("tool", {}) if isinstance(workflow, dict) else {}
                middleware_decisions = (
                    attempt.get("program", {}).get("middleware_decisions", {})
                    if isinstance(attempt.get("program"), dict)
                    else {}
                )
                executions.append(
                    {
                        "execution_index": execution_index,
                        "seed": int(attempt.get("seed", execution_index)),
                        "constraint": {
                            "constraint_id": constraint_id,
                            "constraint_type": str(source.get("constraint_type", "unknown")),
                            "text": str(source.get("text", constraint_id)),
                        },
                        "trajectory_decisions": {
                            "component_decisions": middleware_decisions,
                            "selected_skill": workflow.get("skill_name"),
                            "selected_prompt_family": workflow.get("selected_prompt_family"),
                            "memory_retrieval": workflow.get("memory_retrieval", {}),
                            "repair_action": attempt.get("repair_action"),
                        },
                        "tool_calls": [
                            {
                                "tool_name": tool.get("name") or generation.get("backend"),
                                "tool_type": tool.get("type", "image_generation_backend"),
                                "generation_parameters": copy.deepcopy(
                                    generation.get("generation_parameters", tool.get("generation_parameters", {}))
                                ),
                                "selected_model": generation.get("selected_model"),
                            }
                        ],
                        "visual_observation": {
                            "seed": attempt.get("seed"),
                            "image_uri_present": bool(generation.get("image_uri")),
                            "backend": generation.get("backend"),
                            "prompt_channels": copy.deepcopy(attempt.get("prompt_channels", {})),
                        },
                        "visual_verification": copy.deepcopy(row),
                        "outcome": "pass" if passed else "failure",
                        "failure_symptom": None if passed else str(row.get("symptom", "unknown")),
                    }
                )
            if executions:
                conditions = sorted({row["seed"] for row in executions})
                instances.append(
                    {
                        "instance_index": instance_index,
                        "constraint": {
                            "constraint_id": constraint_id,
                            "constraint_type": str(source.get("constraint_type", "unknown")),
                            "text": str(source.get("text", constraint_id)),
                        },
                        "executions": executions,
                        "stochastic_conditions": conditions,
                        "expected_k": self.stochastic_evidence_k,
                        "complete_for_expected_k": len(conditions) == self.stochastic_evidence_k,
                    }
                )
        if not instances:
            raise ValueError("fast-loop Localizer requires at least one constraint-indexed execution")
        evidence = {
            "schema": TRACE_EVIDENCE_SCHEMA,
            "constraint": {
                "constraint_ids": sorted(all_ids),
                "constraint_types": sorted({row["constraint"]["constraint_type"] for row in instances}),
                "texts": sorted({row["constraint"]["text"] for row in instances}),
                "failure_symptoms": sorted(
                    {
                        str(execution["failure_symptom"])
                        for instance in instances
                        for execution in instance["executions"]
                        if execution["failure_symptom"]
                    }
                ),
            },
            "constraint_instances": instances,
            "aggregation": {
                "observed_execution_count": sum(len(row["executions"]) for row in instances),
                "distinct_stochastic_condition_count": len(
                    {
                        execution["seed"]
                        for instance in instances
                        for execution in instance["executions"]
                    }
                ),
                "stochastic_conditions": sorted(
                    {
                        execution["seed"]
                        for instance in instances
                        for execution in instance["executions"]
                    }
                ),
                "expected_k": self.stochastic_evidence_k,
                "exploration_complete_for_expected_k": all(
                    row["complete_for_expected_k"] for row in instances
                ),
                "failure_support_complete_for_all_k": all(
                    len(
                        {
                            execution["seed"]
                            for execution in row["executions"]
                            if execution["outcome"] == "failure"
                        }
                    )
                    == self.stochastic_evidence_k
                    for row in instances
                ),
            },
        }
        validate_trace_evidence(evidence, expected_k=self.stochastic_evidence_k, allow_incomplete=False)
        return evidence

    def _apply_generation_patch(
        self,
        program: Dict[str, Any],
        patch: GenerationPatch,
    ) -> Dict[str, Any]:
        patched = copy.deepcopy(program)
        path = tuple(patch.path)
        if patch.operation == "append_prompt_clause":
            current = str(patched["workflow"].get("t5_prompt") or "").strip()
            clause = " ".join(str(patch.value).strip().split())
            next_prompt = f"{current} {clause}".strip()
            patched["workflow"]["t5_prompt"] = next_prompt
            segments = patched["workflow"].get("t5_prompt_segments")
            segment_types = patched["workflow"].get("t5_prompt_segment_types")
            if isinstance(segments, list) and isinstance(segment_types, list):
                segments.append(clause)
                segment_types.append("llm_repair")
            else:
                patched["workflow"]["t5_prompt_segments"] = [next_prompt]
                patched["workflow"]["t5_prompt_segment_types"] = ["llm_repair"]
            prompt_provenance(patched["workflow"])
        else:
            node: Any = patched
            for part in path[:-1]:
                if not isinstance(node, dict) or part not in node:
                    raise ValueError(f"generation patch path does not exist: {'.'.join(path)}")
                node = node[part]
            if not isinstance(node, dict):
                raise ValueError(f"generation patch parent is not an object: {'.'.join(path)}")
            node[path[-1]] = patch.value
        patched["fast_loop_repair"] = {
            "source": "llm_localizer_proposer",
            "repair_type": patch.operation,
            "target_component": patch.target_component,
            "patch": patch.to_dict(),
            "prompt_changed": path[0] == "workflow",
            "official_labels_used": False,
            "repository_mutated": False,
            "image_generator_mutated": False,
        }
        return patched

    def _prompt_for_program(self, program: Dict[str, Any]) -> str:
        workflow = program.get("workflow", {}) if isinstance(program, dict) else {}
        return str(workflow.get("compiled_prompt") or "")

    def _latency_record(self, generation_latency: float, inspection_latency: float, attempt_started: float) -> Dict[str, Any]:
        inspection_by_source = dict(getattr(self.inspector, "last_latency_by_source", {}) or {})
        if not inspection_by_source and inspection_latency > 0:
            inspection_by_source[self._inspector_source()] = round(inspection_latency, 6)
        return {
            "generation_seconds": round(generation_latency, 6),
            "inspection_seconds": round(inspection_latency, 6),
            "attempt_seconds": round(time.perf_counter() - attempt_started, 6),
            "inspection_by_source_seconds": inspection_by_source,
        }

    def _inspector_source(self) -> str:
        config = getattr(self.inspector, "config", None)
        if isinstance(config, dict) and config.get("name"):
            return str(config["name"])
        return self.inspector.__class__.__name__

    def _failures_from_inspection(self, inspection: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        failures = []
        for cid, row in inspection.items():
            if isinstance(row, dict) and not self._row_effectively_passed(row):
                failures.append(self._actionable_failure_view(cid, row))
        return failures

    def _strict_failures_from_inspection(self, inspection: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return all non-confirmed rows, including soft semantic abstentions."""
        return [
            self._actionable_failure_view(cid, row)
            for cid, row in inspection.items()
            if isinstance(row, dict) and not bool(row.get("passed", False))
        ]

    def _actionable_failure_view(self, constraint_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """Lift nested verifier diagnostics without changing their provenance."""
        merged = {"constraint_id": constraint_id, **row}
        subresults = row.get("inspection_subresults", {})
        if not isinstance(subresults, dict):
            return merged
        ignored_sources = {
            str(source) for source in row.get("ignored_for_merge_sources", [])
        } if isinstance(row.get("ignored_for_merge_sources"), list) else set()
        list_fields = (
            "missing_required_objects",
            "count_mismatches",
            "forbidden_objects_present",
            "attribute_binding_mismatches",
            "attribute_binding_unknowns",
            "spatial_relation_mismatches",
            "spatial_relation_unknowns",
            "semantic_checks",
            "failure_subtypes",
        )
        for field in list_fields:
            lifted: List[Any] = list(merged.get(field, [])) if isinstance(merged.get(field), list) else []
            for source, subresult in subresults.items():
                # A diagnostic verifier that does not own this capability must
                # not steer repair.  Its evidence remains in the trace for audit.
                if str(source) in ignored_sources:
                    continue
                values = subresult.get(field, []) if isinstance(subresult, dict) else []
                for value in values if isinstance(values, list) else []:
                    if value not in lifted:
                        lifted.append(value)
            if lifted:
                merged[field] = lifted
        return merged


    def _passed(self, inspection: Dict[str, Dict[str, Any]]) -> bool:
        return bool(inspection) and all(self._row_effectively_passed(row) for row in inspection.values())

    def _strictly_confirmed(self, inspection: Dict[str, Dict[str, Any]]) -> bool:
        return bool(inspection) and all(
            isinstance(row, dict) and bool(row.get("passed", False))
            for row in inspection.values()
        )

    def _row_effectively_passed(self, row: Dict[str, Any]) -> bool:
        if not isinstance(row, dict):
            return False
        soft_abstention = (
            str(row.get("evidence_status", "")).strip().lower() == "unknown"
            and row.get("hard_evidence_required") is False
            and str(row.get("symptom", "")).strip().lower() == "missing_inspection"
        )
        if not bool(row.get("passed", False)) and not soft_abstention:
            return False
        if bool(row.get("not_applicable", False)):
            return True
        for key in (
            "missing_required_objects",
            "count_mismatches",
            "forbidden_objects_present",
            "spatial_relation_mismatches",
            "attribute_binding_mismatches",
        ):
            value = row.get(key)
            if isinstance(value, list) and value:
                return False
        if soft_abstention:
            return True
        explicit_score = self._numeric_constraint_score(row)
        if explicit_score is None:
            return True
        return explicit_score >= 0.50

    def _score(self, inspection: Dict[str, Dict[str, Any]]) -> float:
        if not inspection:
            return -1.0
        values = []
        for row in inspection.values():
            if not isinstance(row, dict):
                continue
            explicit_score = self._combined_constraint_score(row)
            if explicit_score is not None:
                values.append(explicit_score)
                continue
            passed = 1.0 if bool(row.get("passed", False)) else 0.0
            confidence = row.get("confidence", 1.0)
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                confidence = 1.0
            values.append(passed * float(confidence))
        return sum(values) / len(values) if values else -1.0

    def _attempt_is_better(
        self,
        passed: bool,
        score: float,
        inspection: Dict[str, Dict[str, Any]],
        best_score: float,
        best_inspection: Dict[str, Dict[str, Any]],
    ) -> bool:
        # A backend/inspection exception records an empty inspection.  Such a
        # candidate has no evidence at all; treating its empty diagnostic
        # lists as zero omissions/failures would let it erase a previously
        # generated and inspected image.
        if not inspection:
            return False
        # Establish a real inspected baseline before comparing diagnostics.
        # An empty sentinel has no hard omissions only because it has no
        # evidence; it must not outrank a genuine failed candidate that
        # correctly reports a missing category.
        if not best_inspection:
            return bool(inspection)
        score = self._score(inspection)
        best_score = self._score(best_inspection)
        best_passed = self._passed(best_inspection)
        if passed != best_passed:
            return passed
        strictly_confirmed = self._strictly_confirmed(inspection)
        best_strictly_confirmed = self._strictly_confirmed(best_inspection)
        if strictly_confirmed != best_strictly_confirmed:
            return strictly_confirmed
        omissions = self._hard_omission_count(inspection)
        best_omissions = self._hard_omission_count(best_inspection)
        if omissions != best_omissions:
            # A wholly absent required category makes the conjunction fail
            # regardless of a soft semantic probability elsewhere.  Count
            # each canonical class once because spatial rows may repeat the
            # object owner's diagnostics for audit provenance.
            return omissions < best_omissions
        failure_count = self._failure_count(inspection)
        best_failure_count = self._failure_count(best_inspection)
        if failure_count != best_failure_count:
            return failure_count < best_failure_count
        weakest = self._weakest_constraint_score(inspection)
        best_weakest = self._weakest_constraint_score(best_inspection)
        if abs(weakest - best_weakest) >= 0.15:
            # T2I compositional prompts are conjunctions.  A candidate with one
            # very weak object/attribute/phrase should not beat a more balanced
            # one merely because other easy constraints score near 1.0.
            return weakest > best_weakest
        if passed and not strictly_confirmed:
            # Soft abstentions are passes without hard evidence.  Do not let
            # calibration-sized changes in one VLM probability replace the
            # simpler seed-control candidate once the weakest constraint is
            # comparable.
            return score > best_score and score - best_score >= 0.08
        if score != best_score:
            # This mean uses a monotonic, normalized expected/found alignment
            # for the hard inventory row and the unchanged verifier scores for
            # other owners.  It balances count against attribute/relation
            # evidence instead of making either family lexicographically
            # absolute.
            return score > best_score
        if weakest != best_weakest:
            # GenEval prompts are conjunctions.  Prefer improving the weakest
            # internally observed constraint instead of allowing several easy
            # high scores to hide one near-zero count/relation/attribute.
            return weakest > best_weakest
        return self._failure_count(inspection) < self._failure_count(best_inspection)

    def _weakest_constraint_score(self, inspection: Dict[str, Dict[str, Any]]) -> float:
        values: List[float] = []
        for row in inspection.values():
            if not isinstance(row, dict) or bool(row.get("not_applicable", False)):
                continue
            value = self._combined_constraint_score(row)
            if value is None:
                value = 1.0 if self._row_effectively_passed(row) else 0.0
            values.append(value)
        return min(values) if values else -1.0

    def _hard_omission_count(self, inspection: Dict[str, Dict[str, Any]]) -> int:
        missing_classes = set()
        for row in inspection.values():
            if not isinstance(row, dict):
                continue
            missing = row.get("missing_required_objects")
            if not isinstance(missing, list):
                continue
            for value in missing:
                if isinstance(value, dict):
                    name = str(value.get("class") or value.get("object") or "").strip().lower()
                else:
                    name = str(value).strip().lower()
                if name:
                    missing_classes.add(name)
        return len(missing_classes)


    def _hard_inventory_alignment(self, row: Dict[str, Any]) -> float:
        error, expected_total, has_evidence = self._hard_inventory_metrics({"inventory": row})
        if not has_evidence:
            return 0.0
        return max(0.0, 1.0 - (float(error) / max(1, expected_total)))

    def _hard_inventory_metrics(self, inspection: Dict[str, Dict[str, Any]]) -> tuple[int, int, bool]:
        errors_by_class: Dict[str, int] = {}
        expected_by_class: Dict[str, int] = {}
        has_inventory_evidence = False
        for row in inspection.values():
            if not isinstance(row, dict):
                continue
            if isinstance(row.get("object_counts"), dict):
                has_inventory_evidence = True
                for name, found in row["object_counts"].items():
                    key = str(name).strip().lower()
                    if key:
                        expected_by_class.setdefault(key, int(found or 0))
            mismatch_rows = row.get("count_mismatches", [])
            for value in mismatch_rows if isinstance(mismatch_rows, list) else []:
                if not isinstance(value, dict):
                    continue
                name = str(value.get("class") or value.get("object") or "").strip().lower()
                if not name:
                    continue
                expected = int(value.get("expected", 0) or 0)
                found = int(value.get("found", 0) or 0)
                expected_by_class[name] = expected
                errors_by_class[name] = max(errors_by_class.get(name, 0), abs(found - expected))
            missing_rows = row.get("missing_required_objects", [])
            for value in missing_rows if isinstance(missing_rows, list) else []:
                if not isinstance(value, dict):
                    continue
                name = str(value.get("class") or value.get("object") or "").strip().lower()
                if not name:
                    continue
                expected = int(value.get("expected", value.get("count", 1)) or 1)
                found = int(value.get("found", 0) or 0)
                expected_by_class[name] = expected
                errors_by_class[name] = max(errors_by_class.get(name, 0), abs(found - expected))
        has_evidence = has_inventory_evidence or bool(errors_by_class)
        if not has_evidence:
            return 10**9, 0, False
        return sum(errors_by_class.values()), sum(expected_by_class.values()), True

    def _failure_count(self, inspection: Dict[str, Dict[str, Any]]) -> int:
        if not inspection:
            return 10**9
        return sum(1 for row in inspection.values() if isinstance(row, dict) and not self._row_effectively_passed(row))

    def _numeric_constraint_score(self, row: Dict[str, Any]) -> float | None:
        for key in ("selection_score", "constraint_score", "alignment_score"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            return max(0.0, min(1.0, float(value)))
        return None

    def _combined_constraint_score(self, row: Dict[str, Any]) -> float | None:
        scores: List[float] = []
        if isinstance(row.get("object_counts"), dict):
            scores.append(self._hard_inventory_alignment(row))
        numeric = self._numeric_constraint_score(row)
        if numeric is not None:
            scores.append(numeric)
        subresults = row.get("inspection_subresults", {})
        if isinstance(subresults, dict):
            for subresult in subresults.values():
                if not isinstance(subresult, dict):
                    continue
                sub_score = (
                    self._hard_inventory_alignment(subresult)
                    if isinstance(subresult.get("object_counts"), dict)
                    else self._numeric_constraint_score(subresult)
                )
                if sub_score is not None:
                    scores.append(sub_score)
        return min(scores) if scores else None

    def _failure_family(self, failures: Sequence[Dict[str, Any]]) -> str:
        families = {self._family_for_failure(f) for f in failures}
        families.discard("unknown")
        if not families:
            return "unknown"
        if len(families) == 1:
            return next(iter(families))
        return "mixed"

    def _family_for_failure(self, failure: Dict[str, Any]) -> str:
        if failure.get("spatial_relation_unknowns"):
            return "spatial"
        if str(failure.get("evidence_status", "")).strip().lower() == "unknown":
            return "tools"
        return self._family_for_symptom(str(failure.get("symptom", "unknown")))


    def _family_for_symptom(self, symptom: str) -> str:
        if symptom in {"missing_required_object", "object_count_mismatch", "forbidden_object_present"}:
            return "count"
        if symptom == "attribute_binding_mismatch":
            return "attribute"
        if symptom == "spatial_relation_mismatch":
            return "spatial"
        if symptom == "action_relation_mismatch":
            return "action"
        if symptom in {"missing_inspection", "inspection_error", "harness_execution_error"}:
            return "tools"
        if symptom.startswith("low_"):
            return "policy"
        if symptom in {"negative_constraint_missing", "constraint_not_in_contract"}:
            return "policy"
        return "unknown"
