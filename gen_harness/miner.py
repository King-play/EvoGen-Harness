from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .schema import VisualExperience, Weakness
from .trace_protocol import TRACE_EVIDENCE_SCHEMA, validate_trace_evidence


class WeaknessMiner:
    def __init__(
        self,
        min_support: int = 2,
        image_metric_threshold: float = 0.85,
        expected_stochastic_conditions: int | None = None,
    ):
        self.min_support = min_support
        self.image_metric_threshold = image_metric_threshold
        self.expected_stochastic_conditions = expected_stochastic_conditions

    def mine(self, experiences: Iterable[VisualExperience]) -> List[Weakness]:
        all_experiences = list(experiences)
        clusters: Dict[Tuple[str, str, str], List[VisualExperience]] = defaultdict(list)
        for exp in all_experiences:
            result = exp.verification_result or {}
            evidence_status = str(result.get("evidence_status", "")).strip().lower()
            single_verifier = bool(result.get("single_verifier_evidence", False))
            partitioned_owner_mismatch = (
                single_verifier
                and str(result.get("merge_strategy", "")).strip().lower() == "capability_partitioned"
                and bool(str(result.get("capability_owner", "")).strip())
                and evidence_status == "mismatch"
                and not bool(result.get("verifier_conflict", False))
            )
            # Self-evolution may learn from a declared capability owner even
            # when that capability is semantic rather than detector-hard. The
            # owner contract is the evidence boundary for that constraint:
            # explicit mismatches are admissible, while unknown/conflicting
            # results and undeclared single-verifier failures remain
            # diagnostics only.
            eligible_failure = (
                not result.get("passed")
                and evidence_status not in {"unknown", "confirmed"}
                and (not single_verifier or partitioned_owner_mismatch)
            )
            if eligible_failure:
                component = _formal_component_for_mining(exp)
                symptom = exp.failure_symptom or "unknown"
                key = (component, exp.constraint_type, symptom)
                _append_unique(clusters[key], exp)
        for _component, ctype, symptom in self._image_metric_weaknesses(exp):
            _append_unique(clusters[("unknown", ctype, symptom)], exp)

        weaknesses: List[Weakness] = []
        for (component, ctype, symptom), rows in sorted(clusters.items()):
            if len(rows) < self.min_support:
                continue
            wid = f"weak_{component}_{ctype}_{symptom}_{len(rows)}"
            weaknesses.append(
                Weakness(
                    weakness_id=wid,
                    suspected_component=component,
                    constraint_type=ctype,
                    failure_symptom=symptom,
                    support_count=len(rows),
                    support_experience_ids=[r.experience_id for r in rows],
                    hypothesis=self._hypothesis(component, ctype, symptom),
                    evidence={
                        "task_ids": sorted({r.task_id for r in rows}),
                        "constraint_ids": sorted({r.constraint_id for r in rows}),
                        "stochastic_conditions": sorted(
                            {
                                (r.visual_result or {}).get("seed")
                                for r in rows
                                if (r.visual_result or {}).get("seed") is not None
                            },
                            key=str,
                        ),
                        "num_stochastic_conditions": len(
                            {
                                (r.visual_result or {}).get("seed")
                                for r in rows
                                if (r.visual_result or {}).get("seed") is not None
                            }
                        ),
                        "localization_source": _localization_source(component, rows),
                        "examples": [_trace_example(r) for r in rows[:12]],
                        "trace_evidence": _trace_evidence(
                            _trace_rows_for_cluster(rows, all_experiences),
                            expected_stochastic_conditions=self.expected_stochastic_conditions,
                            failure_rows=rows,
                        ),
                    },
                )
            )
        return weaknesses

    def _image_metric_weaknesses(self, exp: VisualExperience) -> List[Tuple[str, str, str]]:
        result = exp.verification_result or {}
        metrics = result.get("image_metrics", {}) if isinstance(result, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        out: List[Tuple[str, str, str]] = []
        for metric, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if float(value) >= self.image_metric_threshold:
                continue
            out.append(("unknown", str(metric), f"low_{metric}"))
        return out

    def _hypothesis(self, component: str, ctype: str, symptom: str) -> str:
        return (
            f"Repeated {symptom} failures for {ctype} require the LLM HarnessLocalizer "
            "to determine responsibility before a patch is proposed."
        )


def _formal_component_for_mining(exp: VisualExperience) -> str:
    attribution = exp.metadata.get("attribution") if isinstance(exp.metadata, dict) else None
    if isinstance(attribution, dict) and attribution.get("mode") == "llm_harness_localizer":
        localization = attribution.get("localization", {})
        frontier = localization.get("frontier", []) if isinstance(localization, dict) else []
        if isinstance(frontier, list) and frontier:
            return str(frontier[0])
    return "unknown"


def _append_unique(rows: List[VisualExperience], exp: VisualExperience) -> None:
    # Constraint-indexed records from one task/seed share a stochastic
    # execution. Count that execution once so repeated constraints in a single
    # image cannot masquerade as trajectory aggregation across omega_k.
    seed = (exp.visual_result or {}).get("seed")
    if all(
        row.experience_id != exp.experience_id
        and (row.task_id, (row.visual_result or {}).get("seed")) != (exp.task_id, seed)
        for row in rows
    ):
        rows.append(exp)


def _trace_example(exp: VisualExperience) -> Dict[str, object]:
    row = exp.to_compact_dict()
    row.pop("suspected_component", None)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata.pop("attribution", None)
        metadata.pop("failure_diagnosis", None)
        row["metadata"] = metadata
    return row


def _trace_evidence(
    rows: List[VisualExperience],
    *,
    expected_stochastic_conditions: int | None,
    failure_rows: List[VisualExperience],
) -> Dict[str, object]:
    """Build the explicit constraint-indexed E_t contract used by TRACE.

    The raw artifact retains enough provenance for audit. The
    Localizer sanitizer later removes task-level identifiers and benchmark
    metadata before this structure is sent to an LLM.
    """

    constraint_types = sorted({str(row.constraint_type) for row in rows})
    failure_symptoms = sorted({str(row.failure_symptom or "unknown") for row in rows})
    constraint_ids = sorted({str(row.constraint_id) for row in rows})
    seeds = [
        (row.visual_result or {}).get("seed")
        for row in rows
        if (row.visual_result or {}).get("seed") is not None
    ]
    distinct_seeds = sorted(set(seeds), key=str)
    constraint_texts = sorted(
        {
            str(row.constraint_text).strip()
            for row in rows
            if str(row.constraint_text).strip()
        }
    )
    grouped: Dict[Tuple[str, str], List[VisualExperience]] = defaultdict(list)
    for row in rows:
        grouped[(row.task_id, row.constraint_id)].append(row)
    constraint_instances = []
    flattened_execution_count = 0
    for instance_index, (_instance_key, instance_rows) in enumerate(
        sorted(grouped.items(), key=lambda item: str(item[0]))
    ):
        instance_seeds = sorted(
            {
                (row.visual_result or {}).get("seed")
                for row in instance_rows
                if (row.visual_result or {}).get("seed") is not None
            },
            key=str,
        )
        executions = []
        for execution_index, row in enumerate(
            sorted(instance_rows, key=lambda item: str((item.visual_result or {}).get("seed")))
        ):
            compact = row.to_compact_dict()
            visual_result = compact.get("visual_result", {})
            if not isinstance(visual_result, dict):
                visual_result = {}
            visual_observation = {
                key: visual_result.get(key)
                for key in ("seed", "backend", "selected_model", "candidate_family")
                if key in visual_result
            }
            passed = bool((row.verification_result or {}).get("passed", False))
            executions.append(
                {
                    "execution_index": execution_index,
                    "seed": (row.visual_result or {}).get("seed"),
                    "constraint": {
                        "constraint_id": row.constraint_id,
                        "constraint_type": row.constraint_type,
                        "text": row.constraint_text,
                    },
                    "trajectory_decisions": compact.get("component_decisions", {}),
                    "tool_calls": compact.get("tool_calls", []),
                    "visual_observation": visual_observation,
                    "visual_verification": _trace_verification(compact.get("verification_result", {})),
                    "outcome": "pass" if passed else "failure",
                    "failure_symptom": None if passed else row.failure_symptom,
                }
            )
        expected = int(expected_stochastic_conditions) if expected_stochastic_conditions is not None else None
        constraint_instances.append(
            {
                "instance_index": instance_index,
                "constraint": {
                    "constraint_id": instance_rows[0].constraint_id,
                    "constraint_type": instance_rows[0].constraint_type,
                    "text": instance_rows[0].constraint_text,
                },
                "executions": executions,
                "stochastic_conditions": instance_seeds,
                "expected_k": expected,
                "complete_for_expected_k": expected is None or len(instance_seeds) == expected,
            }
        )
        flattened_execution_count += len(executions)
    expected = int(expected_stochastic_conditions) if expected_stochastic_conditions is not None else None
    evidence = {
        "schema": TRACE_EVIDENCE_SCHEMA,
        "constraint": {
            "constraint_ids": constraint_ids,
            "constraint_types": constraint_types,
            "texts": constraint_texts,
            "failure_symptoms": failure_symptoms,
        },
        "constraint_instances": constraint_instances,
        "aggregation": {
            "observed_execution_count": flattened_execution_count,
            "distinct_stochastic_condition_count": len(distinct_seeds),
            "stochastic_conditions": distinct_seeds,
            "expected_k": expected,
            "exploration_complete_for_expected_k": expected is None
            or all(instance["complete_for_expected_k"] for instance in constraint_instances),
            "failure_support_complete_for_all_k": expected is None
            or all(
                len(
                    {
                        execution["seed"]
                        for execution in instance["executions"]
                        if execution["outcome"] == "failure"
                    }
                )
                == expected
                for instance in constraint_instances
            ),
        },
    }
    validate_trace_evidence(
        evidence,
        expected_k=expected,
        allow_incomplete=expected is None,
    )
    return evidence


def _trace_rows_for_cluster(
    failure_rows: List[VisualExperience],
    all_experiences: List[VisualExperience],
) -> List[VisualExperience]:
    """Expand failures to all K executions for each affected task/constraint."""

    instances = {(row.task_id, row.constraint_id) for row in failure_rows}
    return [
        row
        for row in all_experiences
        if (row.task_id, row.constraint_id) in instances
    ]


def _trace_verification(value: object) -> Dict[str, object]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "passed",
        "symptom",
        "confidence",
        "evidence_status",
        "inspection_source",
        "merge_strategy",
        "capability_owner",
        "single_verifier_evidence",
        "verifier_conflict",
        "hard_evidence_required",
        "count_mismatches",
        "missing_required_objects",
        "forbidden_objects_present",
        "attribute_mismatches",
        "spatial_mismatches",
        "action_mismatches",
        "inspection_subresults",
    }
    return {key: value[key] for key in allowed if key in value}


def _localization_source(component: str, rows: List[VisualExperience]) -> str:
    if component != "unknown" and rows and all(_formal_component_for_mining(row) == component for row in rows):
        return "llm_harness_localizer"
    return "llm_required"
