from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from .io import expand_env_vars, read_json


TRACE_PROTOCOL_SCHEMA = "gen_harness.trace_protocol.v1"
TRACE_EVIDENCE_SCHEMA = "gen_harness.trace_evidence.v1"
TRACE_LOCALIZER_OUTPUT_SCHEMA = "gen_harness.trace_localizer_output.v1"

TRACE_DEFAULT_FRONTIER_SIZE = 3
TRACE_DEFAULT_K = 4
TRACE_DEFAULT_CANDIDATE_BUDGET = 3
TRACE_DEFAULT_MAX_DEPTH = 3
TRACE_DEFAULT_PRESERVATION_WEIGHT = 0.5
TRACE_DEFAULT_EXECUTION_COST_WEIGHT = 0.05
TRACE_DEFAULT_REGRESSION_EPSILON = 0.01
TRACE_DEFAULT_LLM_MODEL_SNAPSHOT = "gpt-4.1-2025-04-14"

EDITABLE_RESPONSIBILITIES = ("policy", "tools", "skills", "middleware", "memory")
LOCALIZER_SCORE_KEYS = (*EDITABLE_RESPONSIBILITIES, "no_patch")
TRACE_EVIDENCE_KEYS = {"schema", "constraint", "constraint_instances", "aggregation"}

PAPER_DATASET_PROTOCOL = {
    "name": "parti_prompts",
    "stream": "P2",
    "target_count": 500,
    "heldout_count": 100,
    "preservation_count": 100,
}


def validate_trace_seeds(
    seeds: Sequence[int],
    *,
    expected_k: int = TRACE_DEFAULT_K,
    allow_ablation: bool = False,
) -> list[int]:
    """Validate the stochastic execution frontier used by formal TRACE."""

    normalized = list(dict.fromkeys(int(seed) for seed in seeds))
    if not normalized:
        raise ValueError("TRACE requires at least one stochastic execution seed")
    if len(normalized) != len(seeds):
        raise ValueError("TRACE seed list must not contain duplicate stochastic conditions")
    if not allow_ablation and len(normalized) != int(expected_k):
        raise ValueError(
            f"formal TRACE requires exactly K={expected_k} distinct stochastic conditions; "
            "use an explicit seed-ablation opt-in for sensitivity experiments"
        )
    return normalized


def validate_localizer_output(
    row: Any,
    *,
    expected_constraint_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Validate the exact structured output contract for the Harness Localizer."""

    if not isinstance(row, dict):
        raise ValueError("localizer output must be a JSON object")
    expected_keys = {"scores", "constraint_attributions", "rationale"}
    extra = sorted(set(row) - expected_keys)
    missing = sorted(expected_keys - set(row))
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing fields: {missing}")
        if extra:
            detail.append(f"unexpected fields: {extra}")
        raise ValueError("localizer output schema mismatch; " + "; ".join(detail))

    scores = row["scores"]
    if not isinstance(scores, dict):
        raise ValueError("localizer output field 'scores' must be an object")
    score_keys = set(scores)
    expected_score_keys = set(LOCALIZER_SCORE_KEYS)
    extra_scores = sorted(score_keys - expected_score_keys)
    missing_scores = sorted(expected_score_keys - score_keys)
    if missing_scores or extra_scores:
        detail = []
        if missing_scores:
            detail.append(f"missing score keys: {missing_scores}")
        if extra_scores:
            detail.append(f"unexpected score keys: {extra_scores}")
        raise ValueError("localizer score schema mismatch; " + "; ".join(detail))

    normalized_scores: Dict[str, float] = {}
    for name in LOCALIZER_SCORE_KEYS:
        value = scores[name]
        normalized_scores[name] = _bounded_score(value, f"localizer score for {name}")

    attributions = row["constraint_attributions"]
    if not isinstance(attributions, list) or not attributions:
        raise ValueError("localizer output field 'constraint_attributions' must be a non-empty list")
    expected_attr_keys = {"constraint_id", "scores", "primary_component", "confidence", "evidence_summary"}
    normalized_attributions = []
    seen_constraint_ids = set()
    for index, attribution in enumerate(attributions):
        if not isinstance(attribution, dict):
            raise ValueError(f"constraint_attributions[{index}] must be an object")
        extra_attr = sorted(set(attribution) - expected_attr_keys)
        missing_attr = sorted(expected_attr_keys - set(attribution))
        if missing_attr or extra_attr:
            detail = []
            if missing_attr:
                detail.append(f"missing fields: {missing_attr}")
            if extra_attr:
                detail.append(f"unexpected fields: {extra_attr}")
            raise ValueError(f"constraint_attributions[{index}] schema mismatch; " + "; ".join(detail))
        constraint_id = str(attribution["constraint_id"]).strip()
        if not constraint_id:
            raise ValueError(f"constraint_attributions[{index}].constraint_id must be non-empty")
        if constraint_id in seen_constraint_ids:
            raise ValueError(f"duplicate localizer attribution for constraint_id: {constraint_id}")
        seen_constraint_ids.add(constraint_id)
        attr_scores = attribution["scores"]
        if not isinstance(attr_scores, dict):
            raise ValueError(f"constraint_attributions[{index}].scores must be an object")
        extra_attr_scores = sorted(set(attr_scores) - expected_score_keys)
        missing_attr_scores = sorted(expected_score_keys - set(attr_scores))
        if missing_attr_scores or extra_attr_scores:
            detail = []
            if missing_attr_scores:
                detail.append(f"missing score keys: {missing_attr_scores}")
            if extra_attr_scores:
                detail.append(f"unexpected score keys: {extra_attr_scores}")
            raise ValueError(f"constraint_attributions[{index}].scores schema mismatch; " + "; ".join(detail))
        normalized_attr_scores = {
            name: _bounded_score(attr_scores[name], f"constraint_attributions[{index}].scores.{name}")
            for name in LOCALIZER_SCORE_KEYS
        }
        primary = str(attribution["primary_component"]).strip()
        if primary not in LOCALIZER_SCORE_KEYS:
            raise ValueError(f"constraint_attributions[{index}].primary_component is not a valid responsibility")
        confidence = _bounded_score(attribution["confidence"], f"constraint_attributions[{index}].confidence")
        evidence_summary = attribution["evidence_summary"]
        if not isinstance(evidence_summary, str) or not evidence_summary.strip():
            raise ValueError(f"constraint_attributions[{index}].evidence_summary must be a non-empty string")
        if len(evidence_summary) > 2000:
            raise ValueError(f"constraint_attributions[{index}].evidence_summary exceeds the 2000-character schema limit")
        normalized_attributions.append(
            {
                "constraint_id": constraint_id,
                "scores": normalized_attr_scores,
                "primary_component": primary,
                "confidence": confidence,
                "evidence_summary": evidence_summary.strip(),
            }
        )

    if expected_constraint_ids is not None:
        expected_ids = {str(value).strip() for value in expected_constraint_ids if str(value).strip()}
        observed_ids = {row["constraint_id"] for row in normalized_attributions}
        if observed_ids != expected_ids:
            missing_ids = sorted(expected_ids - observed_ids)
            extra_ids = sorted(observed_ids - expected_ids)
            detail = []
            if missing_ids:
                detail.append(f"missing constraint attributions: {missing_ids}")
            if extra_ids:
                detail.append(f"unexpected constraint attributions: {extra_ids}")
            raise ValueError("localizer constraint attribution coverage mismatch; " + "; ".join(detail))

    rationale = row["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("localizer output field 'rationale' must be a non-empty string")
    if len(rationale) > 4000:
        raise ValueError("localizer rationale exceeds the 4000-character schema limit")
    return {
        "scores": normalized_scores,
        "constraint_attributions": normalized_attributions,
        "rationale": rationale.strip(),
    }


def _bounded_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return numeric


def validate_trace_evidence(
    evidence: Any,
    *,
    expected_k: int | None = None,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Validate the constraint-indexed E_t evidence passed to the Localizer."""

    if not isinstance(evidence, dict):
        raise ValueError("TRACE evidence must be a JSON object")
    if evidence.get("schema") != TRACE_EVIDENCE_SCHEMA:
        raise ValueError(f"TRACE evidence must declare schema={TRACE_EVIDENCE_SCHEMA!r}")
    extra = sorted(set(evidence) - TRACE_EVIDENCE_KEYS)
    missing = sorted(TRACE_EVIDENCE_KEYS - set(evidence))
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing fields: {missing}")
        if extra:
            detail.append(f"unexpected fields: {extra}")
        raise ValueError("TRACE evidence schema mismatch; " + "; ".join(detail))

    constraint = evidence["constraint"]
    if not isinstance(constraint, dict):
        raise ValueError("TRACE evidence field 'constraint' must be an object")
    for key in ("constraint_ids", "constraint_types", "texts", "failure_symptoms"):
        if not isinstance(constraint.get(key), list):
            raise ValueError(f"TRACE evidence constraint field {key!r} must be a list")

    instances = evidence["constraint_instances"]
    if not isinstance(instances, list) or not instances:
        raise ValueError("TRACE evidence must contain at least one constraint instance")
    aggregation = evidence["aggregation"]
    if not isinstance(aggregation, dict):
        raise ValueError("TRACE evidence field 'aggregation' must be an object")

    normalized_expected = int(expected_k) if expected_k is not None else aggregation.get("expected_k")
    if normalized_expected is not None and int(normalized_expected) < 1:
        raise ValueError("TRACE evidence expected_k must be positive")
    if expected_k is not None and aggregation.get("expected_k") != int(expected_k):
        raise ValueError(
            f"TRACE evidence expected_k mismatch: expected {expected_k}, "
            f"found {aggregation.get('expected_k')}"
        )

    observed_execution_count = 0
    all_seeds = []
    complete_instances = 0
    failure_complete_instances = 0
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise ValueError(f"TRACE evidence constraint instance {index} must be an object")
        required_instance_keys = {
            "instance_index",
            "constraint",
            "executions",
            "stochastic_conditions",
            "expected_k",
            "complete_for_expected_k",
        }
        if set(instance) != required_instance_keys:
            raise ValueError(f"TRACE evidence constraint instance {index} has an invalid schema")
        instance_constraint = instance["constraint"]
        if not isinstance(instance_constraint, dict) or set(instance_constraint) != {
            "constraint_id",
            "constraint_type",
            "text",
        }:
            raise ValueError(f"TRACE evidence constraint instance {index} has an invalid constraint")
        executions = instance["executions"]
        if not isinstance(executions, list) or not executions:
            raise ValueError(f"TRACE evidence constraint instance {index} has no executions")
        seeds = []
        failure_seeds = []
        for execution in executions:
            if not isinstance(execution, dict):
                raise ValueError("TRACE evidence execution must be an object")
            required_execution_keys = {
                "execution_index",
                "seed",
                "constraint",
                "trajectory_decisions",
                "tool_calls",
                "visual_observation",
                "visual_verification",
                "outcome",
                "failure_symptom",
            }
            if set(execution) != required_execution_keys:
                raise ValueError("TRACE evidence execution has an invalid schema")
            seed = execution["seed"]
            if seed is None and normalized_expected is None:
                pass
            elif isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("TRACE evidence execution seed must be an integer")
            if execution["outcome"] not in {"pass", "failure"}:
                raise ValueError("TRACE evidence execution outcome must be 'pass' or 'failure'")
            if not isinstance(execution["trajectory_decisions"], dict):
                raise ValueError("TRACE evidence trajectory_decisions must be an object")
            if not isinstance(execution["tool_calls"], list):
                raise ValueError("TRACE evidence tool_calls must be a list")
            if not isinstance(execution["visual_observation"], dict):
                raise ValueError("TRACE evidence visual_observation must be an object")
            if not isinstance(execution["visual_verification"], dict):
                raise ValueError("TRACE evidence visual_verification must be an object")
            if execution["outcome"] == "failure" and not str(execution["failure_symptom"] or "").strip():
                raise ValueError("TRACE evidence failures must include failure_symptom")
            if seed is not None:
                seeds.append(seed)
                all_seeds.append(seed)
            if execution["outcome"] == "failure":
                failure_seeds.append(seed)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"TRACE evidence constraint instance {index} repeats a stochastic condition")
        declared_conditions = instance["stochastic_conditions"]
        if declared_conditions != sorted(set(seeds), key=str):
            raise ValueError(f"TRACE evidence constraint instance {index} has inconsistent stochastic_conditions")
        instance_expected = instance["expected_k"]
        if instance_expected != normalized_expected:
            raise ValueError(f"TRACE evidence constraint instance {index} has inconsistent expected_k")
        complete = normalized_expected is None or len(seeds) == normalized_expected
        if bool(instance["complete_for_expected_k"]) != complete:
            raise ValueError(f"TRACE evidence constraint instance {index} has an invalid completeness flag")
        if complete:
            complete_instances += 1
        if normalized_expected is None or len(set(failure_seeds)) == normalized_expected:
            failure_complete_instances += 1
        observed_execution_count += len(executions)

    expected_complete = normalized_expected is None or complete_instances == len(instances)
    if aggregation.get("observed_execution_count") != observed_execution_count:
        raise ValueError("TRACE evidence observed_execution_count does not match executions")
    if aggregation.get("distinct_stochastic_condition_count") != len(set(all_seeds)):
        raise ValueError("TRACE evidence distinct_stochastic_condition_count does not match executions")
    if aggregation.get("stochastic_conditions") != sorted(set(all_seeds), key=str):
        raise ValueError("TRACE evidence stochastic_conditions does not match executions")
    if aggregation.get("expected_k") != normalized_expected:
        raise ValueError("TRACE evidence aggregation expected_k is inconsistent")
    if aggregation.get("exploration_complete_for_expected_k") is not expected_complete:
        raise ValueError("TRACE evidence exploration completeness flag is invalid")
    expected_failure_complete = normalized_expected is None or failure_complete_instances == len(instances)
    if aggregation.get("failure_support_complete_for_all_k") is not expected_failure_complete:
        raise ValueError("TRACE evidence failure-support completeness flag is invalid")
    if not allow_incomplete and not expected_complete:
        raise ValueError("TRACE evidence does not contain K executions for every constraint instance")
    return evidence


def validate_paper_dataset(
    tasks: Iterable[Any],
    *,
    protocol: str,
) -> Dict[str, Any]:
    """Validate the public paper split contract without reading benchmark labels."""

    rows = list(tasks)
    split_counts: Dict[str, int] = {}
    task_ids = set()
    for task in rows:
        task_id = str(getattr(task, "task_id", "") or "")
        if not task_id:
            raise ValueError("TRACE dataset tasks must have non-empty task_id values")
        if task_id in task_ids:
            raise ValueError(f"TRACE dataset contains duplicate task_id: {task_id}")
        task_ids.add(task_id)
        metadata = getattr(task, "metadata", {}) or {}
        split = str(metadata.get("split", "")).strip()
        if split not in {"target", "heldout", "preservation"}:
            raise ValueError(f"TRACE dataset task {task_id} has invalid split {split!r}")
        split_counts[split] = split_counts.get(split, 0) + 1

    if protocol == "paper_p2":
        expected = {
            "target": PAPER_DATASET_PROTOCOL["target_count"],
            "heldout": PAPER_DATASET_PROTOCOL["heldout_count"],
            "preservation": PAPER_DATASET_PROTOCOL["preservation_count"],
        }
        if split_counts != expected:
            raise ValueError(
                "paper_p2 TRACE requires exact target/heldout/preservation counts "
                f"{expected}, found {split_counts}"
            )
    elif protocol != "development":
        raise ValueError(f"unsupported TRACE dataset protocol: {protocol}")

    return {
        "schema": "gen_harness.trace_dataset_validation.v1",
        "protocol": protocol,
        "dataset": dict(PAPER_DATASET_PROTOCOL) if protocol == "paper_p2" else {"name": "development"},
        "num_tasks": len(rows),
        "split_counts": split_counts,
        "task_ids_unique": True,
    }


def validate_fixed_llm_config(config: Dict[str, Any], *, role: str) -> Dict[str, str]:
    """Require an explicit GPT-4.1 model and immutable snapshot for paper TRACE."""

    model = str(config.get("model", "")).strip()
    snapshot = str(config.get("model_snapshot", "")).strip()
    if not model or not snapshot:
        raise ValueError(
            f"paper TRACE requires an explicit {role} model and model_snapshot; "
            "do not use an unpinned model alias"
        )
    if not model.lower().startswith("gpt-4.1") or snapshot.lower() != TRACE_DEFAULT_LLM_MODEL_SNAPSHOT:
        raise ValueError(
            f"paper TRACE requires GPT-4.1 for {role}; found model={model!r}, snapshot={snapshot!r}"
        )
    return {"model": model, "model_snapshot": snapshot}


def load_trace_protocol_config(path: str | Path) -> Dict[str, Any]:
    config = expand_env_vars(read_json(path))
    if not isinstance(config, dict) or config.get("schema") != TRACE_PROTOCOL_SCHEMA:
        raise ValueError(f"TRACE protocol config must declare schema={TRACE_PROTOCOL_SCHEMA!r}")
    return config
