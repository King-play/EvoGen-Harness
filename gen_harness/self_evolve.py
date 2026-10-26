from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import read_json, read_jsonl, write_json
from .component_audit import ComponentBoundaryAuditor


SELF_EVOLVE_SCHEMA = "gen_harness.self_evolve_manifest.v1"


def _reference_trajectory_policy_record(summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
    summary_policy = summary.get("reference_trajectory_policy", {}) if isinstance(summary, dict) else {}
    allowed_sources = summary_policy.get("allowed_sources") if isinstance(summary_policy, dict) else None
    if not isinstance(allowed_sources, list) or not allowed_sources:
        allowed_sources = ["non_benchmark_dev_tasks", "internal_baseline_traces", "human_authored_dev_traces"]
    return {
        "official_benchmark_reference_trajectories_allowed": False,
        "allowed_sources": [str(source) for source in allowed_sources],
        "official_labels_used_for_generation_or_selection": False,
        "official_evaluator_used_for_generation_or_selection": False,
        "benchmark_tasks_allowed": False,
        "leakage_safe": True,
    }


def build_self_evolve_manifest(
    output_dir: str | Path,
    *,
    write_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Build an auditable self-evolution manifest from a slow-loop run.

    This is deliberately a Gen-Harness manifest, not a copy of AHE's coding-agent
    workspace format.  The invariant is the same: every evolution step must be
    backed by failure evidence, a root-cause hypothesis, a targeted patch, a
    predicted impact, and a validation gate result.
    """

    out = Path(output_dir)
    self_evolve_summary = _read_json_optional(out / "self_evolve_run_summary.json")
    if self_evolve_summary:
        manifest = _build_iterative_self_evolve_manifest(out, self_evolve_summary)
        if write_path is not None:
            write_json(write_path, manifest)
        return manifest

    summary = _read_json_optional(out / "slow_loop_summary.json")
    evolution_log = _read_json_optional(out / "harness_evolution_log.json")
    trace_certificate = _read_json_optional(out / "trace_certificate.json")
    if not trace_certificate and isinstance(evolution_log, dict):
        embedded_trace = evolution_log.get("trace_certificate")
        trace_certificate = embedded_trace if isinstance(embedded_trace, dict) else {}
    experiences = read_jsonl(out / "experiences.jsonl")
    weaknesses = read_jsonl(out / "weaknesses.jsonl")
    patches = read_jsonl(out / "patches.jsonl")
    reports = read_jsonl(out / "validation_reports.jsonl")
    diagnoses = read_jsonl(out / "failure_diagnoses.jsonl")
    diagnoses_by_experience = {
        str(row.get("experience_id") or ""): row
        for row in diagnoses
        if isinstance(row, dict)
    }
    reports_by_patch = {
        str(row.get("patch_id") or row.get("patch", {}).get("patch_id") or ""): row
        for row in reports
        if isinstance(row, dict)
    }
    promoted_by_patch = {
        str(row.get("patch_id") or ""): bool(row.get("promoted_to_harness", False))
        for row in evolution_log.get("evolution_steps", [])
        if isinstance(row, dict)
    }

    changes = []
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        evidence_summary = patch.get("evidence_summary", {}) if isinstance(patch.get("evidence_summary"), dict) else {}
        weakness = _matching_weakness(patch, weaknesses)
        report = reports_by_patch.get(str(patch.get("patch_id") or ""), {})
        changes.append(_change_row(
            patch,
            weakness,
            evidence_summary,
            report,
            diagnoses_by_experience,
            promoted_to_harness=promoted_by_patch.get(str(patch.get("patch_id") or ""), False),
        ))

    num_accepted = sum(1 for row in changes if row.get("gate", {}).get("accepted") is True)
    num_promoted = sum(1 for row in changes if row.get("gate", {}).get("promoted_to_harness") is True)
    harness_root = evolution_log.get("harness_root") or summary.get("harness_root")
    tasks_path = evolution_log.get("tasks_path") or summary.get("tasks_path")
    component_audit = _component_audit_for(harness_root, tasks_path)
    manifest = {
        "schema": SELF_EVOLVE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(out),
        "harness_root": harness_root,
        "reference_model": _reference_model_record(),
        "reference_frameworks": _reference_framework_records(),
        "reference_trajectory_policy": _reference_trajectory_policy_record(summary),
        "loop": {
            "name": "Gen-Harness deterministically gated self-evolution",
            "stages": [
                "evaluate/generate visual experiences",
                "mine repeated weaknesses",
                "propose component-scoped patches",
                "validate on target/heldout/preservation gates",
                "promote only accepted patches when explicitly enabled",
                "write regression memory for rejected patches",
            ],
            "fully_autonomous": bool(evolution_log.get("promote", False)),
            "human_gated": not bool(evolution_log.get("promote", False)),
            "stop_condition": "stop when target metric improves over the fixed baseline without heldout/preservation regression",
        },
        "leakage_and_fallback_guards": {
            "benchmark_tasks_allowed": bool(evolution_log.get("allow_benchmark_tasks", summary.get("allow_benchmark_tasks", False))),
            "benchmark_tasks_refused_by_default": True,
            "official_evaluator_used_for_generation_or_selection": False,
            "official_labels_used_for_prompt_compilation": False,
            "fallback_image_substitution_allowed": False,
            "component_boundary_audit_passed": bool(component_audit.get("passed")),
        },
        "inputs": {
            "tasks_path": tasks_path,
            "seeds": evolution_log.get("seeds") or summary.get("seeds", []),
            "fast_loop_attempts": evolution_log.get("fast_loop_attempts") or summary.get("fast_loop_attempts"),
            "policy_extractor_manifest": evolution_log.get("policy_extractor_manifest")
            or summary.get("policy_extractor_manifest", {}),
            "validation_metric_mode": evolution_log.get("validation_metric_mode") or summary.get("validation_metric_mode"),
            "patch_validation_runtime": evolution_log.get("patch_validation_runtime") or summary.get("patch_validation_runtime"),
            "patch_validation_budget": evolution_log.get("patch_validation_budget") or summary.get("patch_validation_budget"),
        },
        "observability": {
            "num_experiences": len(experiences),
            "num_failed_experiences": sum(1 for row in experiences if not _passed(row)),
            "num_weaknesses": len(weaknesses),
            "num_patch_proposals": len(patches),
            "num_validation_reports": len(reports),
            "num_failure_diagnoses": len(diagnoses),
            "artifacts": {
                "experiences": str(out / "experiences.jsonl"),
                "weaknesses": str(out / "weaknesses.jsonl"),
                "patches": str(out / "patches.jsonl"),
                "validation_reports": str(out / "validation_reports.jsonl"),
                "failure_diagnoses": str(out / "failure_diagnoses.jsonl"),
                "harness_evolution_log": str(out / "harness_evolution_log.json"),
                "trace_certificate": str(out / "trace_certificate.json"),
            },
        },
        "trace_certificate": trace_certificate,
        "component_boundary_audit": component_audit,
        "failure_attribution": {
            "schema": "gen_harness.self_evolve_failure_attribution.v1",
            "source": "internal Gen-Harness traces and lightweight verifier only",
            "formal_component_field": "llm_harness_localizer_frontier_only",
            "frontier_source": "llm_harness_localizer",
            "official_labels_used": False,
            "fallback_used": any(bool(row.get("fallback_used")) for row in diagnoses if isinstance(row, dict)),
            "leakage_safe": all(bool(row.get("leakage_safe", True)) for row in diagnoses if isinstance(row, dict)),
        },
        "changes": changes,
        "gate_summary": {
            "num_accepted": num_accepted,
            "num_rejected": max(0, len(changes) - num_accepted),
            "num_promoted": num_promoted,
            "promotion_requires_explicit_flag": True,
            "verdict": "promoted" if num_promoted else ("accepted_not_promoted" if num_accepted else "no_accepted_patch"),
        },
    }
    if write_path is not None:
        write_json(write_path, manifest)
    return manifest


def _build_iterative_self_evolve_manifest(out: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    rounds = summary.get("rounds", [])
    if not isinstance(rounds, list):
        rounds = []
    round_manifests: List[Dict[str, Any]] = []
    changes: List[Dict[str, Any]] = []
    for round_record in rounds:
        if not isinstance(round_record, dict):
            continue
        round_index = round_record.get("round")
        manifest_path = _round_manifest_path(out, round_record)
        round_manifest = _read_json_optional(manifest_path)
        if round_manifest:
            round_manifests.append({
                "round": round_index,
                "path": str(manifest_path),
                "schema": round_manifest.get("schema"),
                "observability": round_manifest.get("observability", {}),
                "failure_attribution": round_manifest.get("failure_attribution", {}),
                "trace_certificate": _trace_certificate_summary(round_manifest.get("trace_certificate", {})),
                "gate_summary": round_manifest.get("gate_summary", {}),
            })
            for change in round_manifest.get("changes", []) if isinstance(round_manifest.get("changes", []), list) else []:
                if isinstance(change, dict):
                    changes.append({"round": round_index, **change})
    component_audit = _component_audit_for(summary.get("harness_root"), summary.get("tasks_path"))
    num_accepted = sum(1 for row in changes if row.get("gate", {}).get("accepted") is True)
    num_promoted = sum(1 for row in changes if row.get("gate", {}).get("promoted_to_harness") is True)
    observability = _aggregate_round_observability(round_manifests)
    return {
        "schema": SELF_EVOLVE_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(out),
        "harness_root": summary.get("harness_root"),
        "reference_model": _reference_model_record(),
        "reference_frameworks": _reference_framework_records(),
        "reference_trajectory_policy": _reference_trajectory_policy_record(summary),
        "loop": {
            "name": "Gen-Harness iterative self-evolution",
            "stages": [
                "repeat slow-loop rounds",
                "fast-loop baseline generation and explicit local repair inside each exploration pass",
                "collect natural failures",
                "diagnose component-level root causes",
                "mine repeated weaknesses",
                "propose component-scoped patches",
                "validate on target/heldout/preservation gates",
                "promote accepted patches only when explicitly enabled",
            ],
            "fully_autonomous": bool(summary.get("promote_accepted", False)),
            "human_gated": not bool(summary.get("promote_accepted", False)),
            "stop_reason": summary.get("stop_reason"),
            "stop_condition": "stop when no internal failures remain or no safe accepted patch can be promoted",
        },
        "leakage_and_fallback_guards": {
            "benchmark_tasks_allowed": bool(summary.get("leakage_and_fallback_guards", {}).get("allow_benchmark_tasks", False)) if isinstance(summary.get("leakage_and_fallback_guards"), dict) else False,
            "benchmark_tasks_refused_by_default": True,
            "official_evaluator_used_for_generation_or_selection": False,
            "official_labels_used_for_prompt_compilation": False,
            "fallback_image_substitution_allowed": False,
            "component_boundary_audit_passed": bool(component_audit.get("passed")),
        },
        "inputs": {
            "tasks_path": summary.get("tasks_path"),
            "seeds": summary.get("seeds", []),
            "fast_loop_attempts": summary.get("fast_loop_attempts"),
            "policy_extractor_manifest": summary.get("policy_extractor_manifest", {}),
            "validation_metric_mode": summary.get("validation_metric_mode"),
            "max_rounds": summary.get("max_rounds"),
            "promote_accepted": summary.get("promote_accepted"),
        },
        "observability": {
            **observability,
            "num_rounds": len(rounds),
            "num_round_manifests": len(round_manifests),
            "artifacts": {
                "self_evolve_run_summary": str(out / "self_evolve_run_summary.json"),
                "round_manifests": [row["path"] for row in round_manifests],
            },
        },
        "trace_certificate": _aggregate_trace_certificates(round_manifests),
        "component_boundary_audit": component_audit,
        "failure_attribution": {
            "schema": "gen_harness.self_evolve_failure_attribution.v1",
            "source": "internal Gen-Harness traces and lightweight verifier only",
            "formal_component_field": "llm_harness_localizer_frontier_only",
            "frontier_source": "llm_harness_localizer",
            "official_labels_used": False,
            "fallback_used": any(
                bool((row.get("failure_attribution") or {}).get("fallback_used"))
                for row in round_manifests
                if isinstance(row, dict)
            ),
            "leakage_safe": all(
                bool((row.get("failure_attribution") or {}).get("leakage_safe", True))
                for row in round_manifests
                if isinstance(row, dict)
            ),
        },
        "changes": changes,
        "gate_summary": {
            "num_accepted": num_accepted,
            "num_rejected": max(0, len(changes) - num_accepted),
            "num_promoted": num_promoted,
            "promotion_requires_explicit_flag": True,
            "verdict": "promoted" if num_promoted else ("accepted_not_promoted" if num_accepted else "no_accepted_patch"),
        },
        "round_manifests": round_manifests,
    }


def _round_manifest_path(out: Path, round_record: Dict[str, Any]) -> Path:
    artifacts = round_record.get("artifacts", {})
    if isinstance(artifacts, dict) and artifacts.get("self_evolve_manifest"):
        return Path(str(artifacts["self_evolve_manifest"]))
    output_dir = round_record.get("output_dir")
    if output_dir:
        return Path(str(output_dir)) / "self_evolve_manifest.json"
    return out / f"round_{int(round_record.get('round', 0) or 0):02d}" / "self_evolve_manifest.json"


def _aggregate_round_observability(round_manifests: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = {
        "num_experiences": 0,
        "num_failed_experiences": 0,
        "num_weaknesses": 0,
        "num_patch_proposals": 0,
        "num_validation_reports": 0,
        "num_failure_diagnoses": 0,
    }
    for row in round_manifests:
        obs = row.get("observability", {})
        if not isinstance(obs, dict):
            continue
        for key in totals:
            value = obs.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return totals


def _trace_certificate_summary(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    localization = value.get("localization", {}) if isinstance(value.get("localization"), dict) else {}
    proposals = value.get("patch_proposals", {}) if isinstance(value.get("patch_proposals"), dict) else {}
    validation = value.get("validation", {}) if isinstance(value.get("validation"), dict) else {}
    promotion = value.get("promotion", {}) if isinstance(value.get("promotion"), dict) else {}
    return {
        "schema": value.get("schema"),
        "flow": value.get("flow", []),
        "num_localizations": localization.get("num_localizations"),
        "num_no_patch": localization.get("num_no_patch"),
        "num_patches": proposals.get("num_patches"),
        "all_component_scoped": proposals.get("all_component_scoped"),
        "num_reports": validation.get("num_reports"),
        "target_heldout_preservation_present": validation.get("target_heldout_preservation_present"),
        "official_evaluator_used": validation.get("official_evaluator_used"),
        "official_scores_used": validation.get("official_scores_used"),
        "fallback_used": validation.get("fallback_used"),
        "num_promoted": promotion.get("num_promoted"),
        "accepted_only": promotion.get("accepted_only"),
    }


def _aggregate_trace_certificates(round_manifests: List[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = [
        row.get("trace_certificate", {})
        for row in round_manifests
        if isinstance(row, dict) and isinstance(row.get("trace_certificate"), dict)
    ]
    return {
        "schema": "gen_harness.trace_certificate.aggregate.v1",
        "num_rounds": len(round_manifests),
        "num_round_certificates": len([row for row in summaries if row]),
        "rounds": summaries,
        "all_rounds_certified": len(summaries) == len(round_manifests) and all(
            row.get("schema") == "gen_harness.trace_certificate.v1"
            and row.get("all_component_scoped") is not False
            and row.get("target_heldout_preservation_present") is not False
            and row.get("official_evaluator_used") is False
            and row.get("official_scores_used") is False
            and row.get("fallback_used") is False
            and row.get("accepted_only") is not False
            for row in summaries
        ),
    }


def _reference_model_record() -> Dict[str, Any]:
    return {
        "source": "agentic-harness-engineering",
        "used_as": "conceptual_reference_only",
        "copied_runtime": False,
        "local_path_recorded": False,
    }


def _reference_framework_records() -> List[Dict[str, Any]]:
    return [
        {
            "source": "agentic-harness-engineering",
            "paper": "HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution",
            "used_as": "conceptual_reference_only",
            "adopted_invariants": [
                "failure evidence",
                "root-cause diagnosis",
                "targeted harness patch",
                "predicted impact",
                "validation gate",
                "regression-aware promotion",
            ],
            "copied_runtime": False,
            "local_path_recorded": False,
            "official_benchmark_reference_trajectories_allowed": False,
        },
        {
            "source": "JIT",
            "used_as": "conceptual_reference_only",
            "adopted_invariants": [
                "fixed harness component protocol",
                "patch proposal before validation",
                "complete artifact provenance",
            ],
            "copied_runtime": False,
            "local_path_recorded": False,
            "official_benchmark_reference_trajectories_allowed": False,
        },
    ]


def _component_audit_for(harness_root: Any, tasks_path: Any) -> Dict[str, Any]:
    if harness_root:
        return ComponentBoundaryAuditor().audit(harness_root, tasks_path=tasks_path)
    return {"schema": "gen_harness.component_boundary_audit.v1", "passed": False, "issues": [{"code": "missing_harness_root"}]}


def _read_json_optional(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _passed(row: Dict[str, Any]) -> bool:
    result = row.get("verification_result", {}) if isinstance(row, dict) else {}
    return bool(result.get("passed")) if isinstance(result, dict) else False


def _matching_weakness(patch: Dict[str, Any], weaknesses: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    supporting = set(str(v) for v in patch.get("supporting_experience", []) if v)
    for weakness in weaknesses:
        ids = set(str(v) for v in weakness.get("support_experience_ids", []) if v)
        if supporting and supporting & ids:
            return weakness
    return {}


def _change_row(
    patch: Dict[str, Any],
    weakness: Dict[str, Any],
    evidence_summary: Dict[str, Any],
    report: Dict[str, Any],
    diagnoses_by_experience: Dict[str, Dict[str, Any]],
    *,
    promoted_to_harness: bool = False,
) -> Dict[str, Any]:
    validation = report.get("validation", report)
    if not isinstance(validation, dict):
        validation = {}
    supporting = [str(v) for v in patch.get("supporting_experience", []) if v]
    diagnoses = [diagnoses_by_experience[eid] for eid in supporting if eid in diagnoses_by_experience]
    return {
        "change_id": patch.get("patch_id"),
        "component": patch.get("target_component"),
        "artifact": patch.get("changed_artifact"),
        "changed_fields": patch.get("changed_fields", []),
        "operation": patch.get("operation"),
        "failure_evidence": {
            "support_count": weakness.get("support_count") or evidence_summary.get("support_count"),
            "supporting_experience": patch.get("supporting_experience", []),
            "failure_symptom": weakness.get("failure_symptom") or evidence_summary.get("failure_symptom"),
            "constraint_type": weakness.get("constraint_type") or evidence_summary.get("constraint_type"),
            "diagnostics": evidence_summary.get("diagnostics", []),
            "failure_diagnoses": diagnoses[:5],
        },
        "root_cause": (
            weakness.get("hypothesis")
            or evidence_summary.get("hypothesis")
            or (diagnoses[0].get("root_cause") if diagnoses else None)
            or patch.get("payload", {}).get("proposal_rationale", {}).get("why_this_component")
        ),
        "targeted_fix": {
            "payload": patch.get("payload", {}),
            "rollback_boundary": patch.get("rollback_boundary"),
        },
        "predicted_impact": {
            "improvement": patch.get("predicted_improvement"),
            "possible_regression": patch.get("possible_regression"),
            "target_validation_set": patch.get("target_validation_set"),
            "preservation_set": patch.get("preservation_set"),
            "held_out_set": patch.get("held_out_set"),
        },
        "gate": {
            "accepted": bool(report.get("accepted", False)),
            "promoted_to_harness": bool(promoted_to_harness or report.get("promoted_to_harness", False)),
            "validation_report": report,
            "regression_risk": patch.get("regression_risk", {}),
        },
    }
