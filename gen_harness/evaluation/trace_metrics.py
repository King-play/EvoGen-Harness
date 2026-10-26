from __future__ import annotations

from typing import Any, Iterable


RESPONSIBILITIES = ("policy", "tools", "skills", "middleware", "memory")
NO_PATCH = "no_patch"


def summarize_attribution(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute Appendix A.1 attribution and compound-failure metrics."""
    rows = list(cases)
    labels = (*RESPONSIBILITIES, NO_PATCH)
    counts = {gold: {pred: 0 for pred in labels} for gold in labels}
    for index, row in enumerate(rows):
        gold = str(row.get("ground_truth") or row.get("gold_component") or "").strip().lower()
        pred = str(row.get("predicted") or row.get("predicted_component") or "").strip().lower()
        if gold in labels and pred in labels:
            counts[gold][pred] += 1
        elif not row.get("gold_components"):
            raise ValueError(f"attribution case {index} has unsupported labels: gold={gold!r}, predicted={pred!r}")

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for label in RESPONSIBILITIES:
        tp = counts[label][label]
        fp = sum(counts[other][label] for other in labels if other != label)
        fn = sum(counts[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "attribution_accuracy": recall,
            "f1": f1,
        }

    no_patch_total = sum(counts[NO_PATCH].values())
    no_patch_accuracy = counts[NO_PATCH][NO_PATCH] / no_patch_total if no_patch_total else 0.0
    row_normalized_confusion = {
        gold: {
            pred: value / max(1, sum(counts[gold].values()))
            for pred, value in pred_counts.items()
        }
        for gold, pred_counts in counts.items()
    }

    compound = [row for row in rows if len(row.get("gold_components", [])) > 1]
    cause_recalls = []
    both_recovered = 0
    repair_success = 0
    for row in compound:
        gold = {str(value).lower() for value in row["gold_components"]}
        frontier = {str(value).lower() for value in row.get("frontier", [])}
        recovered = {str(value).lower() for value in row.get("recovered_components", [])}
        cause_recalls.append(len(gold & frontier) / len(gold))
        both_recovered += int(gold <= recovered)
        repair_success += int(bool(row.get("repair_success")))

    return {
        "schema": "gen_harness.trace_attribution_metrics.v1",
        "num_cases": len(rows),
        "responsibilities": list(RESPONSIBILITIES),
        "per_class": per_class,
        "macro_attribution_f1": sum(f1_values) / len(f1_values),
        "no_patch_accuracy": no_patch_accuracy,
        "confusion_counts": counts,
        "row_normalized_confusion": row_normalized_confusion,
        "compound": {
            "num_cases": len(compound),
            "cause_recall_at_frontier": sum(cause_recalls) / len(cause_recalls) if cause_recalls else 0.0,
            "both_recovered": both_recovered / len(compound) if compound else 0.0,
            "repair_success": repair_success / len(compound) if compound else 0.0,
        },
    }


def summarize_responsibility_updates(
    patches: Iterable[dict[str, Any]],
    reports: Iterable[dict[str, Any]],
    evolution_steps: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Compute paper Update Share, Commit Rate, and retained held-out gain."""
    patch_rows = list(patches)
    report_by_id = {str(row.get("patch_id")): row for row in reports}
    promoted = {
        str(row.get("patch_id"))
        for row in evolution_steps
        if row.get("promoted_to_harness") is True
    }
    proposed = {name: 0 for name in RESPONSIBILITIES}
    accepted = {name: 0 for name in RESPONSIBILITIES}
    committed = {name: 0 for name in RESPONSIBILITIES}
    heldout_gains = {name: [] for name in RESPONSIBILITIES}
    for patch in patch_rows:
        component = str(patch.get("target_component") or "")
        if component not in proposed:
            continue
        proposed[component] += 1
        patch_id = str(patch.get("patch_id"))
        report = report_by_id.get(patch_id, {})
        if report.get("accepted") is True:
            accepted[component] += 1
        if patch_id in promoted:
            committed[component] += 1
            gain = report.get("heldout_delta")
            if report.get("accepted") is True and isinstance(gain, (int, float)) and not isinstance(gain, bool):
                heldout_gains[component].append(float(gain))
    total_committed = sum(committed.values())
    return {
        "schema": "gen_harness.trace_responsibility_metrics.v1",
        "num_proposed": sum(proposed.values()),
        "num_accepted": sum(accepted.values()),
        "num_committed": total_committed,
        "per_responsibility": {
            name: {
                "proposed": proposed[name],
                "accepted": accepted[name],
                "committed": committed[name],
                "update_share": committed[name] / total_committed if total_committed else 0.0,
                "commit_rate": committed[name] / proposed[name] if proposed[name] else 0.0,
                "mean_heldout_gain": (
                    sum(heldout_gains[name]) / len(heldout_gains[name])
                    if heldout_gains[name]
                    else 0.0
                ),
            }
            for name in RESPONSIBILITIES
        },
    }


def summarize_paired_outcomes(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute Appendix A.1 held-out gain and preservation regression rate.

    Each row represents one task after averaging over its matched stochastic
    conditions and contains `split`, `before_score`, `after_score`, and an
    optional per-task `threshold` (default 0.5).
    """
    paired = list(rows)
    heldout_deltas = []
    preservation_previously_successful = 0
    preservation_regressed = 0
    for index, row in enumerate(paired):
        split = str(row.get("split") or "")
        before = row.get("before_score")
        after = row.get("after_score")
        threshold = row.get("threshold", 0.5)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (before, after, threshold)):
            raise ValueError(f"paired outcome {index} requires numeric before_score, after_score, and threshold")
        if split == "heldout":
            heldout_deltas.append(float(after) - float(before))
        elif split == "preservation" and float(before) >= float(threshold):
            preservation_previously_successful += 1
            preservation_regressed += int(float(after) < float(threshold))
    return {
        "schema": "gen_harness.trace_paired_outcome_metrics.v1",
        "num_rows": len(paired),
        "heldout_gain": sum(heldout_deltas) / len(heldout_deltas) if heldout_deltas else 0.0,
        "preservation_previously_successful": preservation_previously_successful,
        "preservation_regressed": preservation_regressed,
        "regression_rate": (
            preservation_regressed / preservation_previously_successful
            if preservation_previously_successful
            else 0.0
        ),
        "regression_percent": (
            100.0 * preservation_regressed / preservation_previously_successful
            if preservation_previously_successful
            else 0.0
        ),
    }


def summarize_efficiency(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Average the paper's per-prompt inference and search-efficiency metrics.

    Input rows are prompt-level observations. `candidate_edits` is optional
    because it applies to the search-efficiency analysis rather than ordinary
    frozen-harness inference.
    """
    observations = list(rows)
    required = ("wall_clock_seconds", "generator_calls", "llm_vlm_calls")
    values: dict[str, list[float]] = {name: [] for name in (*required, "candidate_edits")}
    for index, row in enumerate(observations):
        metric_row = row.get("efficiency", row.get("evolution_efficiency", row))
        if not isinstance(metric_row, dict):
            raise ValueError(f"efficiency row {index} must be an object or contain an efficiency object")
        for name in required:
            value = metric_row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
                raise ValueError(f"efficiency row {index} requires non-negative numeric {name}")
            values[name].append(float(value))
        candidate_edits = metric_row.get(
            "candidate_edits",
            metric_row.get("candidate_edits_per_evolution_task", metric_row.get("candidate_edits_evaluated")),
        )
        if candidate_edits is not None:
            if isinstance(candidate_edits, bool) or not isinstance(candidate_edits, (int, float)) or float(candidate_edits) < 0:
                raise ValueError(f"efficiency row {index} requires non-negative numeric candidate_edits")
            values["candidate_edits"].append(float(candidate_edits))

    def mean(name: str) -> float | None:
        return sum(values[name]) / len(values[name]) if values[name] else None

    return {
        "schema": "gen_harness.trace_efficiency_metrics.v1",
        "num_prompts": len(observations),
        "average_wall_clock_seconds_per_prompt": mean("wall_clock_seconds") or 0.0,
        "average_generator_calls_per_prompt": mean("generator_calls") or 0.0,
        "average_llm_vlm_calls_per_prompt": mean("llm_vlm_calls") or 0.0,
        "average_candidate_edits_per_task": mean("candidate_edits"),
        "num_candidate_edit_observations": len(values["candidate_edits"]),
    }
