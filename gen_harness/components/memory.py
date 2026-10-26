from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from ..io import append_jsonl, read_jsonl
from ..schema import PatchManifest, VisualExperience, Weakness

MAX_MEMORY_ROW_BYTES = 131_072


class VisualMemory:
    """Persistent visual experience memory.

    Memory is an active component: it stores failures, retrieves related cases,
    records validated patches, and preserves rejected/regression-inducing edits.
    """

    component_type = "memory"

    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def write_experiences(self, experiences: Iterable[VisualExperience]) -> None:
        rows = list(experiences)
        append_jsonl(self.memory_dir / "visual_experiences.jsonl", [self._bounded_experience_row(e) for e in rows])
        failures = [self._bounded_experience_row(e) for e in rows if not e.verification_result.get("passed")]
        if failures:
            append_jsonl(self.memory_dir / "failure_clusters.jsonl", failures)

    def write_weaknesses(self, weaknesses: Iterable[Weakness]) -> None:
        append_jsonl(self.memory_dir / "weaknesses.jsonl", [w.to_dict() for w in weaknesses])

    def write_patch_result(self, patch: PatchManifest, report: Dict[str, object]) -> None:
        row = patch.to_dict()
        row["validation_report"] = report
        if self._validated_patch_report(report):
            append_jsonl(self.memory_dir / "validated_patches.jsonl", [row])
        else:
            append_jsonl(self.memory_dir / "regression_cases.jsonl", [row])

    def _validated_patch_report(self, report: Dict[str, object]) -> bool:
        if not isinstance(report, dict) or report.get("accepted") is not True:
            return False
        scope = report.get("validation_scope")
        guards = report.get("validation_guards")
        if not isinstance(scope, dict) or not isinstance(guards, dict):
            return False
        if scope.get("target_heldout_preservation_present") is not True:
            return False
        if not report.get("seeds"):
            return False
        required_splits = {"target", "heldout", "preservation"}
        before = report.get("before")
        after = report.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            return False
        if not required_splits <= set(before) or not required_splits <= set(after):
            return False
        return (
            guards.get("target_improved") is True
            and guards.get("heldout_improved") is True
            and guards.get("heldout_non_regressing") is True
            and guards.get("preservation_regression_bounded") is True
            and guards.get("positive_utility") is True
            and guards.get("matched_stochastic_conditions") is True
            and guards.get("official_evaluator_used") is False
            and guards.get("official_scores_used") is False
            and guards.get("fallback_used") is False
            and guards.get("internal_verifier_only") is True
        )

    def retrieve_failures(self, *, component: str | None = None, constraint_type: str | None = None, limit: int = 50) -> List[Dict[str, object]]:
        rows = read_jsonl(self.memory_dir / "failure_clusters.jsonl")
        if component:
            rows = [r for r in rows if r.get("suspected_component") == component]
        if constraint_type:
            rows = [r for r in rows if r.get("constraint_type") == constraint_type]
        return rows[-limit:]

    def retrieve_validated_patches(
        self,
        *,
        component: str | None = None,
        constraint_types: Iterable[str] | None = None,
        failure_symptoms: Iterable[str] | None = None,
        limit: int = 50,
    ) -> List[Dict[str, object]]:
        rows = read_jsonl(self.memory_dir / "validated_patches.jsonl")
        if component:
            rows = [r for r in rows if r.get("target_component") == component]
        rows = self._filter_patch_rows(rows, constraint_types=constraint_types, failure_symptoms=failure_symptoms)
        return rows[-limit:]

    def retrieve_regression_cases(
        self,
        *,
        component: str | None = None,
        constraint_types: Iterable[str] | None = None,
        failure_symptoms: Iterable[str] | None = None,
        limit: int = 50,
    ) -> List[Dict[str, object]]:
        rows = read_jsonl(self.memory_dir / "regression_cases.jsonl")
        if component:
            rows = [r for r in rows if r.get("target_component") == component]
        rows = self._filter_patch_rows(rows, constraint_types=constraint_types, failure_symptoms=failure_symptoms)
        return rows[-limit:]

    def _filter_patch_rows(
        self,
        rows: List[Dict[str, object]],
        *,
        constraint_types: Iterable[str] | None = None,
        failure_symptoms: Iterable[str] | None = None,
    ) -> List[Dict[str, object]]:
        cset = {str(v) for v in constraint_types or []}
        sset = {str(v) for v in failure_symptoms or []}
        if cset:
            rows = [r for r in rows if not r.get("evidence_summary") or r.get("evidence_summary", {}).get("constraint_type") in cset]
        if sset:
            rows = [r for r in rows if not r.get("evidence_summary") or r.get("evidence_summary", {}).get("failure_symptom") in sset]
        return rows

    def retrieve_prompt_guidelines(
        self,
        *,
        task_family: str | None = None,
        prompt_family: str | None = None,
        constraint_types: Iterable[str] | None = None,
        failure_symptoms: Iterable[str] | None = None,
        limit: int = 20,
    ) -> List[Dict[str, object]]:
        rows = read_jsonl(self.memory_dir / "prompt_guidelines.jsonl")
        if task_family:
            rows = [r for r in rows if r.get("task_family") in {task_family, "*"}]
        if prompt_family:
            rows = [r for r in rows if r.get("prompt_family") in {prompt_family, "*", None, ""}]
        cset = {str(v) for v in constraint_types or []}
        if cset:
            rows = [r for r in rows if not r.get("constraint_type") or r.get("constraint_type") in cset or r.get("constraint_type") == "*"]
        sset = {str(v) for v in failure_symptoms or []}
        if sset:
            rows = [r for r in rows if not r.get("failure_symptom") or r.get("failure_symptom") in sset or r.get("failure_symptom") == "*"]
        return rows[-limit:]

    def describe(self) -> Dict[str, object]:
        return {"component_name": "visual_memory", "component_type": "memory", "memory_dir": str(self.memory_dir)}

    def _bounded_experience_row(self, exp: VisualExperience) -> Dict[str, object]:
        row = exp.to_compact_dict()
        size = len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        if size <= MAX_MEMORY_ROW_BYTES:
            return row
        minimal: Dict[str, object] = {
            "experience_id": exp.experience_id,
            "task_id": exp.task_id,
            "constraint_id": exp.constraint_id,
            "constraint_type": exp.constraint_type,
            "constraint_text": exp.constraint_text[:500],
            "failure_symptom": exp.failure_symptom,
            "suspected_component": exp.suspected_component,
            "visual_result": {
                key: value
                for key, value in (exp.visual_result or {}).items()
                if key in {"image_uri", "image_path", "seed", "backend", "selected_model"}
            },
            "verification_result": {
                key: value
                for key, value in (exp.verification_result or {}).items()
                if key in {
                    "passed",
                    "symptom",
                    "confidence",
                    "selection_score",
                    "constraint_score",
                    "evidence_status",
                    "inspection_source",
                    "merge_strategy",
                }
            },
            "metadata": {
                key: value
                for key, value in (exp.metadata or {}).items()
                if key in {"split", "family", "source", "seed"}
            },
            "memory_compaction": {
                "schema": "gen_harness.memory_compaction.v1",
                "reason": "compact_row_exceeded_max_bytes",
                "compact_row_bytes": size,
                "max_memory_row_bytes": MAX_MEMORY_ROW_BYTES,
            },
        }
        return minimal
