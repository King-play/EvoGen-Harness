from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .schema import VisualExperience


DIAGNOSIS_SCHEMA = "gen_harness.failure_diagnosis.v1"


@dataclass
class FailureDiagnosis:
    schema: str
    experience_id: str
    task_id: str
    constraint_id: str
    constraint_type: str
    failure_symptom: str
    suspected_component: str
    confidence: float | None
    root_cause: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    next_action: Dict[str, Any] = field(default_factory=dict)
    leakage_safe: bool = True
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "experience_id": self.experience_id,
            "task_id": self.task_id,
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type,
            "failure_symptom": self.failure_symptom,
            "suspected_component": self.suspected_component,
            "confidence": self.confidence,
            "root_cause": self.root_cause,
            "evidence": self.evidence,
            "next_action": self.next_action,
            "leakage_safe": self.leakage_safe,
            "fallback_used": self.fallback_used,
        }


class FailureDebugger:
    """Record observed failures and any explicit LLM localization evidence.

    This debugger consumes only Gen-Harness runtime traces and internal
    inspection outputs.  It never reads benchmark answers, official evaluator
    VQA rows, or post-hoc official scores.  It is therefore safe to use inside
    slow-loop evolution.
    """

    def diagnose_many(self, experiences: Iterable[VisualExperience]) -> List[FailureDiagnosis]:
        return [self.diagnose(exp) for exp in experiences if not exp.verification_result.get("passed")]

    def diagnose(self, exp: VisualExperience) -> FailureDiagnosis:
        symptom = str(exp.failure_symptom or exp.verification_result.get("symptom") or "unknown")
        ctype = str(exp.constraint_type or "unknown")
        evidence = self._evidence(exp, symptom)
        component = self._component(exp, symptom, evidence)
        root_cause = self._root_cause(exp, component, symptom, evidence)
        return FailureDiagnosis(
            schema=DIAGNOSIS_SCHEMA,
            experience_id=exp.experience_id,
            task_id=exp.task_id,
            constraint_id=exp.constraint_id,
            constraint_type=ctype,
            failure_symptom=symptom,
            suspected_component=component,
            confidence=self._confidence(exp, component, evidence),
            root_cause=root_cause,
            evidence=evidence,
            next_action=self._next_action(component, symptom, ctype),
            leakage_safe=self._leakage_safe(exp),
            fallback_used=self._fallback_used(exp),
        )

    def summary(self, diagnoses: Iterable[FailureDiagnosis]) -> Dict[str, Any]:
        rows = list(diagnoses)
        by_component: Dict[str, int] = {}
        by_symptom: Dict[str, int] = {}
        for row in rows:
            by_component[row.suspected_component] = by_component.get(row.suspected_component, 0) + 1
            by_symptom[row.failure_symptom] = by_symptom.get(row.failure_symptom, 0) + 1
        return {
            "schema": "gen_harness.failure_diagnosis_summary.v1",
            "num_diagnoses": len(rows),
            "by_component": dict(sorted(by_component.items())),
            "by_symptom": dict(sorted(by_symptom.items())),
            "leakage_safe": all(row.leakage_safe for row in rows),
            "fallback_used": any(row.fallback_used for row in rows),
        }

    def _component(self, exp: VisualExperience, symptom: str, evidence: Dict[str, Any]) -> str:
        return self._attribution_component(exp) or "unknown"

    def _root_cause(self, exp: VisualExperience, component: str, symptom: str, evidence: Dict[str, Any]) -> str:
        attribution = exp.metadata.get("attribution", {}) if isinstance(exp.metadata, dict) else {}
        localization = attribution.get("localization", {}) if isinstance(attribution, dict) else {}
        rationale = localization.get("rationale") if isinstance(localization, dict) else None
        if component != "unknown" and rationale:
            return str(rationale)
        return f"Observed {symptom}; the persistent cause has not been localized."

    def _next_action(self, component: str, symptom: str, ctype: str) -> Dict[str, Any]:
        return {"type": "review_localized_responsibility" if component != "unknown" else "collect_more_internal_evidence", "target": ctype, "symptom": symptom}

    def _evidence(self, exp: VisualExperience, symptom: str) -> Dict[str, Any]:
        result = exp.verification_result if isinstance(exp.verification_result, dict) else {}
        decisions = exp.component_decisions if isinstance(exp.component_decisions, dict) else {}
        workflow = decisions.get("skills", {}) if isinstance(decisions.get("skills"), dict) else {}
        middleware = decisions.get("middleware", {}) if isinstance(decisions.get("middleware"), dict) else {}
        tool_calls = exp.tool_calls if isinstance(exp.tool_calls, list) else []
        fast_loop = exp.metadata.get("fast_loop") if isinstance(exp.metadata, dict) else None
        evidence: Dict[str, Any] = {
            "inspection_symptom": symptom,
            "constraint_text": exp.constraint_text,
            "verification_confidence": result.get("confidence"),
        }
        for key in ("missing_required_objects", "count_mismatches", "forbidden_objects_present", "attribute_mismatches", "spatial_mismatches", "action_mismatches"):
            if result.get(key):
                evidence[key] = result[key]
        missing_steps = workflow.get("missing_steps")
        if missing_steps:
            evidence["missing_steps"] = missing_steps
        routing_warning = middleware.get("routing_warning")
        if routing_warning:
            evidence["routing_warning"] = routing_warning
        if tool_calls:
            evidence["tool_names"] = [str(row.get("tool_name")) for row in tool_calls if isinstance(row, dict)]
        if isinstance(fast_loop, dict):
            attempts = fast_loop.get("attempts", [])
            attempt_errors = [
                str(attempt.get("error"))
                for attempt in attempts
                if isinstance(attempt, dict) and attempt.get("error")
            ] if isinstance(attempts, list) else []
            evidence["fast_loop"] = {
                "num_attempts": len(attempts) if isinstance(attempts, list) else 0,
                "passed": bool(fast_loop.get("passed")),
                "selected_attempt": fast_loop.get("selected_attempt"),
                "last_decision": self._last_fast_loop_decision(attempts),
            }
            if attempt_errors:
                evidence["attempt_error_count"] = len(attempt_errors)
                evidence["attempt_errors"] = attempt_errors[:5]
        return evidence


    def _attribution_component(self, exp: VisualExperience) -> str | None:
        attribution = exp.metadata.get("attribution", {}) if isinstance(exp.metadata, dict) else {}
        if not isinstance(attribution, dict) or attribution.get("mode") != "llm_harness_localizer":
            return None
        localization = attribution.get("localization", {})
        if not isinstance(localization, dict):
            return None
        frontier = localization.get("frontier", [])
        if not localization.get("no_patch_preferred") and frontier:
            component = str(frontier[0])
            if component in {"policy", "tools", "skills", "middleware", "memory"}:
                return component
        return None

    def _confidence(self, exp: VisualExperience, component: str, evidence: Dict[str, Any]) -> float | None:
        attribution = exp.metadata.get("attribution", {}) if isinstance(exp.metadata, dict) else {}
        localization = attribution.get("localization", {}) if isinstance(attribution, dict) else {}
        scores = localization.get("scores", {}) if isinstance(localization, dict) else {}
        value = scores.get(component) if isinstance(scores, dict) else None
        if isinstance(value, (float, int)) and not isinstance(value, bool) and 0 <= value <= 1:
            return float(value)
        return None

    def _last_fast_loop_decision(self, attempts: Any) -> Dict[str, Any] | None:
        if not isinstance(attempts, list) or not attempts:
            return None
        last = attempts[-1]
        if isinstance(last, dict) and isinstance(last.get("repair_decision"), dict):
            return last["repair_decision"]
        return None

    def _leakage_safe(self, exp: VisualExperience) -> bool:
        text = repr({"metadata": exp.metadata, "verification_result": exp.verification_result}).lower()
        forbidden = ("vqa_list", "official_answer", "soft_tifa_answer", "question_answers")
        return not any(token in text for token in forbidden)

    def _fallback_used(self, exp: VisualExperience) -> bool:
        text = repr({"metadata": exp.metadata, "tool_calls": exp.tool_calls, "visual_result": exp.visual_result}).lower()
        return "fallback_image" in text or "fallback image" in text
