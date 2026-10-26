from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from ..json_extract import parse_json_object_from_text
from ..leakage_guard import reject_forbidden_payload_context
from ..inspection import InspectionRunner
from ..schema import GenerationTask, VisualConstraint
from ..tool_payloads import (
    constraints_for_external_tool,
    public_constraint_ids,
    public_inspection_payload,
)


class OpenAIVisionInspectionRunner(InspectionRunner):
    """OpenAI-compatible vision judge for Gen-Harness constraint results."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config["base_url"].rstrip("/")
        self.endpoint = config.get("endpoint") or f"{self.base_url}/chat/completions"
        self.headers = dict(config.get("headers", {}))
        self.timeout = int(config.get("timeout", 300))

    def inspect(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int = 0) -> Dict[str, Dict[str, Any]]:
        image_uri = generation.get("image_uri")
        if not image_uri:
            raise ValueError(f"generation result does not contain image_uri: {generation}")
        constraints = self._constraints_for_inspection(task, program)
        public_constraint_ids = self._public_constraint_ids(constraints)
        body = {
            "model": self.config["model"],
            "temperature": self.config.get("temperature", 0),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._user_prompt(task, program, constraints, public_constraint_ids)},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                    ],
                },
            ],
        }
        extra_body = self.config.get("extra_body", {})
        reject_forbidden_payload_context(extra_body, context="vision inspection extra_body")
        for key, value in extra_body.items():
            body[key] = value
        result = self._post_json(self.endpoint, body)
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = self._parse_json(content)
        constraint_results = parsed.get("constraint_results", parsed)
        image_metrics = parsed.get("image_metrics", {}) if isinstance(parsed, dict) else {}
        return self._normalize_results(
            task,
            program,
            constraint_results,
            image_metrics,
            constraints=constraints,
            public_constraint_ids=public_constraint_ids,
        )

    def _system_prompt(self) -> str:
        return (
            "You are a strict visual quality inspector. Return only valid JSON. "
            "Evaluate the actual generated image, not the prompt text alone. "
            "Return top-level image_metrics with scores from 0 to 1 where higher is better: "
            "overall_quality, aesthetic_quality, prompt_alignment, identity_consistency, "
            "style_consistency, artifact_free, text_logo_absence. "
            "Also judge each constraint independently. Each constraint result must contain "
            "passed:boolean, symptom:string, confidence:number between 0 and 1."
        )

    def _user_prompt(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        constraints: list[VisualConstraint] | None = None,
        public_constraint_ids: Dict[str, str] | None = None,
    ) -> str:
        constraints = constraints if constraints is not None else self._constraints_for_inspection(task, program)
        public_constraint_ids = public_constraint_ids or self._public_constraint_ids(constraints)
        payload = public_inspection_payload(
            task,
            program,
            {},
            constraints=constraints,
            public_ids=public_constraint_ids,
        )
        payload.pop("image_uri", None)
        payload.pop("image_path", None)
        payload.pop("generation", None)
        expected = {
            "image_metrics": {
                "overall_quality": "number",
                "aesthetic_quality": "number",
                "prompt_alignment": "number",
                "identity_consistency": "number",
                "style_consistency": "number",
                "artifact_free": "number",
                "text_logo_absence": "number",
            },
            "constraint_results": {
                "constraint_id": {"passed": "boolean", "symptom": "string", "confidence": "number"}
            },
        }
        return (
            "Inspect the generated image for the following task. "
            "Return JSON exactly in this shape: "
            + json.dumps(expected, ensure_ascii=False)
            + ". Use 0 for identity_consistency or style_consistency when the task has no identity/style requirement.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def _constraints_for_inspection(self, task: GenerationTask, program: Dict[str, Any]) -> list[VisualConstraint]:
        return constraints_for_external_tool(task, program)

    def _normalize_results(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        rows: Dict[str, Any],
        image_metrics: Dict[str, Any] | None = None,
        *,
        constraints: list[VisualConstraint] | None = None,
        public_constraint_ids: Dict[str, str] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        metric_names = {
            "overall_quality",
            "aesthetic_quality",
            "prompt_alignment",
            "identity_consistency",
            "style_consistency",
            "artifact_free",
            "text_logo_absence",
        }
        normalized_image_metrics = self._normalize_metrics(image_metrics or {}, metric_names)
        out: Dict[str, Dict[str, Any]] = {}
        constraints = constraints if constraints is not None else self._constraints_for_inspection(task, program)
        public_constraint_ids = public_constraint_ids or self._public_constraint_ids(constraints)
        for c in constraints:
            public_id = public_constraint_ids[c.constraint_id]
            raw = rows.get(public_id, {}) if isinstance(rows, dict) else {}
            if not raw and isinstance(rows, dict):
                raw = rows.get(c.constraint_id, {})
            row_metrics = dict(normalized_image_metrics)
            if not row_metrics:
                row_metrics = self._normalize_metrics(raw.get("image_metrics", {}) if isinstance(raw, dict) else {}, metric_names)
            if not row_metrics:
                row_metrics = self._normalize_metrics(raw if isinstance(raw, dict) else {}, metric_names)
            out[c.constraint_id] = {
                "passed": bool(raw.get("passed", False)),
                "symptom": str(raw.get("symptom", "pass" if raw.get("passed") else "failed_visual_check")),
                "confidence": self._score(raw.get("confidence", 0.0)),
                "image_metrics": row_metrics,
            }
            for name, value in row_metrics.items():
                out[c.constraint_id][name] = value
            for key, value in raw.items():
                if key not in out[c.constraint_id]:
                    out[c.constraint_id][key] = value
        return out

    def _public_constraint_ids(self, constraints: list[VisualConstraint]) -> Dict[str, str]:
        return public_constraint_ids(constraints)

    def _normalize_metrics(self, raw: Dict[str, Any], metric_names: set[str]) -> Dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        return {name: self._score(raw[name]) for name in metric_names if name in raw}

    def _score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, score))

    def _parse_json(self, content: str) -> Dict[str, Any]:
        return parse_json_object_from_text(content)

    def _post_json(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        max_retries = int(self.config.get("max_retries", 2))
        retry_statuses = set(self.config.get("retry_statuses", [429, 500, 502, 503, 504]))
        last_error = None
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"vision inspection request failed: HTTP {exc.code}: {detail}")
                if exc.code not in retry_statuses or attempt >= max_retries:
                    raise last_error from exc
                time.sleep(float(self.config.get("retry_backoff", 2.0)) * (attempt + 1))
            except (urllib.error.URLError, http.client.RemoteDisconnected) as exc:
                last_error = RuntimeError(f"vision inspection request failed: connection error: {exc}")
                if attempt >= max_retries:
                    raise last_error from exc
                time.sleep(float(self.config.get("retry_backoff", 2.0)) * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"vision inspection request failed: no response")
