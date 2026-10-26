from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from .base import HarnessComponent
from ..io import expand_env_vars, read_json
from ..json_extract import parse_json_object_from_text
from ..leakage_guard import (
    reject_forbidden_payload_context,
    reject_forbidden_trace_text,
    sanitize_formal_trace_metadata,
)
from ..schema import GenerationTask, VisualConstraint
from ..tool_payloads import public_references_for_llm


class PolicyExtractor:
    """Interface for converting natural-language intent into visual constraints."""

    name = "policy_extractor"

    def extract(self, task: GenerationTask, policy_artifact: Dict[str, Any]) -> List[VisualConstraint]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"component_name": self.name, "component_type": "policy_extractor"}


class LLMPolicyExtractor(PolicyExtractor):
    """LLM-backed prompt-to-contract extractor.

    The extractor converts public task prompt text into typed visual
    constraints. It never receives official evaluator outputs, hidden labels,
    benchmark answers, image scores, or task-specific post-hoc feedback.
    """

    name = "openai_chat_policy_extractor"

    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict) or config.get("type") != "openai_chat_policy_extractor":
            raise ValueError("LLMPolicyExtractor requires an openai_chat_policy_extractor config")
        self.config = config
        self.base_url = str(config["base_url"]).rstrip("/")
        self.endpoint = str(config.get("endpoint") or f"{self.base_url}/chat/completions")
        self.model = str(config["model"])
        self.timeout = int(config.get("timeout", 120))
        self.max_schema_retries = max(0, int(config.get("max_schema_retries", 2)))
        self.max_request_retries = max(0, int(config.get("max_request_retries", 2)))

    def extract(self, task: GenerationTask, policy_artifact: Dict[str, Any]) -> List[VisualConstraint]:
        ontology = set(policy_artifact.get("constraint_ontology", []))
        correction: str | None = None
        for attempt in range(self.max_schema_retries + 1):
            request_error: Exception | None = None
            for request_attempt in range(self.max_request_retries + 1):
                try:
                    if correction is None:
                        parsed = self._request_constraints(task, policy_artifact)
                    else:
                        parsed = self._request_constraints(task, policy_artifact, correction=correction)
                    request_error = None
                    break
                except (TimeoutError, urllib.error.URLError) as exc:
                    request_error = exc
                    if request_attempt >= self.max_request_retries:
                        raise
            if request_error is not None:
                raise request_error
            try:
                rows = parsed.get("constraints")
                if not isinstance(rows, list):
                    raise ValueError("LLM policy extractor JSON must contain a constraints list")
                constraints = [
                    self._constraint_from_row(row, index, ontology=ontology)
                    for index, row in enumerate(rows)
                ]
                self._validate_constraint_references(constraints)
                return constraints
            except ValueError as exc:
                if not self._is_retryable_schema_error(exc):
                    raise
                if attempt >= self.max_schema_retries:
                    raise
                correction = (
                    "The previous JSON failed strict contract validation. "
                    f"Repair the complete constraints list without omitting any visual requirement. "
                    f"Validation error: {exc}"
                )
        raise RuntimeError("unreachable policy extraction retry state")

    @staticmethod
    def _is_retryable_schema_error(exc: ValueError) -> bool:
        message = str(exc).lower()
        return "forbidden token" not in message and "provenance" not in message

    def describe(self) -> Dict[str, Any]:
        row = super().describe()
        row["model"] = self.model
        row["official_feedback_used"] = False
        row["fallback_used"] = False
        row["fresh_extraction_per_task"] = True
        return row

    def _request_constraints(
        self,
        task: GenerationTask,
        policy_artifact: Dict[str, Any],
        *,
        correction: str | None = None,
    ) -> Dict[str, Any]:
        body = {
            "model": self.model,
            "temperature": self.config.get("temperature", 0.0),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": "Extract visual constraints from the public user prompt. Return strict JSON only.",
                            "expected_schema": {
                                "constraints": [
                                    {
                                        "constraint_id": "policy_llm_0",
                                        "text": "verifiable visual requirement",
                                        "constraint_type": "one of ontology",
                                        "target": "optional target string or null",
                                        "required_reference_role": None,
                                        "verification_rule": "optional verifier instruction or null",
                                        "preservation": [],
                                        "metadata": {
                                            "required_objects": [],
                                            "attribute_bindings": [],
                                            "spatial_relations": [],
                                            "action_relations": [],
                                            "prompt_family": "optional declared family from the harness skill artifact or null"
                                        },
                                    }
                                ]
                            },
                            "allowed_constraint_types": sorted(policy_artifact.get("constraint_ontology", [])),
                            "task": {
                                "prompt": task.prompt,
                                "task_family": task.task_family,
                                "metadata": _safe_prompt_metadata(task.metadata),
                                "references": public_references_for_llm(task),
                            },
                            "requirements": [
                                "Use only prompt text, public metadata, and declared references.",
                                "Do not use official benchmark scores, hidden labels, benchmark answers, evaluator results, post-hoc selection, fallback images, or task-id-specific rules.",
                                "Every count constraint must include metadata.required_objects rows with class and integer count.",
                                "Every attribute constraint must include metadata.attribute_bindings rows with object and attribute.",
                                "Every spatial constraint must include metadata.spatial_relations rows with subject, relation, and object.",
                                "Every action constraint must include metadata.action_relations rows with subject, action, and object.",
                                "Treat metadata.required_objects.class values as the canonical entity identifiers for this contract.",
                                "Use the exact same canonical entity identifier in every attribute, spatial, and action row; do not pluralize, abbreviate, or paraphrase it.",
                                "A canonical class is the bare object noun phrase. Keep color, material, texture, and shape modifiers out of class values: emit class flower plus attribute glass, never class glass flower.",
                                "Every explicit visual modifier in the public prompt must appear in metadata.attribute_bindings for the affected canonical class.",
                                "Spatial and action constraints must also include metadata.required_objects rows for every entity they mention.",
                                "Use empty arrays for metadata fields that are not relevant; never omit the metadata object.",
                                "If the public prompt and typed constraints clearly select a reusable prompt family, set metadata.prompt_family to that family name; otherwise use null. Never invent a task-specific family.",
                                "Prefer compact typed constraints: count/object requirements, attributes, spatial relations, actions, text, negative, quality, aesthetic, prompt_alignment.",
                            ],
                            **({"schema_correction": correction} if correction else {}),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
        }
        extra_body = self.config.get("extra_body", {})
        reject_forbidden_payload_context(extra_body, context="LLM policy extra_body")
        for key, value in extra_body.items():
            body[key] = value
        response = _post_policy_chat_json(self.config, self.endpoint, body, self.timeout)
        parsed = parse_json_object_from_text(_policy_chat_content(response))
        _reject_policy_forbidden(parsed)
        return parsed

    def _constraint_from_row(self, row: Any, index: int, *, ontology: set[str]) -> VisualConstraint:
        if not isinstance(row, dict):
            raise ValueError("each LLM policy constraint must be an object")
        _reject_policy_forbidden(row)
        constraint_type = str(row.get("constraint_type", "")).strip()
        if constraint_type not in ontology:
            raise ValueError(f"LLM policy constraint type is not in ontology: {constraint_type}")
        constraint_id = str(row.get("constraint_id") or f"policy_llm_{index}")
        text = str(row.get("text") or "").strip()
        if not text:
            raise ValueError("LLM policy constraint text must be non-empty")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata = dict(metadata)
        for key in ("required_objects", "attribute_bindings", "spatial_relations", "action_relations"):
            value = metadata.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"LLM policy metadata.{key} must be a list")
            metadata[key] = value
        self._validate_structured_metadata(constraint_type, metadata)
        metadata["source"] = "policy_extractor.openai_chat"
        metadata["official_feedback_used"] = False
        preservation = row.get("preservation", [])
        if preservation is None:
            preservation = []
        if not isinstance(preservation, list) or not all(isinstance(item, str) for item in preservation):
            raise ValueError("LLM policy preservation must be a list of strings")
        return VisualConstraint(
            constraint_id=constraint_id,
            text=text,
            constraint_type=constraint_type,
            target=_optional_str(row.get("target")),
            required_reference_role=_optional_str(row.get("required_reference_role")),
            verification_rule=_optional_str(row.get("verification_rule")),
            preservation=list(preservation),
            metadata=metadata,
        )

    @staticmethod
    def _validate_structured_metadata(constraint_type: str, metadata: Dict[str, Any]) -> None:
        if constraint_type == "count":
            rows = metadata["required_objects"]
            if not rows or any(
                not isinstance(row, dict)
                or not str(row.get("class", "")).strip()
                or int(row.get("count", 0) or 0) < 1
                for row in rows
            ):
                raise ValueError("LLM count constraint requires metadata.required_objects with positive counts")
        elif constraint_type == "attribute":
            rows = metadata["attribute_bindings"]
            if not rows or any(
                not isinstance(row, dict)
                or not str(row.get("object", "")).strip()
                or not str(row.get("attribute", "")).strip()
                for row in rows
            ):
                raise ValueError("LLM attribute constraint requires metadata.attribute_bindings")
        elif constraint_type == "spatial":
            rows = metadata["spatial_relations"]
            if not rows or any(
                not isinstance(row, dict)
                or not str(row.get("subject", "")).strip()
                or not str(row.get("relation", "")).strip()
                or not str(row.get("object", "")).strip()
                for row in rows
            ):
                raise ValueError("LLM spatial constraint requires metadata.spatial_relations")
        elif constraint_type == "action":
            rows = metadata["action_relations"]
            if not rows or any(
                not isinstance(row, dict)
                or not str(row.get("subject", "")).strip()
                or not str(row.get("action", "")).strip()
                or not str(row.get("object", "")).strip()
                for row in rows
            ):
                raise ValueError("LLM action constraint requires metadata.action_relations")

    @staticmethod
    def _validate_constraint_references(constraints: List[VisualConstraint]) -> None:
        """Require one exact entity vocabulary across the complete contract.

        The policy extractor is the owner of the entity vocabulary.  Rejecting
        a relation that uses a surface variant such as ``birds`` when the
        inventory declares ``bird`` forces the LLM to repair its structured
        output instead of silently routing the scene to a weaker backend.
        """

        entity_names = {
            str(row.get("class", "")).strip().lower()
            for constraint in constraints
            for row in constraint.metadata.get("required_objects", [])
            if isinstance(row, dict) and str(row.get("class", "")).strip()
        }
        if not entity_names:
            return

        references = []
        for constraint in constraints:
            metadata = constraint.metadata
            references.extend(
                ("attribute", row.get("object"))
                for row in metadata.get("attribute_bindings", [])
                if isinstance(row, dict)
            )
            references.extend(
                ("spatial subject", row.get("subject"))
                for row in metadata.get("spatial_relations", [])
                if isinstance(row, dict)
            )
            references.extend(
                ("spatial object", row.get("object"))
                for row in metadata.get("spatial_relations", [])
                if isinstance(row, dict)
            )
            references.extend(
                ("action subject", row.get("subject"))
                for row in metadata.get("action_relations", [])
                if isinstance(row, dict)
            )
            references.extend(
                ("action object", row.get("object"))
                for row in metadata.get("action_relations", [])
                if isinstance(row, dict)
            )

        for role, value in references:
            normalized = str(value or "").strip().lower()
            if normalized not in entity_names:
                available = ", ".join(sorted(entity_names))
                raise ValueError(
                    f"LLM policy {role} {value!r} is not an exact canonical entity identifier; "
                    f"use one of [{available}]"
                )

    def _system_prompt(self) -> str:
        return (
            "You are the Policy component in Gen-Harness. "
            "Convert public prompt text into typed visual constraints for downstream skills, tools, middleware, memory, and internal verification. "
            "You do not generate images, judge images, propose code patches, or see official benchmark feedback. "
            "Do not use official evaluator outputs, hidden labels, benchmark answers, post-hoc image selection, fallback images, regex recipes, or task-id-specific rules. "
            "For every count, attribute, spatial, or action constraint, emit the corresponding structured metadata arrays required by the user schema. "
            "Build one canonical entity vocabulary in metadata.required_objects and reuse each class value exactly in every attribute, spatial, and action reference; never switch between singular and plural surface forms. "
            "Canonical class values must be bare object noun phrases, never color/material/texture modifiers: use class flower plus attribute glass, never class glass flower. "
            "Every explicit color, material, texture, or shape modifier in the prompt must be represented in attribute_bindings. "
            "Include the required_objects inventory in every spatial or action constraint so the contract is self-contained and referentially complete. "
            "Return only valid JSON."
        )


class VisualPolicy(HarnessComponent):
    """Compiles user/task intent into explicit visual contracts.

    This component owns constraint ontology, reference requirements,
    verification policy, and preservation policy. It is patchable at file level.
    """

    component_type = "policy"

    def __init__(
        self,
        visual_contract_path: str | Path,
        reference_policy_path: str | Path,
        preservation_policy_path: str | Path,
        *,
        policy_extractor_config: str | Path | None = None,
        policy_extractor: PolicyExtractor | None = None,
    ):
        super().__init__(visual_contract_path)
        self.reference_policy_path = Path(reference_policy_path)
        self.preservation_policy_path = Path(preservation_policy_path)
        from ..io import read_json
        self.reference_policy = read_json(self.reference_policy_path)
        self.preservation_policy = read_json(self.preservation_policy_path)
        if policy_extractor is not None and policy_extractor_config is not None:
            raise ValueError("provide either policy_extractor or policy_extractor_config, not both")
        if policy_extractor is not None:
            self.extractor = policy_extractor
        elif policy_extractor_config is not None:
            self.extractor = build_policy_extractor(policy_extractor_config)
        else:
            raise ValueError("policy_extractor_config is required for VisualPolicy construction")

    def compile_contract(self, task: GenerationTask) -> Dict[str, Any]:
        ontology = set(self.artifact.get("constraint_ontology", []))
        constraints: List[Dict[str, Any]] = []
        required_references: Dict[str, str] = {}
        required_evidence: List[Dict[str, str]] = []
        preservation: Dict[str, List[str]] = {}
        verification: Dict[str, str] = {}
        negative_constraints: List[str] = list(self.artifact.get("negative_constraints", []))

        actual_extractor = self._extractor_for_task(task)
        task_constraints = actual_extractor.extract(task, self.artifact)
        constraint_source = "policy_extractor"
        for c in task_constraints:
            if c.constraint_type not in ontology:
                raise ValueError(f"constraint type not in ontology: {c.constraint_type}")
            constraint_row = c.to_dict()
            constraint_row["metadata"] = sanitize_formal_trace_metadata(constraint_row.get("metadata", {}))
            constraints.append(constraint_row)
            if c.required_reference_role:
                required_references[c.constraint_id] = c.required_reference_role
            if c.constraint_type in self.artifact.get("evidence_required_for", []):
                required_evidence.append({"constraint_id": c.constraint_id, "constraint_type": c.constraint_type, "query": c.text})
            if c.target and c.constraint_type == "negative" and self.artifact.get("auto_include_negative_constraints", False):
                if c.target not in negative_constraints:
                    negative_constraints.append(c.target)
            rule = c.verification_rule or self.artifact.get("verification_rules", {}).get(c.constraint_type)
            if rule:
                verification[c.constraint_id] = rule
            preservation[c.constraint_id] = c.preservation or self.preservation_policy.get("default_preservation", [])

        return {
            "component": self.describe(),
            "task_id": task.task_id,
            "task_metadata": sanitize_formal_trace_metadata(task.metadata),
            "constraints": constraints,
            "required_references": required_references,
            "required_evidence": required_evidence,
            "negative_constraints": negative_constraints,
            "verification_criteria": verification,
            "preservation_criteria": preservation,
            "reference_policy": self.reference_policy,
            "prompt_governance": self.artifact.get("prompt_governance", {}),
            "policy_extractor": actual_extractor.describe(),
            "constraint_source": constraint_source,
        }

    def extract_constraints(self, task: GenerationTask) -> List[VisualConstraint]:
        """Convert natural-language intent into structured VisualConstraint rows."""

        return self._extractor_for_task(task).extract(task, self.artifact)

    def _extractor_for_task(self, task: GenerationTask) -> PolicyExtractor:
        resolver = getattr(self.extractor, "extractor_for_task", None)
        if callable(resolver):
            return resolver(task)
        return self.extractor

    def responsible_decisions(self) -> List[str]:
        return ["constraint extraction", "constraint ontology", "contract compilation", "reference requirements", "verification criteria", "preservation criteria"]


def build_policy_extractor(config_path: str | Path) -> PolicyExtractor:
    cfg = expand_env_vars(read_json(config_path))
    extractor_type = cfg.get("type")
    if extractor_type == "openai_chat_policy_extractor":
        return LLMPolicyExtractor(cfg)
    raise ValueError(f"unsupported policy extractor config type: {extractor_type}")


def _post_policy_chat_json(config: Dict[str, Any], endpoint: str, body: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", **dict(config.get("headers", {}))}
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM policy request failed: HTTP {exc.code}: {detail}") from exc


def _policy_chat_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"LLM policy response did not contain choices: {response}")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"LLM policy response did not contain message content: {response}")
    return content


def _reject_policy_forbidden(value: Any) -> None:
    reject_forbidden_trace_text(value, context="LLM policy output")
    reject_forbidden_payload_context(value, context="LLM policy output")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_prompt_metadata(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    forbidden = {
        "answer",
        "answers",
        "benchmark",
        "category",
        "complex_subtype",
        "dataset",
        "eval",
        "evaluation",
        "generation_input",
        "geneval_index",
        "geneval_metadata",
        "geneval_tag",
        "geneval2_index",
        "hidden_label",
        "hidden_labels",
        "image_index",
        "label",
        "labels",
        "official_eval",
        "question_answers",
        "score",
        "scores",
        "source_index",
        "split",
        "task_id",
        "vqa_list",
    }
    forbidden_fragments = (
        "answer",
        "benchmark",
        "category",
        "eval",
        "geneval",
        "label",
        "score",
        "source_index",
        "split",
        "task_id",
        "vqa",
    )
    safe: Dict[str, Any] = {}
    for key, value in metadata.items():
        normalized = str(key).strip().lower()
        if normalized in forbidden or any(fragment in normalized for fragment in forbidden_fragments):
            continue
        safe[str(key)] = _safe_prompt_metadata_value(value)
    return safe


def _safe_prompt_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _safe_prompt_metadata(value)
    if isinstance(value, list):
        return [_safe_prompt_metadata_value(item) for item in value]
    return value
