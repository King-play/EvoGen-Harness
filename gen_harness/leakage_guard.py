from __future__ import annotations

import json
from typing import Any, Iterable


PROVENANCE_KEY_EXACT = {
    "answer",
    "answers",
    "benchmark",
    "category",
    "complex_subtype",
    "dataset",
    "eval",
    "evaluation",
    "experience_id",
    "fallback_image",
    "fallback_used",
    "file_path",
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
    "local_path",
    "official_eval",
    "official_evaluator",
    "official_score",
    "original_prompt",
    "question_answers",
    "score",
    "scores",
    "selected_result",
    "selected_score",
    "source_index",
    "split",
    "support_experience_ids",
    "task_id",
    "task_ids",
    "task_metadata",
    "vqa_list",
}

PROVENANCE_KEY_FRAGMENTS = (
    "answer",
    "benchmark",
    "category",
    "eval",
    "fallback",
    "geneval",
    "hidden_label",
    "image_path",
    "label",
    "official",
    "posthoc",
    "post_hoc",
    "score",
    "selected_result",
    "selected_score",
    "source_index",
    "task_id",
    "vqa",
)

FORBIDDEN_TRACE_TEXT = {
    "official_score",
    "official scores",
    "official evaluator",
    "benchmark answer",
    "benchmark label",
    "hidden label",
    "fallback_image",
    "fallback image",
    "posthoc",
    "post-hoc",
    "selected_score",
    "selected_result",
    "task id",
    "task_id",
}

FORBIDDEN_TRACE_ID_PREFIXES = (
    "geneval_",
    "geneval2_",
    "t2icompbenchpp_",
)


def reject_forbidden_trace_text(value: Any, *, context: str) -> None:
    text = json.dumps(value, ensure_ascii=False).lower()
    for token in FORBIDDEN_TRACE_TEXT:
        if token in text:
            raise ValueError(f"{context} contains forbidden token: {token}")
    matched = _first_forbidden_trace_identifier(text)
    if matched:
        raise ValueError(f"{context} contains forbidden task-specific identifier: {matched}")


def sanitize_llm_payload_context(value: Any) -> Any:
    """Remove task/benchmark provenance from structured payloads sent to LLMs."""

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if provenance_key_is_forbidden(key_text):
                continue
            sanitized = sanitize_llm_payload_context(child)
            if isinstance(sanitized, str) and _is_forbidden_payload_text(sanitized):
                continue
            out[key_text] = sanitized
        return out
    if isinstance(value, list):
        return [sanitize_llm_payload_context(item) for item in value]
    return value


def sanitize_formal_trace_metadata(value: Any) -> Any:
    """Remove benchmark/evaluator provenance before writing formal traces.

    Public bookkeeping such as split names is allowed in run artifacts, but
    benchmark labels, source indices, evaluator answers, scores, official
    metadata, fallback fields, and task-specific identifiers are not.
    """

    return _sanitize_metadata_value(value, allow_keys={"generation_input", "split", "task_family"})


def reject_forbidden_payload_context(value: Any, *, context: str) -> None:
    sanitized = sanitize_llm_payload_context(value)
    if sanitized != value:
        raise ValueError(f"{context} contains task, benchmark, score, official, or fallback provenance")
    reject_forbidden_trace_text(value, context=context)


def _sanitize_metadata_value(value: Any, *, allow_keys: set[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            key_lower = key_text.strip().lower()
            if key_lower not in allow_keys and provenance_key_is_forbidden(key_text):
                continue
            sanitized = _sanitize_metadata_value(child, allow_keys=allow_keys)
            if isinstance(sanitized, str) and _is_forbidden_payload_text(sanitized):
                continue
            out[key_text] = sanitized
        return out
    if isinstance(value, list):
        return [_sanitize_metadata_value(item, allow_keys=allow_keys) for item in value]
    return value


def provenance_key_is_forbidden(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in PROVENANCE_KEY_EXACT or any(fragment in normalized for fragment in PROVENANCE_KEY_FRAGMENTS)


def _first_forbidden_trace_identifier(text: str) -> str:
    for token in _identifier_tokens(text):
        if any(token.startswith(prefix) for prefix in FORBIDDEN_TRACE_ID_PREFIXES):
            return token
    return ""


def _identifier_tokens(text: str) -> Iterable[str]:
    current: list[str] = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            current.append(char)
        elif current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


def _is_forbidden_payload_text(value: str) -> bool:
    lowered = value.lower()
    return bool(_first_forbidden_trace_identifier(lowered)) or any(token in lowered for token in FORBIDDEN_TRACE_TEXT)
