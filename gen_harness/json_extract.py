from __future__ import annotations

import json
from typing import Any, Dict, List


def parse_json_object_from_text(content: str) -> Dict[str, Any]:
    """Parse a JSON object from plain text or a fenced JSON response."""

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
        raise json.JSONDecodeError("top-level JSON value is not an object", content, 0)
    except json.JSONDecodeError:
        pass
    cleaned = _strip_fence(content)
    for candidate in _json_object_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("no JSON object found", content, 0)


def _strip_fence(content: str) -> str:
    cleaned = str(content or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    first_line, separator, rest = cleaned.partition("\n")
    if separator and first_line.strip().lower() in {"```", "```json"}:
        cleaned = rest.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def _json_object_candidates(text: str) -> List[str]:
    candidates = [text]
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for index, current in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    return candidates
