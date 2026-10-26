from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VisualConstraint:
    constraint_id: str
    text: str
    constraint_type: str
    target: Optional[str] = None
    required_reference_role: Optional[str] = None
    verification_rule: Optional[str] = None
    preservation: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualConstraint":
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationTask:
    task_id: str
    prompt: str
    task_family: str
    constraints: List[VisualConstraint]
    references: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationTask":
        constraints = [VisualConstraint.from_dict(c) for c in data.get("constraints", [])]
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            task_family=data.get("task_family", "unknown"),
            constraints=constraints,
            references=data.get("references", {}),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["constraints"] = [c.to_dict() for c in self.constraints]
        return d


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualExperience:
    experience_id: str
    task_id: str
    constraint_id: str
    constraint_type: str
    constraint_text: str
    component_decisions: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    visual_result: Dict[str, Any]
    verification_result: Dict[str, Any]
    failure_symptom: Optional[str]
    suspected_component: Optional[str]
    repair_action: Optional[Dict[str, Any]] = None
    repair_outcome: Optional[Dict[str, Any]] = None
    regressed_constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualExperience":
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_compact_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["component_decisions"] = _compact_component_decisions(row.get("component_decisions", {}))
        row["tool_calls"] = _compact_tool_calls(row.get("tool_calls", []))
        row["visual_result"] = _compact_value(row.get("visual_result", {}), max_depth=3)
        row["verification_result"] = _compact_value(row.get("verification_result", {}), max_depth=4)
        row["repair_action"] = _compact_value(row.get("repair_action"), max_depth=3)
        row["repair_outcome"] = _compact_value(row.get("repair_outcome"), max_depth=3)
        row["metadata"] = _compact_metadata(row.get("metadata", {}))
        return row


@dataclass
class Weakness:
    weakness_id: str
    suspected_component: str
    constraint_type: str
    failure_symptom: str
    support_count: int
    support_experience_ids: List[str]
    hypothesis: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compact_metadata(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_value(value, max_depth=3)
    out = dict(value)
    fast_loop = out.get("fast_loop")
    if isinstance(fast_loop, dict):
        attempts = fast_loop.get("attempts", [])
        out["fast_loop"] = {
            key: _compact_value(val, max_depth=2)
            for key, val in fast_loop.items()
            if key != "attempts"
        }
        if isinstance(attempts, list):
            out["fast_loop"]["attempts"] = [
                _compact_attempt(row)
                for row in attempts[:5]
                if isinstance(row, dict)
            ]
            if len(attempts) > 5:
                out["fast_loop"]["attempts_truncated"] = len(attempts) - 5
    return _compact_value(out, max_depth=4)


def _compact_component_decisions(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_value(value, max_depth=3)
    out = _compact_value(value, max_depth=3)
    tools = value.get("tools", {})
    if isinstance(tools, dict):
        generation_artifact = tools.get("generation_artifact", {})
        if isinstance(generation_artifact, dict) and isinstance(generation_artifact.get("tools"), dict):
            tool_rows: Dict[str, Any] = {}
            for name, tool in generation_artifact["tools"].items():
                if not isinstance(tool, dict):
                    continue
                tool_rows[str(name)] = {
                    key: _compact_value(tool.get(key), max_depth=2)
                    for key in ("name", "type", "capabilities", "actual_capabilities", "cost")
                    if key in tool
                }
            if isinstance(out, dict):
                out.setdefault("tools", {})
                if isinstance(out["tools"], dict):
                    out["tools"].setdefault("generation_artifact", {})
                    if isinstance(out["tools"]["generation_artifact"], dict):
                        out["tools"]["generation_artifact"]["tools"] = tool_rows
    return out


def _compact_attempt(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _compact_value(row.get(key), max_depth=2)
        for key in (
            "attempt",
            "seed",
            "passed",
            "score",
            "error",
            "repair_decision",
            "repair_action",
            "repair_outcome",
            "prompt",
            "prompt_channels",
        )
        if key in row
    }


def _compact_tool_calls(value: Any) -> Any:
    if not isinstance(value, list):
        return _compact_value(value, max_depth=2)
    rows = []
    for row in value[:5]:
        if not isinstance(row, dict):
            rows.append(_compact_value(row, max_depth=2))
            continue
        result = row.get("result", {})
        arguments = row.get("arguments", {})
        rows.append(
            {
                "tool_name": row.get("tool_name"),
                "arguments": _compact_value(arguments, max_depth=2),
                "result": _compact_value(result, max_depth=2),
            }
        )
    if len(value) > 5:
        rows.append({"truncated_tool_calls": len(value) - 5})
    return rows


def _compact_value(value: Any, *, max_depth: int, max_list_items: int = 8, max_string_chars: int = 500) -> Any:
    if max_depth <= 0:
        if isinstance(value, (dict, list)):
            return {"truncated": True, "type": type(value).__name__}
        return _compact_scalar(value, max_string_chars=max_string_chars)
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (key, child) in enumerate(value.items()):
            if idx >= max_list_items:
                out["truncated_keys"] = len(value) - max_list_items
                break
            out[str(key)] = _compact_value(child, max_depth=max_depth - 1, max_list_items=max_list_items, max_string_chars=max_string_chars)
        return out
    if isinstance(value, list):
        rows = [
            _compact_value(child, max_depth=max_depth - 1, max_list_items=max_list_items, max_string_chars=max_string_chars)
            for child in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            rows.append({"truncated_items": len(value) - max_list_items})
        return rows
    return _compact_scalar(value, max_string_chars=max_string_chars)


def _compact_scalar(value: Any, *, max_string_chars: int) -> Any:
    if isinstance(value, str) and len(value) > max_string_chars:
        return value[:max_string_chars] + f"...<truncated {len(value) - max_string_chars} chars>"
    return value


@dataclass
class PatchManifest:
    patch_id: str
    target_component: str
    changed_artifact: str
    changed_fields: List[str]
    operation: str
    payload: Dict[str, Any]
    supporting_experience: List[str]
    predicted_improvement: str
    possible_regression: str
    target_validation_set: str
    preservation_set: str
    held_out_set: str
    rollback_boundary: str
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    regression_risk: Dict[str, Any] = field(default_factory=dict)
    edit_operator: Optional[str] = None
    edit_key: Optional[str] = None
    old_value: Any = None
    new_value: Any = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatchManifest":
        row = dict(data)
        row.pop("canonical_edit", None)
        return cls(**row)

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["canonical_edit"] = self.canonical_edit()
        return row

    def canonical_edit(self) -> Dict[str, Any]:
        """Return the paper interface δ=(z,o,k,v_old,v_new)."""
        operator = self.edit_operator or {
            "set_path": "REPLACE",
            "append_unique": "ADD",
            "append_jsonl_unique": "ADD",
            "delete_path": "DELETE",
        }.get(self.operation, self.operation.upper())
        path = self.payload.get("path") if isinstance(self.payload, dict) else None
        key = self.edit_key or (".".join(str(part) for part in path) if isinstance(path, list) else self.changed_artifact)
        new_value = self.new_value
        if new_value is None and isinstance(self.payload, dict):
            new_value = self.payload.get("value", self.payload.get("row"))
        return {
            "responsibility": self.target_component,
            "operator": operator,
            "key": key,
            "old_value": self.old_value,
            "new_value": new_value,
        }
