from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

from .base import HarnessComponent
from ..io import read_json
from ..schema import GenerationTask


class VisualMiddleware(HarnessComponent):
    """Connects policy, skills, references, tools, provenance, and routing."""

    component_type = "middleware"

    def __init__(self, skill_selector_path: str | Path, reference_binder_path: str | Path, tool_router_path: str | Path):
        super().__init__(skill_selector_path)
        self.reference_binder_path = Path(reference_binder_path)
        self.tool_router_path = Path(tool_router_path)
        self.reference_binder = read_json(self.reference_binder_path)
        self.tool_router = read_json(self.tool_router_path)

    def select_skill(self, task: GenerationTask) -> str:
        return self.artifact.get("task_family_to_skill", {}).get(task.task_family, self.artifact.get("default_skill"))

    def bind_references(self, task: GenerationTask, contract: Dict[str, Any], evidence: Dict[str, Any] | None = None) -> Dict[str, str]:
        mode = self.reference_binder.get("mode", "role_aware")
        refs = dict(task.references)
        if evidence:
            for ref in evidence.get("references", []):
                ref_id = ref.get("id")
                if ref_id:
                    refs.setdefault(ref_id, ref)
        bindings: Dict[str, str] = {}
        if mode == "role_aware":
            bindings = self._bind_role_aware(refs)
        elif mode == "priority_rules":
            for rule in self.reference_binder.get("rules", []):
                role = rule["role"]
                for ref_id, meta in refs.items():
                    if meta.get("role") == role or any(k in ref_id.lower() for k in rule.get("id_keywords", [])):
                        bindings[role] = ref_id
                        break
        elif mode == "first_reference_for_all":
            first_ref = next(iter(refs), None)
            for role in self.reference_binder.get("roles", []):
                if first_ref:
                    bindings[role] = first_ref
        else:
            raise ValueError(f"unsupported reference binding mode: {mode}")
        required = contract.get("required_references", {})
        missing = [role for role in required.values() if role not in bindings]
        if missing and self.reference_binder.get("strict_required_references", True):
            raise ValueError(f"missing required reference roles: {missing}")
        return bindings

    def route_tool(self, task: GenerationTask, contract: Dict[str, Any], tool_metadata: Dict[str, Any]) -> Dict[str, Any]:
        needs = set(self._required_capabilities(task, contract))
        tools = tool_metadata.get("tools", {})
        route_key = "+".join(sorted(needs))
        preferred = self.tool_router.get("routes", {}).get(route_key)
        if not preferred and "identity_conditioning" in needs:
            preferred = self.tool_router.get("identity_tool")
        if not preferred:
            preferred = self.tool_router.get("default_tool")

        route_names = []
        if preferred:
            route_names.append(preferred)
        route_names.extend(tools.keys())

        seen = set()
        first_available = None
        evaluated = []
        for name in route_names:
            if name in seen or name not in tools:
                continue
            seen.add(name)
            tool = dict(tools[name])
            missing = self._missing_capabilities(tool, needs)
            evaluated.append({"name": name, "missing_capabilities": missing})
            if first_available is None:
                first_available = (tool, missing)
            if not missing:
                tool["routing_diagnostics"] = {
                    "route_key": route_key,
                    "preferred_tool": preferred,
                    "required_capabilities": sorted(needs),
                    "evaluated_routes": evaluated,
                    "rerouted_from_preferred": bool(preferred and name != preferred),
                }
                return tool
            if name == preferred and self.tool_router.get("strict_preferred_route", False):
                tool["routing_warning"] = {"missing_capabilities": missing, "route_key": route_key}
                tool["routing_diagnostics"] = {
                    "route_key": route_key,
                    "preferred_tool": preferred,
                    "required_capabilities": sorted(needs),
                    "evaluated_routes": evaluated,
                    "strict_preferred_route": True,
                }
                return tool

        if first_available is not None:
            tool, missing = first_available
            tool["routing_warning"] = {"missing_capabilities": missing, "route_key": route_key}
            tool["routing_diagnostics"] = {
                "route_key": route_key,
                "preferred_tool": preferred,
                "required_capabilities": sorted(needs),
                "evaluated_routes": evaluated,
                "rerouted_from_preferred": False,
            }
            return tool
        raise ValueError(f"no generation tools are registered; required capabilities: {sorted(needs)}")



    def _bind_role_aware(self, refs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        bindings: Dict[str, str] = {}
        for ref_id, meta in refs.items():
            role = meta.get("role")
            if role and role not in bindings:
                bindings[role] = ref_id
        return bindings

    def _missing_capabilities(self, tool: Dict[str, Any], needs: Set[str]) -> List[str]:
        caps = tool.get("capabilities", {})
        return sorted(cap for cap in needs if not caps.get(cap, False))

    def trace_decisions(self, skill_name: str, bindings: Dict[str, str], tool: Dict[str, Any]) -> Dict[str, Any]:
        trace = {
            "component": self.describe(),
            "selected_skill": skill_name,
            "reference_bindings": bindings,
            "selected_tool": tool.get("name"),
            "routing_capabilities": tool.get("capabilities", {}),
            "routing_warning": tool.get("routing_warning"),
            "routing_diagnostics": tool.get("routing_diagnostics"),
        }
        return trace

    def _required_capabilities(self, task: GenerationTask, contract: Dict[str, Any] | None = None) -> List[str]:
        needs = ["text_to_image"]
        constraint_types = [c.constraint_type for c in task.constraints]
        if contract:
            constraint_types.extend(str(c.get("constraint_type")) for c in contract.get("constraints", []))
        if any(ctype == "identity" for ctype in constraint_types):
            needs.append("identity_conditioning")
        if any(ctype == "style" for ctype in constraint_types):
            needs.append("style_conditioning")
        return needs

    def responsible_decisions(self) -> List[str]:
        return ["skill selection", "reference binding", "tool routing", "provenance tracing", "budget and rollback control"]
