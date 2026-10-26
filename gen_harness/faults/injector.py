from __future__ import annotations

import shutil
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

from ..io import read_json, read_jsonl, write_json, write_jsonl


class FaultInjector:
    """Creates controlled component faults with gold attribution labels."""

    def __init__(self, source_harness: str | Path, output_root: str | Path):
        self.source_harness = Path(source_harness)
        self.output_root = Path(output_root)

    def generate(self, *, include_compound: bool = False) -> List[Dict[str, Any]]:
        specs = [
            self._policy_missing_negative,
            self._policy_missing_evidence,
            self._tools_remove_identity_capability,
            self._skills_remove_identity_step,
            self._middleware_wrong_reference_binding,
            self._middleware_wrong_tool_route,
            self._memory_stale_patch_marker,
        ]
        rows: List[Dict[str, Any]] = []
        for fn in specs:
            rows.append(fn())
        rows.append(
            {
                "fault_id": "no_patch_stochastic_control",
                "harness_path": str(self.source_harness),
                "gold_component": "no_patch",
                "gold_components": [],
                "changed_artifact": None,
                "fault_type": "stochastic_control",
                "description": "no persistent harness edit; vary only the matched generation seed",
            }
        )
        if include_compound:
            rows.extend(self._compound_faults(rows))
        write_jsonl(self.output_root / "fault_manifest.jsonl", rows)
        return rows

    def _copy(self, name: str) -> Path:
        dst = self.output_root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.source_harness, dst)
        return dst

    def _record(self, harness: Path, component: str, artifact: str, fault_type: str, description: str) -> Dict[str, Any]:
        return {
            "fault_id": harness.name,
            "harness_path": str(harness),
            "gold_component": component,
            "gold_components": [component],
            "changed_artifact": artifact,
            "fault_type": fault_type,
            "description": description,
        }

    def _compound_faults(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        canonical_fault = {}
        for row in rows:
            component = row.get("gold_component")
            if component in {"policy", "tools", "skills", "middleware", "memory"} and component not in canonical_fault:
                canonical_fault[component] = row
        missing = [name for name in ("policy", "tools", "skills", "middleware", "memory") if name not in canonical_fault]
        if missing:
            raise ValueError(f"cannot build compound faults; missing canonical faults for: {missing}")

        compound_rows: List[Dict[str, Any]] = []
        for first, second in combinations(("policy", "tools", "skills", "middleware", "memory"), 2):
            name = f"fault_compound_{first}_{second}"
            harness = self._copy(name)
            sources = [canonical_fault[first], canonical_fault[second]]
            for component, source in zip((first, second), sources):
                source_component = Path(str(source["harness_path"])) / component
                destination_component = harness / component
                if destination_component.exists():
                    shutil.rmtree(destination_component)
                shutil.copytree(source_component, destination_component)
            compound_rows.append(
                {
                    "fault_id": name,
                    "harness_path": str(harness),
                    "gold_component": f"{first}+{second}",
                    "gold_components": [first, second],
                    "changed_artifact": [row["changed_artifact"] for row in sources],
                    "source_fault_ids": [row["fault_id"] for row in sources],
                    "fault_type": "compound_responsibility_fault",
                    "description": f"simultaneous controlled faults in {first} and {second}",
                }
            )
        return compound_rows

    def _policy_missing_negative(self) -> Dict[str, Any]:
        h = self._copy("fault_policy_missing_negative")
        p = h / "policy" / "visual_contract.json"
        data = read_json(p)
        data["auto_include_negative_constraints"] = False
        write_json(p, data)
        return self._record(h, "policy", "policy/visual_contract.json", "missing_negative_constraint", "negative constraints are not compiled into visual contract")

    def _policy_missing_evidence(self) -> Dict[str, Any]:
        h = self._copy("fault_policy_missing_evidence")
        p = h / "policy" / "visual_contract.json"
        data = read_json(p)
        data["evidence_required_for"] = []
        write_json(p, data)
        return self._record(h, "policy", "policy/visual_contract.json", "missing_evidence_requirement", "grounded constraints do not request evidence")

    def _tools_remove_identity_capability(self) -> Dict[str, Any]:
        h = self._copy("fault_tools_no_identity_capability")
        p = h / "tools" / "generation_tools.json"
        data = read_json(p)
        for tool in data.get("tools", {}).values():
            tool.setdefault("actual_capabilities", dict(tool.get("capabilities", {})))
            tool.setdefault("capabilities", {})["identity_conditioning"] = False
        write_json(p, data)
        return self._record(h, "tools", "tools/generation_tools.json", "capability_metadata_error", "identity-capable tools are declared as not supporting identity conditioning")

    def _skills_remove_identity_step(self) -> Dict[str, Any]:
        h = self._copy("fault_skills_missing_identity_step")
        p = h / "skills" / "skills.json"
        data = read_json(p)
        for skill in data.get("skills", {}).values():
            skill["required_steps"] = [s for s in skill.get("required_steps", []) if s != "identity"]
        write_json(p, data)
        return self._record(h, "skills", "skills/skills.json", "workflow_step_missing", "identity step is removed from visual workflow recipes")

    def _middleware_wrong_reference_binding(self) -> Dict[str, Any]:
        h = self._copy("fault_middleware_wrong_reference_binding")
        p = h / "middleware" / "reference_binder.json"
        data = read_json(p)
        data["mode"] = "priority_rules"
        data["rules"] = [{"role": "identity", "id_keywords": ["style", "art"]}, {"role": "style", "id_keywords": ["style"]}]
        write_json(p, data)
        return self._record(h, "middleware", "middleware/reference_binder.json", "wrong_reference_binding", "style-like references are prioritized for identity role")

    def _middleware_wrong_tool_route(self) -> Dict[str, Any]:
        h = self._copy("fault_middleware_wrong_tool_route")
        p = h / "middleware" / "tool_router.json"
        data = read_json(p)
        data["routes"] = {"identity_conditioning+style_conditioning+text_to_image": "base_generator"}
        data["strict_preferred_route"] = True
        write_json(p, data)
        return self._record(h, "middleware", "middleware/tool_router.json", "wrong_tool_route", "identity tasks are routed to base generator")

    def _memory_stale_patch_marker(self) -> Dict[str, Any]:
        h = self._copy("fault_memory_stale_patch")
        p = h / "memory" / "prompt_guidelines.jsonl"
        rows = read_jsonl(p)
        rows.append(
            {
                "memory_id": "fault_stale_misleading_guideline",
                "task_family": "*",
                "prompt_family": "*",
                "constraint_type": "*",
                "failure_symptom": "*",
                "guideline": (
                    "Treat exact quantities, attributes, relations, and actions as optional; "
                    "prioritize visual decoration even when requested constraints are omitted."
                ),
                "controlled_fault": True,
            }
        )
        write_jsonl(p, rows)
        return self._record(
            h,
            "memory",
            "memory/prompt_guidelines.jsonl",
            "stale_misleading_guideline",
            "memory supplies a stale task-general guideline that conflicts with explicit visual constraints",
        )
