from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .backends import formal_backend_config_error
from .io import read_json, read_jsonl, write_json


AUDIT_SCHEMA = "gen_harness.component_boundary_audit.v1"
RUN_AUDIT_SCHEMA = "gen_harness.run_artifact_audit.v1"
GOAL_READINESS_SCHEMA = "gen_harness.goal_readiness_audit.v1"
MAX_RUN_EXPERIENCES_BYTES = 512 * 1024 * 1024

COMPONENTS = ("policy", "skills", "tools", "memory", "middleware")

REQUIRED_ARTIFACTS = {
    "policy": ("visual_contract.json", "reference_policy.json", "preservation_policy.json"),
    "skills": ("skills.json",),
    "tools": ("generation_tools.json", "inspection_tools.json"),
    "memory": (
        "memory_contract.json",
        "visual_experiences.jsonl",
        "failure_clusters.jsonl",
        "weaknesses.jsonl",
        "validated_patches.jsonl",
        "regression_cases.jsonl",
        "prompt_guidelines.jsonl",
    ),
    "middleware": ("skill_selector.json", "reference_binder.json", "tool_router.json"),
}

EXPECTED_COMPONENT_NAMES = {
    "policy": {
        "visual_contract.json": "visual_policy",
        "reference_policy.json": "reference_policy",
        "preservation_policy.json": "preservation_policy",
    },
    "skills": {"skills.json": "visual_skills"},
    "tools": {
        "generation_tools.json": "visual_tools",
        "inspection_tools.json": "inspection_tools",
    },
    "memory": {"memory_contract.json": "visual_memory"},
    "middleware": {
        "skill_selector.json": "skill_selector",
        "reference_binder.json": "reference_binder",
        "tool_router.json": "tool_router",
    },
}

EXPECTED_COMPONENT_TYPES = {
    "policy": {
        "visual_contract.json": "policy",
        "reference_policy.json": "policy",
        "preservation_policy.json": "policy",
    },
    "skills": {"skills.json": "skills"},
    "tools": {
        "generation_tools.json": "tools",
        "inspection_tools.json": "tools",
    },
    "memory": {"memory_contract.json": "memory"},
    "middleware": {
        "skill_selector.json": "middleware",
        "reference_binder.json": "middleware",
        "tool_router.json": "middleware",
    },
}

RESPONSIBILITIES = {
    "policy": (
        "constraint ontology",
        "visual contract compilation",
        "reference/evidence/preservation policy",
    ),
    "skills": (
        "task workflow recipes",
        "prompt family rules",
        "constraint-to-prompt compilation",
    ),
    "tools": (
        "generation tool capability declarations",
        "inspection tool capability declarations",
        "tool execution surface",
    ),
    "memory": (
        "experience records",
        "weakness records",
        "validated patches",
        "regression cases",
        "prompt guidelines",
    ),
    "middleware": (
        "skill selection",
        "reference binding",
        "tool routing",
        "runtime provenance wiring",
    ),
}

FORBIDDEN_KEYS = {
    "policy": {
        "skills",
        "prompt_families",
        "routes",
        "default_tool",
        "vqa_list",
        "questions",
        "answers",
        "answer",
    },
    "skills": {
        "routes",
        "default_tool",
        "identity_tool",
        "constraint_ontology",
        "vqa_list",
        "questions",
        "answers",
        "answer",
    },
    "tools": {
        "task_family_to_skill",
        "prompt_families",
        "prompt_rules",
        "constraint_ontology",
        "vqa_list",
        "questions",
        "answers",
        "answer",
    },
    "middleware": {
        "constraint_ontology",
        "prompt_families",
        "prompt_rules",
        "capabilities",
        "vqa_list",
        "questions",
        "answers",
        "answer",
    },
    "memory": {
        "vqa_list",
        "questions",
        "answers",
        "answer",
        "official_answer",
        "soft_tifa_answer",
    },
}

FORBIDDEN_VALUE_TOKENS = (
    "fallback_image",
    "fallback image",
    "use official evaluator for selection",
    "official labels for prompt",
    "soft_tifa_answer",
)

FORBIDDEN_FORMAL_INTERFACE_TOKENS = (
    "t2icompbenchpp_scene_templates",
    "t2icompbenchpp_scene_template_family",
    "t2icompbenchpp_contextual_scene_presence",
)

BENCHMARK_LABEL_KEYS = {
    "vqa_list",
    "questions",
    "question_answers",
    "answers",
    "answer",
    "skills",
    "atom_count",
    "official_answer",
    "soft_tifa_answer",
}

RUN_BENCHMARK_LABEL_KEYS = BENCHMARK_LABEL_KEYS - {"skills"}
RUN_BENCHMARK_PROVENANCE_KEYS = {
    "benchmark",
    "category",
    "complex_subtype",
    "dataset",
    "geneval_index",
    "geneval_metadata",
    "geneval_tag",
    "geneval2_index",
    "image_index",
    "source_benchmark",
    "source_index",
    "source_tag",
}


@dataclass
class AuditIssue:
    severity: str
    component: str
    artifact: str
    code: str
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "component": self.component,
            "artifact": self.artifact,
            "code": self.code,
            "message": self.message,
            "evidence": self.evidence,
        }


class ComponentBoundaryAuditor:
    """Audit that Gen-Harness has real, non-overlapping first-class components.

    The audit is intentionally structural and leakage-safe: it inspects harness
    artifacts and optional prompt-only task files.  It does not call benchmark
    evaluators and does not read official GenEval2 answers for generation,
    routing, selection, or prompt compilation.
    """

    def audit(
        self,
        harness_root: str | Path,
        *,
        tasks_path: str | Path | None = None,
        write_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        root = Path(harness_root)
        issues: List[AuditIssue] = []
        loaded = self._load_artifacts(root, issues)
        self._audit_component_names(loaded, issues)
        self._audit_component_types(loaded, issues)
        self._audit_boundary_contracts(loaded, issues)
        self._audit_forbidden_overlap(loaded, issues)
        self._audit_cross_component_contracts(loaded, issues)
        self._audit_memory_files(root, issues)
        task_audit = self._audit_tasks(tasks_path, issues) if tasks_path else {"checked": False}
        report = {
            "schema": AUDIT_SCHEMA,
            "harness_root": str(root),
            "components": {
                component: {
                    "responsibilities": list(RESPONSIBILITIES[component]),
                    "required_artifacts": list(REQUIRED_ARTIFACTS[component]),
                    "present": bool((root / component).exists()),
                }
                for component in COMPONENTS
            },
            "tasks": task_audit,
            "checks": {
                "component_artifacts_exist": not any(i.code == "missing_artifact" for i in issues),
                "component_names_match": not any(i.code == "wrong_component_name" for i in issues),
                "component_types_match": not any(i.code == "wrong_component_type" for i in issues),
                "boundary_contracts_present": not any(i.code == "boundary_contract_incomplete" for i in issues),
                "responsibilities_non_overlapping": not any(
                    i.code in {"forbidden_key_overlap", "forbidden_value_token", "benchmark_specific_formal_interface"}
                    for i in issues
                ),
                "policy_skills_tools_aligned": not any(i.code.startswith("contract_") for i in issues),
                "benchmark_labels_absent": not any(
                    i.code in {"benchmark_label_leak", "benchmark_specific_formal_interface"}
                    for i in issues
                ),
                "fallback_absent": not any(i.code == "fallback_detected" for i in issues),
            },
            "passed": not any(i.severity == "error" for i in issues),
            "issues": [i.to_dict() for i in issues],
        }
        if write_path is not None:
            write_json(write_path, report)
        return report

    def _load_artifacts(self, root: Path, issues: List[AuditIssue]) -> Dict[str, Dict[str, Any]]:
        loaded: Dict[str, Dict[str, Any]] = {component: {} for component in COMPONENTS}
        for component, artifacts in REQUIRED_ARTIFACTS.items():
            cdir = root / component
            if not cdir.exists():
                issues.append(AuditIssue("error", component, str(cdir), "missing_component_dir", f"missing component directory: {cdir}"))
                continue
            for artifact in artifacts:
                path = cdir / artifact
                if not path.exists():
                    issues.append(AuditIssue("error", component, artifact, "missing_artifact", f"missing required artifact: {path}"))
                    continue
                if artifact.endswith(".json"):
                    try:
                        loaded[component][artifact] = read_json(path)
                    except Exception as exc:
                        issues.append(AuditIssue("error", component, artifact, "invalid_json", f"invalid JSON artifact: {exc}"))
        return loaded

    def _audit_component_names(self, loaded: Dict[str, Dict[str, Any]], issues: List[AuditIssue]) -> None:
        for component, expected_by_artifact in EXPECTED_COMPONENT_NAMES.items():
            for artifact, expected in expected_by_artifact.items():
                data = loaded.get(component, {}).get(artifact)
                if not isinstance(data, dict):
                    continue
                found = data.get("component_name")
                if found != expected:
                    issues.append(
                        AuditIssue(
                            "error",
                            component,
                            artifact,
                            "wrong_component_name",
                            f"{artifact} must declare component_name={expected!r}, found {found!r}",
                        )
                    )

    def _audit_component_types(self, loaded: Dict[str, Dict[str, Any]], issues: List[AuditIssue]) -> None:
        for component, expected_by_artifact in EXPECTED_COMPONENT_TYPES.items():
            for artifact, expected in expected_by_artifact.items():
                data = loaded.get(component, {}).get(artifact)
                if not isinstance(data, dict):
                    continue
                found = data.get("component_type")
                if found != expected:
                    issues.append(
                        AuditIssue(
                            "error",
                            component,
                            artifact,
                            "wrong_component_type",
                            f"{artifact} must declare component_type={expected!r}, found {found!r}",
                        )
                    )

    def _audit_boundary_contracts(self, loaded: Dict[str, Dict[str, Any]], issues: List[AuditIssue]) -> None:
        required_fields = ("owns", "reads", "writes", "must_not_own")
        for component, artifacts in loaded.items():
            for artifact, data in artifacts.items():
                if not isinstance(data, dict):
                    continue
                contract = data.get("boundary_contract")
                if not isinstance(contract, dict):
                    issues.append(
                        AuditIssue(
                            "error",
                            component,
                            artifact,
                            "boundary_contract_incomplete",
                            f"{artifact} must declare a boundary_contract",
                            {"missing": list(required_fields)},
                        )
                    )
                    continue
                missing = [
                    field
                    for field in required_fields
                    if not isinstance(contract.get(field), list)
                    or not contract.get(field)
                    or not all(isinstance(item, str) and item.strip() for item in contract.get(field, []))
                ]
                if missing:
                    issues.append(
                        AuditIssue(
                            "error",
                            component,
                            artifact,
                            "boundary_contract_incomplete",
                            f"{artifact} has an incomplete boundary_contract",
                            {"missing_or_empty": missing},
                        )
                    )

    def _audit_forbidden_overlap(self, loaded: Dict[str, Dict[str, Any]], issues: List[AuditIssue]) -> None:
        for component, artifacts in loaded.items():
            for artifact, data in artifacts.items():
                for path, key, value in self._walk(data):
                    if key in FORBIDDEN_KEYS[component]:
                        issues.append(
                            AuditIssue(
                                "error",
                                component,
                                artifact,
                                "forbidden_key_overlap",
                                f"{component} artifact contains key owned by another component or benchmark labels: {'.'.join(path)}",
                                {"key": key, "path": path},
                            )
                        )
                    if str(key) in FORBIDDEN_FORMAL_INTERFACE_TOKENS:
                        issues.append(
                            AuditIssue(
                                "error",
                                component,
                                artifact,
                                "benchmark_specific_formal_interface",
                                f"{component} artifact contains benchmark-specific formal interface key: {'.'.join(path)}",
                                {"key": str(key), "path": path},
                            )
                        )
                    if isinstance(value, str):
                        lowered = value.lower()
                        if "fallback" in lowered:
                            issues.append(
                                AuditIssue(
                                    "error",
                                    component,
                                    artifact,
                                    "fallback_detected",
                                    f"{component} artifact contains fallback text at {'.'.join(path)}",
                                    {"path": path, "value": value[:200]},
                                )
                            )
                        for token in FORBIDDEN_VALUE_TOKENS:
                            if token in lowered:
                                issues.append(
                                    AuditIssue(
                                        "error",
                                        component,
                                        artifact,
                                        "forbidden_value_token",
                                        f"{component} artifact contains forbidden token {token!r}",
                                        {"path": path, "value": value[:200]},
                                    )
                                )

    def _audit_cross_component_contracts(self, loaded: Dict[str, Dict[str, Any]], issues: List[AuditIssue]) -> None:
        policy = loaded.get("policy", {}).get("visual_contract.json", {})
        skills = loaded.get("skills", {}).get("skills.json", {})
        generation_tools = loaded.get("tools", {}).get("generation_tools.json", {})
        inspection_tools = loaded.get("tools", {}).get("inspection_tools.json", {})
        skill_selector = loaded.get("middleware", {}).get("skill_selector.json", {})
        tool_router = loaded.get("middleware", {}).get("tool_router.json", {})
        ontology = set(policy.get("constraint_ontology", [])) if isinstance(policy, dict) else set()
        skill_rows = skills.get("skills", {}) if isinstance(skills, dict) else {}
        inspection_caps = self._inspection_capabilities(inspection_tools)

        for skill_name, skill in skill_rows.items() if isinstance(skill_rows, dict) else []:
            for step in skill.get("required_steps", []):
                if step not in ontology:
                    issues.append(AuditIssue("error", "skills", "skills.json", "contract_skill_step_not_in_policy", f"skill {skill_name!r} requires step absent from policy ontology: {step}"))
                if step not in inspection_caps:
                    issues.append(AuditIssue("error", "tools", "inspection_tools.json", "contract_skill_step_not_inspectable", f"skill {skill_name!r} requires step not declared inspectable: {step}"))

        known_skills = set(skill_rows.keys()) if isinstance(skill_rows, dict) else set()
        for task_family, skill_name in (skill_selector.get("task_family_to_skill", {}) or {}).items():
            if skill_name not in known_skills:
                issues.append(
                    AuditIssue(
                        "error",
                        "middleware",
                        "skill_selector.json",
                        "contract_unknown_skill_route",
                        f"task family {task_family!r} routes to unknown skill {skill_name!r}",
                    )
                )
        default_skill = skill_selector.get("default_skill")
        if default_skill and default_skill not in known_skills:
            issues.append(AuditIssue("error", "middleware", "skill_selector.json", "contract_unknown_default_skill", f"default skill is unknown: {default_skill!r}"))

        tool_rows = generation_tools.get("tools", {}) if isinstance(generation_tools, dict) else {}
        known_tools = set(tool_rows.keys()) if isinstance(tool_rows, dict) else set()
        for key in ("default_tool", "identity_tool"):
            tool_name = tool_router.get(key)
            if tool_name and tool_name not in known_tools:
                issues.append(AuditIssue("error", "middleware", "tool_router.json", "contract_unknown_tool_route", f"{key} points to unknown tool {tool_name!r}"))

    def _audit_memory_files(self, root: Path, issues: List[AuditIssue]) -> None:
        memory_dir = root / "memory"
        if not memory_dir.exists():
            return
        for path in sorted(memory_dir.glob("*.jsonl")):
            for idx, row in enumerate(read_jsonl(path), start=1):
                for walk_path, key, value in self._walk(row):
                    if key in FORBIDDEN_KEYS["memory"]:
                        issues.append(
                            AuditIssue(
                                "error",
                                "memory",
                                path.name,
                                "benchmark_label_leak",
                                f"memory row {idx} contains benchmark/evaluator label key: {'.'.join(walk_path)}",
                                {"line": idx, "key": key},
                            )
                        )
                    if isinstance(value, str) and "fallback" in value.lower():
                        issues.append(
                            AuditIssue(
                                "error",
                                "memory",
                                path.name,
                                "fallback_detected",
                                f"memory row {idx} contains fallback text: {'.'.join(walk_path)}",
                                {"line": idx, "value": value[:200]},
                            )
                        )

    def _audit_tasks(self, tasks_path: str | Path | None, issues: List[AuditIssue]) -> Dict[str, Any]:
        if tasks_path is None:
            return {"checked": False}
        path = Path(tasks_path)
        rows = read_jsonl(path)
        leaked_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            found = sorted(set(self._keys(row)) & BENCHMARK_LABEL_KEYS)
            if found:
                leaked_rows.append({"line": idx, "keys": found})
                issues.append(
                    AuditIssue(
                        "error",
                        "tasks",
                        str(path),
                        "benchmark_label_leak",
                        f"task row {idx} contains benchmark/evaluator label keys: {found}",
                        {"line": idx, "keys": found},
                    )
                )
        return {
            "checked": True,
            "path": str(path),
            "num_rows": len(rows),
            "prompt_only": not leaked_rows,
            "leaked_rows": leaked_rows[:20],
        }

    def _inspection_capabilities(self, artifact: Dict[str, Any]) -> set[str]:
        caps: set[str] = set()
        for tool in (artifact.get("tools", {}) or {}).values():
            raw = tool.get("capabilities", [])
            if isinstance(raw, dict):
                caps.update(str(k) for k, v in raw.items() if v)
            elif isinstance(raw, list):
                caps.update(str(v) for v in raw)
        return caps

    def _walk(self, obj: Any, path: Sequence[str] = ()) -> Iterable[tuple[List[str], str, Any]]:
        if isinstance(obj, dict):
            for key, value in obj.items():
                current = [*path, str(key)]
                yield current, str(key), value
                yield from self._walk(value, current)
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                yield from self._walk(value, [*path, str(idx)])

    def _keys(self, obj: Any) -> Iterable[str]:
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield str(key)
                yield from self._keys(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from self._keys(value)


class RunArtifactAuditor:
    """Audit generated run artifacts for leakage and fallback evidence.

    This audit is intended for smoke/full run acceptance.  It checks the task
    file and Gen-Harness experience traces.  It does not inspect official
    benchmark result files because those are allowed only after generation for
    reporting.
    """

    def audit(
        self,
        *,
        tasks_path: str | Path | None = None,
        experiences_path: str | Path | None = None,
        inspection_config_path: str | Path | None = None,
        backend_config_path: str | Path | None = None,
        write_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        issues: List[AuditIssue] = []
        task_audit = self._audit_tasks(tasks_path, issues) if tasks_path else {"checked": False}
        experience_audit = self._audit_experiences(experiences_path, issues) if experiences_path else {"checked": False}
        summary_audit = self._audit_run_summaries(experiences_path, issues) if experiences_path else {"checked": False}
        inspection_audit = self._audit_runtime_config(inspection_config_path, "inspection_config", issues) if inspection_config_path else {"checked": False}
        backend_audit = self._audit_runtime_config(backend_config_path, "backend_config", issues) if backend_config_path else {"checked": False}
        report = {
            "schema": RUN_AUDIT_SCHEMA,
            "tasks": task_audit,
            "experiences": experience_audit,
            "summaries": summary_audit,
            "inspection_config": inspection_audit,
            "backend_config": backend_audit,
            "checks": {
                "benchmark_labels_absent": not any(i.code == "benchmark_label_leak" for i in issues),
                "fallback_absent": not any(i.code == "fallback_detected" for i in issues),
                "official_evaluator_absent_from_internal_loop": not any(i.code == "official_evaluator_in_internal_loop" for i in issues),
                "internal_verifier_only": not any(i.code == "official_evaluator_in_internal_loop" for i in issues),
                "benchmark_conditioning_absent": not any(i.code == "benchmark_conditioned_generation" for i in issues),
                "runtime_component_provenance_complete": not any(i.code == "runtime_component_provenance_incomplete" for i in issues),
                "experience_trace_size_bounded": not any(i.code == "experience_trace_too_large" for i in issues),
                "formal_policy_extractor_provenance_present": not any(i.code == "formal_policy_extractor_provenance_missing" for i in issues),
                "formal_policy_extractor_is_llm": not any(i.code in {"legacy_policy_extractor_in_formal_trace", "formal_policy_extractor_not_llm"} for i in issues),
                "formal_backend_uses_real_generator": not any(i.code == "unsupported_formal_generator_backend" for i in issues),
                "formal_inspection_uses_real_visual_evidence": not any(i.code == "formal_inspection_uses_metadata_only_diagnostic" for i in issues),
            },
            "passed": not any(i.severity == "error" for i in issues),
            "issues": [i.to_dict() for i in issues],
        }
        if write_path is not None:
            write_json(write_path, report)
        return report

    def _audit_tasks(self, tasks_path: str | Path, issues: List[AuditIssue]) -> Dict[str, Any]:
        return ComponentBoundaryAuditor()._audit_tasks(tasks_path, issues)

    def _audit_experiences(self, experiences_path: str | Path, issues: List[AuditIssue]) -> Dict[str, Any]:
        path = Path(experiences_path)
        if not path.is_file():
            issues.append(
                AuditIssue(
                    "error",
                    "run",
                    str(path),
                    "experience_trace_missing",
                    "formal run audit requires an existing Visual Experience JSONL trace",
                )
            )
            return {
                "checked": False,
                "path": str(path),
                "num_rows": None,
                "reason": "experience_trace_missing",
            }
        if path.exists() and path.stat().st_size > MAX_RUN_EXPERIENCES_BYTES:
            size_mib = path.stat().st_size / (1024 * 1024)
            limit_mib = MAX_RUN_EXPERIENCES_BYTES / (1024 * 1024)
            issues.append(
                AuditIssue(
                    "error",
                    "run",
                    str(path),
                    "experience_trace_too_large",
                    f"experience trace is too large for safe audit: {size_mib:.1f} MiB > {limit_mib:.1f} MiB",
                    {"size_mib": round(size_mib, 3), "limit_mib": round(limit_mib, 3)},
                )
            )
            return {
                "checked": False,
                "path": str(path),
                "num_rows": None,
                "size_mib": round(size_mib, 3),
                "max_size_mib": round(limit_mib, 3),
                "reason": "experience_trace_too_large",
            }
        rows = read_jsonl(path)
        fallback_rows = []
        leak_rows = []
        official_rows = []
        benchmark_conditioning_rows = []
        legacy_policy_rows = []
        provenance_rows = []
        for idx, row in enumerate(rows, start=1):
            export_metadata_allowed = self._export_metadata_allowed(row)
            if "export_metadata" in row and not export_metadata_allowed:
                leak_rows.append({"line": idx, "keys": ["export_metadata"]})
                issues.append(
                    AuditIssue(
                        "error",
                        "run",
                        str(path),
                        "benchmark_label_leak",
                        f"experience row {idx} contains export_metadata outside an export/generation result row",
                        {"line": idx, "keys": ["export_metadata"]},
                    )
                )
            missing_provenance = self._missing_component_provenance(row)
            if missing_provenance:
                provenance_rows.append({"line": idx, "missing": missing_provenance})
                issues.append(
                    AuditIssue(
                        "error",
                        "run",
                        str(path),
                        "runtime_component_provenance_incomplete",
                        f"experience row {idx} does not record complete runtime component provenance: {missing_provenance}",
                        {"line": idx, "missing": missing_provenance},
                    )
                )
            found_keys = self._run_benchmark_label_keys(row)
            if found_keys:
                leak_rows.append({"line": idx, "keys": found_keys})
                issues.append(
                    AuditIssue(
                        "error",
                        "run",
                        str(path),
                        "benchmark_label_leak",
                        f"experience row {idx} contains benchmark/evaluator label keys: {found_keys}",
                        {"line": idx, "keys": found_keys},
                    )
                )
            for walk_path, key, value in ComponentBoundaryAuditor()._walk(row):
                if export_metadata_allowed and walk_path and str(walk_path[0]) == "export_metadata":
                    continue
                key_lowered = str(key).lower()
                lowered = value.lower() if isinstance(value, str) else ""
                if key_lowered in {"fallback_used", "fallback_image_substitution_allowed"} and value is True:
                    fallback_rows.append({"line": idx, "path": walk_path, "value": value})
                    issues.append(
                        AuditIssue(
                            "error",
                            "run",
                            str(path),
                            "fallback_detected",
                            f"experience row {idx} explicitly reports fallback usage at {'.'.join(walk_path)}",
                            {"line": idx, "path": walk_path},
                        )
                    )
                    continue
                if key_lowered in {
                    "official_labels_used",
                    "official_evaluator_used",
                    "official_scores_used",
                    "uses_official_labels",
                    "uses_official_eval_outputs",
                    "uses_scores",
                    "official_evaluator_used_for_generation_or_selection",
                    "official_labels_used_for_prompt_compilation",
                } and value is True:
                    official_rows.append({"line": idx, "path": walk_path, "value": value})
                    issues.append(
                        AuditIssue(
                            "error",
                            "run",
                            str(path),
                            "official_evaluator_in_internal_loop",
                            f"experience row {idx} explicitly reports official benchmark signal usage at {'.'.join(walk_path)}",
                            {"line": idx, "path": walk_path},
                        )
                    )
                    continue
                fallback_field_is_active = value is True or (
                    isinstance(value, str)
                    and value.strip().lower() not in {"", "false", "0", "no", "none", "null"}
                )
                if (
                    ("fallback_image" in key_lowered or "fallback image" in key_lowered)
                    and fallback_field_is_active
                ):
                    fallback_rows.append({"line": idx, "path": walk_path, "value": str(value)[:200]})
                    issues.append(
                        AuditIssue(
                            "error",
                            "run",
                            str(path),
                            "fallback_detected",
                            f"experience row {idx} contains fallback image field at {'.'.join(walk_path)}",
                            {"line": idx, "path": walk_path},
                        )
                    )
                    continue
                if not isinstance(value, str):
                    continue
                if "fallback_image" in lowered or "fallback image" in lowered:
                    fallback_rows.append({"line": idx, "path": walk_path, "value": value[:200]})
                    issues.append(
                        AuditIssue(
                            "error",
                            "run",
                            str(path),
                            "fallback_detected",
                            f"experience row {idx} contains fallback image text at {'.'.join(walk_path)}",
                            {"line": idx, "path": walk_path},
                        )
                    )
                if self._looks_like_official_internal_evaluator(lowered):
                    official_rows.append({"line": idx, "path": walk_path, "value": value[:200]})
                    issues.append(
                        AuditIssue(
                            "error",
                            "run",
                            str(path),
                            "official_evaluator_in_internal_loop",
                            f"experience row {idx} references an official evaluator inside internal trace: {'.'.join(walk_path)}",
                            {"line": idx, "path": walk_path},
                        )
                    )
                if self._looks_like_benchmark_conditioning(lowered) and not self._benchmark_string_allowed(walk_path):
                    benchmark_conditioning_rows.append({"line": idx, "path": walk_path, "value": value[:200]})
                    issues.append(
                        AuditIssue(
                            "error",
                            "run",
                            str(path),
                            "benchmark_conditioned_generation",
                            f"experience row {idx} contains benchmark-specific generation/selection token: {'.'.join(walk_path)}",
                            {"line": idx, "path": walk_path},
                        )
                    )
                if "policy_extractor.rule_based" in lowered or "rule_based_policy_extractor" in lowered:
                    legacy_policy_rows.append({"line": idx, "path": walk_path, "value": value[:200]})
                    issues.append(
                        AuditIssue(
                            "error",
                            "run",
                            str(path),
                            "legacy_policy_extractor_in_formal_trace",
                            f"experience row {idx} uses the legacy rule-based policy extractor in formal trace: {'.'.join(walk_path)}",
                            {"line": idx, "path": walk_path},
                        )
                    )
        return {
            "checked": True,
            "path": str(path),
            "num_rows": len(rows),
            "leak_rows": leak_rows[:20],
            "fallback_rows": fallback_rows[:20],
            "official_internal_rows": official_rows[:20],
            "benchmark_conditioning_rows": benchmark_conditioning_rows[:20],
            "legacy_policy_rows": legacy_policy_rows[:20],
            "provenance_rows": provenance_rows[:20],
        }

    def _audit_run_summaries(self, experiences_path: str | Path, issues: List[AuditIssue]) -> Dict[str, Any]:
        root = Path(experiences_path).parent
        candidates = [
            root / "suite_evaluation_summary.json",
            root / "run_summary.json",
            root / "geneval_generation_summary.json",
            root / "geneval2_generation_summary.json",
            root / "wise_generation_summary.json",
            root / "t2icompbenchpp_generation_summary.json",
            root / "slow_loop_summary.json",
            root / "self_evolve_run_summary.json",
            root / "self_evolve_manifest.json",
            root / "round_input_snapshot.json",
            root / "quality_summary.json",
        ]
        checked = []
        missing = []
        non_llm = []
        for path in candidates:
            if not path.is_file():
                continue
            data = read_json(path)
            manifest = self._summary_policy_extractor_manifest(data)
            checked.append({"path": str(path), "policy_extractor_manifest": manifest})
            if not manifest:
                missing.append(str(path))
                issues.append(
                    AuditIssue(
                        "error",
                        "run",
                        str(path),
                        "formal_policy_extractor_provenance_missing",
                        "run summary does not record policy extractor provenance",
                    )
                )
                continue
            extractor = manifest.get("policy_extractor", {}) if isinstance(manifest, dict) else {}
            extractor_name = extractor.get("component_name") if isinstance(extractor, dict) else None
            if extractor_name != "openai_chat_policy_extractor" or manifest.get("formal_llm_policy_extractor") is not True:
                non_llm.append(str(path))
                issues.append(
                    AuditIssue(
                        "error",
                        "run",
                        str(path),
                        "formal_policy_extractor_not_llm",
                        "run summary does not prove that formal policy extraction used the LLM extractor",
                        {"policy_extractor_manifest": manifest},
                    )
                )
            if (
                path.name
                in {
                    "geneval_generation_summary.json",
                    "geneval2_generation_summary.json",
                    "wise_generation_summary.json",
                    "t2icompbenchpp_generation_summary.json",
                }
                and isinstance(data, dict)
            ):
                methods = data.get("methods", {})
                if isinstance(methods, dict):
                    for method_name, method_summary in methods.items():
                        if not isinstance(method_summary, dict) or int(method_summary.get("num_rows", 0) or 0) <= 0:
                            continue
                        if int(method_summary.get("num_internal_inspection_attempts", 0) or 0) <= 0:
                            issues.append(
                                AuditIssue(
                                    "error",
                                    "run",
                                    str(path),
                                    "formal_fast_loop_missing",
                                    f"formal benchmark method {method_name!r} did not execute the required LLM Fast Loop",
                                )
                            )
        if not checked:
            issues.append(
                AuditIssue(
                    "error",
                    "run",
                    str(root),
                    "formal_policy_extractor_provenance_missing",
                    "formal run audit requires a run summary or manifest that records policy extractor provenance",
                )
            )
        return {
            "checked": True,
            "directory": str(root),
            "num_summaries": len(checked),
            "checked_summaries": checked,
            "missing_policy_extractor_manifest": missing,
            "non_llm_policy_extractor_summaries": non_llm,
        }

    def _summary_policy_extractor_manifest(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        manifest = data.get("policy_extractor_manifest")
        if isinstance(manifest, dict):
            return manifest
        inputs = data.get("inputs")
        if isinstance(inputs, dict) and isinstance(inputs.get("policy_extractor_manifest"), dict):
            return inputs["policy_extractor_manifest"]
        return {}

    def _run_benchmark_label_keys(self, row: Dict[str, Any]) -> List[str]:
        """Distinguish internal verifier replies from benchmark answers.

        Atomic internal VLM probes store a literal yes/no `answer` below a
        `semantic_checks` node. They are produced from the generated image and
        are not benchmark labels. Every answer-like key elsewhere remains a
        hard leakage error.
        """
        found = set()
        export_metadata_allowed = self._export_metadata_allowed(row)
        for walk_path, key, _ in ComponentBoundaryAuditor()._walk(row):
            if export_metadata_allowed and walk_path and str(walk_path[0]) == "export_metadata":
                continue
            name = str(key)
            if name not in RUN_BENCHMARK_LABEL_KEYS and name not in RUN_BENCHMARK_PROVENANCE_KEYS:
                continue
            if name == "answer" and "semantic_checks" in [str(part) for part in walk_path[:-1]]:
                continue
            found.add(name)
        return sorted(found)

    def _export_metadata_allowed(self, row: Dict[str, Any]) -> bool:
        if not isinstance(row, dict) or "export_metadata" not in row:
            return False
        if row.get("official_eval_status") in {"pending", "not_exportable"} and "method" in row:
            return True
        if "generation" in row and "program" in row and "method" in row:
            return True
        return False

    def _missing_component_provenance(self, row: Dict[str, Any]) -> List[str]:
        decisions = row.get("component_decisions")
        if not isinstance(decisions, dict):
            return [f"component_decisions.{component}" for component in COMPONENTS] + ["component_decisions.component_sources"]
        missing: List[str] = []
        for component in COMPONENTS:
            value = decisions.get(component)
            if not isinstance(value, dict):
                missing.append(f"component_decisions.{component}")
        sources = decisions.get("component_sources")
        if not isinstance(sources, dict):
            missing.append("component_decisions.component_sources")
        else:
            for component in COMPONENTS:
                source = sources.get(component)
                if not isinstance(source, str) or not source.strip():
                    missing.append(f"component_decisions.component_sources.{component}")
        return missing

    def _audit_runtime_config(self, config_path: str | Path, component: str, issues: List[AuditIssue]) -> Dict[str, Any]:
        path = Path(config_path)
        data = read_json(path)
        if component == "backend_config" and isinstance(data, dict):
            backend_error = formal_backend_config_error(data)
            if backend_error:
                issues.append(
                    AuditIssue(
                        "error",
                        component,
                        str(path),
                        "unsupported_formal_generator_backend",
                        backend_error,
                        {
                            "backend_type": data.get("type"),
                            "backend_name": data.get("name"),
                        },
                    )
                )
        if component == "inspection_config":
            self._audit_inspection_config(data, path, component, issues)
        for walk_path, key, value in ComponentBoundaryAuditor()._walk(data):
            key_lowered = str(key).lower()
            if key_lowered in {"fallback_used", "fallback_image_substitution_allowed"} and value is True:
                issues.append(
                    AuditIssue(
                        "error",
                        component,
                        str(path),
                        "fallback_detected",
                        f"{component} config explicitly enables fallback usage at {'.'.join(walk_path)}",
                    )
                )
            if key_lowered in {
                "official_labels_used",
                "official_evaluator_used",
                "official_scores_used",
                "uses_official_labels",
                "uses_official_eval_outputs",
                "uses_scores",
                "official_evaluator_used_for_generation_or_selection",
                "official_labels_used_for_prompt_compilation",
            } and value is True:
                issues.append(
                    AuditIssue(
                        "error",
                        component,
                        str(path),
                        "official_evaluator_in_internal_loop",
                        f"{component} config explicitly enables official benchmark signal usage at {'.'.join(walk_path)}",
                    )
                )
        command_strings = []
        for walk_path, key, value in ComponentBoundaryAuditor()._walk(data):
            if key not in {"command", "persistent_command"} or not isinstance(value, list):
                continue
            command = " ".join(str(part) for part in value)
            command_strings.append(command)
            lowered = command.lower()
            if self._looks_like_official_internal_evaluator(lowered):
                issues.append(
                    AuditIssue(
                        "error",
                        component,
                        str(path),
                        "official_evaluator_in_internal_loop",
                        f"{component} command references an official evaluator: {'.'.join(walk_path)}",
                        {"command": command},
                    )
                )
            if "fallback_image" in lowered or "fallback image" in lowered:
                issues.append(
                    AuditIssue(
                        "error",
                        component,
                        str(path),
                        "fallback_detected",
                        f"{component} command references fallback image substitution: {'.'.join(walk_path)}",
                        {"command": command},
                    )
                )
            if self._looks_like_benchmark_conditioning(command):
                issues.append(
                    AuditIssue(
                        "error",
                        component,
                        str(path),
                        "benchmark_conditioned_generation",
                        f"{component} command references benchmark-specific generation/selection settings: {'.'.join(walk_path)}",
                        {"command": command},
                    )
                )
        return {"checked": True, "path": str(path), "num_commands": len(command_strings)}

    def _audit_inspection_config(self, data: Any, path: Path, component: str, issues: List[AuditIssue]) -> None:
        configs = data.get("inspectors", [data]) if isinstance(data, dict) else []
        for index, cfg in enumerate(configs):
            if not isinstance(cfg, dict):
                continue
            typ = cfg.get("type")
            label = f"inspectors.{index}" if isinstance(data, dict) and isinstance(data.get("inspectors"), list) else "root"
            if typ == "object_requirements":
                issues.append(
                    AuditIssue(
                        "error",
                        component,
                        str(path),
                        "formal_inspection_uses_metadata_only_diagnostic",
                        f"{label} uses object_requirements, which reads backend metadata instead of real visual evidence",
                    )
                )
            if typ == "detector_object_requirements" and "command" not in cfg and "persistent_command" not in cfg:
                issues.append(
                    AuditIssue(
                        "error",
                        component,
                        str(path),
                        "formal_inspection_uses_metadata_only_diagnostic",
                        f"{label} uses detector_object_requirements without command/persistent_command",
                    )
                )
            if typ == "lightweight_alignment" and "command" not in cfg and "persistent_command" not in cfg:
                issues.append(
                    AuditIssue(
                        "error",
                        component,
                        str(path),
                        "formal_inspection_uses_metadata_only_diagnostic",
                        f"{label} uses lightweight_alignment without command/persistent_command",
                    )
                )

    def _looks_like_official_internal_evaluator(self, text: str) -> bool:
        return (
            "evaluate_geneval" in text
            or "soft_tifa" in text
            or "official_soft_tifa" in text
            or "geneval2_data.jsonl" in text
        )

    def _looks_like_benchmark_conditioning(self, text: str) -> bool:
        lowered = str(text).lower()
        forbidden_tokens = (
            "genharness_geneval2_",
            "genharness_geneval_",
            "policy_extractor.t2icompbenchpp",
            "raw_geneval2_",
            "raw_geneval_",
            "t2icompbenchpp_",
        )
        return any(token in lowered for token in forbidden_tokens)

    def _benchmark_string_allowed(self, walk_path: Sequence[str]) -> bool:
        if not walk_path:
            return False
        if len(walk_path) == 1 and str(walk_path[0]) == "task_id":
            return True
        return False


class GoalReadinessAuditor:
    """Audit whether a self-evolve -> smoke -> full GenEval2 run can support the goal claim."""

    def audit(
        self,
        run_root: str | Path,
        *,
        smoke_min_am: float = 88.0,
        smoke_min_gm: float = 67.0,
        full_target_gm: float = 72.0,
        write_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        root = Path(run_root)
        issues: List[AuditIssue] = []
        self_evolve_dir = root / "self_evolve_dev"
        smoke_dir = root / "geneval2_smoke32"
        full_dir = root / "geneval2_full800"

        runtime_strategy = self._audit_runtime_strategy(root / "runtime_strategy_env.json", issues)
        self_evolve = self._audit_self_evolve(self_evolve_dir, issues)
        smoke = self._audit_eval_stage(
            smoke_dir,
            score_name="official_soft_tifa_batched_smoke32.json",
            issues=issues,
            min_am=smoke_min_am,
            min_gm=smoke_min_gm,
            require_full_score=False,
        )
        full = self._audit_eval_stage(
            full_dir,
            score_name="official_soft_tifa_batched_full800.json",
            issues=issues,
            min_gm=full_target_gm,
            require_full_score=True,
        )

        checks = {
            "runtime_strategy_recorded": runtime_strategy.get("passed", False),
            "self_evolve_completed": self_evolve.get("passed", False),
            "smoke_passed": smoke.get("passed", False),
            "full_passed": full.get("passed", False),
            "full_target_gm_reached": bool(full.get("target_gm_reached", False)),
            "leakage_and_fallback_absent": not any(
                i.code in {
                    "benchmark_label_leak",
                    "fallback_detected",
                    "official_evaluator_in_internal_loop",
                    "benchmark_conditioned_generation",
                }
                for i in issues
            ),
        }
        report = {
            "schema": GOAL_READINESS_SCHEMA,
            "run_root": str(root),
            "thresholds": {
                "smoke_min_am": smoke_min_am,
                "smoke_min_gm": smoke_min_gm,
                "full_target_gm": full_target_gm,
            },
            "runtime_strategy": runtime_strategy,
            "self_evolve": self_evolve,
            "smoke": smoke,
            "full": full,
            "checks": checks,
            "passed": all(checks.values()) and not any(i.severity == "error" for i in issues),
            "issues": [i.to_dict() for i in issues],
        }
        if write_path is not None:
            write_json(write_path, report)
        return report

    def _audit_runtime_strategy(self, path: Path, issues: List[AuditIssue]) -> Dict[str, Any]:
        data = self._read_required_json(path, "runtime_strategy", issues)
        if not isinstance(data, dict):
            return {"checked": True, "path": str(path), "passed": False}
        strategy_env = data.get("strategy_env", {})
        bad_keys = [str(key) for key in strategy_env if RunArtifactAuditor()._looks_like_benchmark_conditioning(str(key))]
        semantic_keys = [str(key) for key in strategy_env if str(key).startswith("GENHARNESS_SELF_EVOLVE_")]
        versioned_keys = [
            str(key)
            for key in strategy_env
            if str(key).startswith("GENHARNESS_V")
            and len(str(key)) > len("GENHARNESS_V")
            and str(key)[len("GENHARNESS_V")].isdigit()
        ]
        passed = bool(data.get("benchmark_conditioned_env_absent", False)) and not bad_keys and bool(semantic_keys) and not versioned_keys
        if not passed:
            if bad_keys or not bool(data.get("benchmark_conditioned_env_absent", False)):
                issues.append(
                    AuditIssue(
                        "error",
                        "runtime_strategy",
                        str(path),
                        "benchmark_conditioned_generation",
                        "runtime strategy contains benchmark-conditioned env settings",
                        {"bad_keys": bad_keys},
                    )
                )
            if not semantic_keys:
                issues.append(
                    AuditIssue(
                        "error",
                        "runtime_strategy",
                        str(path),
                        "self_evolve_strategy_unproven",
                        "runtime strategy does not record semantic GENHARNESS_SELF_EVOLVE_* controls",
                    )
                )
            if versioned_keys:
                issues.append(
                    AuditIssue(
                        "error",
                        "runtime_strategy",
                        str(path),
                        "versioned_strategy_env_detected",
                        "runtime strategy records historical versioned controls instead of semantic self-evolve controls",
                        {"versioned_keys": versioned_keys},
                    )
                )
        return {
            "checked": True,
            "path": str(path),
            "passed": passed,
            "num_strategy_env": len(strategy_env) if isinstance(strategy_env, dict) else 0,
            "bad_keys": bad_keys,
            "semantic_keys": semantic_keys,
            "versioned_keys": versioned_keys,
        }

    def _audit_self_evolve(self, root: Path, issues: List[AuditIssue]) -> Dict[str, Any]:
        before = self._read_required_json(root / "component_audit_before.json", "self_evolve", issues)
        after = self._read_required_json(root / "component_audit_after.json", "self_evolve", issues)
        summary = self._read_required_json(root / "output" / "self_evolve_run_summary.json", "self_evolve", issues)
        manifest = self._read_required_json(root / "self_evolve_manifest.json", "self_evolve", issues)
        passed = True
        for name, data in {"before": before, "after": after}.items():
            if isinstance(data, dict) and not bool(data.get("passed", False)):
                passed = False
                issues.append(AuditIssue("error", "self_evolve", str(root), "component_audit_failed", f"self-evolve {name} component audit failed"))
            elif not isinstance(data, dict):
                passed = False
        if isinstance(summary, dict):
            if summary.get("schema") != "gen_harness.self_evolve_run.v1":
                passed = False
                issues.append(AuditIssue("error", "self_evolve", str(root), "invalid_schema", "self-evolve summary schema is missing or invalid"))
            if int(summary.get("fast_loop_attempts", 0) or 0) < 2:
                passed = False
                issues.append(AuditIssue("error", "self_evolve", str(root), "fast_loop_not_enabled", "self-evolve summary does not prove fast-loop attempts >= 2"))
            if not self._formal_policy_extractor_manifest_is_llm(summary.get("policy_extractor_manifest", {})):
                passed = False
                issues.append(
                    AuditIssue(
                        "error",
                        "self_evolve",
                        str(root),
                        "formal_policy_extractor_not_llm",
                        "self-evolve summary does not prove that prompt-only policy extraction used the LLM extractor",
                        {"policy_extractor_manifest": summary.get("policy_extractor_manifest", {})},
                    )
                )
            guards = summary.get("leakage_and_fallback_guards", {})
            if not isinstance(guards, dict) or any(
                bool(guards.get(key, False))
                for key in (
                    "allow_benchmark_tasks",
                    "official_evaluator_used_for_generation_or_selection",
                    "official_labels_used_for_prompt_compilation",
                    "fallback_image_substitution_allowed",
                )
            ):
                passed = False
                issues.append(AuditIssue("error", "self_evolve", str(root), "self_evolve_guard_failed", "self-evolve leakage/fallback guards are not clean"))
            if not self._reference_trajectory_policy_is_clean(summary.get("reference_trajectory_policy", {})):
                passed = False
                issues.append(
                    AuditIssue(
                        "error",
                        "self_evolve",
                        str(root),
                        "reference_trajectory_policy_unsafe",
                        "self-evolve summary does not prove reference trajectories are non-benchmark and leakage-safe",
                    )
                )
        else:
            passed = False
        if isinstance(manifest, dict):
            if not self._manifest_proves_self_evolve_chain(manifest, issues, root):
                passed = False
            if isinstance(summary, dict) and not self._manifest_proves_round_aggregation(manifest, summary, issues, root):
                passed = False
            if isinstance(summary, dict) and not self._self_evolve_completion_is_clean(summary, manifest, issues, root):
                passed = False
            manifest_guards = manifest.get("leakage_and_fallback_guards", {})
            if isinstance(manifest_guards, dict) and bool(manifest_guards.get("fallback_used", False)):
                passed = False
                issues.append(AuditIssue("error", "self_evolve", str(root), "fallback_detected", "self-evolve manifest reports fallback usage"))
            inputs = manifest.get("inputs", {})
            policy_manifest = inputs.get("policy_extractor_manifest", {}) if isinstance(inputs, dict) else {}
            if not self._formal_policy_extractor_manifest_is_llm(policy_manifest):
                passed = False
                issues.append(
                    AuditIssue(
                        "error",
                        "self_evolve",
                        str(root),
                        "formal_policy_extractor_not_llm",
                        "self-evolve manifest does not prove that prompt-only policy extraction used the LLM extractor",
                        {"policy_extractor_manifest": policy_manifest},
                    )
                )
        else:
            passed = False
        return {
            "checked": True,
            "path": str(root),
            "passed": passed,
            "stop_reason": summary.get("stop_reason") if isinstance(summary, dict) else None,
            "num_rounds": len(summary.get("rounds", [])) if isinstance(summary, dict) and isinstance(summary.get("rounds"), list) else None,
            "policy_extractor_manifest": summary.get("policy_extractor_manifest", {}) if isinstance(summary, dict) else {},
        }

    def _self_evolve_completion_is_clean(
        self,
        summary: Dict[str, Any],
        manifest: Dict[str, Any],
        issues: List[AuditIssue],
        root: Path,
    ) -> bool:
        stop_reason = str(summary.get("stop_reason") or "").strip()
        incomplete_stop_reasons = {
            "",
            "initial_component_audit_failed",
            "final_component_audit_failed",
            "no_patch_proposals",
            "no_accepted_patches",
            "accepted_patch_not_promoted",
        }
        if stop_reason in incomplete_stop_reasons:
            issues.append(
                AuditIssue(
                    "error",
                    "self_evolve",
                    str(root),
                    "self_evolve_stop_reason_incomplete",
                    "self-evolve stopped before a clean no-failure state or an effective promoted patch",
                    {"stop_reason": stop_reason},
                )
            )
            return False

        observability = manifest.get("observability", {})
        gate_summary = manifest.get("gate_summary", {})
        failed = int(observability.get("num_failed_experiences", 0) or 0) if isinstance(observability, dict) else 0
        patch_proposals = int(observability.get("num_patch_proposals", 0) or 0) if isinstance(observability, dict) else 0
        accepted = int(gate_summary.get("num_accepted", 0) or 0) if isinstance(gate_summary, dict) else 0
        promoted = int(gate_summary.get("num_promoted", 0) or 0) if isinstance(gate_summary, dict) else 0
        promote_expected = bool(summary.get("promote_accepted", False))
        if failed > 0 and patch_proposals > 0 and accepted <= 0:
            issues.append(
                AuditIssue(
                    "error",
                    "self_evolve",
                    str(root),
                    "self_evolve_no_effective_patch",
                    "self-evolve observed failures and patch proposals but no accepted patch",
                    {"failed": failed, "patch_proposals": patch_proposals, "accepted": accepted},
                )
            )
            return False
        if failed > 0 and promote_expected and promoted <= 0:
            issues.append(
                AuditIssue(
                    "error",
                    "self_evolve",
                    str(root),
                    "self_evolve_no_promoted_patch",
                    "self-evolve was configured to promote accepted patches but no patch was promoted",
                    {"failed": failed, "promoted": promoted},
                )
            )
            return False
        return True

    def _reference_trajectory_policy_is_clean(self, policy: Any) -> bool:
        if not isinstance(policy, dict):
            return False
        if policy.get("official_benchmark_reference_trajectories_allowed") is not False:
            return False
        if policy.get("official_labels_used_for_generation_or_selection") is not False:
            return False
        if policy.get("official_evaluator_used_for_generation_or_selection") is not False:
            return False
        if policy.get("benchmark_tasks_allowed") is not False:
            return False
        if policy.get("leakage_safe") is not True:
            return False
        allowed_sources = policy.get("allowed_sources", [])
        if not isinstance(allowed_sources, list) or not allowed_sources:
            return False
        allowed = {"non_benchmark_dev_tasks", "internal_baseline_traces", "human_authored_dev_traces"}
        return {str(source) for source in allowed_sources}.issubset(allowed)

    def _formal_policy_extractor_manifest_is_llm(self, manifest: Any) -> bool:
        if not isinstance(manifest, dict):
            return False
        extractor = manifest.get("policy_extractor")
        if not isinstance(extractor, dict):
            return False
        return (
            extractor.get("component_name") == "openai_chat_policy_extractor"
            and manifest.get("formal_llm_policy_extractor") is True
            and manifest.get("policy_extractor_configured") is True
        )

    def _reference_frameworks_are_clean(self, frameworks: Any) -> bool:
        if not isinstance(frameworks, list):
            return False
        required_sources = {"agentic-harness-engineering", "JIT"}
        seen: set[str] = set()
        for row in frameworks:
            if not isinstance(row, dict):
                return False
            source = str(row.get("source") or "").strip()
            if source in required_sources:
                seen.add(source)
            if row.get("used_as") != "conceptual_reference_only":
                return False
            if row.get("copied_runtime") is not False:
                return False
            if row.get("local_path_recorded") is not False:
                return False
            if "local_path" in row:
                return False
            if row.get("official_benchmark_reference_trajectories_allowed") is not False:
                return False
            invariants = row.get("adopted_invariants", [])
            if not isinstance(invariants, list) or not invariants:
                return False
        return required_sources.issubset(seen)

    def _manifest_proves_self_evolve_chain(self, manifest: Dict[str, Any], issues: List[AuditIssue], root: Path) -> bool:
        passed = True
        if manifest.get("schema") != "gen_harness.self_evolve_manifest.v1":
            issues.append(AuditIssue("error", "self_evolve", str(root), "invalid_schema", "self-evolve manifest schema is missing or invalid"))
            passed = False
        reference_model = manifest.get("reference_model", {})
        if not isinstance(reference_model, dict) or reference_model.get("copied_runtime") is not False or reference_model.get("local_path_recorded") is not False:
            issues.append(AuditIssue("error", "self_evolve", str(root), "reference_policy_unproven", "manifest does not prove AHE/reference code was conceptual-only without local path recording"))
            passed = False
        if not self._reference_frameworks_are_clean(manifest.get("reference_frameworks", [])):
            issues.append(
                AuditIssue(
                    "error",
                    "self_evolve",
                    str(root),
                    "reference_frameworks_unproven",
                    "manifest does not prove AHE/HarnessEvolve and JIT were used only as clean conceptual references",
                )
            )
            passed = False
        if not self._reference_trajectory_policy_is_clean(manifest.get("reference_trajectory_policy", {})):
            issues.append(
                AuditIssue(
                    "error",
                    "self_evolve",
                    str(root),
                    "reference_trajectory_policy_unsafe",
                    "self-evolve manifest does not prove reference trajectories are non-benchmark and leakage-safe",
                )
            )
            passed = False
        attribution = manifest.get("failure_attribution", {})
        if (
            not isinstance(attribution, dict)
            or attribution.get("official_labels_used") is not False
            or attribution.get("fallback_used") is not False
            or attribution.get("leakage_safe") is not True
            or attribution.get("formal_component_field") != "llm_harness_localizer_frontier_only"
            or attribution.get("frontier_source") != "llm_harness_localizer"
        ):
            issues.append(AuditIssue("error", "self_evolve", str(root), "failure_attribution_unproven", "manifest does not prove leakage-safe failure attribution"))
            passed = False
        observability = manifest.get("observability", {})
        if not isinstance(observability, dict):
            issues.append(AuditIssue("error", "self_evolve", str(root), "observability_missing", "manifest does not include self-evolve observability counts"))
            return False
        failed = int(observability.get("num_failed_experiences", 0) or 0)
        diagnoses = int(observability.get("num_failure_diagnoses", 0) or 0)
        if failed > 0 and diagnoses <= 0:
            issues.append(AuditIssue("error", "self_evolve", str(root), "failure_diagnosis_missing", "manifest has failures but no failure diagnoses"))
            passed = False
        if not self._trace_certificate_is_clean(manifest.get("trace_certificate", {})):
            issues.append(
                AuditIssue(
                    "error",
                    "self_evolve",
                    str(root),
                    "trace_certificate_unproven",
                    "manifest does not prove TRACE localization, NO-PATCH, scoped patching, validation, and promotion gates",
                )
            )
            passed = False
        changes = manifest.get("changes", [])
        if changes is None:
            changes = []
        if not isinstance(changes, list):
            issues.append(AuditIssue("error", "self_evolve", str(root), "self_evolve_changes_invalid", "manifest changes must be a list"))
            return False
        if failed > 0 and int(observability.get("num_patch_proposals", 0) or 0) > 0 and not changes:
            issues.append(AuditIssue("error", "self_evolve", str(root), "self_evolve_chain_missing", "manifest has patch proposals but no change chain rows"))
            passed = False
        for idx, change in enumerate(changes, start=1):
            if not isinstance(change, dict):
                issues.append(AuditIssue("error", "self_evolve", str(root), "self_evolve_change_invalid", f"manifest change {idx} is not an object"))
                passed = False
                continue
            missing = self._missing_change_chain_fields(change)
            if missing:
                issues.append(
                    AuditIssue(
                        "error",
                        "self_evolve",
                        str(root),
                        "self_evolve_chain_incomplete",
                        f"manifest change {idx} does not prove the failure->root-cause->patch->impact->gate chain",
                        {"change_index": idx, "missing": missing},
                    )
                )
                passed = False
            scope_missing = self._missing_change_component_scope_fields(change)
            if scope_missing:
                issues.append(
                    AuditIssue(
                        "error",
                        "self_evolve",
                        str(root),
                        "self_evolve_component_scope_incomplete",
                        f"manifest change {idx} does not prove a single valid component-owned patch scope",
                        {"change_index": idx, "missing": scope_missing},
                    )
                )
                passed = False
        return passed

    def _trace_certificate_is_clean(self, certificate: Any) -> bool:
        if not isinstance(certificate, dict):
            return False
        schema = certificate.get("schema")
        if schema == "gen_harness.trace_certificate.aggregate.v1":
            if certificate.get("all_rounds_certified") is not True:
                return False
            rounds = certificate.get("rounds", [])
            return isinstance(rounds, list) and bool(rounds)
        if schema != "gen_harness.trace_certificate.v1":
            return False
        required_flow = {
            "constraint_indexed_visual_experience",
            "llm_harness_localization_with_no_patch",
            "responsibility_scoped_patch_proposal",
            "target_heldout_preservation_validation",
            "regression_safe_promotion_only",
        }
        if not required_flow.issubset({str(item) for item in certificate.get("flow", []) if isinstance(item, str)}):
            return False
        localization = certificate.get("localization", {})
        proposals = certificate.get("patch_proposals", {})
        validation = certificate.get("validation", {})
        promotion = certificate.get("promotion", {})
        guards = certificate.get("leakage_and_fallback_guards", {})
        if (
            not isinstance(localization, dict)
            or localization.get("no_patch_supported") is not True
            or localization.get("frontier_source") != "llm_harness_localizer"
        ):
            return False
        if not isinstance(proposals, dict) or proposals.get("all_component_scoped") is not True:
            return False
        if not isinstance(validation, dict):
            return False
        if int(validation.get("num_reports", 0) or 0) > 0 and validation.get("target_heldout_preservation_present") is not True:
            return False
        if validation.get("official_evaluator_used") is not False or validation.get("official_scores_used") is not False:
            return False
        if validation.get("fallback_used") is not False:
            return False
        if not isinstance(promotion, dict) or promotion.get("accepted_only") is not True:
            return False
        if not isinstance(guards, dict):
            return False
        return (
            guards.get("official_labels_used_for_prompt_compilation") is False
            and guards.get("official_evaluator_used_for_generation_or_selection") is False
            and guards.get("fallback_image_substitution_allowed") is False
        )

    def _manifest_proves_round_aggregation(
        self,
        manifest: Dict[str, Any],
        summary: Dict[str, Any],
        issues: List[AuditIssue],
        root: Path,
    ) -> bool:
        rounds = summary.get("rounds", [])
        if not isinstance(rounds, list):
            rounds = []
        expected_rounds = len([row for row in rounds if isinstance(row, dict)])
        if expected_rounds <= 0:
            return True

        observability = manifest.get("observability", {})
        round_manifests = manifest.get("round_manifests", [])
        if not isinstance(observability, dict) or not isinstance(round_manifests, list):
            issues.append(
                AuditIssue(
                    "error",
                    "self_evolve",
                    str(root),
                    "self_evolve_round_manifest_missing",
                    "self-evolve manifest does not aggregate per-round manifests",
                    {"expected_rounds": expected_rounds},
                )
            )
            return False

        observed_rounds = int(observability.get("num_rounds", 0) or 0)
        observed_manifests = int(observability.get("num_round_manifests", 0) or 0)
        if observed_rounds != expected_rounds or observed_manifests != expected_rounds or len(round_manifests) != expected_rounds:
            issues.append(
                AuditIssue(
                    "error",
                    "self_evolve",
                    str(root),
                    "self_evolve_round_manifest_missing",
                    "self-evolve manifest round aggregation does not match self-evolve summary",
                    {
                        "expected_rounds": expected_rounds,
                        "observed_rounds": observed_rounds,
                        "observed_round_manifests": observed_manifests,
                        "round_manifest_rows": len(round_manifests),
                    },
                )
            )
            return False

        bad_round_schemas = [
            {"index": idx, "schema": row.get("schema") if isinstance(row, dict) else None}
            for idx, row in enumerate(round_manifests, start=1)
            if not isinstance(row, dict) or row.get("schema") != "gen_harness.self_evolve_manifest.v1"
        ]
        if bad_round_schemas:
            issues.append(
                AuditIssue(
                    "error",
                    "self_evolve",
                    str(root),
                    "self_evolve_round_manifest_invalid",
                    "one or more aggregated round manifests have invalid schema",
                    {"bad_round_schemas": bad_round_schemas},
                )
            )
            return False
        return True

    def _missing_change_chain_fields(self, change: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        evidence = change.get("failure_evidence")
        if not isinstance(evidence, dict) or not evidence.get("supporting_experience"):
            missing.append("failure_evidence.supporting_experience")
        if not isinstance(evidence, dict) or not evidence.get("failure_diagnoses"):
            missing.append("failure_evidence.failure_diagnoses")
        root_cause = change.get("root_cause")
        if not root_cause:
            missing.append("root_cause")
        fix = change.get("targeted_fix")
        if not isinstance(fix, dict) or not fix.get("payload"):
            missing.append("targeted_fix.payload")
        impact = change.get("predicted_impact")
        if not isinstance(impact, dict) or not impact.get("improvement"):
            missing.append("predicted_impact.improvement")
        gate = change.get("gate")
        if not isinstance(gate, dict) or "accepted" not in gate:
            missing.append("gate.accepted")
        return missing

    def _missing_change_component_scope_fields(self, change: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        component = str(change.get("component") or "").strip()
        artifact = str(change.get("artifact") or "").strip()
        operation = str(change.get("operation") or "").strip()
        changed_fields = change.get("changed_fields")
        if component not in COMPONENTS:
            missing.append("component")
        if not artifact:
            missing.append("artifact")
        elif component in COMPONENTS and artifact not in REQUIRED_ARTIFACTS[component]:
            missing.append("artifact_owned_by_component")
        if not isinstance(changed_fields, list) or not changed_fields:
            missing.append("changed_fields")
        if operation not in {"set_path", "append_unique", "append_jsonl_unique", "delete_path"}:
            missing.append("operation")
        fix = change.get("targeted_fix", {})
        payload = fix.get("payload") if isinstance(fix, dict) else None
        schema_validation = payload.get("schema_validation") if isinstance(payload, dict) else None
        if not isinstance(schema_validation, dict):
            missing.append("targeted_fix.payload.schema_validation")
        else:
            if schema_validation.get("target_component") != component:
                missing.append("targeted_fix.payload.schema_validation.target_component")
            if schema_validation.get("changed_artifact") != artifact:
                missing.append("targeted_fix.payload.schema_validation.changed_artifact")
        rollback_boundary = fix.get("rollback_boundary") if isinstance(fix, dict) else None
        if not isinstance(rollback_boundary, str) or not rollback_boundary.startswith(f"{component}/"):
            missing.append("targeted_fix.rollback_boundary")
        return missing

    def _audit_eval_stage(
        self,
        root: Path,
        *,
        score_name: str,
        issues: List[AuditIssue],
        min_am: float | None = None,
        min_gm: float | None = None,
        require_full_score: bool,
    ) -> Dict[str, Any]:
        component = self._read_required_json(root / "component_audit.json", "geneval2_eval", issues)
        run_audit = self._read_required_json(root / "run_artifact_audit.json", "geneval2_eval", issues)
        image_map_path = root / "geneval2_image_map.json"
        score = self._read_required_json(root / score_name, "geneval2_eval", issues)
        image_map = self._audit_image_map_stage(
            image_map_path,
            score=score,
            require_full_score=require_full_score,
            issues=issues,
        )
        passed = True
        if isinstance(component, dict) and not bool(component.get("passed", False)):
            passed = False
            issues.append(AuditIssue("error", "geneval2_eval", str(root), "component_audit_failed", f"{root.name} component audit failed"))
        elif not isinstance(component, dict):
            passed = False
        if isinstance(run_audit, dict) and not bool(run_audit.get("passed", False)):
            passed = False
            issues.append(AuditIssue("error", "geneval2_eval", str(root), "run_artifact_audit_failed", f"{root.name} run artifact audit failed"))
        elif not isinstance(run_audit, dict):
            passed = False
        if not image_map.get("passed", False):
            passed = False
        am = self._score_value(score, "soft_tifa_am")
        gm = self._score_value(score, "soft_tifa_gm")
        if min_am is not None and (am is None or am < min_am):
            passed = False
            issues.append(AuditIssue("error", "geneval2_eval", str(root), "soft_tifa_am_below_threshold", f"{root.name} AM below threshold", {"am": am, "threshold": min_am}))
        if min_gm is not None and (gm is None or gm < min_gm):
            passed = False
            issues.append(AuditIssue("error", "geneval2_eval", str(root), "soft_tifa_gm_below_threshold", f"{root.name} GM below threshold", {"gm": gm, "threshold": min_gm}))
        if require_full_score and gm is None:
            passed = False
        return {
            "checked": True,
            "path": str(root),
            "passed": passed,
            "soft_tifa_am": am,
            "soft_tifa_gm": gm,
            "target_gm_reached": bool(gm is not None and min_gm is not None and gm >= min_gm),
            "image_map": image_map,
        }

    def _audit_image_map_stage(
        self,
        image_map_path: Path,
        *,
        score: Dict[str, Any] | None,
        require_full_score: bool,
        issues: List[AuditIssue],
    ) -> Dict[str, Any]:
        if not image_map_path.is_file():
            issues.append(AuditIssue("error", "geneval2_eval", str(image_map_path), "missing_required_artifact", f"missing required artifact: {image_map_path}"))
            return {"checked": True, "path": str(image_map_path), "passed": False}

        manifest_path = image_map_path.with_suffix(image_map_path.suffix + ".manifest.json")
        manifest = self._read_required_json(manifest_path, "geneval2_eval", issues)
        if not isinstance(manifest, dict):
            return {
                "checked": True,
                "path": str(image_map_path),
                "manifest_path": str(manifest_path),
                "passed": False,
            }

        passed = True
        schema = manifest.get("schema")
        if schema != "gen_harness.geneval2_image_map.v1":
            passed = False
            issues.append(AuditIssue("error", "geneval2_eval", str(manifest_path), "invalid_schema", "GenEval2 image-map manifest schema is missing or invalid"))

        num_benchmark = self._positive_int(manifest.get("num_benchmark_prompts"))
        num_mapped = self._positive_int(manifest.get("num_mapped_prompts"))
        missing_indices = manifest.get("missing_indices", [])
        if not isinstance(missing_indices, list):
            missing_indices = ["invalid_missing_indices"]
            passed = False
            issues.append(AuditIssue("error", "geneval2_eval", str(manifest_path), "image_map_manifest_invalid", "GenEval2 image-map manifest missing_indices must be a list"))

        if require_full_score:
            if manifest.get("require_all") is not True:
                passed = False
                issues.append(
                    AuditIssue(
                        "error",
                        "geneval2_eval",
                        str(manifest_path),
                        "image_map_not_require_all",
                        "full GenEval2 image map was not exported with require_all=true",
                    )
                )
            if num_benchmark is None or num_mapped is None or num_mapped != num_benchmark or missing_indices:
                passed = False
                issues.append(
                    AuditIssue(
                        "error",
                        "geneval2_eval",
                        str(manifest_path),
                        "image_map_incomplete",
                        "full GenEval2 image map does not prove every benchmark prompt has a generated image",
                        {
                            "num_benchmark_prompts": num_benchmark,
                            "num_mapped_prompts": num_mapped,
                            "num_missing": len(missing_indices),
                            "first_missing_indices": missing_indices[:20],
                        },
                    )
                )
        else:
            samples = self._positive_int(score.get("samples")) if isinstance(score, dict) else None
            if num_mapped is None or num_mapped <= 0:
                passed = False
                issues.append(AuditIssue("error", "geneval2_eval", str(manifest_path), "image_map_incomplete", "smoke GenEval2 image map has no mapped prompts"))
            if samples is not None and num_mapped is not None and num_mapped < samples:
                passed = False
                issues.append(
                    AuditIssue(
                        "error",
                        "geneval2_eval",
                        str(manifest_path),
                        "image_map_incomplete",
                        "smoke GenEval2 image map maps fewer prompts than the official smoke score reports",
                        {"num_mapped_prompts": num_mapped, "score_samples": samples},
                    )
                )

        return {
            "checked": True,
            "path": str(image_map_path),
            "manifest_path": str(manifest_path),
            "passed": passed,
            "require_all": manifest.get("require_all"),
            "num_benchmark_prompts": num_benchmark,
            "num_mapped_prompts": num_mapped,
            "num_missing": len(missing_indices),
        }

    def _positive_int(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _read_required_json(self, path: Path, component: str, issues: List[AuditIssue]) -> Dict[str, Any] | None:
        if not path.is_file():
            issues.append(AuditIssue("error", component, str(path), "missing_required_artifact", f"missing required artifact: {path}"))
            return None
        try:
            data = read_json(path)
        except Exception as exc:
            issues.append(AuditIssue("error", component, str(path), "invalid_json", f"invalid JSON artifact: {exc}"))
            return None
        return data if isinstance(data, dict) else None

    def _score_value(self, data: Dict[str, Any] | None, key: str) -> float | None:
        if not isinstance(data, dict):
            return None
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
