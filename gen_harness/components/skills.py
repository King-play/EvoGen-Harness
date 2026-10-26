from __future__ import annotations

from typing import Any, Dict, List

from .base import HarnessComponent
from ..schema import GenerationTask
from ..spatial_prompt import canonical_spatial_clause, normalize_relation, prompt_provenance


class VisualSkillLibrary(HarnessComponent):
    """Reusable visual workflow recipes and prompt-reference programs."""

    component_type = "skills"


    def get_skill(self, name: str) -> Dict[str, Any]:
        skills = self.artifact.get("skills", {})
        if name not in skills:
            raise KeyError(f"unknown skill: {name}")
        return skills[name]

    def compile_workflow(
        self,
        task: GenerationTask,
        skill_name: str,
        contract: Dict[str, Any],
        bindings: Dict[str, str],
        memory_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        skill = self.get_skill(skill_name)
        contract_constraints = list(contract.get("constraints", []))
        missing_steps = []
        required_steps = set(skill.get("required_steps", []))
        for c in contract_constraints:
            ctype = str(c.get("constraint_type"))
            if ctype in {"identity", "style", "layout", "material", "count", "spatial", "action", "attribute", "text", "negative"}:
                if ctype not in required_steps:
                    missing_steps.append(ctype)
        family = self._selected_prompt_family(task, skill, contract)
        compiled_prompt = self._render_prompt(task, skill, contract, bindings, memory_context or {})
        workflow = {
            "component": self.describe(),
            "skill": skill,
            "skill_name": skill_name,
            "prompt_template": skill.get("template"),
            "selected_prompt_family": family,
            "required_steps": list(required_steps),
            "missing_steps": sorted(set(missing_steps)),
            "contract_coverage": {str(c.get("constraint_id")): str(c.get("constraint_type")) in required_steps for c in contract_constraints},
            "reference_bindings": bindings,
            "compiled_prompt": compiled_prompt,
            "memory_retrieval": self._memory_retrieval_trace(memory_context or {}, skill, family),
        }
        if skill.get("name") == "compositional_generation":
            clip_segments, clip_segment_types = self._render_compact_compositional_clip_segments(
                task,
                skill,
                contract,
                bindings,
                memory_context or {},
            )
            workflow["clip_prompt_segments"] = clip_segments
            workflow["clip_prompt_segment_types"] = clip_segment_types
            workflow["clip_prompt"] = " ".join(clip_segments)
            workflow["t5_prompt"] = compiled_prompt
            # Keep semantic ownership in the long channel as well as CLIP.
            # Otherwise a large object inventory can consume T5's context and
            # silently remove later spatial/action clauses.
            t5_parts = self._render_compositional_t5_parts(task, skill, contract, bindings, memory_context or {})
            workflow["t5_prompt_segments"] = [value for _, value in t5_parts]
            workflow["t5_prompt_segment_types"] = [family for family, _ in t5_parts]
        else:
            # Every backend channel is compiled here.  Backends are consumers;
            # they must never invent a prompt or fall back to task text.
            workflow["clip_prompt"] = compiled_prompt
            workflow["t5_prompt"] = compiled_prompt
        workflow["memory_used"] = bool(workflow["memory_retrieval"]["applicable_records"])
        prompt_provenance(workflow)
        return workflow


    def _memory_retrieval_trace(self, memory_context: Dict[str, Any], skill: Dict[str, Any], family: str) -> Dict[str, Any]:
        """Report records that can actually affect this compiled workflow.

        Merely loading built-in or unrelated memory rows is not evidence that
        Memory changed generation.  This trace deliberately counts only
        retrieved failure/patch/regression rows; static default guidelines are
        already represented by the skill artifact and are not double-counted.
        """

        counts = {
            key: len(memory_context.get(key, []))
            for key in ("retrieved_failures", "validated_patches", "regression_cases")
            if isinstance(memory_context.get(key, []), list)
        }
        applied_guidelines = self._compact_memory_guidelines(memory_context)
        applied_patch_rules = self._compact_validated_rules(
            memory_context,
            skill,
            family,
            rule_fields=("prompt_rules", "clip_rules"),
        )
        return {
            "record_counts": counts,
            "applicable_records": len(applied_guidelines) + len(applied_patch_rules),
            "applied_guidelines": applied_guidelines,
            "applied_patch_rules": applied_patch_rules,
            "prompt_guidelines_retrieved": len(memory_context.get("prompt_guidelines", []))
            if isinstance(memory_context.get("prompt_guidelines", []), list)
            else 0,
            "static_guidelines_count_as_memory_use": False,
        }

    def _render_prompt(self, task: GenerationTask, skill: Dict[str, Any], contract: Dict[str, Any], bindings: Dict[str, str], memory_context: Dict[str, Any]) -> str:
        if skill.get("name") == "prompt_enhancement":
            return self._render_enhanced_prompt(task, skill, contract, memory_context)
        if skill.get("name") == "compositional_generation":
            return self._render_compositional_prompt(task, skill, contract, bindings, memory_context)
        lines = [
            "Generate one image from the following initial prompt.",
            "",
            "Initial prompt:",
            task.prompt,
            "",
        ]
        self._append_reference_bindings(lines, task, bindings)
        lines.append("Required visual constraints:")
        for c in contract.get("constraints", []):
            lines.append(f"- [{c['constraint_type']}] {c['text']}")
        return "\n".join(lines)

    def _render_compositional_prompt(self, task: GenerationTask, skill: Dict[str, Any], contract: Dict[str, Any], bindings: Dict[str, str], memory_context: Dict[str, Any]) -> str:
        return " ".join(value for _, value in self._render_compositional_t5_parts(task, skill, contract, bindings, memory_context))

    def _render_compositional_t5_parts(
        self,
        task: GenerationTask,
        skill: Dict[str, Any],
        contract: Dict[str, Any],
        bindings: Dict[str, str],
        memory_context: Dict[str, Any],
    ) -> List[tuple[str, str]]:
        constraints = contract.get("constraints", [])
        object_constraints = self._object_prompt_constraints(constraints)
        family = self._selected_prompt_family(task, skill, contract)
        typed_parts: List[tuple[str, str]] = [("source", self._sentence(task.prompt))]
        if object_constraints:
            include_rows: List[Dict[str, Any]] = []
            exclude_rows: List[Any] = []
            for c in object_constraints:
                meta = c.get("metadata", {})
                include = meta.get("required_objects", meta.get("include", [])) if isinstance(meta, dict) else []
                exclude = meta.get("forbidden_objects", meta.get("exclude", [])) if isinstance(meta, dict) else []
                for obj in include:
                    if isinstance(obj, dict) and obj.get("class"):
                        include_rows.append(obj)
                for obj in exclude:
                    exclude_rows.append(obj)
            typed_parts.extend(
                ("object", clause) for clause in self._compact_object_requirements(
                    include_rows,
                    exclude_rows,
                )
            )
        typed_parts.extend(
            ("attribute", clause)
            for clause in self._compact_attribute_requirements(
                task,
                contract,
            )
        )
        typed_parts.extend(("spatial", clause) for clause in self._compact_spatial_requirements(task, skill, contract, family=family))
        typed_parts.extend(("action", clause) for clause in self._compact_action_requirements(task, skill, contract))
        typed_parts.extend(("negative", clause) for clause in self._compact_text_and_negative_constraints(constraints))
        typed_parts.extend(("reference", clause) for clause in self._compact_reference_bindings(task, bindings))
        typed_parts.extend(("repair", clause) for clause in self._compact_skill_rules(skill, family))
        typed_parts.extend(("repair", clause) for clause in self._compact_memory_guidelines(memory_context))
        typed_parts.extend(("repair", clause) for clause in self._compact_validated_rules(memory_context, skill, family))
        typed_parts.append(("presentation", self._presentation_clause(task, contract, family=family)))
        non_object_constraints = [c for c in constraints if c not in object_constraints]
        for c in non_object_constraints:
            ctype = str(c.get("constraint_type"))
            if ctype not in {"attribute", "spatial", "action", "text", "negative"} and c.get("text"):
                typed_parts.append((ctype if ctype else "unspecified", self._sentence(str(c["text"]))))
        output: List[tuple[str, str]] = []
        seen = set()
        for segment_type, value in typed_parts:
            clause = self._sentence(value)
            key = clause.lower()
            if clause and key not in seen:
                seen.add(key)
                output.append((segment_type, clause))
        return output

    def _render_compact_compositional_clip_segments(self, task: GenerationTask, skill: Dict[str, Any], contract: Dict[str, Any], bindings: Dict[str, str], memory_context: Dict[str, Any]) -> tuple[List[str], List[str]]:
        constraints = contract.get("constraints", [])
        object_constraints = self._object_prompt_constraints(constraints)
        family = self._selected_prompt_family(task, skill, contract)
        if self._uses_attribute_scene_clip(task, skill, contract, family):
            return self._render_attribute_scene_clip_segments(task, skill, contract, family)
        typed_parts: List[tuple[str, str]] = [("source", self._sentence(task.prompt))]
        if object_constraints:
            include_rows: List[Dict[str, Any]] = []
            exclude_rows: List[Any] = []
            for c in object_constraints:
                meta = c.get("metadata", {})
                include = meta.get("required_objects", meta.get("include", [])) if isinstance(meta, dict) else []
                exclude = meta.get("forbidden_objects", meta.get("exclude", [])) if isinstance(meta, dict) else []
                for obj in include:
                    if isinstance(obj, dict) and obj.get("class"):
                        include_rows.append(obj)
                for obj in exclude:
                    exclude_rows.append(obj)
            typed_parts.extend(
                ("object", clause) for clause in self._compact_object_requirements(
                    include_rows,
                    exclude_rows,
                )
            )
        typed_parts.extend(("spatial", clause) for clause in self._compact_spatial_requirements(task, skill, contract, family=family, compact=True))
        typed_parts.extend(("repair", clause) for clause in self._compact_clip_repair_rules(memory_context, skill, family))
        typed_parts.extend(
            ("attribute", clause)
            for clause in self._compact_attribute_requirements(
                task,
                contract,
            )
        )
        typed_parts.extend(("action", clause) for clause in self._compact_action_requirements(task, skill, contract))
        typed_parts.extend(("negative", clause) for clause in self._compact_text_and_negative_constraints(constraints))
        typed_parts.extend(("reference", clause) for clause in self._compact_reference_bindings(task, bindings))
        typed_parts.append(("presentation", self._short_presentation_clause(task, contract, family=family)))
        # Preserve complete semantic clauses. The generation adapter uses both
        # model tokenizers to select the longest fitting prefix and records any
        # omitted low-priority clauses in the trace.
        segments: List[str] = []
        segment_types: List[str] = []
        seen = set()
        for segment_type, value in typed_parts:
            clause = self._sentence(value)
            key = clause.lower()
            if clause and key not in seen:
                seen.add(key)
                segments.append(clause)
                segment_types.append(segment_type)
        if any(
            segment_type == "spatial"
            and "one of each only" in segment.lower()
            and ("full group" in segment.lower() or " all " in f" {segment.lower()} ")
            for segment, segment_type in zip(segments, segment_types)
        ):
            raise ValueError(f"plural count conflicts with a singular spatial anchor for task {task.task_id}")
        return segments, segment_types

    def _uses_attribute_scene_clip(self, task: GenerationTask, skill: Dict[str, Any], contract: Dict[str, Any], family: str) -> bool:
        if not self._has_contextual_scene_presence(contract):
            return False
        family_config = self._prompt_family_config(skill, family)
        scene_families = family_config.get("scene_preserving_constraint_families", []) if isinstance(family_config, dict) else []
        route_family = self._constraint_route_family(contract)
        if route_family not in {str(value).strip().lower() for value in scene_families if str(value).strip()}:
            return False
        constraint_types = {str(row.get("constraint_type")) for row in contract.get("constraints", []) if isinstance(row, dict)}
        return family == "attribute_binding" and "attribute" in constraint_types

    def _render_attribute_scene_clip_segments(self, task: GenerationTask, skill: Dict[str, Any], contract: Dict[str, Any], family: str) -> tuple[List[str], List[str]]:
        object_rows: List[Dict[str, Any]] = []
        for constraint in self._object_prompt_constraints(contract.get("constraints", [])):
            meta = constraint.get("metadata", {}) if isinstance(constraint, dict) else {}
            rows = meta.get("required_objects", []) if isinstance(meta, dict) else []
            for row in rows:
                if isinstance(row, dict) and row.get("class"):
                    object_rows.append(row)
        bindings = self._attribute_bindings(task, contract)
        parts: List[tuple[str, str]] = []
        if self._uses_attribute_scene_prompt(skill, family, contract):
            parts.append(("source", self._sentence(task.prompt)))
        if bindings:
            rendered = self._render_attribute_bindings(bindings)
            parts.append(("attribute", f"Exact object colors and materials: {rendered}."))
        if object_rows:
            objects = []
            for row in object_rows:
                count = int(row.get("count", 1) or 1)
                name = str(row.get("class") or "").strip()
                if not name:
                    continue
                objects.append(f"{count} {name}" if count > 1 else name)
            if objects:
                parts.append(("object", "Visible objects: " + ", ".join(objects) + "."))
        family_config = self._prompt_family_config(skill, family)
        presentation_key = "scene_preserving_presentation" if self._uses_attribute_scene_prompt(skill, family, contract) else "isolated_presentation"
        presentation = str(family_config.get(presentation_key, "Large separated objects on a plain white background.") if isinstance(family_config, dict) else "Large separated objects on a plain white background.")
        parts.append(("presentation", presentation))
        segments = [self._sentence(text) for _, text in parts if self._sentence(text)]
        types = [segment_type for segment_type, text in parts if self._sentence(text)]
        return segments, types

    def _uses_attribute_scene_prompt(self, skill: Dict[str, Any], family: str, contract: Dict[str, Any]) -> bool:
        if not self._has_contextual_scene_presence(contract):
            return False
        family_config = self._prompt_family_config(skill, family)
        scene_families = family_config.get("scene_preserving_constraint_families", []) if isinstance(family_config, dict) else []
        route_family = self._constraint_route_family(contract)
        return route_family in {str(value).strip().lower() for value in scene_families if str(value).strip()}

    @staticmethod
    def _has_contextual_scene_presence(contract: Dict[str, Any]) -> bool:
        constraints = contract.get("constraints", []) if isinstance(contract, dict) else []
        if not isinstance(constraints, list):
            return False
        has_attribute = False
        has_prompt_alignment = False
        required_classes: set[str] = set()
        for row in constraints:
            if not isinstance(row, dict):
                continue
            ctype = str(row.get("constraint_type", "")).strip().lower()
            has_attribute = has_attribute or ctype == "attribute"
            has_prompt_alignment = has_prompt_alignment or ctype == "prompt_alignment"
            metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
            if isinstance(metadata, dict) and metadata.get("contextual_scene_presence") is True:
                return True
            if isinstance(metadata, dict):
                required = metadata.get("required_objects", [])
                if isinstance(required, list):
                    for obj in required:
                        if isinstance(obj, dict) and obj.get("class"):
                            required_classes.add(str(obj["class"]).strip().lower())
        if has_attribute and has_prompt_alignment and len(required_classes) >= 5:
            return True
        return False

    def _constraint_route_family(self, contract: Dict[str, Any]) -> str:
        types = {str(row.get("constraint_type", "")) for row in contract.get("constraints", []) if isinstance(row, dict)}
        semantic = types - {"count", "prompt_alignment"}
        if len(semantic) > 1:
            return "complex"
        if semantic:
            return next(iter(semantic))
        return "count" if "count" in types else "default"

    @staticmethod
    def _prompt_family_config(skill: Dict[str, Any], family: str | None) -> Dict[str, Any]:
        families = skill.get("prompt_families", {}) if isinstance(skill, dict) else {}
        row = families.get(family) if isinstance(families, dict) else None
        return row if isinstance(row, dict) else {}

    def _compact_clip_repair_rules(self, memory_context: Dict[str, Any], skill: Dict[str, Any], family: str | None = None, *, max_rules: int = 3) -> List[str]:
        """Keep a tiny set of evolved rules in the short CLIP channel.

        The split-prompt backend primarily consumes the short channel, so any
        accepted skill refinement must be visible there as well as in the
        detailed T5 prompt.
        """
        rules: List[str] = []
        source_groups = [self._compact_skill_rules(skill, family, prefer_recent=(family == "object_relations"))]
        family_clip_rules = self._compact_family_clip_rules(skill, family)
        if family_clip_rules:
            source_groups.append(family_clip_rules)
        source_groups.append(self._compact_validated_rules(memory_context, skill, family, rule_fields=("clip_rules", "prompt_rules")))
        for source_rules in source_groups:
            for rule in source_rules:
                text = str(rule).strip()
                if not text or text in rules:
                    continue
                rules.append(text)
                if len(rules) >= max_rules:
                    return rules
        return rules

    def _compact_family_clip_rules(self, skill: Dict[str, Any], family: str | None = None) -> List[str]:
        rules: List[str] = []
        for rule in skill.get("clip_rules", []):
            text = str(rule).strip()
            if text and text not in rules:
                rules.append(text)
        families = skill.get("prompt_families", {})
        if isinstance(families, dict):
            for key in ("shared", family):
                family_config = families.get(key, {})
                if not isinstance(family_config, dict):
                    continue
                for rule in family_config.get("clip_rules", []):
                    text = str(rule).strip()
                    if text and text not in rules:
                        rules.append(text)
        return rules

    def _compact_object_requirements(self, include_rows: List[Dict[str, Any]], exclude_rows: List[Any]) -> List[str]:
        """Render the declared inventory without inventing object-specific layouts."""
        required: Dict[str, tuple[str, int]] = {}
        for obj in include_rows:
            name = str(obj.get("class") or "").strip()
            if not name:
                continue
            count = obj.get("count", 1)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("required object counts must be non-negative integers")
            key = name.casefold()
            if key not in required or count > required[key][1]:
                required[key] = (name, count)
        lines = [f"Exactly {count} visible instances of {name}." for name, count in required.values()]
        for obj in exclude_rows:
            name = str(obj.get("class", "") if isinstance(obj, dict) else obj).strip()
            if name and name.casefold() not in required:
                lines.append(f"No {name}.")
        return lines


    def _compact_attribute_requirements(self, task: GenerationTask, contract: Dict[str, Any]) -> List[str]:
        """Preserve arbitrary attributes without a color/material vocabulary."""
        return self._dedupe_sentences([
            f"Attribute binding: {binding['object']} has the requested attribute {binding['attribute']}"
            for binding in self._attribute_bindings(task, contract)
        ])

    def _presentation_clause(self, task: GenerationTask, contract: Dict[str, Any], *, family: str | None = None) -> str:
        if self._is_prompt_alignment_only(contract):
            return "Natural coherent scene; preserve the original prompt semantics; no text, logos, watermarks, or distortions."
        if self._allows_spatial_occlusion(contract):
            return "Clean image; required objects recognizable; preserve the requested occlusion/depth relation; no extra objects, text, logos, watermarks, reflections, or distortions."
        return "Clean image; required objects fully visible; no extra objects, text, logos, watermarks, reflections, occlusion, or distortions."

    def _short_presentation_clause(self, task: GenerationTask, contract: Dict[str, Any], *, family: str | None = None) -> str:
        if self._is_prompt_alignment_only(contract):
            return "Natural coherent scene matching the prompt."
        if self._allows_spatial_occlusion(contract):
            return "Plain image; requested occlusion/depth relation preserved."
        if family == "object_relations":
            return "Keep the image plain."
        return "Large separated objects on a plain white background."

    def _allows_spatial_occlusion(self, contract: Dict[str, Any]) -> bool:
        for rel in self._spatial_relations(None, contract):
            if normalize_relation(rel.get("relation")) == "behind":
                return True
        return False

    def _is_prompt_alignment_only(self, contract: Dict[str, Any]) -> bool:
        constraints = contract.get("constraints", []) if isinstance(contract, dict) else []
        types = [str(row.get("constraint_type")) for row in constraints if isinstance(row, dict)]
        return types == ["prompt_alignment"]

    def _render_attribute_bindings(self, bindings: List[Dict[str, str]]) -> str:
        return ", ".join(
            f"{binding['attribute']} {binding['object']}"
            for binding in bindings
            if binding.get("object") and binding.get("attribute")
        )


    def _compact_spatial_requirements(
        self,
        task: GenerationTask,
        skill: Dict[str, Any],
        contract: Dict[str, Any],
        family: str | None = None,
        *,
        compact: bool = False,
    ) -> List[str]:
        relations = []
        for rel in self._spatial_relations(task, contract):
            subject = str(rel.get("subject") or "").strip()
            target = str(rel.get("object") or "").strip()
            relation = normalize_relation(rel.get("relation"))
            if subject and target and relation:
                relations.append((subject, relation, target))
        counts: Dict[str, int] = {}
        for constraint in contract.get("constraints", []):
            metadata = constraint.get("metadata", {}) if isinstance(constraint, dict) else {}
            for row in metadata.get("required_objects", []) if isinstance(metadata, dict) else []:
                if isinstance(row, dict) and row.get("class"):
                    key = str(row["class"]).strip().lower()
                    counts[key] = max(counts.get(key, 0), int(row.get("count", 1) or 1))
        clauses = [
            canonical_spatial_clause(
                subject,
                relation,
                target,
                compact=compact,
                subject_count=counts.get(subject.lower(), 1),
                target_count=counts.get(target.lower(), 1),
            )
            for subject, relation, target in relations
        ]
        repair_clauses: List[str] = []
        families = skill.get("prompt_families", {}) if isinstance(skill, dict) else {}
        # Spatial constraints always consume the object-relations repair
        # artifact. A compositional task also has an object/count constraint,
        # so gating this artifact on one exclusive prompt family made promoted
        # spatial patches inert for realistic multi-constraint prompts.
        family_config = families.get("object_relations", {}) if isinstance(families, dict) else {}
        repairs = family_config.get("relation_repairs", {}) if isinstance(family_config, dict) else {}
        if not isinstance(repairs, dict):
            return clauses
        for subject, relation, target in relations:
            if counts.get(subject.lower(), 1) > 1 or counts.get(target.lower(), 1) > 1:
                continue
            axis = (
                "horizontal" if relation in {"left of", "right of"}
                else "vertical" if relation in {"above", "below"}
                else "depth" if relation in {"in front of", "behind"}
                else ""
            )
            relation_key = relation.replace(" ", "_")
            template = str(repairs.get(relation_key) or repairs.get(axis) or "").strip()
            if not template:
                continue
            repair_clauses.append(self._sentence(template.format(subject=subject, target=target)))
        # The compact anchor now leaves enough CLIP context for the evolved
        # repair, so preserve the natural contract-then-repair order.
        return self._dedupe([*clauses, *repair_clauses])


    def _compact_action_requirements(self, task: GenerationTask, skill: Dict[str, Any], contract: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        families = skill.get("prompt_families", {}) if isinstance(skill, dict) else {}
        family = families.get("action_interaction", {}) if isinstance(families, dict) else {}
        repairs = family.get("action_repairs", {}) if isinstance(family, dict) else {}
        templates = family.get("action_templates", {}) if isinstance(family, dict) else {}
        for rel in self._action_relations(task, contract):
            subject = str(rel.get("subject") or "").strip()
            target = str(rel.get("object") or "").strip()
            action = str(rel.get("action") or "").replace("_", " ").strip()
            if not subject or not target or not action:
                continue
            template = templates.get(action) if isinstance(templates, dict) else None
            if not isinstance(template, str) or not template.strip():
                template = "Action: {subject} clearly {action} {target}; visible interaction."
            lines.append(self._sentence(template.format(subject=subject, target=target, action=action)))
            if isinstance(repairs, dict):
                template = str(repairs.get("template") or "").strip()
                if template:
                    lines.append(self._sentence(template.format(subject=subject, target=target, action=action)))
        return lines

    def _compact_text_and_negative_constraints(self, constraints: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        for c in constraints:
            ctype = str(c.get("constraint_type"))
            text = str(c.get("text") or "").strip()
            if not text:
                continue
            if ctype == "text":
                lines.append(f"Render exact legible text: {text}.")
            elif ctype == "negative":
                lines.append(f"No: {text}.")
            elif ctype == "spatial":
                meta = c.get("metadata", {}) if isinstance(c, dict) else {}
                rows = meta.get("spatial_relations", []) if isinstance(meta, dict) else []
                if not rows:
                    lines.append(self._sentence(text))
        return lines

    def _compact_reference_bindings(self, task: GenerationTask, bindings: Dict[str, str]) -> List[str]:
        lines: List[str] = []
        for role, ref in sorted(bindings.items()):
            ref_meta = task.references.get(ref, {})
            ref_uri = ref_meta.get("uri", ref)
            lines.append(f"Use reference {ref} for {role}: {ref_uri}.")
        return lines

    def _compact_validated_rules(
        self,
        memory_context: Dict[str, Any],
        skill: Dict[str, Any],
        family: str,
        *,
        rule_fields: tuple[str, ...] = ("prompt_rules",),
    ) -> List[str]:
        validated_patches = memory_context.get("validated_patches", []) if isinstance(memory_context, dict) else []
        rules = self._validated_prompt_rules(validated_patches, skill.get("name"), family, rule_fields=rule_fields)
        if not rules:
            rules = self._validated_prompt_rules(validated_patches, skill.get("name"), None, rule_fields=rule_fields)
        return rules

    def _compact_skill_rules(self, skill: Dict[str, Any], family: str | None = None, *, prefer_recent: bool = False) -> List[str]:
        rules = list(skill.get("prompt_rules", []))
        families = skill.get("prompt_families", {})
        for key in ("shared", family):
            config = families.get(key, {}) if isinstance(families, dict) else {}
            if isinstance(config, dict):
                rules.extend(config.get("prompt_rules", []))
        result = self._dedupe([str(rule).strip() for rule in rules])
        return list(reversed(result)) if prefer_recent else result

    def _compact_memory_guidelines(self, memory_context: Dict[str, Any]) -> List[str]:
        return self._dedupe([
            str(row["guideline"]).strip()
            for row in memory_context.get("prompt_guidelines", [])
            if isinstance(row, dict) and row.get("guideline")
        ])


    def _sentence(self, text: str) -> str:
        cleaned = " ".join(str(text).strip().split())
        if not cleaned:
            return ""
        return cleaned if cleaned.endswith((".", "!", "?")) else cleaned + "."

    def _dedupe_sentences(self, values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            text = self._sentence(value)
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
        return out

    def _validated_prompt_rules(
        self,
        patches: List[Dict[str, Any]],
        skill_name: str | None,
        prompt_family: str | None = None,
        *,
        rule_fields: tuple[str, ...] = ("prompt_rules",),
    ) -> List[str]:
        rules: List[str] = []
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            if patch.get("target_component") != "skills" or patch.get("changed_artifact") != "skills.json":
                continue
            payload = patch.get("payload", {})
            path = payload.get("path", []) if isinstance(payload, dict) else []
            if len(path) >= 3 and path[0] == "skills":
                if skill_name and path[1] not in {skill_name, "*"}:
                    continue
                if len(path) >= 5 and path[2] == "prompt_families":
                    if prompt_family and path[3] not in {prompt_family, "*"}:
                        continue
                    if path[4] not in rule_fields:
                        continue
                elif path[2] not in rule_fields:
                    continue
                value = payload.get("value")
                if isinstance(value, str) and value not in rules:
                    rules.append(value)
                elif isinstance(value, list):
                    for item in value:
                        text = str(item).strip()
                        if text and text not in rules:
                            rules.append(text)
        return rules

    def _selected_prompt_family(self, task: GenerationTask, skill: Dict[str, Any], contract: Dict[str, Any]) -> str:
        families = skill.get("prompt_families", {})
        if not isinstance(families, dict) or not families:
            return "default"
        constraint_types = [str(c.get("constraint_type")) for c in contract.get("constraints", [])]
        declared_families = {
            str((c.get("metadata") or {}).get("prompt_family", "")).strip()
            for c in contract.get("constraints", [])
            if isinstance(c, dict) and isinstance(c.get("metadata"), dict)
        }
        declared_families.discard("")
        if len(declared_families) == 1:
            declared = next(iter(declared_families))
            if declared in families:
                return declared
        if constraint_types == ["prompt_alignment"]:
            return "scene_composition" if "scene_composition" in families else "default"
        if "text" in constraint_types:
            return "text_rendering" if "text_rendering" in families else "default"
        if "identity" in constraint_types or "style" in constraint_types:
            return "reference_binding" if "reference_binding" in families else "default"
        if "action" in constraint_types and "action_interaction" in families:
            return "action_interaction"
        if "spatial" in constraint_types and self._spatial_relations(task, contract):
            return "object_relations" if "object_relations" in families else "default"
        if "attribute" in constraint_types:
            route_family = self._constraint_route_family(contract)
            for family_name, family_config in families.items():
                if not isinstance(family_config, dict):
                    continue
                scene_families = family_config.get("scene_preserving_constraint_families", [])
                if route_family in {str(value).strip().lower() for value in scene_families if str(value).strip()}:
                    return str(family_name)
        if "count" in constraint_types and self._has_repeated_required_object(contract):
            defaults = skill.get("family_selection_defaults", {}) if isinstance(skill, dict) else {}
            repeated_count_family = str(defaults.get("repeated_count", "")).strip() if isinstance(defaults, dict) else ""
            if repeated_count_family in families:
                return repeated_count_family
            if "counting_general" in families:
                return "counting_general"
        if "attribute" in constraint_types:
            return "attribute_binding" if "attribute_binding" in families else "default"
        if "clutter_and_occlusion_control" in families:
            return "clutter_and_occlusion_control"
        if "scene_composition" in families:
            return "scene_composition"
        return "default"

    def _has_repeated_required_object(self, contract: Dict[str, Any]) -> bool:
        for constraint in contract.get("constraints", []):
            if not isinstance(constraint, dict):
                continue
            metadata = constraint.get("metadata", {})
            required = metadata.get("required_objects", []) if isinstance(metadata, dict) else []
            for row in required if isinstance(required, list) else []:
                if isinstance(row, dict) and int(row.get("count", 1) or 1) > 1:
                    return True
        return False


    def _spatial_relations(self, task: GenerationTask, contract: Dict[str, Any]) -> List[Dict[str, str]]:
        relations: List[Dict[str, str]] = []
        for constraint in contract.get("constraints", []):
            cmeta = constraint.get("metadata", {}) if isinstance(constraint, dict) else {}
            rows = cmeta.get("spatial_relations", []) if isinstance(cmeta, dict) else []
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("subject") and row.get("object"):
                        relations.append({
                            "subject": str(row["subject"]),
                            "relation": str(row.get("relation", "")).replace("_", " "),
                            "object": str(row["object"]),
                        })
        return relations

    def _action_relations(self, task: GenerationTask, contract: Dict[str, Any]) -> List[Dict[str, str]]:
        relations: List[Dict[str, str]] = []
        for constraint in contract.get("constraints", []):
            cmeta = constraint.get("metadata", {}) if isinstance(constraint, dict) else {}
            rows = cmeta.get("action_relations", []) if isinstance(cmeta, dict) else []
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("subject") and row.get("object") and row.get("action"):
                        relations.append({
                            "subject": str(row["subject"]),
                            "action": str(row.get("action", "")).replace("_", " "),
                            "object": str(row["object"]),
                        })
        return relations

    def _attribute_bindings(self, task: GenerationTask, contract: Dict[str, Any]) -> List[Dict[str, str]]:
        bindings: List[Dict[str, str]] = []
        for constraint in contract.get("constraints", []):
            cmeta = constraint.get("metadata", {}) if isinstance(constraint, dict) else {}
            rows = cmeta.get("attribute_bindings", []) if isinstance(cmeta, dict) else []
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("object") and row.get("attribute"):
                        bindings.append({"object": str(row["object"]), "attribute": str(row["attribute"])})
        return bindings

    def _dedupe(self, values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            if value and value not in seen:
                seen.add(value)
                out.append(value)
        return out


    def _has_object_requirements(self, constraint: Dict[str, Any]) -> bool:
        meta = constraint.get("metadata", {})
        if not isinstance(meta, dict):
            return False
        return bool(meta.get("required_objects") or meta.get("forbidden_objects") or meta.get("include") or meta.get("exclude"))

    def _object_prompt_constraints(self, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        primary = []
        secondary = []
        for constraint in constraints:
            if not isinstance(constraint, dict) or not self._has_object_requirements(constraint):
                continue
            ctype = str(constraint.get("constraint_type") or "").strip().lower()
            meta = constraint.get("metadata", {}) if isinstance(constraint.get("metadata"), dict) else {}
            schema = str(meta.get("constraint_schema") or "").strip().lower()
            if ctype in {"count", "negative"} or schema == "object_requirements.v1":
                primary.append(constraint)
            else:
                secondary.append(constraint)
        return primary or secondary

    def _append_reference_bindings(self, lines: List[str], task: GenerationTask, bindings: Dict[str, str]) -> None:
        if not bindings:
            return
        lines.append("")
        lines.append("Use these reference roles:")
        for role, ref in sorted(bindings.items()):
            ref_meta = task.references.get(ref, {})
            ref_uri = ref_meta.get("uri", ref)
            lines.append(f"- {role}: {ref} ({ref_uri})")


    def _render_enhanced_prompt(self, task: GenerationTask, skill: Dict[str, Any], contract: Dict[str, Any], memory_context: Dict[str, Any]) -> str:
        governance = contract.get("prompt_governance", {})
        profile = []
        profile.extend(governance.get("default_quality_profile", []))
        profile.extend(governance.get("task_family_profiles", {}).get(task.task_family, []))
        constraints = contract.get("constraints", [])
        negative_constraints = [c for c in constraints if c.get("constraint_type") == "negative"]
        lines = [
            "You are generating the final image, not explaining the prompt.",
            "Use the user's initial prompt as the creative intent, but convert it into a concrete, high-quality image-generation brief.",
            "",
            "Initial user prompt:",
            task.prompt,
            "",
            "Enhanced generation brief:",
            "Create a coherent, polished image that preserves the user's requested scene while making the subject, setting, composition, lighting, style, color palette, and quality bar explicit.",
            "Use professional visual language: clear focal subject, cinematic composition, intentional lighting, harmonious colors, sharp details, realistic spatial layout, and clean final rendering.",
        ]
        if profile:
            lines.append("")
            lines.append("Harness quality controls:")
            for rule in profile:
                lines.append(f"- {rule}")
        prompt_guidelines = memory_context.get("prompt_guidelines", []) if isinstance(memory_context, dict) else []
        if prompt_guidelines:
            lines.append("")
            lines.append("Harness memory guidance:")
            for guideline in prompt_guidelines:
                text = guideline.get("guideline") if isinstance(guideline, dict) else None
                if text:
                    lines.append(f"- {text}")
        retrieved_failures = memory_context.get("retrieved_failures", []) if isinstance(memory_context, dict) else []
        if retrieved_failures:
            lines.append("")
            lines.append("Known failure patterns to avoid:")
            for failure in retrieved_failures:
                symptom = failure.get("failure_symptom") or failure.get("verification_result", {}).get("symptom") if isinstance(failure, dict) else None
                ctype = failure.get("constraint_type") if isinstance(failure, dict) else None
                if symptom:
                    lines.append(f"- {ctype or 'visual'}: {symptom}")
        validated_patches = memory_context.get("validated_patches", []) if isinstance(memory_context, dict) else []
        reusable_rules = self._validated_prompt_rules(validated_patches, skill.get("name"))
        if reusable_rules:
            lines.append("")
            lines.append("Validated harness repair guidance:")
            for rule in reusable_rules:
                lines.append(f"- {rule}")
        if constraints:
            lines.append("")
            lines.append("Evaluation-critical requirements:")
            for c in constraints:
                if c.get("constraint_type") != "negative":
                    lines.append(f"- [{c['constraint_type']}] {c['text']}")
        if negative_constraints:
            lines.append("")
            lines.append("Avoid:")
            for c in negative_constraints:
                lines.append(f"- {c['text']}")
        return "\n".join(lines)

    def responsible_decisions(self) -> List[str]:
        return ["skill selection target", "workflow steps", "prompt-reference program compilation", "repair procedure"]
