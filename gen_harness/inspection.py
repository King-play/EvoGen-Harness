from __future__ import annotations

import base64
import colorsys
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .io import expand_env_vars, read_json
from .tool_payloads import constraints_for_external_tool, public_constraint_ids, public_inspection_payload, remap_public_constraint_results
from .schema import GenerationTask, VisualConstraint
from .spatial_prompt import normalize_relation


def format_inspection_command(
    command: Sequence[Any],
    *,
    input_json: str | None = None,
    output_json: str | None = None,
) -> List[str]:
    """Resolve a configured inspection command without hard-coding local paths.

    Config files should use ``"{python}"`` for the Python interpreter.  Runtime
    deployments can override it with ``GENHARNESS_PYTHON``; otherwise the current
    interpreter is used.  ``${VAR}`` environment placeholders are expanded by the
    existing strict env-var expander.
    """

    python_bin = os.environ.get("GENHARNESS_PYTHON") or sys.executable or "python"
    values = {
        "python": python_bin,
        "input_json": input_json or "",
        "output_json": output_json or "",
    }
    formatted: List[str] = []
    for part in command:
        text = str(part)
        for key, value in values.items():
            text = text.replace("{" + key + "}", value)
        formatted.append(str(expand_env_vars(text)))
    return formatted





def canonical_object_class(name: Any) -> str:
    return " ".join(str(name).strip().lower().split())


def canonical_detector_object_class(name: Any) -> str:
    return canonical_object_class(name)


def constraints_for_internal_inspection(task: GenerationTask, program: Dict[str, Any]) -> List[VisualConstraint]:
    contract = program.get("contract", {}) if isinstance(program, dict) else {}
    rows = contract.get("constraints", []) if isinstance(contract, dict) else []
    if rows:
        return [VisualConstraint.from_dict(row) for row in rows]
    return list(task.constraints)


class InspectionRunner:
    """Interface for real visual inspection tools."""

    def inspect(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int = 0) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError


class PersistentJSONCommandClient:
    """Small newline-delimited JSON client for model-backed inspection workers."""

    def __init__(self, command: Sequence[str]):
        self.command = list(command)
        self.proc: subprocess.Popen[str] | None = None

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        proc = self._ensure_proc()
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        startup_noise: List[str] = []
        while True:
            line = proc.stdout.readline()
            if not line:
                self.close()
                detail = f"; non-JSON output: {' | '.join(startup_noise[-3:])}" if startup_noise else ""
                raise RuntimeError(f"persistent inspection worker exited unexpectedly{detail}")
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                # Some ML libraries write one-time loading reports to stdout.
                # Treat them as startup noise instead of corrupting the first
                # request/response exchange. Bound retained diagnostics only;
                # reading continues until the worker emits its JSON response.
                startup_noise.append(line.strip()[:300])
                if len(startup_noise) > 20:
                    startup_noise.pop(0)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(str(result["error"]))
        return result

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc
        self.proc = subprocess.Popen(
            self.command,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=1,
        )
        return self.proc

    def __del__(self) -> None:
        self.close()


class ExternalCommandInspection(InspectionRunner):
    """Inspection adapter for VLM judges, detectors, OCR, identity metrics, or benchmark scripts.

    The command receives JSON containing task, program, generation result, and
    must output:

    {"constraint_results": {"constraint_id": {"passed": true, "symptom": "pass", "confidence": 0.9}}}
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._persistent_client: PersistentJSONCommandClient | None = None

    def inspect(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int = 0) -> Dict[str, Dict[str, Any]]:
        constraints = constraints_for_external_tool(task, program)
        public_ids = public_constraint_ids(constraints)
        public_input = public_inspection_payload(task, program, generation, constraints=constraints, public_ids=public_ids)
        image_path = generation.get("image_path") or generation.get("image_uri")
        payload = {
            "input": public_input,
            "constraints": public_input.get("constraints", []),
            "seed": seed,
            # Model-backed workers use the persistent inspection envelope and
            # read the image from the top-level generation field. Keep this
            # deliberately narrow: only the materialized image handle crosses
            # the boundary; task/program internals remain sanitized in input.
            "generation": {
                "image_uri": generation.get("image_uri"),
                "image_path": image_path,
            },
        }
        if "persistent_command" in self.config:
            result = self._persistent_request(payload)
            return remap_public_constraint_results(result["constraint_results"], constraints, public_ids)
        with tempfile.TemporaryDirectory(prefix="gen_harness_inspect_") as td:
            td_path = Path(td)
            input_path = td_path / "input.json"
            output_path = td_path / "output.json"
            payload = {**payload, "output_json": str(output_path)}
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            command = format_inspection_command(self.config["command"], input_json=str(input_path), output_json=str(output_path))
            proc = subprocess.run(command, text=True, capture_output=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"inspection command failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
            result = json.loads(output_path.read_text(encoding="utf-8") if output_path.exists() else proc.stdout)
            return remap_public_constraint_results(result["constraint_results"], constraints, public_ids)

    def _persistent_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._persistent_client is None:
            command = format_inspection_command(self.config["persistent_command"])
            self._persistent_client = PersistentJSONCommandClient(command)
        return self._persistent_client.request(payload)


class ObjectRequirementInspection(InspectionRunner):
    """Internal constraint checker for structured object requirements.

    This runner consumes detector/VLM-normalized generation metadata, not any
    benchmark-specific result format. Backends or upstream tools may attach one
    of these fields to the generation result:

    - detected_objects: [{"class": "cat", "count": 2}]
    - objects: ["cat", "cat", "chair"]
    - object_counts: {"cat": 2, "chair": 1}
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def inspect(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int = 0) -> Dict[str, Dict[str, Any]]:
        counts = self._object_counts(generation)
        out: Dict[str, Dict[str, Any]] = {}
        for c in self._constraints_for_inspection(task, program):
            meta = c.metadata or {}
            required = self._required_objects_for_constraint(meta) if isinstance(meta, dict) else []
            forbidden = meta.get("forbidden_objects", meta.get("exclude", [])) if isinstance(meta, dict) else []
            exact_required_counts = str(meta.get("count_mode", "")).lower() == "exact" if isinstance(meta, dict) else False
            if not required and not forbidden:
                out[c.constraint_id] = {
                    "passed": True,
                    "symptom": "pass",
                    "confidence": 1.0,
                    "object_counts": counts,
                    "note": "no structured object requirements for this constraint",
                }
                continue

            missing = []
            count_mismatches = []
            forbidden_present = []
            for obj in required:
                name = canonical_object_class(obj.get("class", ""))
                expected = int(obj.get("count", 1) or 1)
                found = counts.get(name, 0)
                if found <= 0:
                    missing.append({"class": name, "expected": expected, "found": found})
                elif found < expected:
                    count_mismatches.append({"class": name, "expected": expected, "found": found, "direction": "under"})
                elif exact_required_counts and found > expected:
                    count_mismatches.append({"class": name, "expected": expected, "found": found, "direction": "over"})
            required_expected = {
                canonical_object_class(obj.get("class", "")): int(obj.get("count", 1) or 1)
                for obj in required
                if isinstance(obj, dict) and obj.get("class")
            }
            for obj in forbidden:
                name = canonical_object_class(obj.get("class", obj))
                found = counts.get(name, 0)
                threshold = int(obj.get("count", 1) or 1) if isinstance(obj, dict) else 1
                if found >= threshold:
                    expected = required_expected.get(name)
                    if expected is not None and threshold > expected:
                        count_mismatches.append({"class": name, "expected": expected, "found": found, "direction": "over"})
                    else:
                        forbidden_present.append({"class": name, "forbidden_at_or_above": threshold, "found": found})

            passed = not missing and not count_mismatches and not forbidden_present
            constraint_score = self._object_requirement_score(required, forbidden, counts)
            if missing:
                symptom = "missing_required_object"
            elif count_mismatches:
                symptom = "object_count_mismatch"
            elif forbidden_present:
                symptom = "forbidden_object_present"
            else:
                symptom = "pass"
            out[c.constraint_id] = {
                "passed": passed,
                "symptom": symptom,
                "confidence": constraint_score,
                "constraint_score": constraint_score,
                "object_counts": counts,
                "missing_required_objects": missing,
                "count_mismatches": count_mismatches,
                "forbidden_objects_present": forbidden_present,
            }
        return out

    def _required_objects_for_constraint(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        required: List[Dict[str, Any]] = []
        # Explicit contract inventory is authoritative.  Relation and attribute
        # records only recover object names absent from that inventory; they must
        # never introduce a second, implicit count=1 requirement for the same
        # class.
        seen_classes = set()
        for obj in meta.get("required_objects", meta.get("include", [])) if isinstance(meta, dict) else []:
            if not isinstance(obj, dict) or not obj.get("class"):
                continue
            name = canonical_object_class(obj["class"])
            count = max(1, int(obj.get("count", 1) or 1))
            if name not in seen_classes:
                seen_classes.add(name)
                required.append({"class": name, "count": count})
        for obj in self._objects_implied_by_relations_and_attributes(meta):
            if obj["class"] not in seen_classes:
                seen_classes.add(obj["class"])
                required.append(obj)
        return required

    def _objects_implied_by_relations_and_attributes(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        seen = set()
        for rel in meta.get("spatial_relations", []) if isinstance(meta.get("spatial_relations"), list) else []:
            if not isinstance(rel, dict):
                continue
            for key in ("subject", "object"):
                name = canonical_object_class(rel.get(key) or "")
                if name and name not in seen:
                    seen.add(name)
                    objects.append({"class": name, "count": 1})
        for binding in meta.get("attribute_bindings", []) if isinstance(meta.get("attribute_bindings"), list) else []:
            if not isinstance(binding, dict):
                continue
            name = canonical_object_class(binding.get("object") or "")
            if name and name not in seen:
                seen.add(name)
                objects.append({"class": name, "count": 1})
        return objects

    def _constraints_for_inspection(self, task: GenerationTask, program: Dict[str, Any]) -> List[VisualConstraint]:
        return constraints_for_internal_inspection(task, program)

    def _object_counts(self, generation: Dict[str, Any]) -> Dict[str, int]:
        if not isinstance(generation, dict):
            return {}
        if isinstance(generation.get("object_counts"), dict):
            return {canonical_object_class(k): int(v) for k, v in generation["object_counts"].items()}
        counts: Dict[str, int] = {}
        detected = generation.get("detected_objects")
        if isinstance(detected, list):
            for obj in detected:
                if isinstance(obj, dict):
                    name = obj.get("class") or obj.get("name")
                    count = int(obj.get("count", 1) or 1)
                else:
                    name = obj
                    count = 1
                if name:
                    key = canonical_object_class(name)
                    counts[key] = counts.get(key, 0) + count
        objects = generation.get("objects")
        if isinstance(objects, list):
            for name in objects:
                key = canonical_object_class(name)
                if key:
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def _object_requirement_score(self, required: Any, forbidden: Any, counts: Dict[str, int]) -> float:
        scores = []
        for obj in required if isinstance(required, list) else []:
            if not isinstance(obj, dict) or not obj.get("class"):
                continue
            name = canonical_object_class(obj["class"])
            expected = max(1, int(obj.get("count", 1) or 1))
            found = max(0, int(counts.get(name, 0) or 0))
            if found == expected:
                scores.append(1.0)
            elif found < expected:
                scores.append(found / expected)
            else:
                scores.append(max(0.0, 1.0 - ((found - expected) / expected)))
        for obj in forbidden if isinstance(forbidden, list) else []:
            name = canonical_object_class(obj.get("class", obj)) if isinstance(obj, dict) else canonical_object_class(obj)
            if not name:
                continue
            found = max(0, int(counts.get(name, 0) or 0))
            scores.append(1.0 if found == 0 else 0.0)
        return sum(scores) / len(scores) if scores else 1.0


class DetectorObjectRequirementInspection(InspectionRunner):
    """Adapter from generic detector output to object_requirements.v1 results.

    The configured command receives task/program/generation JSON and should emit
    either {"object_counts": {...}} or {"detected_objects": [{"class": "cat"}]}.
    This adapter then reuses ObjectRequirementInspection so downstream feedback
    remains constraint-level and benchmark-agnostic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.object_inspector = ObjectRequirementInspection(config)
        self._persistent_client: PersistentJSONCommandClient | None = None

    def inspect(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int = 0) -> Dict[str, Dict[str, Any]]:
        constraints = self.object_inspector._constraints_for_inspection(task, program)
        if not any(self._needs_detector(c) for c in constraints):
            return {
                c.constraint_id: {
                    "passed": True,
                    "symptom": "pass",
                    "confidence": 1.0,
                    "not_applicable": True,
                    "reason": "constraint has no detector-checkable object or spatial requirements",
                    "inspection_source": self.config.get("name", "detector_object_requirements"),
                }
                for c in constraints
            }
        detector_result = self._run_detector(task, program, generation, seed=seed)
        detector_aliases = self._detector_aliases_for_constraints(constraints)
        normalized_generation = dict(generation or {})
        if isinstance(detector_result.get("object_counts"), dict):
            normalized_generation["object_counts"] = self._normalize_detector_object_counts(
                detector_result["object_counts"],
                detector_aliases,
            )
        if isinstance(detector_result.get("detected_objects"), list):
            normalized_generation["detected_objects"] = self._normalize_detector_detections(
                detector_result["detected_objects"],
                detector_aliases,
            )
        if isinstance(detector_result.get("objects"), list):
            normalized_generation["objects"] = detector_result["objects"]
        normalized_generation["inspection_source"] = {
            "type": "detector_object_requirements",
            "name": self.config.get("name"),
            "raw_result": detector_result,
        }
        result = self.object_inspector.inspect(task, program, normalized_generation, seed=seed)
        spatial_result = self._inspect_spatial_relations(task, program, detector_result, detector_aliases=detector_aliases)
        attribute_result = self._inspect_attribute_bindings(task, program, detector_result, generation)
        for cid, row in {**spatial_result, **attribute_result}.items():
            previous = result.get(cid, {"passed": True, "symptom": "pass", "confidence": 1.0})
            merged = {**previous, **row}
            merged["passed"] = bool(previous.get("passed", False)) and bool(row.get("passed", False))
            if not merged["passed"] and row.get("symptom") != "pass":
                merged["symptom"] = row.get("symptom")
            elif not merged["passed"] and previous.get("symptom") != "pass":
                merged["symptom"] = previous.get("symptom")
            previous_score = self._bounded_score(previous.get("selection_score", previous.get("constraint_score", previous.get("confidence"))))
            row_score = self._bounded_score(row.get("selection_score", row.get("constraint_score", row.get("confidence"))))
            scores = [score for score in (previous_score, row_score) if score is not None]
            if scores:
                merged["constraint_score"] = min(scores)
                merged["confidence"] = min(scores)
                merged["selection_score"] = min(scores)
            result[cid] = merged
        self._mark_unsupported_constraints_not_applicable(result, constraints, detector_result)
        self._mark_uncertain_constraints_unknown(result, constraints, detector_result)
        self._apply_unexpected_object_gate(result, detector_result)
        self._stamp_detector_evidence_status(result)
        return result

    def _stamp_detector_evidence_status(self, result: Dict[str, Dict[str, Any]]) -> None:
        if not bool(self.config.get("hard_evidence_required", False)):
            return
        for row in result.values():
            if bool(row.get("not_applicable", False)):
                continue
            row["hard_evidence_required"] = True
            if row.get("evidence_status") not in {"confirmed", "mismatch", "unknown"}:
                row["evidence_status"] = "confirmed" if bool(row.get("passed", False)) else "mismatch"

    def _mark_uncertain_constraints_unknown(
        self,
        result: Dict[str, Dict[str, Any]],
        constraints: Sequence[VisualConstraint],
        detector_result: Dict[str, Any],
    ) -> None:
        """Abstain when an open-vocabulary detector score is ambiguous."""

        uncertain = {
            canonical_object_class(value)
            for value in detector_result.get("uncertain_classes", [])
            if str(value).strip()
        }
        if not uncertain:
            return
        for constraint in constraints:
            affected = sorted(self._constraint_referenced_classes(constraint) & uncertain)
            if not affected:
                continue
            row = dict(result.get(constraint.constraint_id, {}))
            row.update({
                "passed": False,
                "symptom": "missing_inspection",
                "confidence": min(0.49, float(row.get("confidence", 0.0) or 0.0)),
                "constraint_score": min(0.49, float(row.get("constraint_score", 0.0) or 0.0)),
                "selection_score": min(0.49, float(row.get("selection_score", 0.0) or 0.0)),
                "hard_evidence_required": True,
                "evidence_status": "unknown",
                "uncertain_classes": affected,
                "reason": "open-vocabulary detector confidence is inside the configured uncertainty band",
                "inspection_source": self.config.get("name", "detector_object_requirements"),
            })
            result[constraint.constraint_id] = row

    def _mark_unsupported_constraints_not_applicable(
        self,
        result: Dict[str, Dict[str, Any]],
        constraints: Sequence[VisualConstraint],
        detector_result: Dict[str, Any],
    ) -> None:
        if not bool(self.config.get("skip_unsupported_classes", False)):
            return
        unsupported = {
            canonical_object_class(value)
            for value in detector_result.get("unsupported_classes", [])
            if str(value).strip()
        }
        if not unsupported:
            return
        for constraint in constraints:
            referenced = self._constraint_referenced_classes(constraint)
            affected = sorted(referenced & unsupported)
            if not affected:
                continue
            supported = sorted(referenced - unsupported)
            if not supported or not self._constraint_has_fully_supported_check(constraint, set(supported)):
                result[constraint.constraint_id] = {
                    "passed": True,
                    "symptom": "pass",
                    "confidence": 1.0,
                    "not_applicable": True,
                    "reason": "no complete check in this constraint is supported by the detector vocabulary",
                    "unsupported_classes": affected,
                    "supported_classes": supported,
                    "inspection_source": self.config.get("name", "detector_object_requirements"),
                }
                continue

            # Preserve usable evidence class-by-class.  An unsupported class
            # must not erase a supported detector result, and must not be
            # treated as a zero-count detector observation either.
            row = dict(result.get(constraint.constraint_id, {}))
            supported_set = set(supported)
            for key in ("missing_required_objects", "count_mismatches", "forbidden_objects_present"):
                values = row.get(key, [])
                if isinstance(values, list):
                    row[key] = [value for value in values if self._diagnostic_class(value) in supported_set]
            for key in ("attribute_binding_mismatches", "attribute_binding_unknowns", "attribute_color_scores"):
                values = row.get(key, [])
                if isinstance(values, list):
                    row[key] = [value for value in values if self._diagnostic_class(value) in supported_set]
            for key in ("spatial_relation_mismatches", "spatial_relation_unknowns", "spatial_relations_passed"):
                values = row.get(key, [])
                if isinstance(values, list):
                    row[key] = [
                        value for value in values
                        if isinstance(value, dict)
                        and canonical_object_class(value.get("subject", "")) in supported_set
                        and canonical_object_class(value.get("object", "")) in supported_set
                    ]
            row["object_counts"] = {
                name: count
                for name, count in row.get("object_counts", {}).items()
                if canonical_object_class(name) in supported_set
            } if isinstance(row.get("object_counts"), dict) else {}
            failure_keys = (
                "missing_required_objects",
                "count_mismatches",
                "forbidden_objects_present",
                "attribute_binding_mismatches",
                "attribute_binding_unknowns",
                "spatial_relation_mismatches",
                "spatial_relation_unknowns",
            )
            failures = any(row.get(key) for key in failure_keys)
            row["passed"] = not failures
            if row.get("missing_required_objects"):
                row["symptom"] = "missing_required_object"
            elif row.get("count_mismatches"):
                row["symptom"] = "object_count_mismatch"
            elif row.get("forbidden_objects_present"):
                row["symptom"] = "forbidden_object_present"
            elif row.get("attribute_binding_unknowns") or row.get("spatial_relation_unknowns"):
                row["symptom"] = "missing_inspection"
            elif row.get("attribute_binding_mismatches"):
                row["symptom"] = "attribute_binding_mismatch"
            elif row.get("spatial_relation_mismatches"):
                row["symptom"] = "spatial_relation_mismatch"
            else:
                row["symptom"] = "pass"
            row["partially_applicable"] = True
            row["unsupported_classes"] = affected
            row["supported_classes"] = supported
            row["reason"] = "detector evidence retained only for supported classes"
            row["inspection_source"] = self.config.get("name", "detector_object_requirements")
            result[constraint.constraint_id] = row

    def _constraint_has_fully_supported_check(self, constraint: VisualConstraint, supported: set[str]) -> bool:
        """Whether at least one atomic check can be evaluated end-to-end."""

        meta = constraint.metadata if isinstance(constraint.metadata, dict) else {}
        if constraint.constraint_type == "spatial":
            relations = meta.get("spatial_relations", [])
            relations = relations if isinstance(relations, list) else []
            return any(
                isinstance(relation, dict)
                and canonical_object_class(relation.get("subject", "")) in supported
                and canonical_object_class(relation.get("object", "")) in supported
                for relation in relations
            )
        if constraint.constraint_type == "attribute":
            bindings = meta.get("attribute_bindings", [])
            bindings = bindings if isinstance(bindings, list) else []
            return any(
                isinstance(binding, dict)
                and canonical_object_class(binding.get("object", "")) in supported
                for binding in bindings
            )
        return bool(supported)

    def _diagnostic_class(self, value: Any) -> str:
        if not isinstance(value, dict):
            return canonical_object_class(value)
        for key in ("class", "object", "subject"):
            if value.get(key):
                return canonical_object_class(value[key])
        return ""

    def _constraint_referenced_classes(self, constraint: VisualConstraint) -> set[str]:
        meta = constraint.metadata if isinstance(constraint.metadata, dict) else {}
        names: set[str] = set()
        for key in ("required_objects", "forbidden_objects", "include", "exclude"):
            rows = meta.get(key)
            for row in rows if isinstance(rows, list) else []:
                if isinstance(row, dict) and row.get("class"):
                    names.add(canonical_object_class(row["class"]))
                elif isinstance(row, str):
                    names.add(canonical_object_class(row))
        rows = meta.get("spatial_relations")
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            for key in ("subject", "object"):
                if row.get(key):
                    names.add(canonical_object_class(row[key]))
        rows = meta.get("attribute_bindings")
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("object"):
                names.add(canonical_object_class(row["object"]))
        return names

    def _detector_aliases_for_constraints(self, constraints: Sequence[VisualConstraint]) -> Dict[str, str]:
        referenced: set[str] = set()
        for constraint in constraints:
            referenced.update(self._constraint_referenced_classes(constraint))
        aliases: Dict[str, str] = {}
        for name in referenced:
            detector_name = canonical_detector_object_class(name)
            if detector_name != name and detector_name not in referenced:
                aliases[detector_name] = name
        return aliases

    def _normalize_detector_object_counts(self, counts: Dict[str, Any], detector_aliases: Dict[str, str]) -> Dict[str, int]:
        normalized: Dict[str, int] = {}
        for key, value in counts.items():
            detector_name = canonical_object_class(key)
            count = int(value)
            normalized[detector_name] = normalized.get(detector_name, 0) + count
            original_name = detector_aliases.get(detector_name)
            if original_name:
                normalized[original_name] = normalized.get(original_name, 0) + count
        return normalized

    def _normalize_detector_detections(self, detections: List[Dict[str, Any]], detector_aliases: Dict[str, str]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            row = dict(det)
            detector_name = canonical_object_class(row.get("class", ""))
            row["class"] = detector_name
            normalized.append(row)
            original_name = detector_aliases.get(detector_name)
            if original_name:
                alias_row = dict(row)
                alias_row["class"] = original_name
                alias_row["detector_alias_of"] = detector_name
                normalized.append(alias_row)
        return normalized

    def _apply_unexpected_object_gate(
        self,
        result: Dict[str, Dict[str, Any]],
        detector_result: Dict[str, Any],
    ) -> None:
        """Optionally reject prominent COCO objects not requested by the prompt.

        This is an opt-in selection-safety gate.  It uses the same detector as
        object/geometry verification and has no benchmark-specific class list.
        """

        if not bool(self.config.get("reject_unrequested_detected_objects", False)):
            return
        allowed = {
            str(value).strip().lower()
            for value in detector_result.get("classes", [])
            if str(value).strip()
        }
        threshold = float(self.config.get("unexpected_object_threshold", 0.80))
        overlap_suppression = self._bounded_score(
            self.config.get("unexpected_object_overlap_suppression")
        )
        allowed_detections = [
            detection
            for detection in detector_result.get("detected_objects", [])
            if isinstance(detection, dict)
            and str(detection.get("class") or "").strip().lower() in allowed
        ]
        unexpected = []
        suppressed = []
        for detection in detector_result.get("all_detected_objects", []):
            if not isinstance(detection, dict):
                continue
            class_name = str(detection.get("class") or "").strip().lower()
            score = self._bounded_score(detection.get("score"))
            if class_name and class_name not in allowed and score is not None and score >= threshold:
                overlap = max(
                    (
                        self._bbox_intersection_over_smaller(
                            detection.get("bbox"), allowed_detection.get("bbox")
                        )
                        for allowed_detection in allowed_detections
                    ),
                    default=0.0,
                )
                if overlap_suppression is not None and overlap >= overlap_suppression:
                    suppressed.append({
                        "class": class_name,
                        "score": score,
                        "bbox": detection.get("bbox"),
                        "intersection_over_smaller": overlap,
                    })
                    continue
                unexpected.append({
                    "class": class_name,
                    "score": score,
                    "bbox": detection.get("bbox"),
                })
        if suppressed:
            for row in result.values():
                row["suppressed_overlapping_detections"] = suppressed
                row["unexpected_object_overlap_suppression"] = overlap_suppression
        if not unexpected:
            return
        for row in result.values():
            row["passed"] = False
            row["symptom"] = "unexpected_detected_object"
            row["unexpected_detected_objects"] = unexpected
            row["unexpected_object_threshold"] = threshold
            row["evidence_status"] = "mismatch"
            previous_score = self._bounded_score(
                row.get("selection_score", row.get("constraint_score", row.get("confidence")))
            )
            gated_score = min(previous_score, 0.49) if previous_score is not None else 0.49
            row["confidence"] = gated_score
            row["constraint_score"] = gated_score
            row["selection_score"] = gated_score

    @staticmethod
    def _bbox_intersection_over_smaller(first: Any, second: Any) -> float:
        """Return intersection divided by the smaller box area.

        Detector label aliases often describe the same visual region.  IoU is
        unnecessarily low when one alias box is nested inside another, whereas
        intersection-over-smaller is close to one.  Invalid boxes deliberately
        return zero so they can never bypass the extra-object veto.
        """

        if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
            return 0.0
        if len(first) < 4 or len(second) < 4:
            return 0.0
        try:
            ax1, ay1, ax2, ay2 = [float(value) for value in first[:4]]
            bx1, by1, bx2, by2 = [float(value) for value in second[:4]]
        except (TypeError, ValueError):
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if area_a <= 0.0 or area_b <= 0.0:
            return 0.0
        intersection = (
            max(0.0, min(ax2, bx2) - max(ax1, bx1))
            * max(0.0, min(ay2, by2) - max(ay1, by1))
        )
        return intersection / min(area_a, area_b)

    def _bounded_score(self, value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return max(0.0, min(1.0, float(value)))

    def _needs_detector(self, constraint: VisualConstraint) -> bool:
        metadata = constraint.metadata if isinstance(constraint.metadata, dict) else {}
        return bool(
            metadata.get("required_objects")
            or metadata.get("forbidden_objects")
            or metadata.get("include")
            or metadata.get("exclude")
            or metadata.get("spatial_relations")
            or metadata.get("attribute_bindings")
            or constraint.constraint_type in {"count", "spatial"}
        )

    def _inspect_spatial_relations(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        detector_result: Dict[str, Any],
        *,
        detector_aliases: Dict[str, str] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        detections = detector_result.get("detected_objects", []) if isinstance(detector_result, dict) else []
        if not isinstance(detections, list):
            detections = []
        aliases = detector_aliases or {}
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for det in detections:
            if not isinstance(det, dict) or not det.get("class") or not isinstance(det.get("bbox"), list):
                continue
            detector_name = canonical_object_class(det["class"])
            by_class.setdefault(detector_name, []).append(det)
            original_name = aliases.get(detector_name)
            if original_name:
                by_class.setdefault(original_name, []).append(det)
        for rows in by_class.values():
            rows.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)

        relations = self._expected_spatial_relations(task, program)
        if not relations:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        relation_exact_objects = self._relation_exact_object_names(task, program)
        for cid, rels in relations.items():
            mismatches = []
            unknowns = []
            passed_relations = []
            relation_scores = []
            for rel in rels:
                subject = canonical_object_class(rel["subject"])
                target = canonical_object_class(rel["object"])
                relation = str(rel["relation"]).replace("_", " ")
                subject_dets = by_class.get(subject, [])
                target_dets = by_class.get(target, [])
                if not subject_dets or not target_dets:
                    relation_scores.append(self._missing_relation_score(bool(subject_dets), bool(target_dets)))
                    unknowns.append({
                        "subject": subject,
                        "relation": relation,
                        "object": target,
                        "reason": "missing_detected_object_for_relation",
                        "subject_found": bool(subject_dets),
                        "object_found": bool(target_dets),
                    })
                    continue
                duplicate = self._relation_duplicate_mismatch(
                    subject,
                    target,
                    subject_dets,
                    target_dets,
                    exact_object_names=relation_exact_objects.get(cid, set()),
                )
                if duplicate is not None:
                    relation_scores.append(float(duplicate["score"]))
                    # Preserve the public diagnostic field while classifying
                    # the evidence itself as unknown: with duplicate
                    # exact-bound objects we cannot know which box is the
                    # intended relation subject/object.
                    mismatches.append(duplicate)
                    unknowns.append(duplicate)
                    continue
                match = self._best_relation_match(subject_dets, target_dets, relation)
                subject_det = match["subject"]
                target_det = match["object"]
                subject_center = match["subject_center"]
                target_center = match["object_center"]
                ok = bool(match["passed"])
                row = {
                    "subject": subject,
                    "relation": relation,
                    "object": target,
                    "subject_bbox": subject_det["bbox"],
                    "object_bbox": target_det["bbox"],
                    "subject_center": subject_center,
                    "object_center": target_center,
                    "relation_detection_score": match["detection_score"],
                }
                if ok:
                    relation_scores.append(1.0)
                    passed_relations.append(row)
                else:
                    relation_scores.append(float(match["score"]))
                    row["reason"] = "bbox_center_relation_mismatch"
                    row["actual_relation"] = self._actual_relation_for_boxes(subject_det, target_det)
                    mismatches.append(row)
            constraint_score = sum(relation_scores) / len(relation_scores) if relation_scores else 0.0
            out[cid] = {
                "passed": not mismatches and not unknowns,
                "symptom": "pass" if not mismatches and not unknowns else "spatial_relation_mismatch",
                "confidence": constraint_score,
                "constraint_score": constraint_score,
                "relation_scores": relation_scores,
                "spatial_relation_mismatches": mismatches,
                "spatial_relation_unknowns": unknowns,
                "spatial_relations_passed": passed_relations,
                "hard_evidence_required": bool(self.config.get("hard_evidence_required", False)),
                "evidence_status": "confirmed" if not mismatches and not unknowns else ("unknown" if unknowns else "mismatch"),
                "inspection_source": self.config.get("name", "detector_object_requirements"),
            }
        return out

    def _inspect_attribute_bindings(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        detector_result: Dict[str, Any],
        generation: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        if not bool(self.config.get("attribute_color_verification", True)):
            return {}
        detections = detector_result.get("detected_objects", []) if isinstance(detector_result, dict) else []
        if not isinstance(detections, list):
            return {}
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for det in detections:
            if not isinstance(det, dict) or not det.get("class") or not isinstance(det.get("bbox"), list):
                continue
            by_class.setdefault(str(det["class"]).strip().lower(), []).append(det)
        for rows in by_class.values():
            rows.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        if not by_class:
            return {}

        constraints = constraints_for_internal_inspection(task, program)
        attribute_constraints = []
        for constraint in constraints:
            if constraint.constraint_type != "attribute":
                continue
            bindings = self._expected_attribute_bindings(constraint)
            if bindings:
                attribute_constraints.append((constraint.constraint_id, bindings))
        if not attribute_constraints:
            return {}

        image_path = self._generation_image_path(generation)
        if image_path is None:
            return {}
        try:
            from PIL import Image  # type: ignore

            image = Image.open(image_path).convert("RGB")
        except Exception:
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        for cid, bindings in attribute_constraints:
            diagnostics = []
            mismatches = []
            unknowns = []
            for binding in bindings:
                obj = binding["object"]
                attribute = binding["attribute"]
                object_detections = by_class.get(obj, [])
                if not object_detections:
                    unknowns.append({
                        "object": obj,
                        "attribute": attribute,
                        "reason": "missing_detected_object_for_attribute",
                    })
                    continue
                match = self._best_attribute_color_match(image, object_detections, attribute, bindings)
                if len(object_detections) > 1:
                    match["ambiguous_duplicate_count"] = len(object_detections)
                    match["selection_score"] = max(0.0, float(match["selection_score"]) - 0.05 * (len(object_detections) - 1))
                diagnostics.append({
                    "object": obj,
                    "attribute": attribute,
                    "positive_color_score": match["positive_score"],
                    "max_negative_color_score": match["max_negative_score"],
                    "selection_score": match["selection_score"],
                    "bbox": match.get("bbox"),
                    "detection_score": match.get("detection_score"),
                    "ambiguous_duplicate_count": match.get("ambiguous_duplicate_count"),
                })
                if match["reason"] == "no_valid_detection_crop":
                    unknowns.append({
                        "object": obj,
                        "attribute": attribute,
                        "reason": match["reason"],
                    })
                elif not match["passed"]:
                    mismatches.append({
                        "object": obj,
                        "attribute": attribute,
                        "reason": match["reason"],
                        "positive_color_score": match["positive_score"],
                        "max_negative_color_score": match["max_negative_score"],
                        "bbox": match.get("bbox"),
                    })
            scores = [float(row["selection_score"]) for row in diagnostics]
            constraint_score = min(scores) if scores else 0.0
            out[cid] = {
                "passed": bool(diagnostics) and not mismatches and not unknowns,
                "symptom": "pass" if diagnostics and not mismatches and not unknowns else ("missing_inspection" if unknowns or not diagnostics else "attribute_binding_mismatch"),
                "confidence": constraint_score,
                "constraint_score": constraint_score,
                "selection_score": constraint_score,
                "attribute_color_scores": diagnostics,
                "attribute_binding_mismatches": mismatches,
                "attribute_binding_unknowns": unknowns,
                "hard_evidence_required": bool(self.config.get("hard_evidence_required", False)),
                "evidence_status": "confirmed" if diagnostics and not mismatches and not unknowns else ("unknown" if unknowns or not diagnostics else "mismatch"),
                "inspection_source": self.config.get("name", "detector_object_requirements"),
            }
        return out

    def _expected_attribute_bindings(self, constraint: VisualConstraint) -> List[Dict[str, str]]:
        meta = constraint.metadata if isinstance(constraint.metadata, dict) else {}
        rows = meta.get("attribute_bindings", []) if isinstance(meta.get("attribute_bindings"), list) else []
        bindings: List[Dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("object") or not row.get("attribute"):
                continue
            attribute = str(row["attribute"]).strip().lower()
            if attribute not in self._known_color_names():
                continue
            bindings.append({"object": str(row["object"]).strip().lower(), "attribute": attribute})
        return bindings

    def _best_attribute_color_match(
        self,
        image: Any,
        detections: List[Dict[str, Any]],
        attribute: str,
        all_bindings: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        threshold = float(self.config.get("attribute_color_threshold", 0.045))
        margin = float(self.config.get("attribute_color_margin", 0.010))
        if bool(self.config.get("attribute_color_compare_all_colors", True)):
            competing_colors = sorted(color for color in self._known_color_names() if color not in {attribute, "grey"})
        else:
            competing_colors = sorted({
                row["attribute"]
                for row in all_bindings
                if row.get("attribute") and row.get("attribute") != attribute and row.get("attribute") in self._known_color_names()
            })
        if (
            bool(self.config.get("attribute_ignore_achromatic_competitors_for_chromatic", False))
            and attribute not in {"black", "white", "gray", "grey"}
        ):
            # Detector boxes are rectangular and often contain a large amount
            # of neutral background around non-rectangular objects. White,
            # black, or gray background pixels are not evidence that a blue
            # kite is the wrong chromatic hue. Keep hue-vs-hue competition
            # (for example red versus orange) while excluding achromatic fill.
            competing_colors = [color for color in competing_colors if color not in {"black", "white", "gray", "grey"}]
        match_mode = str(self.config.get("attribute_match_mode", "best_color")).strip().lower()
        candidate_detections = detections[:1] if match_mode in {"top", "top_confidence", "highest_confidence"} else detections
        best: Dict[str, Any] | None = None
        best_key = (-1.0, -1.0)
        for det in candidate_detections:
            bbox = det.get("bbox")
            if not isinstance(bbox, list):
                continue
            crop = self._crop_detection(image, bbox)
            if crop is None:
                continue
            positive = self._color_fraction(crop, attribute)
            negatives = [self._color_fraction(crop, color) for color in competing_colors]
            max_negative = max(negatives) if negatives else 0.0
            if attribute in {"black", "white", "gray", "grey"}:
                score = positive
            else:
                score = 0.5 + (positive - max_negative)
            selection_score = max(0.0, min(1.0, score))
            detection_score = self._bounded_score(det.get("score")) or 0.0
            key = (selection_score, detection_score)
            if key > best_key:
                passed = positive >= threshold and positive + 1e-9 >= max_negative + margin
                reason = "pass"
                if positive < threshold:
                    reason = "weak_target_color_evidence"
                elif positive + 1e-9 < max_negative + margin:
                    reason = "competing_color_evidence_stronger"
                best_key = key
                best = {
                    "positive_score": max(0.0, min(1.0, positive)),
                    "max_negative_score": max(0.0, min(1.0, max_negative)),
                    "selection_score": selection_score,
                    "detection_score": detection_score,
                    "bbox": bbox,
                    "passed": passed,
                    "reason": reason,
                }
        if best is not None:
            return best
        return {
            "positive_score": 0.0,
            "max_negative_score": 0.0,
            "selection_score": 0.0,
            "detection_score": 0.0,
            "bbox": None,
            "passed": False,
            "reason": "no_valid_detection_crop",
        }

    def _generation_image_path(self, generation: Dict[str, Any]) -> str | None:
        image_path = (generation or {}).get("image_path") or (generation or {}).get("image_uri")
        if not isinstance(image_path, str) or not image_path:
            return None
        if image_path.startswith("data:"):
            if "," not in image_path:
                return None
            header, encoded = image_path.split(",", 1)
            suffix = ".jpg" if "jpeg" in header or "jpg" in header else ".png"
            tmp = tempfile.NamedTemporaryFile(prefix="genharness_attr_color_", suffix=suffix, delete=False)
            tmp.write(base64.b64decode(encoded))
            tmp.close()
            return tmp.name
        path = Path(image_path)
        return str(path) if path.is_file() else None

    def _crop_detection(self, image: Any, bbox: List[Any]) -> Any | None:
        try:
            width, height = image.size
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                return None
            inset_x = (x2 - x1) * 0.08
            inset_y = (y2 - y1) * 0.08
            crop_box = (
                int(max(0.0, x1 + inset_x)),
                int(max(0.0, y1 + inset_y)),
                int(min(float(width), x2 - inset_x)),
                int(min(float(height), y2 - inset_y)),
            )
            crop = image.crop(crop_box)
            if max(crop.size) > 96:
                crop.thumbnail((96, 96))
            return crop
        except Exception:
            return None

    def _color_fraction(self, image: Any, color_name: str) -> float:
        color_name = color_name.strip().lower()
        if color_name == "grey":
            color_name = "gray"
        get_pixels = getattr(image, "get_flattened_data", None)
        pixels = list(get_pixels() if callable(get_pixels) else image.getdata())
        if not pixels:
            return 0.0
        matches = 0
        for pixel in pixels:
            r, g, b = [float(v) / 255.0 for v in pixel[:3]]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            if self._pixel_matches_color(h, s, v, color_name):
                matches += 1
        return matches / float(len(pixels))

    def _pixel_matches_color(self, h: float, s: float, v: float, color_name: str) -> bool:
        if color_name == "red":
            return s >= 0.25 and v >= 0.18 and (h <= 0.045 or h >= 0.94)
        if color_name == "orange":
            # Brown occupies the same hue range as orange but at lower value.
            # Treating every brown pixel as orange made brown impossible to
            # win the all-colors comparison.  Keep the classes disjoint at
            # the core value boundary; bright highlights on a brown object
            # may still count as orange, but no longer dominate its crop.
            return s >= 0.25 and v > 0.62 and 0.045 < h <= 0.115
        if color_name == "yellow":
            return s >= 0.25 and v >= 0.25 and 0.115 < h <= 0.185
        if color_name == "green":
            return s >= 0.20 and v >= 0.16 and 0.20 <= h <= 0.46
        if color_name == "blue":
            return s >= 0.20 and v >= 0.16 and 0.52 <= h <= 0.72
        if color_name == "purple":
            return s >= 0.20 and v >= 0.16 and 0.72 < h <= 0.86
        if color_name == "brown":
            return s >= 0.25 and 0.12 <= v <= 0.62 and 0.045 < h <= 0.12
        if color_name == "black":
            return v <= 0.22
        if color_name == "white":
            return v >= 0.78 and s <= 0.25
        if color_name == "gray":
            return 0.22 < v < 0.82 and s <= 0.18
        return False

    def _known_color_names(self) -> set[str]:
        return {"red", "orange", "yellow", "green", "blue", "purple", "brown", "black", "white", "gray", "grey"}

    def _missing_relation_score(self, subject_found: bool, target_found: bool) -> float:
        if subject_found and target_found:
            return 0.5
        if subject_found or target_found:
            return 0.25
        return 0.0

    def _relation_duplicate_mismatch(
        self,
        subject: str,
        target: str,
        subject_dets: List[Dict[str, Any]],
        target_dets: List[Dict[str, Any]],
        *,
        exact_object_names: set[str],
    ) -> Dict[str, Any] | None:
        duplicates = []
        if subject in exact_object_names and len(subject_dets) > 1:
            duplicates.append({"class": subject, "expected": 1, "found": len(subject_dets)})
        if target in exact_object_names and len(target_dets) > 1:
            duplicates.append({"class": target, "expected": 1, "found": len(target_dets)})
        if not duplicates:
            return None
        detection_scores = []
        for det in subject_dets[:1] + target_dets[:1]:
            value = self._bounded_score(det.get("score"))
            if value is not None:
                detection_scores.append(value)
        score = 0.5 * (sum(detection_scores) / len(detection_scores)) if detection_scores else 0.25
        return {
            "subject": subject,
            "object": target,
            "reason": "ambiguous_duplicate_object_for_relation",
            "duplicate_objects": duplicates,
            "score": max(0.0, min(0.5, score)),
        }

    def _relation_exact_object_names(self, task: GenerationTask, program: Dict[str, Any]) -> Dict[str, set[str]]:
        out: Dict[str, set[str]] = {}
        constraints = constraints_for_internal_inspection(task, program)
        for constraint in constraints:
            if constraint.constraint_type != "spatial":
                continue
            meta = constraint.metadata if isinstance(constraint.metadata, dict) else {}
            if str(meta.get("count_mode", "")).lower() != "exact":
                continue
            names = set()
            for obj in meta.get("required_objects", []) if isinstance(meta.get("required_objects"), list) else []:
                if not isinstance(obj, dict) or not obj.get("class"):
                    continue
                if int(obj.get("count", 1) or 1) == 1:
                    names.add(canonical_object_class(obj["class"]))
            if names:
                out[constraint.constraint_id] = names
        return out

    def _expected_spatial_relations(self, task: GenerationTask, program: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        out: Dict[str, List[Dict[str, str]]] = {}
        constraints = constraints_for_internal_inspection(task, program)
        for constraint in constraints:
            if constraint.constraint_type != "spatial":
                continue
            rows: List[Dict[str, str]] = []
            meta = constraint.metadata if isinstance(constraint.metadata, dict) else {}
            for row in meta.get("spatial_relations", []) if isinstance(meta.get("spatial_relations"), list) else []:
                if isinstance(row, dict) and row.get("subject") and row.get("object"):
                    rows.append({
                        "subject": canonical_object_class(row["subject"]),
                        "relation": normalize_relation(row.get("relation", "")),
                        "object": canonical_object_class(row["object"]),
                    })
            if rows:
                out.setdefault(constraint.constraint_id, []).extend(rows)
        return out

    def _bbox_center(self, bbox: List[Any]) -> List[float]:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def _relation_holds(self, subject_center: List[float], target_center: List[float], relation: str) -> bool:
        margin = float(self.config.get("spatial_margin_pixels", 8.0))
        axis_ratio = float(self.config.get("spatial_primary_axis_ratio", 1.0))
        dx = subject_center[0] - target_center[0]
        dy = subject_center[1] - target_center[1]
        if relation == "left of":
            return dx < -margin and abs(dx) >= abs(dy) * axis_ratio
        if relation == "right of":
            return dx > margin and abs(dx) >= abs(dy) * axis_ratio
        if relation == "above":
            return dy < -margin and abs(dy) >= abs(dx) * axis_ratio
        if relation == "below":
            return dy > margin and abs(dy) >= abs(dx) * axis_ratio
        return True

    def _size_aware_relative_positions(self, subject_bbox: List[Any], target_bbox: List[Any]) -> set[str]:
        """Reproduce GenEval's published size-aware bbox relation definition.

        This is geometry only: it operates on boxes supplied by the configured
        internal detector and never invokes or consumes an official evaluator.
        """

        sx1, sy1, sx2, sy2 = [float(value) for value in subject_bbox[:4]]
        tx1, ty1, tx2, ty2 = [float(value) for value in target_bbox[:4]]
        subject_center = [(sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0]
        target_center = [(tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0]
        subject_dims = [abs(sx2 - sx1), abs(sy2 - sy1)]
        target_dims = [abs(tx2 - tx1), abs(ty2 - ty1)]
        offset = [subject_center[index] - target_center[index] for index in range(2)]
        threshold = float(self.config.get("spatial_size_threshold", 0.1))
        revised = []
        for index, value in enumerate(offset):
            magnitude = max(abs(value) - threshold * (subject_dims[index] + target_dims[index]), 0.0)
            revised.append(magnitude if value >= 0 else -magnitude)
        if all(abs(value) < 1e-3 for value in revised):
            return set()
        norm = (offset[0] ** 2 + offset[1] ** 2) ** 0.5
        if norm <= 0:
            return set()
        dx, dy = revised[0] / norm, revised[1] / norm
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    def _relation_holds_for_boxes(self, subject_det: Dict[str, Any], target_det: Dict[str, Any], relation: str) -> bool:
        mode = str(self.config.get("spatial_geometry_mode", "axis_dominance")).strip().lower()
        if mode == "geneval_size_aware_v1":
            return relation in self._size_aware_relative_positions(subject_det["bbox"], target_det["bbox"])
        return self._relation_holds(self._bbox_center(subject_det["bbox"]), self._bbox_center(target_det["bbox"]), relation)

    def _actual_relation_for_boxes(self, subject_det: Dict[str, Any], target_det: Dict[str, Any]) -> str:
        mode = str(self.config.get("spatial_geometry_mode", "axis_dominance")).strip().lower()
        if mode == "geneval_size_aware_v1":
            relations = self._size_aware_relative_positions(subject_det["bbox"], target_det["bbox"])
            return " and ".join(sorted(relations)) if relations else "none"
        return self._actual_relation(self._bbox_center(subject_det["bbox"]), self._bbox_center(target_det["bbox"]))

    def _relation_score(self, subject_center: List[float], target_center: List[float], relation: str) -> float:
        margin = float(self.config.get("spatial_margin_pixels", 8.0))
        dx = subject_center[0] - target_center[0]
        dy = subject_center[1] - target_center[1]
        components = {
            "left of": (-dx, abs(dy)),
            "right of": (dx, abs(dy)),
            "above": (-dy, abs(dx)),
            "below": (dy, abs(dx)),
        }.get(relation)
        if components is None:
            return 0.5
        primary_delta, orthogonal_delta = components
        directional = max(0.0, min(1.0, 0.5 + (primary_delta / max(margin * 4.0, 1.0)) * 0.5))
        if primary_delta <= 0:
            axis_dominance = 0.0
        else:
            axis_dominance = primary_delta / max(primary_delta + orthogonal_delta, 1.0)
        score = 0.70 * directional + 0.30 * axis_dominance
        return max(0.0, min(0.99, score))

    def _best_relation_match(self, subject_dets: List[Dict[str, Any]], target_dets: List[Dict[str, Any]], relation: str) -> Dict[str, Any]:
        match_mode = str(self.config.get("spatial_match_mode", "best_relation")).strip().lower()
        if match_mode in {"top", "top_confidence", "highest_confidence"}:
            subject_det = subject_dets[0]
            target_det = target_dets[0]
            subject_center = self._bbox_center(subject_det["bbox"])
            target_center = self._bbox_center(target_det["bbox"])
            return {
                "subject": subject_det,
                "object": target_det,
                "subject_center": subject_center,
                "object_center": target_center,
                "score": self._relation_score(subject_center, target_center, relation),
                "detection_score": self._detection_pair_score(subject_det, target_det),
                "passed": self._relation_holds_for_boxes(subject_det, target_det, relation),
            }
        best: Dict[str, Any] | None = None
        best_key = (-1.0, -1.0)
        for subject_det in subject_dets:
            for target_det in target_dets:
                if subject_det is target_det:
                    continue
                subject_center = self._bbox_center(subject_det["bbox"])
                target_center = self._bbox_center(target_det["bbox"])
                relation_score = self._relation_score(subject_center, target_center, relation)
                detection_score = self._detection_pair_score(subject_det, target_det)
                passed = self._relation_holds_for_boxes(subject_det, target_det, relation)
                key = (1.0 if passed else relation_score, detection_score)
                if key > best_key:
                    best_key = key
                    best = {
                        "subject": subject_det,
                        "object": target_det,
                        "subject_center": subject_center,
                        "object_center": target_center,
                        "score": relation_score,
                        "detection_score": detection_score,
                        "passed": passed,
                    }
        if best is not None:
            return best
        subject_det = subject_dets[0]
        target_det = target_dets[0]
        subject_center = self._bbox_center(subject_det["bbox"])
        target_center = self._bbox_center(target_det["bbox"])
        return {
            "subject": subject_det,
            "object": target_det,
            "subject_center": subject_center,
            "object_center": target_center,
            "score": self._relation_score(subject_center, target_center, relation),
            "detection_score": self._detection_pair_score(subject_det, target_det),
            "passed": self._relation_holds_for_boxes(subject_det, target_det, relation),
        }

    def _detection_pair_score(self, subject_det: Dict[str, Any], target_det: Dict[str, Any]) -> float:
        scores = []
        for det in (subject_det, target_det):
            value = det.get("score")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            scores.append(max(0.0, min(1.0, float(value))))
        return sum(scores) / len(scores) if scores else 0.0

    def _actual_relation(self, subject_center: List[float], target_center: List[float]) -> str:
        dx = subject_center[0] - target_center[0]
        dy = subject_center[1] - target_center[1]
        if abs(dx) >= abs(dy):
            return "right of" if dx > 0 else "left of"
        return "below" if dy > 0 else "above"

    def _run_detector(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int) -> Dict[str, Any]:
        constraints = constraints_for_external_tool(task, program)
        public_input = public_inspection_payload(task, program, generation, constraints=constraints)
        payload = {
            "input": public_input,
            "constraints": public_input.get("constraints", []),
            "seed": seed,
            "image_uri": (generation or {}).get("image_uri"),
            "image_path": (generation or {}).get("image_path") or (generation or {}).get("image_uri"),
            "generation": {
                "image_uri": (generation or {}).get("image_uri"),
                "image_path": (generation or {}).get("image_path") or (generation or {}).get("image_uri"),
            },
        }
        if "persistent_command" in self.config:
            if self._persistent_client is None:
                self._persistent_client = PersistentJSONCommandClient(format_inspection_command(self.config["persistent_command"]))
            return self._persistent_client.request(payload)
        if "command" not in self.config:
            passthrough = self.config.get("passthrough_generation_fields", True)
            if passthrough:
                return dict(generation or {})
            raise ValueError("detector_object_requirements inspection requires command or passthrough_generation_fields=true")
        with tempfile.TemporaryDirectory(prefix="gen_harness_detector_") as td:
            td_path = Path(td)
            input_path = td_path / "input.json"
            output_path = td_path / "output.json"
            payload = {**payload, "output_json": str(output_path)}
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            command = format_inspection_command(self.config["command"], input_json=str(input_path), output_json=str(output_path))
            proc = subprocess.run(command, text=True, capture_output=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"detector command failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
            raw = output_path.read_text(encoding="utf-8") if output_path.exists() else proc.stdout
            result = json.loads(raw)
            if "constraint_results" in result:
                raise ValueError("detector_object_requirements expects detector outputs, not constraint_results; use external_command for full inspection results")
            return result


class LightweightAlignmentInspection(InspectionRunner):
    """Lightweight CLIP/SigLIP/OCR-style constraint inspection.

    This adapter is intentionally model-agnostic. It can either consume scores
    already attached by a backend, or call a small external scorer. Expected
    score fields are generic, e.g. generation["alignment_scores"] or command
    output {"scores": {constraint_id: 0.31}}. Text constraints may also be
    checked from generation["ocr_text"] without a VLM judge.
    """

    DEFAULT_TYPES = {"prompt_alignment", "attribute", "spatial", "text", "quality", "aesthetic"}

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.threshold = float(self.config.get("threshold", 0.25))
        self.constraint_types = set(self.config.get("constraint_types", sorted(self.DEFAULT_TYPES)))
        self._persistent_client: PersistentJSONCommandClient | None = None

    def inspect(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int = 0) -> Dict[str, Dict[str, Any]]:
        constraints = self._constraints_for_inspection(task, program)
        applicable_constraints = [c for c in constraints if c.constraint_type in self.constraint_types]
        if applicable_constraints:
            scorer_result = self._score(task, program, generation, seed=seed, constraints=applicable_constraints)
        else:
            scorer_result = {}
        self._remap_public_score_keys(scorer_result, applicable_constraints)
        scores = self._score_map(scorer_result, generation)
        ocr_text = str(scorer_result.get("ocr_text") or (generation or {}).get("ocr_text") or "").lower()
        out: Dict[str, Dict[str, Any]] = {}
        for c in constraints:
            if c.constraint_type not in self.constraint_types:
                out[c.constraint_id] = self._not_applicable_result(c, scores, scorer_result)
                continue
            if c.constraint_type == "text":
                row = self._inspect_text_constraint(c, ocr_text, scores)
            elif c.constraint_type == "attribute":
                row = self._inspect_attribute_constraint(c, scores, scorer_result)
            else:
                row = self._inspect_similarity_constraint(c, scores, scorer_result)
            out[c.constraint_id] = row
        return out

    def _score(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        generation: Dict[str, Any],
        seed: int,
        *,
        constraints: Sequence[VisualConstraint] | None = None,
    ) -> Dict[str, Any]:
        if "command" not in self.config:
            if "persistent_command" not in self.config:
                return dict(generation or {})
        inspection_constraints = list(constraints) if constraints is not None else self._constraints_for_inspection(task, program)
        public_input = public_inspection_payload(task, program, generation, constraints=inspection_constraints)
        payload = {
            "input": public_input,
            "constraints": public_input.get("constraints", []),
            "seed": seed,
            "image_uri": (generation or {}).get("image_uri"),
            "image_path": (generation or {}).get("image_path") or (generation or {}).get("image_uri"),
        }
        if "persistent_command" in self.config:
            if self._persistent_client is None:
                self._persistent_client = PersistentJSONCommandClient(format_inspection_command(self.config["persistent_command"]))
            return self._persistent_client.request(payload)
        with tempfile.TemporaryDirectory(prefix="gen_harness_light_align_") as td:
            td_path = Path(td)
            input_path = td_path / "input.json"
            output_path = td_path / "output.json"
            payload = {**payload, "output_json": str(output_path)}
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            command = format_inspection_command(self.config["command"], input_json=str(input_path), output_json=str(output_path))
            proc = subprocess.run(command, text=True, capture_output=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"lightweight alignment command failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
            raw = output_path.read_text(encoding="utf-8") if output_path.exists() else proc.stdout
            return json.loads(raw)

    def _remap_public_score_keys(self, scorer_result: Dict[str, Any], constraints: Sequence[VisualConstraint]) -> None:
        if not isinstance(scorer_result, dict):
            return
        public_ids = public_constraint_ids(list(constraints))
        reverse = {public: internal for internal, public in public_ids.items()}
        for key in ("constraint_scores", "alignment_scores", "clip_scores", "siglip_scores", "scores"):
            rows = scorer_result.get(key)
            if not isinstance(rows, dict):
                continue
            for public_id, internal_id in reverse.items():
                if public_id in rows and internal_id not in rows:
                    rows[internal_id] = rows[public_id]
            for public_id, value in list(rows.items()):
                if not isinstance(public_id, str) or ":binding:" not in public_id:
                    continue
                head, tail = public_id.split(":binding:", 1)
                internal_head = reverse.get(head)
                if internal_head:
                    rows.setdefault(f"{internal_head}:binding:{tail}", value)

    def _constraints_for_inspection(self, task: GenerationTask, program: Dict[str, Any]) -> List[VisualConstraint]:
        return constraints_for_internal_inspection(task, program)

    def _score_map(self, scorer_result: Dict[str, Any], generation: Dict[str, Any]) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for source in (generation or {}, scorer_result or {}):
            for key in ("constraint_scores", "alignment_scores", "clip_scores", "siglip_scores", "scores"):
                rows = source.get(key)
                if isinstance(rows, dict):
                    for name, value in rows.items():
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            continue
                        merged[str(name)] = float(value)
        return merged

    def _not_applicable_result(self, constraint: VisualConstraint, scores: Dict[str, float], scorer_result: Dict[str, Any]) -> Dict[str, Any]:
        relevant_scores = {}
        for key in (constraint.constraint_id, constraint.constraint_type, constraint.target or "", "prompt_alignment", "overall"):
            if key and key in scores:
                relevant_scores[key] = scores[key]
        return {
            "passed": True,
            "symptom": "pass",
            "confidence": 1.0,
            "not_applicable": True,
            "reason": f"constraint_type {constraint.constraint_type} is not evaluated by this lightweight inspector",
            "inspection_source": self.config.get("name", "lightweight_alignment"),
            "relevant_scores": relevant_scores,
            "raw_result_summary": {
                k: scorer_result.get(k)
                for k in ("model", "scorer", "device", "ocr_engine")
                if k in scorer_result
            },
        }

    def _inspect_text_constraint(self, constraint: VisualConstraint, ocr_text: str, scores: Dict[str, float]) -> Dict[str, Any]:
        required = constraint.metadata.get("required_text", []) if isinstance(constraint.metadata, dict) else []
        missing = []
        for row in required:
            text = str(row.get("text", row) if isinstance(row, dict) else row).strip()
            if text and text.lower() not in ocr_text:
                missing.append(text)
        if missing:
            return {
                "passed": False,
                "symptom": "text_rendering_mismatch",
                "confidence": 1.0,
                "missing_text": missing,
                "ocr_text": ocr_text,
                "inspection_source": "lightweight_alignment",
            }
        if required:
            return {
                "passed": True,
                "symptom": "pass",
                "confidence": 1.0,
                "ocr_text": ocr_text,
                "inspection_source": "lightweight_alignment",
            }
        return self._inspect_similarity_constraint(constraint, scores, {})

    def _inspect_attribute_constraint(self, constraint: VisualConstraint, scores: Dict[str, float], scorer_result: Dict[str, Any]) -> Dict[str, Any]:
        base_row = self._inspect_similarity_constraint(constraint, scores, scorer_result)
        diagnostics = self._attribute_binding_diagnostics(constraint, scores)
        if not diagnostics:
            return base_row
        binding_scores = [float(row["selection_score"]) for row in diagnostics]
        binding_score = min(binding_scores) if binding_scores else 0.0
        base_score = self._row_score_value(base_row)
        selection_score = min(base_score, binding_score) if base_score is not None else binding_score
        margin = float(self.config.get("attribute_binding_margin", 0.0))
        passed_bindings = all(
            float(row.get("positive_score", 0.0)) >= self.threshold
            and float(row.get("positive_score", 0.0)) + 1e-9 >= float(row.get("max_negative_score", 0.0)) + margin
            for row in diagnostics
        )
        mismatches = self._attribute_binding_mismatches(diagnostics, margin)
        passed = bool(base_row.get("passed", False)) and passed_bindings
        row = dict(base_row)
        row.update({
            "passed": passed,
            "symptom": "pass" if passed else "attribute_binding_mismatch",
            "selection_score": max(0.0, min(1.0, selection_score)),
            "attribute_binding_scores": diagnostics,
            "attribute_binding_mismatches": mismatches if not passed else [],
            "attribute_binding_margin": margin,
        })
        return row

    def _attribute_binding_diagnostics(self, constraint: VisualConstraint, scores: Dict[str, float]) -> List[Dict[str, Any]]:
        meta = constraint.metadata if isinstance(constraint.metadata, dict) else {}
        bindings = []
        rows = meta.get("attribute_bindings", [])
        for item in rows if isinstance(rows, list) else []:
            if isinstance(item, dict) and item.get("object") and item.get("attribute"):
                bindings.append({"object": str(item["object"]), "attribute": str(item["attribute"])})
        if not bindings:
            return []
        diagnostics: List[Dict[str, Any]] = []
        cid = constraint.constraint_id
        for index, binding in enumerate(bindings):
            positive_key = f"{cid}:binding:{index}:positive"
            positive_score = scores.get(positive_key)
            if positive_score is None:
                continue
            negative_scores = []
            negative_prefix = f"{cid}:binding:{index}:negative:"
            for key, value in scores.items():
                if key.startswith(negative_prefix):
                    negative_scores.append(float(value))
            max_negative = max(negative_scores) if negative_scores else 0.0
            if negative_scores:
                margin_score = 0.5 + (float(positive_score) - max_negative)
                selection_score = max(0.0, min(1.0, margin_score))
            else:
                selection_score = max(0.0, min(1.0, float(positive_score)))
            diagnostics.append({
                **binding,
                "positive_score": max(0.0, min(1.0, float(positive_score))),
                "max_negative_score": max(0.0, min(1.0, float(max_negative))),
                "selection_score": selection_score,
                "num_negative_prompts": len(negative_scores),
            })
        return diagnostics

    def _attribute_binding_mismatches(self, diagnostics: List[Dict[str, Any]], margin: float) -> List[Dict[str, Any]]:
        mismatches: List[Dict[str, Any]] = []
        for row in diagnostics:
            positive = float(row.get("positive_score", 0.0))
            negative = float(row.get("max_negative_score", 0.0))
            if positive >= self.threshold and positive + 1e-9 >= negative + margin:
                continue
            reason = "weak_positive_binding_score" if positive < self.threshold else "negative_binding_score_stronger"
            mismatches.append({
                "object": row.get("object"),
                "attribute": row.get("attribute"),
                "reason": reason,
                "positive_score": positive,
                "max_negative_score": negative,
                "selection_score": row.get("selection_score"),
            })
        return mismatches

    def _inspect_similarity_constraint(self, constraint: VisualConstraint, scores: Dict[str, float], scorer_result: Dict[str, Any]) -> Dict[str, Any]:
        score = self._lookup_score(constraint, scores)
        if score is None:
            return {
                "passed": False,
                "symptom": "missing_alignment_score",
                "confidence": 0.0,
                "inspection_source": "lightweight_alignment",
                "available_score_keys": sorted(scores),
            }
        passed = score >= self.threshold
        return {
            "passed": passed,
            "symptom": "pass" if passed else self._low_score_symptom(constraint.constraint_type),
            "confidence": min(1.0, max(0.0, score)),
            "alignment_score": score,
            "threshold": self.threshold,
            "inspection_source": "lightweight_alignment",
            "raw_result_summary": {k: scorer_result.get(k) for k in ("model", "scorer", "device") if k in scorer_result},
        }

    def _row_score_value(self, row: Dict[str, Any]) -> float | None:
        for key in ("selection_score", "alignment_score", "constraint_score", "confidence"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            return max(0.0, min(1.0, float(value)))
        return None

    def _lookup_score(self, constraint: VisualConstraint, scores: Dict[str, float]) -> float | None:
        for key in (constraint.constraint_id, constraint.constraint_type, constraint.target or "", "prompt_alignment", "overall"):
            if key and key in scores:
                return scores[key]
        return None

    def _low_score_symptom(self, constraint_type: str) -> str:
        if constraint_type == "attribute":
            return "attribute_binding_mismatch"
        if constraint_type == "spatial":
            return "spatial_relation_mismatch"
        if constraint_type == "text":
            return "text_rendering_mismatch"
        if constraint_type == "aesthetic":
            return "low_aesthetic_quality"
        if constraint_type == "quality":
            return "low_overall_quality"
        return "low_prompt_alignment"


class CompositeInspectionRunner(InspectionRunner):
    """Runs multiple internal/external inspectors and merges constraint results."""

    def __init__(self, inspectors: List[InspectionRunner], config: Dict[str, Any] | None = None):
        if not inspectors:
            raise ValueError("CompositeInspectionRunner requires at least one inspector")
        self.inspectors = inspectors
        self.config = dict(config or {})
        self.last_latency_by_source: Dict[str, float] = {}
        self.last_total_latency_seconds = 0.0

    def inspect(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int = 0) -> Dict[str, Dict[str, Any]]:
        constraints = constraints_for_internal_inspection(task, program)
        constraints_by_id = {constraint.constraint_id: constraint for constraint in constraints}
        applicable_subresults_by_constraint: Dict[str, Dict[str, Dict[str, Any]]] = {}
        subresults_by_constraint: Dict[str, Dict[str, Dict[str, Any]]] = {}
        latency_by_source: Dict[str, float] = {}
        # Later inspectors may consume generic evidence produced by earlier
        # inspectors (for example, a VLM can inspect detector-localized object
        # crops).  Keep this context inside the current inspection call; it is
        # never written into the generation backend result or used by an
        # official evaluator.
        generation_context = dict(generation or {})
        generation_context["upstream_inspection_results"] = {}
        self.last_latency_by_source = {}
        self.last_total_latency_seconds = 0.0
        for inspector in self.inspectors:
            source = self._inspector_source(inspector)
            started = time.perf_counter()
            try:
                result = inspector.inspect(task, program, generation_context, seed=seed)
            finally:
                elapsed = time.perf_counter() - started
                latency_by_source[source] = elapsed
                self.last_latency_by_source = {key: round(value, 6) for key, value in latency_by_source.items()}
                self.last_total_latency_seconds = round(sum(latency_by_source.values()), 6)
            for cid, row in result.items():
                row_copy = dict(row)
                row_copy["latency_seconds"] = round(elapsed, 6)
                subresults = dict(subresults_by_constraint.get(cid, {}))
                subresults[source] = row_copy
                subresults_by_constraint[cid] = subresults
                applicable = not bool(row_copy.get("not_applicable", False))
                if applicable:
                    applicable_rows = dict(applicable_subresults_by_constraint.get(cid, {}))
                    applicable_rows[source] = row_copy
                    applicable_subresults_by_constraint[cid] = applicable_rows
            generation_context["upstream_inspection_results"][source] = {
                cid: dict(row) for cid, row in result.items()
            }
        merged: Dict[str, Dict[str, Any]] = {}
        for cid, subresults in subresults_by_constraint.items():
            merged[cid] = self._merge_constraint_results(
                subresults,
                applicable_subresults_by_constraint.get(cid, {}),
                constraints_by_id.get(cid),
            )
        self.last_latency_by_source = {key: round(value, 6) for key, value in latency_by_source.items()}
        self.last_total_latency_seconds = round(sum(latency_by_source.values()), 6)
        return merged

    def _inspector_source(self, inspector: InspectionRunner) -> str:
        config = getattr(inspector, "config", None)
        if isinstance(config, dict) and config.get("name"):
            return str(config["name"])
        return inspector.__class__.__name__

    def _merge_constraint_results(
        self,
        subresults: Dict[str, Dict[str, Any]],
        applicable_rows: Dict[str, Dict[str, Any]],
        constraint: VisualConstraint | None = None,
    ) -> Dict[str, Any]:
        if not applicable_rows:
            first = next(iter(subresults.values()))
            merged = dict(first)
            merged["passed"] = True
            merged["symptom"] = "pass"
            merged["inspection_subresults"] = dict(subresults)
            return merged

        effective_rows = self._effective_applicable_rows(subresults, applicable_rows, constraint)
        merged: Dict[str, Any] = {}
        for row in effective_rows.values():
            merged = {**merged, **row}

        passed = all(bool(row.get("passed", False)) for row in effective_rows.values())
        merged["passed"] = passed
        merged["symptom"] = "pass" if passed else self._merged_failure_symptom(effective_rows)
        evidence_rows = [
            row for row in effective_rows.values()
            if bool(row.get("hard_evidence_required", False)) or row.get("evidence_status") is not None
        ]
        if evidence_rows:
            statuses = [str(row.get("evidence_status", "unknown")).strip().lower() for row in evidence_rows]
            explicit = {status for status in statuses if status in {"confirmed", "mismatch"}}
            verifier_conflict = len(explicit) > 1
            if "unknown" in statuses or verifier_conflict:
                evidence_status = "unknown"
            elif statuses and all(status == "confirmed" for status in statuses):
                evidence_status = "confirmed"
            else:
                evidence_status = "mismatch"
            merged["hard_evidence_required"] = any(bool(row.get("hard_evidence_required", False)) for row in evidence_rows)
            merged["evidence_status"] = evidence_status
            merged["verifier_conflict"] = verifier_conflict
        merged["inspection_subresults"] = dict(subresults)
        evidence_sources = [
            source for source, row in effective_rows.items()
            if not bool(row.get("not_applicable", False))
            and not bool(row.get("partially_applicable", False))
        ]
        merged["independent_evidence_sources"] = evidence_sources
        merged["num_independent_evidence_sources"] = len(evidence_sources)
        merged["single_verifier_evidence"] = len(evidence_sources) < 2
        ignored_sources = [source for source in applicable_rows if source not in effective_rows]
        if ignored_sources:
            merged["ignored_for_merge_sources"] = ignored_sources
        merge_strategy = str(self.config.get("merge_strategy", "")).strip()
        if merge_strategy:
            merged["merge_strategy"] = merge_strategy
        self._update_selection_score(merged, effective_rows)
        return merged

    def _effective_applicable_rows(
        self,
        subresults: Dict[str, Dict[str, Any]],
        applicable_rows: Dict[str, Dict[str, Any]],
        constraint: VisualConstraint | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        partitioned_rows = self._capability_partition_rows(subresults, constraint)
        if partitioned_rows is not None:
            return partitioned_rows
        has_structured_attribute = any(self._is_structured_attribute_result(row) for row in applicable_rows.values())
        has_structured_spatial = any(self._is_structured_spatial_result(row) for row in applicable_rows.values())
        if not has_structured_attribute and not has_structured_spatial:
            return applicable_rows
        structured_rows = {
            source: row
            for source, row in applicable_rows.items()
            if not self._is_lightweight_alignment_result(source, row)
        }
        return structured_rows or applicable_rows

    def _capability_partition_rows(
        self,
        subresults: Dict[str, Dict[str, Any]],
        constraint: VisualConstraint | None,
    ) -> Dict[str, Dict[str, Any]] | None:
        """Select one predeclared owner from contract type and declared capability."""

        if str(self.config.get("merge_strategy", "")).strip().lower() != "capability_partitioned":
            return None
        owners = self.config.get("capability_owners", {})
        if not isinstance(owners, dict) or constraint is None:
            raise ValueError("capability_partitioned merge requires capability_owners and compiled constraints")
        capability = constraint.constraint_type
        if capability == "spatial":
            meta = constraint.metadata if isinstance(constraint.metadata, dict) else {}
            relations = meta.get("spatial_relations", [])
            relation_names = {
                normalize_relation(row.get("relation", ""))
                for row in relations if isinstance(row, dict)
            }
            two_dimensional_relations = {
                "left of", "right of", "above", "below", "over", "under",
                "on top of", "beneath", "next to", "beside",
            }
            if relation_names and relation_names <= two_dimensional_relations:
                detector_source = str(owners.get("spatial_2d_detector", "")).strip()
                detector_row = subresults.get(detector_source, {})
                detector_supports = (
                    bool(detector_source)
                    and isinstance(detector_row, dict)
                    and not bool(detector_row.get("not_applicable", False))
                    and not bool(detector_row.get("unsupported_classes"))
                )
                if detector_supports:
                    semantic_source = str(owners.get("spatial_2d_open_vocab", "")).strip()
                    semantic_row = subresults.get(semantic_source, {})
                    if isinstance(semantic_row, dict) and not bool(semantic_row.get("not_applicable", False)):
                        return {detector_source: dict(detector_row), semantic_source: dict(semantic_row)}
                    capability = "spatial_2d_detector"
                else:
                    capability = "spatial_2d_open_vocab"
            else:
                capability = "spatial_depth"
        elif capability == "count":
            # For classes fully covered by the detector, retain both independent
            # inspectors.  Agreement can confirm a result; disagreement becomes
            # unknown in _merge_constraint_results.  A partially applicable
            # detector is diagnostic only and must not pretend to verify the
            # complete multi-class constraint.
            detector_source = str(owners.get("count_detector", "")).strip()
            semantic_source = str(owners.get("count", "")).strip()
            detector_row = subresults.get(detector_source, {})
            semantic_row = subresults.get(semantic_source, {})
            detector_complete = (
                bool(detector_source)
                and isinstance(detector_row, dict)
                and not bool(detector_row.get("not_applicable", False))
                and not bool(detector_row.get("partially_applicable", False))
                and not bool(detector_row.get("unsupported_classes"))
            )
            if detector_complete and isinstance(semantic_row, dict) and not bool(semantic_row.get("not_applicable", False)):
                return {detector_source: dict(detector_row), semantic_source: dict(semantic_row)}
        elif capability == "attribute":
            detector_source = str(owners.get("attribute_detector", "")).strip()
            semantic_source = str(owners.get("attribute", "")).strip()
            detector_row = subresults.get(detector_source, {})
            semantic_row = subresults.get(semantic_source, {})
            detector_complete = (
                bool(detector_source)
                and isinstance(detector_row, dict)
                and not bool(detector_row.get("not_applicable", False))
                and not bool(detector_row.get("partially_applicable", False))
                and not bool(detector_row.get("unsupported_classes"))
                and isinstance(detector_row.get("attribute_color_scores"), list)
                and bool(detector_row.get("attribute_color_scores"))
            )
            if detector_complete and isinstance(semantic_row, dict) and not bool(semantic_row.get("not_applicable", False)):
                return {detector_source: dict(detector_row), semantic_source: dict(semantic_row)}
        owner = str(owners.get(capability, "")).strip()
        if not owner:
            raise ValueError(f"no verifier owner configured for capability {capability}")
        row = subresults.get(owner)
        if not isinstance(row, dict) or bool(row.get("not_applicable", False)):
            return {owner: {
                "passed": False,
                "symptom": "inspection_error",
                "confidence": 0.0,
                "hard_evidence_required": True,
                "evidence_status": "unknown",
                "inspection_source": owner,
                "capability_route": capability,
                "reason": f"configured owner {owner} produced no applicable result",
            }}
        selected = dict(row)
        selected["capability_route"] = capability
        selected["capability_owner"] = owner
        return {owner: selected}

    def _is_structured_attribute_result(self, row: Dict[str, Any]) -> bool:
        return isinstance(row.get("attribute_color_scores"), list)

    def _is_structured_spatial_result(self, row: Dict[str, Any]) -> bool:
        return any(
            key in row
            for key in ("relation_scores", "spatial_relation_mismatches", "spatial_relations_passed")
        )

    def _is_lightweight_alignment_result(self, source: str, row: Dict[str, Any]) -> bool:
        inspection_source = str(row.get("inspection_source", "")).lower()
        source_name = str(source).lower()
        return (
            inspection_source == "lightweight_alignment"
            or "lightweight_alignment" in source_name
            or source_name.startswith("openclip")
            or source_name.startswith("siglip")
            or source_name.endswith("_alignment")
        )

    def _merged_failure_symptom(self, effective_rows: Dict[str, Dict[str, Any]]) -> str:
        for row in reversed(list(effective_rows.values())):
            symptom = row.get("symptom")
            if symptom and symptom != "pass":
                return str(symptom)
        return "constraint_failed"

    def _update_selection_score(self, merged_row: Dict[str, Any], applicable_rows: Dict[str, Dict[str, Any]]) -> None:
        scores = [self._row_selection_score(row) for row in applicable_rows.values()]
        scores = [score for score in scores if score is not None]
        if not scores:
            return
        merged_row["selection_score"] = min(scores)

    def _row_selection_score(self, row: Dict[str, Any]) -> float | None:
        for key in ("selection_score", "constraint_score", "alignment_score"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            return max(0.0, min(1.0, float(value)))
        confidence = row.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return 1.0 if bool(row.get("passed", False)) else 0.0
        return max(0.0, min(1.0, float(confidence))) if bool(row.get("passed", False)) else 0.0


def validate_formal_inspection_config(config_path: str | Path) -> Dict[str, Any]:
    """Validate the fixed OWLv2/NVILA verifier stack used by paper TRACE."""
    cfg = read_json(config_path)
    if not isinstance(cfg, dict) or cfg.get("type") != "composite":
        raise ValueError("paper TRACE requires the composite OWLv2/NVILA verifier stack")
    inspectors = cfg.get("inspectors")
    if not isinstance(inspectors, list) or not inspectors:
        raise ValueError("paper TRACE inspection config must declare OWLv2 and NVILA inspectors")
    serialized = json.dumps(cfg, ensure_ascii=False).lower()
    if "owlv2" not in serialized or "nvila" not in serialized:
        raise ValueError("paper TRACE inspection config must contain both OWLv2 and NVILA")
    return cfg


def build_inspector(config_path: str | Path) -> InspectionRunner:
    if config_path is None:
        raise ValueError("--inspection-config is required and must point to a real inspection adapter config")
    cfg = expand_env_vars(read_json(config_path))
    typ = cfg.get("type")
    if typ == "composite":
        return CompositeInspectionRunner([_build_inspector_from_config(item) for item in cfg.get("inspectors", [])], config=cfg)
    return _build_inspector_from_config(cfg)


def _build_inspector_from_config(cfg: Dict[str, Any]) -> InspectionRunner:
    typ = cfg.get("type")
    if typ == "object_requirements":
        if not _diagnostic_inspection_allowed(cfg):
            raise ValueError(
                "object_requirements consumes backend-provided structured metadata and is diagnostic-only; "
                "formal runs must use detector_object_requirements with a real command/persistent_command, "
                "external_command, http, openai_vision_judge, or an explicit diagnostic opt-in"
            )
        return ObjectRequirementInspection(cfg)
    if typ == "detector_object_requirements":
        if (
            "command" not in cfg
            and "persistent_command" not in cfg
            and bool(cfg.get("passthrough_generation_fields", True))
            and not _diagnostic_inspection_allowed(cfg)
        ):
            raise ValueError(
                "detector_object_requirements passthrough_generation_fields reads backend-provided detection metadata "
                "and is diagnostic-only; formal runs must configure command or persistent_command"
            )
        return DetectorObjectRequirementInspection(cfg)
    if typ == "lightweight_alignment":
        if "command" not in cfg and "persistent_command" not in cfg and not _diagnostic_inspection_allowed(cfg):
            raise ValueError(
                "lightweight_alignment without command/persistent_command consumes backend-provided scores and is diagnostic-only; "
                "formal runs must configure a real scorer command"
            )
        return LightweightAlignmentInspection(cfg)
    if typ == "external_command":
        return ExternalCommandInspection(cfg)
    if typ == "http":
        from .adapters.http_inspection import HTTPInspectionRunner

        return HTTPInspectionRunner(cfg)
    if typ == "openai_vision_judge":
        from .adapters.openai_vision_inspection import OpenAIVisionInspectionRunner

        return OpenAIVisionInspectionRunner(cfg)
    raise ValueError(f"unknown or unsupported inspection type: {typ}")


def _diagnostic_inspection_allowed(cfg: Dict[str, Any]) -> bool:
    if bool(cfg.get("allow_diagnostic_inspection_metadata")):
        return True
    return os.environ.get("GENHARNESS_DIAGNOSTIC_ALLOW_METADATA_INSPECTION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
