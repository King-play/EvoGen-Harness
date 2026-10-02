#!/usr/bin/env python3
"""Shared detector helpers for lightweight GenHarness internal verifiers."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any


DETECTOR_CONFOUNDER_QUERIES = {
    "bear": ["teddy bear"],
    "bus": ["train", "truck", "microwave", "refrigerator"],
    "computer keyboard": ["cell phone", "laptop", "piano"],
    "microwave": ["oven", "refrigerator", "tv", "sink"],
    "sink": ["bowl", "cell phone", "toilet"],
    "snowboard": ["skis", "skateboard", "surfboard", "person"],
    "toothbrush": ["skis", "snowboard", "person", "fork", "knife", "spoon"],
}

DETECTOR_CLASS_ALIASES = {
    "persons": "person",
    "people": "person",
    "sheeps": "sheep",
}

DETECTOR_AMBIGUOUS_CLASS_ALIASES = {
    "mug": "cup",
    "mugs": "cup",
}


def expanded_detector_queries(classes: list[str]) -> list[str]:
    out: list[str] = []
    for class_name in classes:
        name = str(class_name).strip().lower()
        if not name or name in out:
            continue
        out.append(name)
        for confounder in DETECTOR_CONFOUNDER_QUERIES.get(name, []):
            if confounder not in out:
                out.append(confounder)
    return out


def filter_confounded_detections(
    detected_objects: list[dict[str, Any]],
    detections_by_class: dict[str, list[dict[str, Any]]],
    *,
    required_classes: list[str] | None = None,
) -> list[dict[str, Any]]:
    required = {str(name).strip().lower() for name in (required_classes or [])}
    filtered: list[dict[str, Any]] = []
    for det in detected_objects:
        class_name = str(det.get("class", "")).lower()
        confounders = DETECTOR_CONFOUNDER_QUERIES.get(class_name, [])
        if not confounders:
            if class_name in required or not _is_confounder_only_class(class_name):
                filtered.append(det)
            continue
        if not any(_overlapping_stronger_confounder(det, detections_by_class.get(confounder, [])) for confounder in confounders):
            filtered.append(det)
    return filtered


def classwise_nms(
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.65,
    containment_threshold: float = 0.90,
) -> list[dict[str, Any]]:
    """Remove same-class duplicate and nested part boxes.

    Open-vocabulary detectors often return one whole-object box plus several
    boxes for contained parts of elongated or articulated objects. Plain IoU
    NMS misses these because the part box is much smaller. Intersection over
    the smaller box removes that duplicate without merging nearby instances.
    """

    threshold = max(0.0, min(1.0, float(iou_threshold)))
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for det in detections:
        class_name = str(det.get("class", "")).strip().lower()
        if class_name not in grouped:
            grouped[class_name] = []
            order.append(class_name)
        grouped[class_name].append(det)
    kept: list[dict[str, Any]] = []
    for class_name in order:
        rows = sorted(grouped[class_name], key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
        class_kept: list[dict[str, Any]] = []
        for row in rows:
            if any(
                bbox_iou(row.get("bbox"), prior.get("bbox")) >= threshold
                or bbox_containment(row.get("bbox"), prior.get("bbox")) >= containment_threshold
                for prior in class_kept
            ):
                continue
            class_kept.append(row)
        kept.extend(class_kept)
    return kept


def bbox_iou(a: Any, b: Any) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def bbox_containment(a: Any, b: Any) -> float:
    """Intersection divided by the smaller box area."""
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def materialize_image(generation: dict[str, Any]) -> str:
    image_path = generation.get("image_path") or generation.get("image_uri")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("generation result does not contain image_path or image_uri")
    if image_path.startswith("data:"):
        if "," not in image_path:
            raise ValueError("invalid data URI image")
        header, encoded = image_path.split(",", 1)
        suffix = ".png"
        if "jpeg" in header or "jpg" in header:
            suffix = ".jpg"
        tmp = tempfile.NamedTemporaryFile(prefix="genharness_detector_", suffix=suffix, delete=False)
        tmp.write(base64.b64decode(encoded))
        tmp.close()
        return tmp.name
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image path does not exist: {path}")
    return str(path)


def required_classes(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    constraints = []
    public_input = payload.get("input", {}) if isinstance(payload.get("input"), dict) else {}
    constraints.extend(public_input.get("constraints", []) if isinstance(public_input.get("constraints"), list) else [])
    task = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
    constraints.extend(task.get("constraints", []) if isinstance(task.get("constraints"), list) else [])
    program = payload.get("program", {}) if isinstance(payload.get("program"), dict) else {}
    contract = program.get("contract", {}) if isinstance(program.get("contract"), dict) else {}
    constraints.extend(contract.get("constraints", []) if isinstance(contract.get("constraints"), list) else [])
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        meta = constraint.get("metadata", {}) if isinstance(constraint.get("metadata"), dict) else {}
        for key in ("required_objects", "include"):
            rows = meta.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("class"):
                        out.append(base_detector_class(str(row["class"])))
        rows = meta.get("spatial_relations")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key in ("subject", "object"):
                    if row.get(key):
                        out.append(base_detector_class(str(row[key])))
        rows = meta.get("attribute_bindings")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("object"):
                    out.append(base_detector_class(str(row["object"])))
    out = normalize_detector_classes(out)
    deduped: list[str] = []
    seen = set()
    for name in out:
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def normalize_detector_class(name: str) -> str:
    return normalize_detector_classes([name])[0] if str(name).strip() else ""


def normalize_detector_classes(names: list[str]) -> list[str]:
    base_names = [base_detector_class(name) for name in names if str(name).strip()]
    base_set = set(base_names)
    out: list[str] = []
    for name in base_names:
        alias = DETECTOR_AMBIGUOUS_CLASS_ALIASES.get(name, name)
        normalized = name if alias != name and alias in base_set else alias
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def base_detector_class(name: str) -> str:
    text = str(name).strip().lower()
    return DETECTOR_CLASS_ALIASES.get(text, text)


def _is_confounder_only_class(class_name: str) -> bool:
    return any(class_name in confounders for confounders in DETECTOR_CONFOUNDER_QUERIES.values())


def _overlapping_stronger_confounder(det: dict[str, Any], confounders: list[dict[str, Any]]) -> bool:
    score = float(det.get("score", 0.0) or 0.0)
    for other in confounders:
        other_score = float(other.get("score", 0.0) or 0.0)
        if other_score >= score and bbox_iou(det.get("bbox"), other.get("bbox")) >= 0.45:
            return True
    return False
