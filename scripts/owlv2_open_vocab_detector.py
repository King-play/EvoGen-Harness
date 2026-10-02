#!/usr/bin/env python3
"""Persistent OWLv2 open-vocabulary detector for Gen-Harness.

The process accepts one Gen-Harness JSON payload per line and returns object
counts plus auditable xyxy boxes.  Counts are computed only after class-wise
NMS.  A class whose best score falls in the uncertainty band is reported as
``uncertain_classes`` so the harness can abstain instead of treating it as an
observed zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detector_utils import classwise_nms, expanded_detector_queries, materialize_image, required_classes


DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"


def load_model(model_name: str, device: str, local_files_only: bool = False) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    # Pin the processor implementation so a transformers upgrade cannot
    # silently change verifier outputs between parent/candidate evaluations.
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(resolved_device).eval()
    return {"torch": torch, "processor": processor, "model": model, "device": resolved_device, "dtype": dtype}


def _label_name(label: Any, queries: list[str]) -> str:
    if isinstance(label, str):
        text = label.strip().lower()
        if text.startswith("a photo of a "):
            return text[len("a photo of a "):]
        return text
    try:
        return queries[int(label)]
    except (TypeError, ValueError, IndexError):
        return ""


def select_detections(
    boxes: list[Any],
    scores: list[Any],
    labels: list[Any],
    queries: list[str],
    *,
    threshold: float,
    uncertainty_threshold: float,
    nms_threshold: float,
) -> tuple[list[dict[str, Any]], list[str], dict[str, float]]:
    raw: list[dict[str, Any]] = []
    best_scores = {name: 0.0 for name in queries}
    for box, score, label in zip(boxes, scores, labels):
        confidence = float(score.item() if hasattr(score, "item") else score)
        class_name = _label_name(label, queries)
        if not class_name or class_name not in best_scores:
            continue
        best_scores[class_name] = max(best_scores[class_name], confidence)
        if confidence < threshold:
            continue
        values = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
        raw.append({
            "class": class_name,
            "count": 1,
            "score": confidence,
            "bbox": [float(value) for value in values[:4]],
            "bbox_format": "xyxy_pixels",
        })
    kept = classwise_nms(raw, iou_threshold=nms_threshold)
    uncertain = sorted(
        name for name, score in best_scores.items()
        if uncertainty_threshold <= score < threshold
    )
    return kept, uncertain, best_scores


def detect(payload: dict[str, Any], state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    classes = required_classes(payload)
    if not classes:
        return {"object_counts": {}, "detected_objects": [], "note": "no detector-checkable classes"}
    queries = expanded_detector_queries(classes)
    image_path = materialize_image(payload.get("generation", {}))
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    text_labels = [[f"a photo of a {name}" for name in queries]]
    inputs = state["processor"](text=text_labels, images=image, return_tensors="pt")
    inputs = {key: value.to(state["device"]) for key, value in inputs.items()}
    with state["torch"].inference_mode():
        outputs = state["model"](**inputs)
    target_sizes = state["torch"].tensor([(image.height, image.width)], device=state["device"])
    processor = state["processor"]
    if hasattr(processor, "post_process_grounded_object_detection"):
        result = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=float(args.uncertainty_threshold),
            text_labels=text_labels,
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
    else:
        result = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=float(args.uncertainty_threshold),
        )[0]
        labels = result.get("labels", [])
    detections, uncertain, best_scores = select_detections(
        list(result.get("boxes", [])),
        list(result.get("scores", [])),
        list(labels),
        queries,
        threshold=float(args.threshold),
        uncertainty_threshold=float(args.uncertainty_threshold),
        nms_threshold=float(args.nms_threshold),
    )
    wanted = set(classes)
    requested_detections = [row for row in detections if row["class"] in wanted]
    counts = {name: 0 for name in classes}
    for row in requested_detections:
        counts[row["class"]] += 1
    return {
        "object_counts": counts,
        "detected_objects": requested_detections,
        "all_detected_objects": detections,
        "detector": "google_owlv2_base_patch16_ensemble",
        "classes": classes,
        "queries": queries,
        "supported_classes": classes,
        "unsupported_classes": [],
        "uncertain_classes": [name for name in uncertain if name in wanted],
        "best_scores": {name: best_scores.get(name, 0.0) for name in classes},
        "threshold": float(args.threshold),
        "uncertainty_threshold": float(args.uncertainty_threshold),
        "nms_threshold": float(args.nms_threshold),
        "device": state["device"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.08)
    parser.add_argument("--nms-threshold", type=float, default=0.55)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.uncertainty_threshold <= args.threshold <= 1.0:
        parser.error("require 0 <= uncertainty-threshold <= threshold <= 1")
    state = load_model(args.model, args.device, args.local_files_only)
    if args.check_model:
        print(json.dumps({"ok": True, "model": args.model, "device": state["device"]}))
        return
    if args.serve:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                result = detect(json.loads(line), state, args)
            except Exception as exc:
                result = {"error": f"{exc.__class__.__name__}: {exc}"}
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required unless --serve is used")
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    Path(args.output).write_text(json.dumps(detect(payload, state, args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
