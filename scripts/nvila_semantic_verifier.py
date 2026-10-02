#!/usr/bin/env python3
"""Persistent NVILA probability verifier for semantic Gen-Harness constraints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detector_utils import materialize_image


DEFAULT_MODEL = "Efficient-Large-Model/NVILA-Lite-2B-Verifier"
SUPPORTED_TYPES = {
    "attribute",
    "action",
    "spatial",
    "prompt_alignment",
    "identity",
    "style",
    "layout",
    "negative",
    "text",
    "quality",
    "aesthetic",
    "material",
}
DEPTH_RELATIONS = {"in front of", "in_front_of", "behind"}


def generation_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    generation = payload.get("generation")
    if isinstance(generation, dict):
        return generation
    return {}


def constraints_for_verification(payload: dict[str, Any]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    public_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}
    task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
    program = payload.get("program") if isinstance(payload.get("program"), dict) else {}
    contract = program.get("contract") if isinstance(program.get("contract"), dict) else {}
    for source in (payload, public_input, task, contract):
        rows = source.get("constraints") if isinstance(source, dict) else None
        if isinstance(rows, list):
            constraints.extend(row for row in rows if isinstance(row, dict))
    return constraints


def metadata_list(constraint: dict[str, Any], key: str) -> list[Any]:
    metadata = constraint.get("metadata") if isinstance(constraint.get("metadata"), dict) else {}
    rows = metadata.get(key)
    return rows if isinstance(rows, list) else []


def prompt_for_payload(payload: dict[str, Any]) -> str:
    for key_path in (
        ("input", "prompt"),
        ("task", "prompt"),
        ("program", "contract", "prompt"),
        ("prompt",),
    ):
        current: Any = payload
        for key in key_path:
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, str) and current.strip():
            return current.strip()
    return ""


def load_model(model_path: str, device: str) -> dict[str, Any]:
    import torch
    import transformers.image_utils as image_utils
    from transformers import AutoModel
    if not hasattr(image_utils, "VideoInput"):
        from typing import Any as TypingAny

        image_utils.VideoInput = TypingAny

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(model_path, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    tokenizer = model.tokenizer
    return {
        "model": model,
        "tokenizer": tokenizer,
        "yes_id": tokenizer.encode("yes", add_special_tokens=False)[0],
        "no_id": tokenizer.encode("no", add_special_tokens=False)[0],
        "device": str(getattr(model, "device", device)),
        "torch": torch,
    }


def binary_probabilities(scores: Any, yes_id: int, no_id: int) -> tuple[float, float]:
    logits = scores[0] if isinstance(scores, (list, tuple)) else scores
    if hasattr(logits, "ndim") and logits.ndim > 1:
        logits = logits[0]
    yes_logit = float(logits[yes_id].detach().float().cpu().item())
    no_logit = float(logits[no_id].detach().float().cpu().item())
    peak = max(yes_logit, no_logit)
    yes_exp = math.exp(yes_logit - peak)
    no_exp = math.exp(no_logit - peak)
    total = yes_exp + no_exp
    return yes_exp / total, no_exp / total


def ask(image: Any, question: str, state: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "Inspect only the supplied image. Do not use outside knowledge or infer hidden objects. "
        f"{question} Respond directly with exactly yes or no."
    )
    with state["torch"].inference_mode():
        # Verifier decisions must be reproducible. Sampling can emit a token
        # that disagrees with the highest yes/no first-token logit.
        answer, scores = state["model"].generate_content([image, prompt], do_sample=False)
    yes_probability, no_probability = binary_probabilities(scores, state["yes_id"], state["no_id"])
    return {
        "answer": str(answer).strip().lower(),
        "yes_probability": yes_probability,
        "no_probability": no_probability,
        "margin": abs(yes_probability - no_probability),
        "question": question,
    }


def semantic_checks(payload: dict[str, Any], constraint: dict[str, Any]) -> list[dict[str, str]]:
    ctype = str(constraint.get("constraint_type") or "").strip().lower()
    checks: list[dict[str, str]] = []
    if ctype == "attribute":
        object_inventory = object_inventory_for_payload(payload)
        for row in metadata_list(constraint, "attribute_bindings"):
            if isinstance(row, dict) and row.get("object") and row.get("attribute"):
                obj, attribute = str(row["object"]), str(row["attribute"])
                checks.append({
                    "kind": "attribute",
                    "subject": obj,
                    "target": attribute,
                    "expected_answer": "yes",
                    "question": f"Is the visible {obj} itself clearly {attribute}?",
                })
                for other in object_inventory:
                    if other.lower() == obj.lower():
                        continue
                    checks.append({
                        "kind": "attribute_exclusivity",
                        "subject": other,
                        "target": attribute,
                        "expected_answer": "no",
                        "question": f"Is the visible {other} clearly {attribute}?",
                    })
    elif ctype == "action":
        for row in metadata_list(constraint, "action_relations"):
            if isinstance(row, dict) and row.get("subject") and row.get("action") and row.get("object"):
                subject, action, obj = str(row["subject"]), str(row["action"]).replace("_", " "), str(row["object"])
                checks.append({
                    "kind": "action",
                    "probe": "relation",
                    "decision": True,
                    "subject": subject,
                    "target": f"{action} {obj}",
                    "question": f"Are the {subject} and {obj} both visible, with the {subject} clearly {action} the {obj}?",
                })
                checks.extend([
                    {
                        "kind": "action_diagnostic", "probe": "object_missing", "decision": False,
                        "subject": subject, "target": obj,
                        "question": f"Are both the {subject} and the {obj} clearly visible and independently recognizable?",
                    },
                    {
                        "kind": "action_diagnostic", "probe": "static_pose", "decision": False,
                        "subject": subject, "target": obj,
                        "question": f"Does the {subject}'s pose clearly show an active, dynamic interaction with the {obj}, rather than a static lineup?",
                    },
                    {
                        "kind": "action_diagnostic", "probe": "missing_contact", "decision": False,
                        "subject": subject, "target": obj,
                        "question": f"Is there clear visible contact or a direct interaction cue between the {subject} and the {obj}?",
                    },
                    {
                        "kind": "action_diagnostic", "probe": "wrong_direction", "decision": False,
                        "subject": subject, "target": obj,
                        "question": f"Do pose, orientation, and motion cues point from the {subject} toward the {obj} in a way consistent with {action}?",
                    },
                ])
    elif ctype == "spatial":
        for row in metadata_list(constraint, "spatial_relations"):
            relation = str(row.get("relation") or "").strip().lower() if isinstance(row, dict) else ""
            if isinstance(row, dict) and relation in DEPTH_RELATIONS and row.get("subject") and row.get("object"):
                subject, obj = str(row["subject"]), str(row["object"])
                phrase = "in front of and closer to the camera than" if relation.replace("_", " ") == "in front of" else "behind and farther from the camera than"
                checks.extend([
                    {
                        "kind": "depth",
                        "subject": subject,
                        "target": f"{relation} {obj}",
                        "question": f"Is the {subject} visibly {phrase} the {obj}?",
                    },
                    {
                        "kind": "role_distinctness",
                        "subject": subject,
                        "target": obj,
                        "question": (
                            f"Are there exactly two distinct semantic roles: one recognizable {subject} and one recognizable {obj}, "
                            f"with the {obj} not merely another {subject}, miniature {subject}, reflection, or depiction?"
                        ),
                    },
                ])
    elif ctype == "prompt_alignment":
        prompt = prompt_for_payload(payload)
        if prompt:
            checks.append({
                "kind": "prompt_alignment",
                "subject": "image",
                "target": prompt,
                "question": f"Does the image clearly and consistently satisfy this description: {prompt}",
            })
        seen_phrases: set[str] = set()
        for phrase in metadata_list(constraint, "prompt_noun_phrases"):
            value = " ".join(str(phrase or "").strip().split())
            if not value or value.lower() in seen_phrases:
                continue
            seen_phrases.add(value.lower())
            checks.append({
                "kind": "prompt_noun_phrase_coverage",
                "subject": "image",
                "target": value,
                "question": (
                    f"Is the visual element or part described by '{value}' clearly visible "
                    "and large enough to inspect in the image?"
                ),
            })
    elif ctype in {
        "identity",
        "style",
        "layout",
        "negative",
        "text",
        "quality",
        "aesthetic",
        "material",
    }:
        text = " ".join(str(constraint.get("text") or "").split())
        if text:
            polarity = "does not contain" if ctype == "negative" else "satisfies"
            checks.append({
                "kind": ctype,
                "subject": "image",
                "target": text,
                "question": f"Does the image clearly {polarity} this visual requirement: {text}?",
            })
    return checks


def object_inventory_for_payload(payload: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    objects: list[str] = []
    for constraint in constraints_for_verification(payload):
        for row in metadata_list(constraint, "required_objects"):
            if not isinstance(row, dict):
                continue
            value = " ".join(str(row.get("class") or "").split())
            if value and value.lower() not in seen:
                seen.add(value.lower())
                objects.append(value)
        for row in metadata_list(constraint, "attribute_bindings"):
            if not isinstance(row, dict):
                continue
            value = " ".join(str(row.get("object") or "").split())
            if value and value.lower() not in seen:
                seen.add(value.lower())
                objects.append(value)
    return objects


def aggregate_result(checks: list[dict[str, Any]], constraint_type: str, pass_threshold: float, reject_threshold: float) -> dict[str, Any]:
    if not checks:
        return {
            "passed": True,
            "symptom": "pass",
            "confidence": 1.0,
            "not_applicable": True,
            "evidence_status": "unknown",
            "reason": "no semantic checks supported by NVILA for this constraint",
            "inspection_source": "nvila_lite_2b_probability_verifier",
        }
    decision_checks = [row for row in checks if bool(row.get("decision", True))] or checks
    yes_min = min(float(row["yes_probability"]) for row in decision_checks)
    no_max = max(float(row["no_probability"]) for row in decision_checks)
    required_min = min(_required_probability(row) for row in decision_checks)
    rejection_max = max(_rejection_probability(row) for row in decision_checks)
    confirmed = required_min >= pass_threshold
    rejected = rejection_max >= reject_threshold
    diagnostic_failures = [
        str(row.get("probe")) for row in checks
        if not bool(row.get("decision", True))
        and _rejection_probability(row) >= reject_threshold
        and row.get("probe")
    ]
    if confirmed:
        evidence_status, passed, symptom = "confirmed", True, "pass"
    elif rejected:
        evidence_status, passed = "mismatch", False
        symptom = {
            "attribute": "attribute_binding_mismatch",
            "action": "action_relation_mismatch",
            "spatial": "spatial_relation_mismatch",
            "prompt_alignment": "low_prompt_alignment",
        }.get(constraint_type, "constraint_failed")
    else:
        evidence_status, passed, symptom = "unknown", False, "missing_inspection"
    # Keep the real probability for candidate ranking. Capping every unknown
    # result at 0.49 erased the ordering signal that NVILA is meant to supply.
    score = required_min
    hard_evidence_required = constraint_type in {"action", "spatial"}
    return {
        "passed": passed,
        "symptom": symptom,
        # Keep actionable diagnoses near the front of the ordered result so
        # compact experience traces retain them for weakness mining/patch
        # prioritization instead of truncating them with verbose raw probes.
        "failure_subtypes": diagnostic_failures,
        "confidence": score,
        "selection_score": score,
        "constraint_score": score,
        "hard_evidence_required": hard_evidence_required,
        "evidence_status": evidence_status,
        "decision_state": "accepted" if confirmed else ("rejected" if rejected else "abstain"),
        "abstained": evidence_status == "unknown",
        "inspection_source": "nvila_lite_2b_probability_verifier",
        "yes_probability_min": required_min,
        "no_probability_max": rejection_max,
        "semantic_checks": checks,
        "required_probability_min": required_min,
        "rejection_probability_max": rejection_max,
        "raw_yes_probability_min": yes_min,
        "raw_no_probability_max": no_max,
        "pass_threshold": pass_threshold,
        "reject_threshold": reject_threshold,
        "reason": f"{len(checks)} atomic NVILA yes/no probability checks",
    }


def _expected_answer(row: dict[str, Any]) -> str:
    expected = str(row.get("expected_answer") or "yes").strip().lower()
    if expected not in {"yes", "no"}:
        return "yes"
    return expected


def _required_probability(row: dict[str, Any]) -> float:
    if _expected_answer(row) == "no":
        return float(row.get("no_probability", 0.0))
    return float(row.get("yes_probability", 0.0))


def _rejection_probability(row: dict[str, Any]) -> float:
    if _expected_answer(row) == "no":
        return float(row.get("yes_probability", 0.0))
    return float(row.get("no_probability", 0.0))


def handle_payload(payload: dict[str, Any], args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(materialize_image(generation_for_payload(payload))).convert("RGB")
    results: dict[str, dict[str, Any]] = {}
    for index, constraint in enumerate(constraints_for_verification(payload)):
        cid = str(constraint.get("constraint_id") or f"constraint_{index}")
        ctype = str(constraint.get("constraint_type") or "").strip().lower()
        if ctype not in SUPPORTED_TYPES:
            results[cid] = aggregate_result([], ctype, args.pass_threshold, args.reject_threshold)
            continue
        specifications = semantic_checks(payload, constraint)
        observations = [{**spec, **ask(image, spec["question"], state)} for spec in specifications]
        results[cid] = aggregate_result(observations, ctype, args.pass_threshold, args.reject_threshold)
    return {
        "scorer": "nvila_lite_2b_probability_verifier",
        "model": args.model,
        "device": state["device"],
        "constraint_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pass-threshold", type=float, default=0.90)
    parser.add_argument("--reject-threshold", type=float, default=0.90)
    args = parser.parse_args()
    state = load_model(args.model, args.device)
    if args.check_model:
        print(json.dumps({"ok": True, "model": args.model, "device": state["device"]}))
        return
    if args.serve:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                result = handle_payload(json.loads(line), args, state)
            except Exception as exc:
                result = {"error": f"{exc.__class__.__name__}: {exc}", "constraint_results": {}}
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if not args.input or not args.output:
        parser.error("--input and --output are required unless --serve is used")
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    Path(args.output).write_text(json.dumps(handle_payload(payload, args, state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
