from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from ..io import write_json, write_jsonl
from .manifest import validate_task_row


OBJECTS = [
    "bench",
    "bottle",
    "book",
    "chair",
    "clock",
    "cup",
    "dog",
    "kite",
    "lamp",
    "mug",
    "pizza",
    "toothbrush",
]
COUNT_OBJECTS = ["apple", "book", "bottle", "chair", "cup", "flower", "mug", "toy car"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
VISUAL_ATTRIBUTES = COLORS + ["wooden", "metal", "plastic", "stone", "spotted", "sparkling"]
RELATIONS = ["left of", "right of", "above", "below", "in front of", "behind"]
ACTIONS = ["playing with", "chasing", "jumping over"]
ACTION_SUBJECTS = ["dog", "cat", "rabbit", "turtle", "bird", "horse", "monkey", "penguin"]
ACTION_TARGETS = ["ball", "toy robot", "box", "hoop", "stuffed bear", "traffic cone", "small cart", "wooden block"]


def build_compositional_suite(
    output_path: str | Path,
    *,
    seed: int = 0,
    target_per_family: int = 8,
    heldout_per_family: int = 8,
    preservation_per_family: int = 4,
    include_action: bool = False,
) -> Dict[str, Any]:
    """Build a deterministic non-benchmark compositional validation suite.

    The suite is intended for Gen-Harness method development before final
    benchmark reporting. It intentionally avoids GenEval prompts, indices,
    official labels, and evaluator outputs.
    """

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for split, per_family in (
        ("target", target_per_family),
        ("heldout", heldout_per_family),
        ("preservation", preservation_per_family),
    ):
        rows.extend(_count_tasks(rng, split, per_family))
        rows.extend(_spatial_tasks(rng, split, per_family))
        rows.extend(_attribute_tasks(rng, split, per_family))
        if include_action:
            rows.extend(_action_tasks(rng, split, per_family))

    for row in rows:
        validate_task_row(row)
        text = str(row)
        if "geneval" in text.lower():
            raise ValueError(f"synthetic suite must not contain GenEval metadata: {row['task_id']}")

    output = Path(output_path)
    write_jsonl(output, rows)
    manifest = {
        "suite": "synthetic_compositional_contracts",
        "purpose": "pre-registered non-GenEval development/heldout/preservation gate",
        "seed": seed,
        "output": str(output),
        "num_tasks": len(rows),
        "splits": _split_counts(rows),
        "families": _family_counts(rows),
        "leakage_controls": [
            "No GenEval prompt metadata is read.",
            "No official evaluator outputs are read.",
            "Rows do not contain GenEval ids, tags, or benchmark metadata.",
            "Tasks are generated from generic object, color, count, spatial, and optional action templates.",
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json") if output.suffix else output.with_name(output.name + ".manifest.json")
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _count_tasks(rng: random.Random, split: str, n: int) -> List[Dict[str, Any]]:
    rows = []
    for idx in range(n):
        name = rng.choice(COUNT_OBJECTS)
        count = 2 + (idx % 6)
        required_objects = [{"class": name, "count": count}]
        if idx % 2:
            secondary = rng.choice([candidate for candidate in OBJECTS if candidate != name])
            required_objects.append({"class": secondary, "count": 1})
            prompt = f"a clean studio photo of {_count_phrase(count, name)} and one {secondary} on a plain surface"
        else:
            prompt = f"a clean studio photo of {_count_phrase(count, name)} on a plain surface"
        rows.append(_task(
            split,
            "count",
            idx,
            prompt,
            [{
                "constraint_id": "object_count_0",
                "text": f"include exactly {_count_phrase(count, name)}",
                "constraint_type": "count",
                "target": "object_count",
                "verification_rule": "internal object requirement inspection",
                "metadata": {
                    "constraint_schema": "object_requirements.v1",
                    "count_mode": "exact",
                    "required_objects": required_objects,
                    "source": "synthetic_compositional_suite",
                },
            }],
        ))
    return rows


def _spatial_tasks(rng: random.Random, split: str, n: int) -> List[Dict[str, Any]]:
    rows = []
    for idx in range(n):
        subject, target = rng.sample(OBJECTS, 2)
        relation = RELATIONS[idx % len(RELATIONS)]
        prompt = f"a simple photo of a {subject} {relation} a {target} on a plain background"
        rows.append(_task(
            split,
            "spatial",
            idx,
            prompt,
            [{
                "constraint_id": "spatial_relation_0",
                "text": f"{subject} must be clearly {relation} {target}",
                "constraint_type": "spatial",
                "target": "spatial_relation",
                "verification_rule": "internal spatial relation inspection",
                "metadata": {
                    "constraint_schema": "spatial_relations.v1",
                    "count_mode": "exact",
                    "required_objects": [{"class": target, "count": 1}, {"class": subject, "count": 1}],
                    "spatial_relations": [{"subject": subject, "relation": relation, "object": target}],
                    "source": "synthetic_compositional_suite",
                },
            }],
        ))
    return rows


def _attribute_tasks(rng: random.Random, split: str, n: int) -> List[Dict[str, Any]]:
    rows = []
    for idx in range(n):
        obj1, obj2 = rng.sample(OBJECTS, 2)
        attribute1 = VISUAL_ATTRIBUTES[(2 * idx) % len(VISUAL_ATTRIBUTES)]
        attribute2 = rng.choice([value for value in VISUAL_ATTRIBUTES if value != attribute1])
        prompt = f"a simple photo of a {attribute1} {obj1} and a {attribute2} {obj2}"
        rows.append(_task(
            split,
            "attribute",
            idx,
            prompt,
            [{
                "constraint_id": "attribute_binding_0",
                "text": f"{obj1} is {attribute1}; {obj2} is {attribute2}",
                "constraint_type": "attribute",
                "target": "attribute_binding",
                "verification_rule": "internal attribute binding inspection",
                "metadata": {
                    "constraint_schema": "attribute_bindings.v1",
                    "count_mode": "exact",
                    "required_objects": [{"class": obj1, "count": 1}, {"class": obj2, "count": 1}],
                    "attribute_bindings": [
                        {"object": obj1, "attribute": attribute1},
                        {"object": obj2, "attribute": attribute2},
                    ],
                    "source": "synthetic_compositional_suite",
                },
            }],
        ))
    return rows


def _action_tasks(rng: random.Random, split: str, n: int) -> List[Dict[str, Any]]:
    rows = []
    for idx in range(n):
        subject = rng.choice(ACTION_SUBJECTS)
        target = rng.choice(ACTION_TARGETS)
        action = rng.choice(ACTIONS)
        prompt = f"a simple photo of a {subject} visibly {action} a {target} in a clean open scene"
        rows.append(_task(
            split,
            "action",
            idx,
            prompt,
            [{
                "constraint_id": "action_relation_0",
                "text": f"{subject} must be visibly {action} {target}",
                "constraint_type": "action",
                "target": "action_relation",
                "verification_rule": "internal action relation inspection",
                "metadata": {
                    "constraint_schema": "action_relations.v1",
                    "count_mode": "exact",
                    "required_objects": [{"class": subject, "count": 1}, {"class": target, "count": 1}],
                    "action_relations": [{"subject": subject, "action": action.replace(" ", "_"), "object": target}],
                    "source": "synthetic_compositional_suite",
                },
            }],
        ))
    return rows


def _task(split: str, family: str, idx: int, prompt: str, constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "task_id": f"synthetic_{split}_{family}_{idx:03d}",
        "prompt": prompt,
        "task_family": "compositional",
        "references": {},
        "constraints": constraints,
        "metadata": {
            "split": split,
            "suite": "synthetic_compositional_contracts",
            "family": family,
            "source": "templated_non_benchmark",
        },
    }


def _count_phrase(count: int, name: str) -> str:
    if count == 1:
        return f"one {name}"
    if name.endswith("s"):
        return f"{count} {name}"
    return f"{count} {name}s"


def _split_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        split = str(row.get("metadata", {}).get("split", "target"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def _family_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        family = str(row.get("metadata", {}).get("family", "unknown"))
        counts[family] = counts.get(family, 0) + 1
    return counts
