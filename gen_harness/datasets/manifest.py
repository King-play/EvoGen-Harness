from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_TASK_FIELDS = {"task_id", "prompt", "task_family", "constraints"}
REQUIRED_CONSTRAINT_FIELDS = {"constraint_id", "text", "constraint_type"}


def load_manifest_rows(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data["tasks"]
    for row in rows:
        validate_task_row(row)
    return rows


def validate_task_row(row: Dict[str, Any]) -> None:
    missing = REQUIRED_TASK_FIELDS - set(row)
    if missing:
        raise ValueError(f"task missing fields {missing}: {row.get('task_id')}")
    if not isinstance(row["constraints"], list) or not row["constraints"]:
        raise ValueError(f"task has no constraints: {row.get('task_id')}")
    for c in row["constraints"]:
        cmissing = REQUIRED_CONSTRAINT_FIELDS - set(c)
        if cmissing:
            raise ValueError(f"constraint missing fields {cmissing}: {row.get('task_id')}")
    split = row.get("metadata", {}).get("split")
    if split not in {"target", "heldout", "preservation", None}:
        raise ValueError(f"invalid split {split}: {row.get('task_id')}")
