from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .io import read_jsonl
from .schema import GenerationTask


def load_tasks(path: str | Path) -> List[GenerationTask]:
    return [GenerationTask.from_dict(row) for row in read_jsonl(path)]


def split_tasks(tasks: Iterable[GenerationTask]) -> dict[str, List[GenerationTask]]:
    out = {"target": [], "heldout": [], "preservation": []}
    for task in tasks:
        split = task.metadata.get("split", "target")
        out.setdefault(split, []).append(task)
    return out
