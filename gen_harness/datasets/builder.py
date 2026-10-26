from __future__ import annotations

import json
import random
from pathlib import Path

from .manifest import load_manifest_rows, validate_task_row


class DatasetBuilder:
    """Builds Gen-Harness task JSONL files from real manifests.

    This builder validates supplied task semantics and
    repartitions user-provided real task rows into target/heldout/preservation
    splits used by patch validation.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def build_splits(
        self,
        input_manifest: str | Path,
        output_jsonl: str | Path,
        target_ratio: float = 0.5,
        heldout_ratio: float = 0.25,
        preservation_ratio: float = 0.25,
    ) -> None:
        rows = load_manifest_rows(input_manifest)
        if abs(target_ratio + heldout_ratio + preservation_ratio - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1")
        rng = random.Random(self.seed)
        rows = list(rows)
        rng.shuffle(rows)
        n = len(rows)
        n_target = int(n * target_ratio)
        n_heldout = int(n * heldout_ratio)
        for i, row in enumerate(rows):
            row = dict(row)
            meta = dict(row.get("metadata", {}))
            if i < n_target:
                meta["split"] = "target"
            elif i < n_target + n_heldout:
                meta["split"] = "heldout"
            else:
                meta["split"] = "preservation"
            row["metadata"] = meta
            validate_task_row(row)
            rows[i] = row
        out = Path(output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
