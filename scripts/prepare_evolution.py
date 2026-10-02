#!/usr/bin/env python3
"""Validate and combine the supplied P2 splits without re-sampling any prompts.

Run from the repository root. This helper never downloads a dataset or calls a
model. It checks the provenance and normalized exact-match scope recorded in
the supplied manifest; it does not claim an all-benchmark near-duplicate audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gen_harness.datasets.evolution import _normalize_prompt
from gen_harness.io import write_json, write_jsonl
from gen_harness.tasks import load_tasks
from gen_harness.trace_protocol import validate_paper_dataset

SPLITS = {"target": 500, "heldout": 100, "preservation": 100}


def validate_splits(root: Path = ROOT) -> tuple[list[dict], dict]:
    root = root.resolve()
    folder = root / "data/evolution"
    manifest_path = folder / "partiprompts_p2_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows: list[dict] = []
    fingerprints: dict[str, str] = {}
    prompt_keys: set[str] = set()
    task_ids: set[str] = set()
    source_indices: set[int] = set()
    for split, count in SPLITS.items():
        path = folder / f"partiprompts_p2_{split}{count}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != count or manifest.get(f"{split}_count") != count:
            raise ValueError(f"{split}: expected {count} rows in both file and manifest")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{split}: each JSONL record must be an object")
            meta = row.get("metadata", {})
            if meta.get("split") != split or meta.get("source") != "PartiPrompts" or meta.get("stream") != "P2":
                raise ValueError(f"{split}: invalid source/split metadata")
            if row.get("constraints") != [] or row.get("references") != {}:
                raise ValueError(f"{split}: expected prompt-only tasks with empty constraints/references")
            prompt = row.get("prompt")
            task_id = row.get("task_id")
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"{split}: empty prompt or task ID")
            key = _normalize_prompt(prompt)
            index = meta.get("source_index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError(f"{split}: invalid source_index")
            if key in prompt_keys or task_id in task_ids or index in source_indices:
                raise ValueError(f"{split}: repeated prompt, task ID, or source_index")
            forbidden = {"label", "labels", "score", "scores", "official_score", "benchmark_labels"}
            if forbidden & (set(row) | set(meta)):
                raise ValueError(f"{split}: evaluation-only metadata found")
            prompt_keys.add(key); task_ids.add(task_id); source_indices.add(index)
        fingerprints[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        all_rows.extend(rows)
    inventory_path = folder / "checksums.json"
    if inventory_path.exists():
        expected = json.loads(inventory_path.read_text(encoding="utf-8"))["files"]
        if fingerprints != expected:
            raise ValueError("evolution split bytes differ from the supplied release checksum inventory")
    benchmark_checks = []
    for relative in manifest.get("benchmark_task_files", []):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("benchmark manifest paths must stay inside the repository")
        benchmark_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        overlaps = {_normalize_prompt(row["prompt"]) for row in benchmark_rows} & prompt_keys
        if overlaps:
            raise ValueError(f"normalized exact overlap found against {relative}")
        benchmark_checks.append({"path": relative, "num_prompts": len(benchmark_rows), "normalized_exact_overlaps": 0})
    report = {
        "schema": "evogen.release_split_check.v1", "passed": True,
        "counts": SPLITS, "total": len(all_rows), "unique_task_ids": len(task_ids),
        "pairwise_disjoint_normalized_prompts": True, "unique_source_indices": len(source_indices),
        "files": fingerprints, "source_manifest": "data/evolution/partiprompts_p2_manifest.json",
        "manifest_filter_rule": manifest.get("filter_rule"), "benchmark_checks": benchmark_checks,
        "scope": "Supplied bytes and recorded normalized exact-match coverage; not a claim of historical-run identity or semantic near-duplicate filtering.",
    }
    return all_rows, report


def prepare(output: Path, *, root: Path = ROOT, force: bool = False) -> dict:
    rows, report = validate_splits(root)
    output = output.resolve()
    if output.is_relative_to((root / "data/evolution").resolve()):
        raise ValueError("write the combined working file outside the supplied data/evolution directory")
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if output.exists() and output.read_text(encoding="utf-8") != encoded and not force:
        raise FileExistsError("output exists with different contents; choose a new path or pass --force")
    write_jsonl(output, rows)
    protocol = validate_paper_dataset(load_tasks(output), protocol="paper_p2")
    report["protocol_validation"] = protocol
    report["combined_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    # Keep reports portable: output filename, not the user's absolute path.
    report["combined_file"] = output.name
    write_json(output.with_suffix(".manifest.json"), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/quickstart/evolution_tasks.jsonl"))
    parser.add_argument("--force", action="store_true", help="replace a different existing combined output")
    args = parser.parse_args()
    try:
        report = prepare(args.output, force=args.force)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
