from __future__ import annotations

import csv
import hashlib
import json
import random
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.request import urlopen

from ..io import read_jsonl, write_json, write_jsonl


DEFAULT_PARTIPROMPTS_URL = "https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv"


def build_partiprompts_evolution_split(
    source: str,
    output_dir: str | Path,
    *,
    benchmark_tasks: Iterable[str | Path] = (),
    seed: int = 2027,
    target_count: int = 500,
    heldout_count: int = 100,
    preservation_count: int = 100,
) -> Dict[str, Any]:
    counts = (target_count, heldout_count, preservation_count)
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("split counts must be non-negative integers")
    if sum(counts) == 0:
        raise ValueError("at least one prompt must be requested")
    source_text = _read_text_source(source)
    source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    prompt_rows = _load_prompt_rows(source_text)
    deduped_rows = _dedupe_prompt_rows(prompt_rows)

    benchmark_paths = [Path(path) for path in benchmark_tasks]
    benchmark_prompt_keys = _load_benchmark_prompt_keys(benchmark_paths)
    filtered_rows = [
        row for row in deduped_rows
        if _normalize_prompt(row["prompt"]) not in benchmark_prompt_keys
    ]

    required = target_count + heldout_count + preservation_count
    if len(filtered_rows) < required:
        raise ValueError(
            f"not enough prompts after filtering: need {required}, found {len(filtered_rows)}"
        )

    rng = random.Random(seed)
    rows = list(filtered_rows)
    rng.shuffle(rows)
    selected = rows[:required]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "partiprompts_p2_target500.jsonl"
    heldout_path = out_dir / "partiprompts_p2_heldout100.jsonl"
    preservation_path = out_dir / "partiprompts_p2_preservation100.jsonl"
    manifest_path = out_dir / "partiprompts_p2_manifest.json"

    target_rows = _task_rows(selected[:target_count], split="target")
    heldout_rows = _task_rows(selected[target_count:target_count + heldout_count], split="heldout")
    preservation_rows = _task_rows(selected[target_count + heldout_count:], split="preservation")
    write_jsonl(target_path, target_rows)
    write_jsonl(heldout_path, heldout_rows)
    write_jsonl(preservation_path, preservation_rows)

    manifest = {
        "schema": "gen_harness.partiprompts_p2_evolution_split.v1",
        "source": source,
        "source_sha256": source_sha256,
        "seed": seed,
        "target_count": len(target_rows),
        "heldout_count": len(heldout_rows),
        "preservation_count": len(preservation_rows),
        "source_prompt_count": len(prompt_rows),
        "deduped_prompt_count": len(deduped_rows),
        "filtered_prompt_count": len(filtered_rows),
        "benchmark_overlap_removed": len(deduped_rows) - len(filtered_rows),
        "benchmark_task_files": [str(path) for path in benchmark_paths],
        "filter_rule": "normalized prompt exact match against provided benchmark task prompts",
        "outputs": {
            "target": str(target_path),
            "heldout": str(heldout_path),
            "preservation": str(preservation_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def _read_text_source(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        with urlopen(source, timeout=60) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def _load_prompt_rows(source_text: str) -> List[Dict[str, str]]:
    sample = source_text.lstrip()[:2048]
    if sample.startswith("[") or sample.startswith("{"):
        try:
            data = json.loads(source_text)
        except json.JSONDecodeError:
            # JSONL contains one complete object per non-empty line.
            rows = [json.loads(line) for line in source_text.splitlines() if line.strip()]
        else:
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = next((data[key] for key in ("tasks", "data", "prompts") if key in data), [data])
            else:
                rows = None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("JSON source must contain a list of prompt objects")
        return [_prompt_row_from_mapping(row, index) for index, row in enumerate(rows)]

    lines = [line for line in source_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty PartiPrompts source")
    if "\t" in lines[0]:
        reader = csv.DictReader(lines, delimiter="\t")
        return [_prompt_row_from_mapping(row, index) for index, row in enumerate(reader)]
    return [_prompt_row_from_mapping(json.loads(line), index) for index, line in enumerate(lines)]


def _prompt_row_from_mapping(row: Dict[str, Any], index: int) -> Dict[str, str]:
    prompt = str(row.get("Prompt") or row.get("prompt") or row.get("text") or row.get("caption") or "").strip()
    if not prompt:
        raise ValueError(f"source row {index} has no prompt")
    return {
        "source_index": str(index),
        "prompt": prompt,
        "category": str(row.get("Category") or row.get("category") or "").strip(),
        "challenge": str(row.get("Challenge") or row.get("challenge") or "").strip(),
        "note": str(row.get("Note") or row.get("note") or "").strip(),
    }


def _dedupe_prompt_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for row in rows:
        key = _normalize_prompt(row["prompt"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _load_benchmark_prompt_keys(paths: Iterable[Path]) -> set[str]:
    keys = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"benchmark task file does not exist: {path}")
        for row in read_jsonl(path):
            prompt = str(row.get("prompt") or "").strip()
            if prompt:
                keys.add(_normalize_prompt(prompt))
    return keys


def _normalize_prompt(prompt: str) -> str:
    table = str.maketrans({char: " " for char in string.punctuation})
    return " ".join(prompt.lower().translate(table).split())


def _task_rows(rows: List[Dict[str, str]], *, split: str) -> List[Dict[str, Any]]:
    return [
        {
            "task_id": f"partiprompts_p2_{split}_{index:04d}",
            "prompt": row["prompt"],
            "task_family": "general_t2i",
            "references": {},
            "constraints": [],
            "metadata": {
                "source": "PartiPrompts",
                "stream": "P2",
                "split": split,
                "source_index": int(row["source_index"]),
                "category": row["category"],
                "challenge": row["challenge"],
                "note": row["note"],
                "generation_input": "prompt_only_no_benchmark_labels",
            },
        }
        for index, row in enumerate(rows)
    ]
