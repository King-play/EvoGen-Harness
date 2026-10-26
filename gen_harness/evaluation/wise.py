from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..io import read_jsonl, write_json, write_jsonl


PROMPT_KEYS = ("prompt", "text", "caption")
ID_KEYS = ("id", "prompt_id", "sample_id", "index")
DOMAIN_KEYS = ("domain", "category")


def build_wise_tasks(
    benchmark_data: str | Path,
    output_path: str | Path,
    *,
    split: str = "heldout",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert WISE prompt records into label-free Gen-Harness tasks.

    Only prompt identity and the public domain are retained.  WiScore labels,
    questions, reference answers, and evaluator fields stay in the separate
    official benchmark file and are never exposed to generation or TRACE.
    """
    rows = _read_benchmark_rows(benchmark_data)
    if limit is not None:
        rows = rows[:limit]
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        prompt = _first_text(row, PROMPT_KEYS)
        if not prompt:
            raise ValueError(f"WISE row {index} has no prompt/text/caption")
        public_id = _first_text(row, ID_KEYS) or str(index)
        if public_id in seen_ids:
            raise ValueError(f"WISE public id is not unique: {public_id}")
        seen_ids.add(public_id)
        domain = _first_text(row, DOMAIN_KEYS)
        tasks.append(
            {
                "task_id": f"wise_{index:05d}",
                "prompt": prompt,
                "task_family": "world_knowledge",
                "references": {},
                "constraints": [],
                "metadata": {
                    "split": split,
                    "benchmark": "wise",
                    "wise_index": index,
                    "wise_id": public_id,
                    "wise_domain": domain,
                    "generation_input": "prompt_only_no_evaluator_labels",
                },
            }
        )
    write_jsonl(output_path, tasks)
    return tasks


def export_wise_manifest(
    tasks_jsonl: str | Path,
    experiences_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    seed: int | None = None,
    require_all: bool = True,
) -> list[dict[str, Any]]:
    """Create a prompt/image manifest for post-generation official WiScore.

    This adapter does not implement or invoke WiScore.  Keeping the official
    evaluator outside the harness prevents benchmark feedback leakage.
    """
    tasks = read_jsonl(tasks_jsonl)
    expected: dict[str, dict[str, Any]] = {}
    for row in tasks:
        task_id = str(row.get("task_id") or "")
        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        if not task_id or meta.get("benchmark") != "wise":
            continue
        expected[task_id] = row
    if not expected:
        raise ValueError("WISE task file contains no benchmark=wise tasks")

    selected: dict[str, str] = {}
    for row in read_jsonl(experiences_jsonl):
        task_id = str(row.get("task_id") or "")
        if task_id not in expected or task_id in selected:
            continue
        row_seed = _row_seed(row)
        if seed is not None and row_seed != seed:
            continue
        image_path = _row_image_path(row)
        if not image_path:
            continue
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing generated WISE image for {task_id}: {path}")
        selected[task_id] = str(path.resolve())

    missing = [task_id for task_id in expected if task_id not in selected]
    if missing and require_all:
        raise ValueError(f"missing generated images for {len(missing)} WISE prompts: {', '.join(missing[:20])}")

    manifest_rows: list[dict[str, Any]] = []
    for task_id, task in expected.items():
        if task_id not in selected:
            continue
        meta = task.get("metadata", {})
        manifest_rows.append(
            {
                "id": meta.get("wise_id"),
                "index": meta.get("wise_index"),
                "domain": meta.get("wise_domain"),
                "prompt": task.get("prompt"),
                "image_path": selected[task_id],
            }
        )
    write_jsonl(output_jsonl, manifest_rows)
    out = Path(output_jsonl)
    write_json(
        out.with_suffix(out.suffix + ".manifest.json"),
        {
            "schema": "gen_harness.wise_export.v1",
            "tasks_jsonl": str(tasks_jsonl),
            "experiences_jsonl": str(experiences_jsonl),
            "output_jsonl": str(out),
            "seed": seed,
            "require_all": require_all,
            "num_benchmark_prompts": len(expected),
            "num_mapped_prompts": len(manifest_rows),
            "missing_task_ids": missing,
            "official_evaluator_used_during_generation": False,
            "official_scores_used_for_selection": False,
        },
    )
    return manifest_rows


def _read_benchmark_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return read_jsonl(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next((payload[key] for key in ("data", "prompts", "items", "samples") if isinstance(payload.get(key), list)), None)
        if rows is None:
            raise ValueError("WISE JSON must be a list or contain data/prompts/items/samples")
    else:
        raise ValueError("WISE benchmark data must be JSON or JSONL objects")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("WISE benchmark rows must be JSON objects")
    return list(rows)


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _row_seed(row: dict[str, Any]) -> int | None:
    visual = row.get("visual_result", {}) if isinstance(row.get("visual_result"), dict) else {}
    value = visual.get("seed", row.get("seed"))
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_image_path(row: dict[str, Any]) -> str:
    visual = row.get("visual_result", {}) if isinstance(row.get("visual_result"), dict) else {}
    raw = visual.get("raw_generation", {}) if isinstance(visual.get("raw_generation"), dict) else {}
    for value in (raw.get("image_path"), raw.get("image_uri"), visual.get("image_uri"), row.get("image_uri")):
        if isinstance(value, str) and value and not value.startswith("data:"):
            return value
    return ""
