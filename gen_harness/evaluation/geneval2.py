from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import read_jsonl, write_json, write_jsonl


def build_geneval2_tasks(
    benchmark_data: str | Path,
    output_path: str | Path,
    *,
    split: str = "heldout",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Convert GenEval2 prompt rows into prompt-only Gen-Harness tasks.

    GenEval2 rows contain evaluator VQA questions and answers. Those labels are
    deliberately not copied into task metadata because task metadata is included
    in Gen-Harness program traces. The official benchmark file should be passed
    separately to the official Soft-TIFA evaluator after image generation.
    """

    rows = read_jsonl(benchmark_data)
    if limit is not None:
        rows = rows[:limit]
    tasks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"GenEval2 row {idx} has no prompt")
        tasks.append(
            {
                "task_id": f"geneval2_{idx:04d}",
                "prompt": prompt,
                "task_family": "compositional",
                "references": {},
                "constraints": [],
                "metadata": {
                    "split": split,
                    "benchmark": "geneval2",
                    "geneval2_index": idx,
                    "geneval2_prompt": prompt,
                    "generation_input": "prompt_only_no_vqa_labels",
                },
            }
        )
    write_jsonl(output_path, tasks)
    return tasks


def export_geneval2_image_map(
    experiences_jsonl: str | Path,
    benchmark_data: str | Path,
    output_json: str | Path,
    *,
    seed: int | None = None,
    require_all: bool = True,
    prompts_file: str | Path | None = None,
) -> dict[str, str]:
    """Export GenHarness experiences to GenEval2's prompt->image path JSON.

    The official GenEval2 evaluator expects a JSON object keyed by exact prompt.
    `experiences.jsonl` contains one row per constraint, so this function
    deduplicates by GenEval2 index and seed while checking that each mapped
    image exists.
    """

    benchmark_rows = read_jsonl(benchmark_data)
    prompts = [str(row.get("prompt") or "").strip() for row in benchmark_rows]
    if any(not prompt for prompt in prompts):
        raise ValueError("GenEval2 benchmark data contains an empty prompt")
    if len(prompts) != len(set(prompts)):
        raise ValueError("GenEval2 prompt keys are not unique")

    if prompts_file is None:
        expected_prompts = prompts
        expected_indices = list(range(len(prompts)))
        prompt_source = "benchmark_data"
    else:
        expected_prompts = _load_prompt_subset(prompts_file)
        prompt_to_index = {prompt: idx for idx, prompt in enumerate(prompts)}
        unknown = [prompt for prompt in expected_prompts if prompt not in prompt_to_index]
        if unknown:
            preview = ", ".join(unknown[:5])
            raise ValueError(f"prompt subset contains {len(unknown)} prompt(s) not found in GenEval2 benchmark data: {preview}")
        expected_indices = [prompt_to_index[prompt] for prompt in expected_prompts]
        prompt_source = str(prompts_file)

    selected: dict[int, tuple[int | None, str]] = {}
    prompt_to_index = {prompt: idx for idx, prompt in enumerate(prompts)}
    for row in read_jsonl(experiences_jsonl):
        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        if meta.get("benchmark") != "geneval2" and isinstance(row.get("export_metadata"), dict):
            meta = row["export_metadata"]
        task_id = str(row.get("task_id") or "")
        prefix, _, suffix = task_id.partition("_")
        if meta.get("benchmark") == "geneval2" and not isinstance(meta.get("geneval2_index"), int):
            if prefix == "geneval2" and suffix.isdigit():
                meta = {**meta, "geneval2_index": int(suffix)}
        if meta.get("benchmark") != "geneval2":
            # Prompt-only generation rows created before benchmark export
            # metadata was propagated still carry the stable public task id.
            # Recover only the explicit GenEval2 id shape; never infer from
            # evaluator labels or image content.
            if prefix == "geneval2" and suffix.isdigit():
                meta = {"benchmark": "geneval2", "geneval2_index": int(suffix)}
        if meta.get("benchmark") != "geneval2":
            continue
        prompt = str(meta.get("geneval2_prompt") or "").strip()
        index = prompt_to_index.get(prompt) if prompt else meta.get("geneval2_index")
        if not isinstance(index, int):
            continue
        if index < 0 or index >= len(prompts):
            raise ValueError(f"GenEval2 index out of range: {index}")
        row_seed = _row_seed(row)
        if seed is not None and row_seed != seed:
            continue
        image_path = _row_image_path(row)
        if not image_path:
            continue
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing generated image for GenEval2 index {index}: {path}")
        if index not in selected:
            selected[index] = (row_seed, str(path.resolve()))

    missing = [idx for idx in expected_indices if idx not in selected]
    if missing and require_all:
        preview = ", ".join(str(idx) for idx in missing[:20])
        raise ValueError(f"missing generated images for {len(missing)} GenEval2 prompts; first missing indices: {preview}")

    image_map = {prompts[idx]: selected[idx][1] for idx in expected_indices if idx in selected}
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, image_map)
    manifest = {
        "schema": "gen_harness.geneval2_image_map.v1",
        "benchmark_data": str(benchmark_data),
        "prompt_source": prompt_source,
        "full_benchmark_prompts": len(prompts),
        "experiences_jsonl": str(experiences_jsonl),
        "output_json": str(out),
        "seed": seed,
        "require_all": require_all,
        "num_benchmark_prompts": len(expected_prompts),
        "num_mapped_prompts": len(image_map),
        "missing_indices": missing,
        "notes": [
            "The map is for official GenEval2 Soft-TIFA scoring after generation.",
            "Official VQA labels are not used by Gen-Harness generation or internal selection.",
        ],
    }
    write_json(out.with_suffix(out.suffix + ".manifest.json"), manifest)
    return image_map


def _load_prompt_subset(path: str | Path) -> list[str]:
    rows = read_jsonl(path)
    prompts = [str(row.get("prompt") or "").strip() for row in rows if isinstance(row, dict)]
    if any(not prompt for prompt in prompts):
        raise ValueError(f"prompt subset contains an empty prompt: {path}")
    if len(prompts) != len(set(prompts)):
        raise ValueError(f"prompt subset contains duplicate prompts: {path}")
    return prompts


def _row_seed(row: dict[str, Any]) -> int | None:
    value = row.get("visual_result", {}).get("seed") if isinstance(row.get("visual_result"), dict) else row.get("seed")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
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
