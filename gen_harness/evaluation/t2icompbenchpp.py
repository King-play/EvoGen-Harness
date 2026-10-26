from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from ..io import read_jsonl, write_json, write_jsonl


T2ICOMPBENCHPP_CATEGORIES = (
    "color",
    "shape",
    "texture",
    "spatial",
    "3d_spatial",
    "numeracy",
    "non_spatial",
    "complex",
)

_COMPLEX_SUBTYPE_FILES = {
    "spatial": "complex_{split}_spatial.txt",
    "action": "complex_{split}_action.txt",
    "spatialaction": "complex_{split}_spatialaction.txt",
}


def build_t2icompbenchpp_tasks(
    dataset_dir: str | Path,
    output_path: str | Path,
    *,
    split: str = "val",
    categories: list[str] | None = None,
    limit_per_category: int | None = None,
) -> list[dict[str, Any]]:
    """Convert T2ICompBench++ prompt txt files into prompt-only tasks.

    The official txt files contain prompts only. The task metadata stores only
    prompt provenance and category identifiers; evaluator scores and labels are
    never copied into the generation task.
    """

    root = Path(dataset_dir)
    selected_categories = categories or list(T2ICOMPBENCHPP_CATEGORIES)
    unknown = [category for category in selected_categories if category not in T2ICOMPBENCHPP_CATEGORIES]
    if unknown:
        raise ValueError(f"unknown T2ICompBench++ categories: {unknown}")

    complex_subtypes = _load_complex_subtypes(root, split)
    tasks: list[dict[str, Any]] = []
    for category in selected_categories:
        path = root / f"{category}_{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        prompts = _read_prompt_lines(path)
        if limit_per_category is not None:
            prompts = prompts[:limit_per_category]
        for source_index, prompt in enumerate(prompts):
            task_id = f"t2icompbenchpp_{category}_{split}_{source_index:04d}"
            metadata = {
                "split": split,
                "benchmark": "t2icompbenchpp",
                "category": category,
                "source_index": source_index,
                "generation_input": "prompt_only_no_official_eval_outputs",
            }
            if category == "complex":
                metadata["complex_subtype"] = complex_subtypes.get(prompt, "mixed")
            tasks.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "task_family": "compositional",
                    "references": {},
                    "constraints": [],
                    "metadata": metadata,
                }
            )
    write_jsonl(output_path, tasks)
    return tasks


def build_t2icompbenchpp_smoke(
    tasks_jsonl: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    total: int = 100,
    seed: str = "t2icompbenchpp-smoke-v1",
    quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a deterministic prompt-only T2ICompBench++ smoke subset.

    The selector reads only public prompts plus category/subtype metadata. It
    does not read generated images, evaluator answers, model scores, official
    labels, or source indices.

    Selection is intentionally benchmark-output blind. Difficulty is estimated
    from prompt structure only, then hashed with category/subtype/prompt text to
    break ties. Source indices are kept only for provenance and export.
    """

    if total < 1:
        raise ValueError("total must be positive")
    rows = read_jsonl(tasks_jsonl)
    if not rows:
        raise ValueError(f"no tasks in {tasks_jsonl}")
    selected_quotas = quotas or {
        "color": 8,
        "shape": 8,
        "texture": 8,
        "spatial": 16,
        "3d_spatial": 14,
        "numeracy": 12,
        "non_spatial": 16,
        "complex": 18,
    }
    if sum(selected_quotas.values()) != total:
        raise ValueError(f"quotas sum to {sum(selected_quotas.values())}, expected total={total}")

    by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in T2ICOMPBENCHPP_CATEGORIES}
    scored: list[dict[str, Any]] = []
    for row in rows:
        category = str(row.get("metadata", {}).get("category", ""))
        if category not in by_category:
            continue
        profile = _prompt_profile(row)
        item = {**row, "metadata": {**row.get("metadata", {}), "smoke_profile": profile}}
        by_category[category].append(item)
        scored.append(item)

    selected: list[dict[str, Any]] = []
    for category, quota in selected_quotas.items():
        candidates = by_category.get(category, [])
        if len(candidates) < quota:
            raise ValueError(f"not enough {category} prompts: have {len(candidates)}, need {quota}")
        ranked = sorted(candidates, key=lambda row: _selection_key(row, seed), reverse=True)
        chosen = ranked[:quota]
        selected.extend(chosen)

    selected.sort(key=lambda row: (str(row["metadata"]["category"]), int(row["metadata"]["source_index"])))
    write_jsonl(output_path, selected)
    manifest = {
        "schema": "gen_harness.t2icompbenchpp_smoke_selection.v1",
        "source_tasks": str(Path(tasks_jsonl)),
        "output": str(Path(output_path)),
        "total": len(selected),
        "seed": seed,
        "quotas": selected_quotas,
        "category_counts": _count_categories(selected),
        "selection_inputs": ["prompt", "category", "complex_subtype"],
        "uses_official_eval_outputs": False,
        "generated_images_used": False,
        "no_replacement_after_generation": True,
        "difficulty_distribution": _count_profile(selected, "difficulty_band"),
        "items": [
            {
                "task_id": row["task_id"],
                "category": row["metadata"]["category"],
                "source_index": row["metadata"]["source_index"],
                "difficulty": row["metadata"]["smoke_profile"]["difficulty"],
                "difficulty_band": row["metadata"]["smoke_profile"]["difficulty_band"],
            }
            for row in selected
        ],
    }
    write_json(manifest_path, manifest)
    return manifest


def prepare_t2icompbenchpp_eval_manifest(
    tasks_jsonl: str | Path,
    experiences_jsonl: str | Path,
    manifest_dir: str | Path,
    sample_root: str | Path,
    *,
    strict_complete: bool = True,
    symlink: bool = True,
) -> dict[str, Any]:
    """Prepare category manifests and image layout for local official scoring."""

    tasks = read_jsonl(tasks_jsonl)
    task_by_id = {row["task_id"]: row for row in tasks}
    generations = {row["task_id"]: row for row in read_jsonl(experiences_jsonl)}
    if strict_complete:
        missing = [task_id for task_id in task_by_id if task_id not in generations]
        if missing:
            raise ValueError(f"missing {len(missing)} generations; first={missing[:5]}")

    manifests = Path(manifest_dir)
    samples = Path(sample_root)
    manifests.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)

    rows_by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in T2ICOMPBENCHPP_CATEGORIES}
    counters = {category: 0 for category in T2ICOMPBENCHPP_CATEGORIES}
    repo_root = Path.cwd().resolve()
    for task in tasks:
        task_id = task["task_id"]
        generation = generations.get(task_id)
        if generation is None:
            continue
        category = str(task.get("metadata", {}).get("category", ""))
        if category not in rows_by_category:
            raise ValueError(f"unknown task category for {task_id}: {category}")
        image_path = _row_image_path(generation)
        if not image_path:
            raise ValueError(f"generation has no image path for {task_id}")
        source_image = _absolute_path(image_path, repo_root)
        if not source_image.is_file():
            raise FileNotFoundError(source_image)

        local_index = counters[category]
        counters[category] += 1
        sample_path = (samples / category / f"{local_index:04d}.png").resolve()
        _place_sample(source_image, sample_path, symlink=symlink)
        rows_by_category[category].append(
            {
                "image_index": local_index,
                "repeat_index": 0,
                "category": category,
                "category_index": local_index,
                "source_index": int(task.get("metadata", {}).get("source_index", local_index)),
                "task_id": task_id,
                "original_prompt": task["prompt"],
                "image_path": str(sample_path),
                "source_image_path": str(source_image),
                "method": generation.get("method", "gen_harness"),
                "protocol": "T2I-CompBench++, 1 image per prompt",
            }
        )

    all_rows: list[dict[str, Any]] = []
    for category, rows in rows_by_category.items():
        path = manifests / f"{category}.jsonl"
        write_jsonl(path, rows)
        all_rows.extend(rows)
    write_jsonl(manifests / "all.jsonl", sorted(all_rows, key=lambda row: (row["category"], row["image_index"])))

    summary = {
        "schema": "gen_harness.t2icompbenchpp_eval_manifest.v1",
        "strict_complete": strict_complete,
        "total_rows": len(all_rows),
        "category_counts": {category: len(rows_by_category[category]) for category in T2ICOMPBENCHPP_CATEGORIES},
        "manifest_dir": str(manifests.resolve()),
        "sample_root": str(samples.resolve()),
        "notes": [
            "This manifest is for official-compatible T2ICompBench++ scoring after image generation.",
            "Official evaluator outputs are not used by Gen-Harness generation or internal selection.",
            "Generated images are exported only from the matching task row; no replacement image is permitted.",
        ],
    }
    write_json(manifests / "summary.json", summary)
    return summary


def _prompt_profile(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    category = str(row.get("metadata", {}).get("category", ""))
    words = _words(prompt)
    noun_like = _noun_like_terms(words)
    relation_count = sum(1 for relation in _RELATION_PHRASES if relation in prompt.lower())
    count_count = sum(1 for word in words if word in _NUMBER_WORDS)
    attribute_count = sum(1 for word in words if word in _ATTRIBUTE_WORDS)
    difficulty = (
        len(noun_like)
        + 1.5 * relation_count
        + count_count
        + 0.7 * attribute_count
        + (1.0 if len(words) >= 14 else 0.0)
        + (1.0 if category in {"spatial", "3d_spatial", "non_spatial", "complex"} else 0.0)
    )
    if difficulty >= 7:
        band = "hard"
    elif difficulty >= 4:
        band = "medium"
    else:
        band = "easy"
    return {
        "difficulty": round(float(difficulty), 4),
        "difficulty_band": band,
        "word_count": len(words),
        "noun_like_count": len(noun_like),
        "relation_count": relation_count,
        "count_word_count": count_count,
        "attribute_count": attribute_count,
    }


def _selection_key(row: dict[str, Any], seed: str) -> tuple[float, float]:
    metadata = row.get("metadata", {})
    difficulty = float(metadata.get("smoke_profile", {}).get("difficulty", 0.0))
    subtype = str(metadata.get("complex_subtype", ""))
    digest = hashlib.sha256(
        f"{seed}|{metadata.get('category')}|{subtype}|{row.get('prompt')}".encode("utf-8")
    ).hexdigest()
    jitter = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    return difficulty, jitter


def _count_categories(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {category: 0 for category in T2ICOMPBENCHPP_CATEGORIES}
    for row in rows:
        category = str(row.get("metadata", {}).get("category"))
        if category in counts:
            counts[category] += 1
    return counts


def _count_profile(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get("metadata", {}).get("smoke_profile", {}).get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalpha() or char in {"'", "-"}:
            current.append(char)
        else:
            if current:
                word = "".join(current).strip("-'")
                if word and word[0].isalpha():
                    words.append(word)
                current = []
    if current:
        word = "".join(current).strip("-'")
        if word and word[0].isalpha():
            words.append(word)
    return words


def _noun_like_terms(words: list[str]) -> list[str]:
    terms: list[str] = []
    for word in words:
        if len(word) < 3 or word in _STOPWORDS:
            continue
        if word.endswith("ing") and word not in _VISUAL_ING_WORDS:
            continue
        if word not in terms:
            terms.append(word)
    return terms


_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "on", "in", "with", "which", "while",
    "near", "next", "to", "from", "for", "by", "at", "as", "its", "their",
    "this", "that", "these", "those", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "is", "are", "was", "were", "be", "being", "been",
    "has", "have", "had", "into", "over", "under", "above", "below", "only",
    "other", "side", "both", "all",
}

_NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight"}

_ATTRIBUTE_WORDS = {
    "black", "blue", "brown", "green", "grey", "gray", "orange", "pink", "purple",
    "red", "white", "yellow", "metallic", "wooden", "plastic", "glass", "rubber",
    "leather", "fluffy", "smooth", "rough", "soft", "hard", "shiny", "silver",
    "circular", "rectangular", "triangular", "oblong", "cylindrical", "conical",
    "pentagonal", "spherical", "cubic",
}

_RELATION_PHRASES = (
    "on top of",
    "on the top of",
    "on the bottom of",
    "on the left of",
    "on the right of",
    "in front of",
    "hidden by",
    "behind",
    "next to",
    "beside",
    "between",
    "above",
    "below",
    "under",
    "near",
)

_VISUAL_ING_WORDS = {
    "building",
    "painting",
    "glittering",
    "glowing",
    "rolling",
    "swirling",
    "sparkling",
    "shimmering",
}

def _read_prompt_lines(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in rows if line]


def _load_complex_subtypes(root: Path, split: str) -> dict[str, str]:
    subtypes: dict[str, str] = {}
    for subtype, template in _COMPLEX_SUBTYPE_FILES.items():
        path = root / template.format(split=split)
        if not path.exists():
            continue
        for prompt in _read_prompt_lines(path):
            subtypes[prompt] = subtype
    return subtypes


def _absolute_path(path: str, base: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _row_image_path(row: dict[str, Any]) -> str:
    generation = row.get("generation", {}) if isinstance(row.get("generation"), dict) else {}
    for value in (generation.get("image_path"), generation.get("image_uri"), row.get("image_uri")):
        if isinstance(value, str) and value and not value.startswith("data:"):
            return value
    return ""


def _place_sample(source: Path, destination: Path, *, symlink: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        current = destination.resolve()
        if current != source.resolve():
            raise FileExistsError(f"refusing to replace {destination}: {current} != {source}")
        return
    if symlink:
        try:
            os.symlink(source, destination)
            return
        except OSError:
            pass
    os.link(source, destination)
