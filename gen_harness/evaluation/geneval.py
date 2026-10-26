from __future__ import annotations

import base64
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..io import read_jsonl, write_json, write_jsonl


GENEVAL_REPO_URL = "https://github.com/djghosh13/geneval"
GENEVAL_RAW_BASE = "https://raw.githubusercontent.com/djghosh13/geneval/main"
GENEVAL_METADATA_REL_PATH = "prompts/evaluation_metadata.jsonl"


TAG_TO_CONSTRAINT_TYPE = {
    "single_object": "count",
    "two_object": "count",
    "counting": "count",
    "colors": "attribute",
    "position": "spatial",
    "color_attr": "attribute",
    "color_attribution": "attribute",
}


def fetch_geneval_dataset(
    output_dir: str | Path,
    *,
    source_base: str = GENEVAL_RAW_BASE,
    metadata_rel_path: str = GENEVAL_METADATA_REL_PATH,
    split: str = "heldout",
    limit: int | None = None,
    overwrite: bool = False,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Download official GenEval prompt metadata and build Gen-Harness tasks.

    This fetches only the benchmark prompt metadata, not generated images or
    detector weights. Official detector evaluation still requires the GenEval
    evaluator checkout and object detector described by the upstream project.
    """

    output = Path(output_dir)
    raw_root = output / "raw" / "geneval"
    metadata_path = raw_root / metadata_rel_path
    if overwrite or not metadata_path.exists():
        _download_file(_join_url(source_base, metadata_rel_path), metadata_path, timeout=timeout)

    task_path = output / "geneval_tasks.jsonl"
    tasks = build_geneval_tasks(metadata_path, task_path, split=split, limit=limit)
    manifest = {
        "dataset": "GenEval",
        "source_repo": GENEVAL_REPO_URL,
        "source_base": source_base,
        "metadata_rel_path": metadata_rel_path,
        "metadata_path": str(metadata_path),
        "tasks_path": str(task_path),
        "split": split,
        "limit": limit,
        "num_tasks": len(tasks),
        "notes": [
            "Downloaded prompt metadata only.",
            "Use the official GenEval repository and detector weights for detector-based scoring.",
        ],
    }
    manifest_path = output / "geneval_fetch_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_geneval_tasks(
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    split: str = "target",
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Convert official GenEval metadata JSONL into Gen-Harness task JSONL.

    The original GenEval row is preserved under metadata.geneval_metadata so
    outputs can later be exported back to the official GenEval evaluator format.
    """

    rows = read_jsonl(metadata_path)
    if limit is not None:
        rows = rows[:limit]

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        tag = str(row.get("tag", "unknown"))
        prompt = str(row["prompt"])
        ctype = TAG_TO_CONSTRAINT_TYPE.get(tag, "prompt_alignment")
        constraint_text = _constraint_text(row)
        tasks.append(
            {
                "task_id": f"geneval_{idx:05d}",
                "prompt": prompt,
                "task_family": "compositional",
                "references": {},
                "constraints": [
                    {
                        "constraint_id": f"geneval_{tag}_0",
                        "text": constraint_text,
                        "constraint_type": ctype,
                        "target": tag,
                        "verification_rule": "official GenEval object-focused evaluator",
                        "preservation": [],
                        "metadata": _constraint_metadata(row),
                    }
                ],
                "metadata": {
                    "split": split,
                    "benchmark": "geneval",
                    "geneval_index": idx,
                    "geneval_tag": tag,
                    "geneval_metadata": row,
                },
            }
        )
    write_jsonl(output_path, tasks)
    return tasks


def export_geneval_images(
    input_jsonl: str | Path,
    output_dir: str | Path,
    *,
    method: str | None = None,
) -> Dict[str, Any]:
    """Export Gen-Harness/quality-exp outputs to official GenEval image layout.

    Official layout:
      IMAGE_FOLDER/00000/metadata.jsonl
      IMAGE_FOLDER/00000/samples/0000.png

    `input_jsonl` may be an experiences JSONL file or a quality_results JSONL
    file. When quality_results contains multiple methods, pass `method` to
    export one method at a time.
    """

    rows = read_jsonl(input_jsonl)
    grouped: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    metadata_by_index: Dict[int, Dict[str, Any]] = {}
    seen = set()

    for row in rows:
        row_method = row.get("method")
        if method is not None and row_method is not None and row_method != method:
            continue
        meta = _export_metadata(row)
        if meta.get("benchmark") != "geneval" and "geneval_metadata" not in meta:
            continue
        if method is not None and row_method is None:
            # Experiences JSONL does not carry a method field; it represents the
            # harness run, so allow method aliases that refer to Gen-Harness.
            if method not in {"gen_harness", "harness", "ours"}:
                continue
        index = int(meta["geneval_index"])
        seed = int(_row_seed(row))
        key = (index, seed, row_method or "gen_harness")
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(index, []).append((seed, row))
        metadata_by_index[index] = dict(meta["geneval_metadata"])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported_images = 0
    exported_prompts = 0
    for index, seed_rows in sorted(grouped.items()):
        prompt_dir = out / f"{index:05d}"
        samples_dir = prompt_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "metadata.jsonl").write_text(
            json.dumps(metadata_by_index[index], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for sample_idx, (row_seed, row) in enumerate(sorted(seed_rows, key=lambda item: item[0])):
            image_ref = _image_ref(row)
            if not image_ref:
                raise ValueError(f"row has no exportable image reference: {row.get('task_id') or row.get('experience_id')}")
            target = samples_dir / f"{sample_idx:04d}{_extension(image_ref)}"
            _write_image(image_ref, target)
            exported_images += 1
        exported_prompts += 1

    summary = {
        "input": str(input_jsonl),
        "output_dir": str(out),
        "method": method,
        "num_prompts": exported_prompts,
        "num_images": exported_images,
    }
    (out / "genharness_geneval_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _join_url(base: str, rel_path: str) -> str:
    return base.rstrip("/") + "/" + rel_path.lstrip("/")


def _download_file(url: str, path: Path, *, timeout: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "gen-harness/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        path.write_bytes(response.read())


def _constraint_text(row: Dict[str, Any]) -> str:
    prompt = str(row["prompt"] if "prompt" in row else "")
    include = _object_list(row.get("include"))
    exclude = _object_list(row.get("exclude"))
    parts = [prompt]
    if include:
        required = ", ".join(_format_required_object(obj) for obj in include)
        parts.append(f"required visible objects: {required}")
    if exclude:
        forbidden = ", ".join(str(obj.get("class", obj)) for obj in exclude)
        parts.append(f"must not contain: {forbidden}")
    return "; ".join(part for part in parts if part)


def _constraint_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    required = _object_list(row.get("include"))
    forbidden = _object_list(row.get("exclude"))
    attributes = _attribute_bindings(row.get("include"))
    relations = _spatial_relations(row.get("include"))
    return {
        "constraint_schema": "object_requirements.v1",
        "required_objects": required,
        "forbidden_objects": forbidden,
        "attribute_bindings": attributes,
        "spatial_relations": relations,
        "source_benchmark": "geneval",
        "source_tag": row.get("tag"),
        "include": required,
        "exclude": forbidden,
        "prompt": row.get("prompt"),
    }


def _object_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict) and item.get("class"):
            out.append({"class": str(item["class"]), "count": int(item.get("count", 1) or 1)})
    return out


def _attribute_bindings(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or not item.get("class") or not item.get("color"):
            continue
        out.append({
            "object": str(item["class"]),
            "attribute": str(item["color"]),
            "attribute_type": "color",
        })
    return out


def _spatial_relations(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or not item.get("class") or "position" not in item:
            continue
        position = item.get("position")
        if not isinstance(position, list) or len(position) != 2:
            continue
        target_index = int(position[1])
        if target_index < 0 or target_index >= len(value):
            continue
        target = value[target_index]
        if not isinstance(target, dict) or not target.get("class"):
            continue
        out.append({
            "subject": str(item["class"]),
            "relation": str(position[0]).replace("_", " "),
            "object": str(target["class"]),
        })
    return out


def _format_required_object(obj: Dict[str, Any]) -> str:
    count = int(obj.get("count", 1) or 1)
    name = str(obj.get("class", "object"))
    if count == 1:
        return f"one {name}"
    return f"{count} {name}s"


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict(row.get("metadata", {}))


def _export_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    export_meta = row.get("export_metadata")
    if isinstance(export_meta, dict):
        return dict(export_meta)
    return _metadata(row)


def _row_seed(row: Dict[str, Any]) -> int:
    if "seed" in row:
        return int(row.get("seed") or 0)
    visual = row.get("visual_result", {}) if isinstance(row.get("visual_result"), dict) else {}
    if "seed" in visual:
        return int(visual.get("seed") or 0)
    generation = row.get("generation", {}) if isinstance(row.get("generation"), dict) else {}
    return int(generation.get("seed", 0) or 0)


def _image_ref(row: Dict[str, Any]) -> str | None:
    if isinstance(row.get("image_uri"), str):
        return row["image_uri"]
    generation = row.get("generation", {}) if isinstance(row.get("generation"), dict) else {}
    for key in ("image_path", "image_uri"):
        value = generation.get(key)
        if isinstance(value, str):
            return value
    visual = row.get("visual_result", {}) if isinstance(row.get("visual_result"), dict) else {}
    if isinstance(visual.get("image_uri"), str):
        return visual["image_uri"]
    raw = visual.get("raw_generation", {}) if isinstance(visual.get("raw_generation"), dict) else {}
    for key in ("image_path", "image_uri"):
        value = raw.get(key)
        if isinstance(value, str):
            return value
    return None


def _extension(image_ref: str) -> str:
    if image_ref.startswith("data:"):
        mime = image_ref.split(";", 1)[0].replace("data:", "")
        return {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp"}.get(mime, ".png")
    suffix = Path(image_ref).suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp"} else ".png"


def _write_image(image_ref: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if image_ref.startswith("data:"):
        if "," not in image_ref:
            raise ValueError("malformed data URI image")
        target.write_bytes(base64.b64decode(image_ref.split(",", 1)[1]))
        return
    src = Path(image_ref)
    if not src.exists():
        raise ValueError(f"GenEval official evaluator requires local image files; not found: {image_ref}")
    shutil.copyfile(src, target)
