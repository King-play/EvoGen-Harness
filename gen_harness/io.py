from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

_ENV_ALIASES = {
    "GENHARNESS_LLM_BASE_URL": ("GEN_HARNESS_BASE_URL", "GEN_HARNESS_GEMINI_BASE_URL"),
    "GENHARNESS_LLM_API_KEY": ("GEN_HARNESS_API_KEY",),
    "GENHARNESS_LLM_MODEL": ("GEN_HARNESS_JUDGE_MODEL",),
    "GENHARNESS_LLM_MODEL_SNAPSHOT": ("GENHARNESS_LLM_MODEL", "GEN_HARNESS_JUDGE_MODEL"),
}


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env_string(obj)
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    return obj


def _expand_env_string(value: str) -> str:
    out: List[str] = []
    index = 0
    while index < len(value):
        start = value.find("${", index)
        if start < 0:
            out.append(value[index:])
            break
        out.append(value[index:start])
        end = value.find("}", start + 2)
        if end < 0:
            out.append(value[start:])
            break
        name = value[start + 2 : end]
        if not _valid_env_name(name):
            out.append(value[start : end + 1])
        else:
            env_name = _resolve_env_name(name)
            if env_name is None:
                raise ValueError(f"environment variable is not set: {name}")
            out.append(os.environ[env_name])
        index = end + 1
    return "".join(out)


def _resolve_env_name(name: str) -> str | None:
    if name in os.environ:
        return name
    for alias in _ENV_ALIASES.get(name, ()):
        if alias in os.environ:
            return alias
    return None


def _valid_env_name(name: str) -> bool:
    if not name:
        return False
    first = name[0]
    if not (first == "_" or first.isalpha()):
        return False
    return all(char == "_" or char.isalnum() for char in name[1:])


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
