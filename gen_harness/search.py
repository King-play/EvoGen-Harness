from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from .io import expand_env_vars, read_json
from .tool_payloads import public_search_input
from .schema import GenerationTask


class SearchBackend:
    """Interface for real evidence/reference search systems."""

    def search(self, task: GenerationTask, contract: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class ExternalCommandSearch(SearchBackend):
    """Search adapter for real retrieval engines.

    The command receives public prompt semantics and a sanitized visual
    contract, then returns evidence and reference candidates. It can wrap web
    search, dataset retrieval, embedding search, image search, or a
    Gen-Searcher-style agent.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def search(self, task: GenerationTask, contract: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="gen_harness_search_") as td:
            td_path = Path(td)
            input_path = td_path / "input.json"
            output_path = td_path / "output.json"
            payload = {"input": public_search_input(task, contract), "output_json": str(output_path)}
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            command = [
                str(part).format(
                    python=os.environ.get("GENHARNESS_PYTHON") or sys.executable or "python",
                    input_json=str(input_path),
                    output_json=str(output_path),
                )
                for part in self.config["command"]
            ]
            proc = subprocess.run(command, text=True, capture_output=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"search command failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
            return json.loads(output_path.read_text(encoding="utf-8") if output_path.exists() else proc.stdout)


def build_search(config_path: str | Path | None, *, required: bool = False) -> SearchBackend:
    if config_path is None:
        raise ValueError("--search-config is required to build a real search/evidence adapter")
    cfg = expand_env_vars(read_json(config_path))
    typ = cfg.get("type")
    if typ == "external_command":
        return ExternalCommandSearch(cfg)
    if typ == "http":
        from .adapters.http_search import HTTPSearchBackend

        return HTTPSearchBackend(cfg)
    raise ValueError(f"unknown or unsupported search type: {typ}")
