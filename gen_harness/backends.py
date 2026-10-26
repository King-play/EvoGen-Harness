from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

from .backend import VisualBackend
from .io import expand_env_vars, read_json
from .leakage_guard import reject_forbidden_payload_context
from .schema import GenerationTask
from .tool_payloads import public_generation_input


FORMAL_GENERATOR_BACKEND_TYPES = frozenset({
    "flow_grpo",
    "flux_diffusers",
    "http",
    "external_command",
    "comfyui",
    "scene_assembly",
})


def formal_backend_config_error(data: Dict[str, Any]) -> str | None:
    """Validate a real frozen generator without fixing one model family."""
    if not isinstance(data, dict):
        return "formal generator config must be a JSON object"
    backend_type = data.get("type")
    if backend_type not in FORMAL_GENERATOR_BACKEND_TYPES:
        return (
            "formal generator must use a real image-generation adapter "
            "(flow_grpo, flux_diffusers, http, external_command, or comfyui)"
        )
    if backend_type in {"flow_grpo", "flux_diffusers"}:
        label = "Flow-GRPO" if backend_type == "flow_grpo" else "FLUX diffusers"
        if data.get("frozen") is not True:
            return f"{label} formal generator must declare frozen=true"
        if not str(data.get("model_family", "")).strip():
            return f"{label} formal generator must declare model_family"
        if backend_type == "flux_diffusers" and not str(data.get("model_path", "")).strip():
            return "FLUX diffusers formal generator must declare model_path"
        return None
    if backend_type == "http":
        metadata = data.get("request_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        model = metadata.get("model") or data.get("model")
        if not str(model).strip():
            return "HTTP formal generator must declare request_metadata.model"
        frozen = metadata.get("frozen", data.get("frozen", False))
        if frozen is not True:
            return "HTTP formal generator must declare frozen=true"
        return None
    if backend_type == "external_command":
        if not isinstance(data.get("command"), list) or not data["command"]:
            return "external formal generator must declare a non-empty command"
        if data.get("frozen") is not True:
            return "external formal generator must declare frozen=true"
        if not str(data.get("model_family", "")).strip():
            return "external formal generator must declare model_family"
        return None
    if backend_type == "scene_assembly":
        source = data.get("source_backend")
        if not isinstance(source, dict):
            return "scene_assembly formal generator must declare source_backend"
        if source.get("type") == "scene_assembly":
            return "scene_assembly source_backend may not recursively be scene_assembly"
        source_error = formal_backend_config_error(source)
        if source_error:
            return f"scene_assembly source_backend invalid: {source_error}"
        text = json.dumps(source, sort_keys=True).lower()
        forbidden = ("gemini", "gpt-image", "dall-e", "openai_image", "flash-image")
        if any(token in text for token in forbidden):
            return "scene_assembly source_backend must be a frozen generator, not an image LLM"
        if data.get("frozen") is not True:
            return "scene_assembly formal generator must declare frozen=true"
        if not str(data.get("model_family", "")).strip():
            return "scene_assembly formal generator must declare model_family"
        return None
    if not str(data.get("workflow_json", "")).strip():
        return "ComfyUI formal generator must declare workflow_json"
    if data.get("frozen") is not True:
        return "ComfyUI formal generator must declare frozen=true"
    if not str(data.get("model_family", "")).strip():
        return "ComfyUI formal generator must declare model_family"
    return None


class ExternalCommandBackend(VisualBackend):
    """Backend adapter for real frozen image-generation systems.

    The command receives a JSON input file path and must emit JSON either to
    stdout or to the configured output path. This lets Gen-Harness integrate
    with diffusers scripts, ComfyUI launchers, SD WebUI clients, or custom VLM
    agents without depending on their Python environments.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="gen_harness_backend_") as td:
            td_path = Path(td)
            input_path = td_path / "input.json"
            output_path = td_path / "output.json"
            payload = {"input": public_generation_input(task, program), "seed": seed, "output_json": str(output_path)}
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
                    f"backend command failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
            if output_path.exists():
                result = json.loads(output_path.read_text(encoding="utf-8"))
            else:
                result = json.loads(proc.stdout)
            result.setdefault("backend", self.config.get("name", "external_command"))
            result.setdefault("seed", seed)
            return result


class ComfyUIBackend(VisualBackend):
    """Minimal real ComfyUI HTTP backend adapter.

    The workflow template must be a ComfyUI prompt JSON. Gen-Harness injects
    prompt text and seed into configured node fields, submits it to ComfyUI, and
    returns generated image metadata from the history endpoint.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server = config.get("server", "http://127.0.0.1:8188").rstrip("/")
        self.workflow = read_json(config["workflow_json"])

    def generate(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
        workflow = json.loads(json.dumps(self.workflow))
        public_input = public_generation_input(task, program)
        reject_forbidden_payload_context(self.config.get("inject", []), context="ComfyUI inject config")
        for edit in self.config.get("inject", []):
            template = str(edit["value"])
            if "{program}" in template or "{task}" in template:
                raise ValueError("ComfyUI inject templates may use {prompt}, {seed}, or {public_input}, not full task/program")
            node = workflow[str(edit["node_id"])]
            target = node
            for key in edit["path"][:-1]:
                target = target[key]
            value = template.format(
                prompt=public_input["generation_prompt"],
                seed=seed,
                public_input=json.dumps(public_input, ensure_ascii=False),
            )
            if edit.get("type") == "int":
                value = int(value)
            target[edit["path"][-1]] = value
        body = json.dumps({"prompt": workflow}).encode("utf-8")
        req = urllib.request.Request(f"{self.server}/prompt", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.config.get("timeout", 60)) as resp:
            prompt_id = json.loads(resp.read().decode("utf-8"))["prompt_id"]
        history = self._poll_history(prompt_id)
        images = []
        for node_out in history.get("outputs", {}).values():
            images.extend(node_out.get("images", []))
        return {"backend": "comfyui", "seed": seed, "prompt_id": prompt_id, "image_uri": images[0] if images else None, "images": images}

    def _poll_history(self, prompt_id: str) -> Dict[str, Any]:
        timeout = self.config.get("timeout", 300)
        interval = self.config.get("poll_interval", 1.0)
        deadline = time.time() + timeout
        while time.time() < deadline:
            with urllib.request.urlopen(f"{self.server}/history/{prompt_id}", timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if prompt_id in data:
                return data[prompt_id]
            time.sleep(interval)
        raise TimeoutError(f"ComfyUI prompt did not finish within {timeout}s: {prompt_id}")


def validate_formal_backend_config(config_path: str | Path) -> Dict[str, Any]:
    """Validate any real frozen generator allowed by the paper protocol."""
    cfg = expand_env_vars(read_json(config_path))
    error = formal_backend_config_error(cfg)
    if error:
        raise ValueError(error)
    return cfg


def build_backend_from_data(cfg: Dict[str, Any]) -> VisualBackend:
    typ = cfg.get("type")
    if typ == "external_command":
        return ExternalCommandBackend(cfg)
    if typ == "comfyui":
        return ComfyUIBackend(cfg)
    if typ == "http":
        from .adapters.http_backend import HTTPImageGenerationBackend

        return HTTPImageGenerationBackend(cfg)
    if typ == "flow_grpo":
        from .adapters.flow_grpo_backend import FlowGRPOBackend

        return FlowGRPOBackend(cfg)
    if typ == "flux_diffusers":
        from .adapters.flux_diffusers_backend import FluxDiffusersBackend

        return FluxDiffusersBackend(cfg)
    if typ == "scene_assembly":
        from .adapters.scene_assembly_backend import SceneAssemblyBackend

        return SceneAssemblyBackend(cfg)
    raise ValueError(
        f"unknown or unsupported backend type: {typ}; formal image generation must use "
        "a real frozen image-generation backend or deterministic scene_assembly wrapper"
    )


def build_backend(config_path: str | Path) -> VisualBackend:
    if config_path is None:
        raise ValueError("--backend-config is required and must point to a real backend adapter config")
    cfg = expand_env_vars(read_json(config_path))
    return build_backend_from_data(cfg)
