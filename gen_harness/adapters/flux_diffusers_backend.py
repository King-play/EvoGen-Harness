from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

from ..backend import VisualBackend
from ..schema import GenerationTask
from ..tool_payloads import public_task_row


class FluxDiffusersBackend(VisualBackend):
    """Local diffusers backend for frozen FLUX-family generators.

    FLUX.1-dev is open-weight and non-commercial licensed, not OSI-style open
    source. This backend keeps that model external to the LLM loop: the LLM
    compiles/localizes/proposes, while diffusers performs image generation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        default_model_path = os.environ.get("GENHARNESS_FLUX_MODEL_ROOT", "models/FLUX.1-dev")
        self.model_path = Path(self.config.get("model_path", default_model_path))
        self.image_cache_dir = Path(self.config.get("image_cache_dir", "runs/flux_diffusers_cache"))
        self.trace_cache_dir = Path(self.config.get("trace_cache_dir", "runs/flux_diffusers_trace_cache"))
        self.local_files_only = bool(self.config.get("local_files_only", True))
        self.torch_dtype = str(self.config.get("torch_dtype", "bfloat16"))
        self.device = str(self.config.get("device", "cuda"))
        self.height = int(self.config.get("height", 1024))
        self.width = int(self.config.get("width", 1024))
        self.num_inference_steps = self.config.get("num_inference_steps", 28)
        self.guidance_scale = self.config.get("guidance_scale", 3.5)
        self.max_sequence_length = self.config.get("max_sequence_length", 512)
        self.min_free_cuda_memory_gb = self.config.get("min_free_cuda_memory_gb")
        self.enable_model_cpu_offload = bool(self.config.get("enable_model_cpu_offload", False))
        self.enable_sequential_cpu_offload = bool(self.config.get("enable_sequential_cpu_offload", False))
        self.enable_attention_slicing = bool(self.config.get("enable_attention_slicing", False))
        self.enable_vae_slicing = bool(self.config.get("enable_vae_slicing", False))
        self.enable_vae_tiling = bool(self.config.get("enable_vae_tiling", False))
        self.reuse_existing_images = bool(self.config.get("reuse_existing_images", False))
        self._pipe = None

    def generate(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
        cached = self._cached_generation(task, program, seed)
        if cached is not None:
            return cached
        pipe = self._load_pipe()
        prompt = self._prompt_for_program(task, program)
        generation_parameters = self._tool_generation_parameters(program)

        generator_device = "cuda" if self.device == "cuda" and self._torch.cuda.is_available() else "cpu"
        generator = self._torch.Generator(device=generator_device).manual_seed(int(seed))
        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "generator": generator,
            "height": int(generation_parameters.get("height", self.height)),
            "width": int(generation_parameters.get("width", self.width)),
        }
        num_inference_steps = generation_parameters.get("num_inference_steps", self.num_inference_steps)
        guidance_scale = generation_parameters.get("guidance_scale", self.guidance_scale)
        max_sequence_length = generation_parameters.get("max_sequence_length", self.max_sequence_length)
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = int(num_inference_steps)
        if guidance_scale is not None:
            kwargs["guidance_scale"] = float(guidance_scale)
        if max_sequence_length is not None:
            kwargs["max_sequence_length"] = int(max_sequence_length)

        image = pipe(**kwargs).images[0]
        image_path = self._save_image(image, task, program, seed)
        result = {
            "backend": self.config.get("name", "flux_diffusers"),
            "seed": seed,
            "prompt": prompt,
            "image_path": str(image_path),
            "image_uri": str(image_path),
            "selected_model": self.config.get("model_family", "FLUX.1-dev"),
            "model_path": str(self.model_path),
            "height": kwargs["height"],
            "width": kwargs["width"],
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "generation_parameters": generation_parameters,
        }
        trace_path = self._save_trace(image_path, task, program, result, seed)
        result["trace_path"] = str(trace_path)
        return result

    def _cached_generation(self, task: GenerationTask, program: Dict[str, Any], seed: int) -> Dict[str, Any] | None:
        if not self.reuse_existing_images:
            return None
        image_path = self._image_path(task, program, seed)
        if not image_path.exists():
            return None
        trace_path = self._trace_path(image_path)
        return {
            "backend": self.config.get("name", "flux_diffusers"),
            "seed": seed,
            "prompt": self._prompt_for_program(task, program),
            "image_path": str(image_path),
            "image_uri": str(image_path),
            "trace_path": str(trace_path),
            "cache_hit": True,
            "image_origin": "reused_frozen_flux_diffusers_source",
            "selected_model": self.config.get("model_family", "FLUX.1-dev"),
            "model_path": str(self.model_path),
        }

    def _load_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        import torch  # type: ignore
        from diffusers import FluxPipeline  # type: ignore

        self._torch = torch
        self._preflight_device(torch)
        dtype = self._torch_dtype(torch)
        model_path = self._require_dir(self.model_path, "FLUX model")
        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
        )
        self._configure_pipe_memory(pipe, cuda_available=torch.cuda.is_available())
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def _torch_dtype(self, torch_module: Any) -> Any:
        if self.torch_dtype == "bfloat16":
            return torch_module.bfloat16
        if self.torch_dtype == "float16":
            return torch_module.float16
        if self.torch_dtype == "float32":
            return torch_module.float32
        raise ValueError(f"unsupported torch_dtype for FLUX backend: {self.torch_dtype}")

    def _configure_pipe_memory(self, pipe: Any, *, cuda_available: bool) -> None:
        if self.device == "cuda" and not cuda_available:
            raise RuntimeError("FLUX diffusers backend requested CUDA, but CUDA is unavailable")
        if self.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if self.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if self.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if self.enable_model_cpu_offload and self.enable_sequential_cpu_offload:
            raise ValueError("enable_model_cpu_offload and enable_sequential_cpu_offload are mutually exclusive")
        if self.enable_model_cpu_offload:
            if not hasattr(pipe, "enable_model_cpu_offload"):
                raise RuntimeError("FLUX low-memory mode requires diffusers/accelerate support for enable_model_cpu_offload")
            pipe.enable_model_cpu_offload()
            return
        if self.enable_sequential_cpu_offload:
            if not hasattr(pipe, "enable_sequential_cpu_offload"):
                raise RuntimeError("FLUX low-memory mode requires diffusers/accelerate support for enable_sequential_cpu_offload")
            pipe.enable_sequential_cpu_offload()
            return
        if hasattr(pipe, "to"):
            pipe.to(self.device)
        else:
            raise RuntimeError("FLUX pipeline does not expose .to(device); execution is unavailable")

    def _prompt_for_program(self, task: GenerationTask, program: Dict[str, Any]) -> str:
        workflow = program.get("workflow", {}) if isinstance(program, dict) else {}
        prompt = str(workflow.get("compiled_prompt") or "").strip()
        if not prompt:
            raise ValueError(f"FLUX generation requires workflow.compiled_prompt for task {task.task_id}")
        return prompt

    def _tool_generation_parameters(self, program: Dict[str, Any]) -> Dict[str, Any]:
        tool = program.get("tool", {}) if isinstance(program, dict) else {}
        params = tool.get("generation_parameters", {}) if isinstance(tool, dict) else {}
        return dict(params) if isinstance(params, dict) else {}

    def _save_image(self, image: Any, task: GenerationTask, program: Dict[str, Any], seed: int) -> Path:
        path = self._image_path(task, program, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def _image_path(self, task: GenerationTask, program: Dict[str, Any], seed: int) -> Path:
        method = str(program.get("method", "gen_harness")) if isinstance(program, dict) else "gen_harness"
        digest = hashlib.sha1(
            repr((task.task_id, method, seed, self._prompt_for_program(task, program))).encode()
        ).hexdigest()[:12]
        return self.image_cache_dir / "images" / method / f"{task.task_id}_seed{seed}_{digest}.png"

    def _trace_path(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.image_cache_dir / "images")
        return self.trace_cache_dir / relative.parent / f"{relative.name}.trace.json"

    def _save_trace(self, image_path: Path, task: GenerationTask, program: Dict[str, Any], result: Dict[str, Any], seed: int) -> Path:
        trace_path = self._trace_path(image_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = {
            "task": public_task_row(task),
            "seed": seed,
            "method": str(program.get("method", "gen_harness")) if isinstance(program, dict) else "gen_harness",
            "final_prompt": result.get("prompt"),
            "image_path": str(image_path),
            "generation": dict(result),
            "program": program,
            "backend_config": {
                "name": self.config.get("name", "flux_diffusers"),
                "type": self.config.get("type", "flux_diffusers"),
                "model_family": self.config.get("model_family", "FLUX.1-dev"),
                "license_note": self.config.get("license_note"),
                "model_path": str(self.model_path),
                "torch_dtype": self.torch_dtype,
                "device": self.device,
                "height": self.height,
                "width": self.width,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "max_sequence_length": self.max_sequence_length,
                "min_free_cuda_memory_gb": self.min_free_cuda_memory_gb,
                "enable_model_cpu_offload": self.enable_model_cpu_offload,
                "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload,
                "enable_attention_slicing": self.enable_attention_slicing,
                "enable_vae_slicing": self.enable_vae_slicing,
                "enable_vae_tiling": self.enable_vae_tiling,
            },
        }
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return trace_path

    def _require_dir(self, path: Path, label: str) -> str:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing local {label} directory: {path}")
        return str(path)

    def _preflight_device(self, torch_module: Any) -> None:
        if self.device != "cuda":
            return
        if self.min_free_cuda_memory_gb is not None:
            nvidia_smi_memory = self._cuda_memory_info_from_nvidia_smi()
            if nvidia_smi_memory is not None:
                self._enforce_min_free_cuda_memory(nvidia_smi_memory)
                return
        if not torch_module.cuda.is_available():
            raise RuntimeError("FLUX diffusers backend requires CUDA unless device is set to cpu")
        if self.min_free_cuda_memory_gb is None:
            return
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        self._enforce_min_free_cuda_memory((float(free_bytes) / (1024**3), float(total_bytes) / (1024**3)))

    def _cuda_memory_info_from_nvidia_smi(self) -> Tuple[float, float] | None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        first_line = next((line.strip() for line in proc.stdout.splitlines() if line.strip()), "")
        if not first_line:
            return None
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) < 2:
            return None
        try:
            free_mib = float(parts[0])
            total_mib = float(parts[1])
        except ValueError:
            return None
        return free_mib / 1024.0, total_mib / 1024.0

    def _enforce_min_free_cuda_memory(self, memory_gb: Tuple[float, float]) -> None:
        free_gb, total_gb = memory_gb
        required_gb = float(self.min_free_cuda_memory_gb)
        if free_gb < required_gb:
            mode = "low-memory" if (self.enable_model_cpu_offload or self.enable_sequential_cpu_offload) else "standard"
            raise RuntimeError(
                f"FLUX {mode} backend needs at least {required_gb:.1f} GiB free CUDA memory before loading, "
                f"but only {free_gb:.1f}/{total_gb:.1f} GiB is free. Free the GPU or enable a FLUX low-memory config."
            )
