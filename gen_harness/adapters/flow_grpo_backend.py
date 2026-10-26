from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from ..backend import VisualBackend
from ..schema import GenerationTask
from ..tool_payloads import public_task_row


class FlowGRPOBackend(VisualBackend):
    """Local Flow-GRPO image generation backend.

    This is a fixed-model backend for fair GenEval runs of Gen-Harness. It does
    not use OctoT2I's Qwen router or NVILA retry loop; Gen-Harness compiles the
    prompt/program and this adapter calls a single local Flow-GRPO generator.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        default_model_root = os.environ.get("GENHARNESS_FLOW_GRPO_MODEL_ROOT", "models")
        self.model_root = Path(self.config.get("model_root", default_model_root))
        self.base_model_path = Path(self.config.get("base_model_path", self.model_root / "stable-diffusion-3.5-medium"))
        self.adapter_path = Path(self.config.get("adapter_path", self.model_root / "Flow-GRPO"))
        self.use_adapter = bool(self.config.get("use_adapter", True))
        self.reuse_existing_images = bool(self.config.get("reuse_existing_images", False))
        self.image_cache_dir = Path(self.config.get("image_cache_dir", "runs/flow_grpo_backend_cache"))
        self.trace_cache_dir = Path(self.config.get("trace_cache_dir", "runs/flow_grpo_backend_trace_cache"))
        self.local_files_only = bool(self.config.get("local_files_only", True))
        self.torch_dtype = self.config.get("torch_dtype", "bfloat16")
        self.num_inference_steps = self.config.get("num_inference_steps")
        self.guidance_scale = self.config.get("guidance_scale")
        self.min_free_cuda_memory_gb = self.config.get("min_free_cuda_memory_gb")
        self.enable_model_cpu_offload = bool(self.config.get("enable_model_cpu_offload", False))
        self.enable_sequential_cpu_offload = bool(self.config.get("enable_sequential_cpu_offload", False))
        self.enable_attention_slicing = bool(self.config.get("enable_attention_slicing", False))
        self.enable_vae_slicing = bool(self.config.get("enable_vae_slicing", False))
        self.enable_vae_tiling = bool(self.config.get("enable_vae_tiling", False))
        self.fast_import_package_distributions = bool(self.config.get("fast_import_package_distributions", True))
        self.clip_max_tokens = int(self.config.get("clip_max_tokens", 72))
        self.t5_max_tokens = int(self.config.get("t5_max_tokens", 256))
        self.split_prompt_channels = bool(self.config.get("split_prompt_channels", False))
        self._pipe = None
        self._last_prompt_token_counts: Dict[str, int] = {}
        self._last_clip_budget_trace: Dict[str, Any] = {}

    def generate(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
        cached = self._cached_generation(task, program, seed)
        if cached is not None:
            return cached
        pipe = self._load_pipe()
        prompt = self._prompt_for_program(task, program)
        generation_parameters = self._tool_generation_parameters(program)
        split_prompt_channels = bool(generation_parameters.get("split_prompt_channels", self.split_prompt_channels))
        clip_max_tokens = int(generation_parameters.get("clip_max_tokens", self.clip_max_tokens))
        sd3_prompts = self._sd3_prompt_kwargs(
            task,
            program,
            prompt,
            split_prompt_channels=split_prompt_channels,
            clip_max_tokens=clip_max_tokens,
        )
        generator = self._torch.Generator(device="cuda").manual_seed(int(seed))
        kwargs: Dict[str, Any] = {"generator": generator, **sd3_prompts}
        num_inference_steps = generation_parameters.get("num_inference_steps", self.num_inference_steps)
        guidance_scale = generation_parameters.get("guidance_scale", self.guidance_scale)
        max_sequence_length = generation_parameters.get("max_sequence_length")
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = int(num_inference_steps)
        if guidance_scale is not None:
            kwargs["guidance_scale"] = float(guidance_scale)
        if max_sequence_length is not None:
            kwargs["max_sequence_length"] = int(max_sequence_length)
        image = pipe(**kwargs).images[0]
        image_path = self._save_image(image, task, program, seed)
        result = {
            "backend": self.config.get("name", "flow_grpo"),
            "seed": seed,
            "prompt": prompt,
            "clip_prompt": sd3_prompts["prompt"],
            "t5_prompt": sd3_prompts["prompt_3"],
            "prompt_token_counts": dict(self._last_prompt_token_counts),
            "clip_prompt_budget": dict(self._last_clip_budget_trace),
            "image_path": str(image_path),
            "image_uri": str(image_path),
            "selected_model": "flow-grpo" if self.use_adapter else "stable-diffusion-3.5-medium-base",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generation_parameters": generation_parameters,
        }
        trace_path = self._save_trace(image_path, task, program, result, seed)
        result["trace_path"] = str(trace_path)
        return result

    def _cached_generation(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any] | None:
        """Replay an exact frozen source sample when explicitly requested.

        This is an experiment/reproducibility switch, not a candidate selector:
        the cache key includes task id, seed, method, and the complete compiled
        source prompt.  A cache hit therefore represents the same deterministic
        Flow-GRPO sample that a fresh invocation would write.
        """

        if not self.reuse_existing_images:
            return None
        image_path = self._image_path(task, program, seed)
        if not image_path.exists():
            return None
        trace_path = self._trace_path(image_path)
        cached_trace: Dict[str, Any] = {}
        if trace_path.exists():
            try:
                parsed = json.loads(trace_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    cached_trace = parsed
            except (OSError, json.JSONDecodeError):
                cached_trace = {}
        generation = cached_trace.get("generation")
        result = dict(generation) if isinstance(generation, dict) else {}
        result.update(
            {
                "backend": self.config.get("name", "flow_grpo"),
                "seed": seed,
                "prompt": self._prompt_for_program(task, program),
                "clip_prompt": str(program.get("workflow", {}).get("clip_prompt") or self._prompt_for_program(task, program)),
                "t5_prompt": str(program.get("workflow", {}).get("t5_prompt") or self._prompt_for_program(task, program)),
                "image_path": str(image_path),
                "image_uri": str(image_path),
                "trace_path": str(trace_path),
                "cache_hit": True,
                "image_origin": "reused_frozen_flow_grpo_source",
                "selected_model": "flow-grpo" if self.use_adapter else "stable-diffusion-3.5-medium-base",
            }
        )
        return result

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        import torch  # type: ignore

        self._torch = torch
        self._preflight_cuda(torch)
        with self._fast_package_distribution_map():
            from diffusers import DiffusionPipeline  # type: ignore
            from peft import PeftModel  # type: ignore

        dtype = torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float16
        base = self._require_dir(self.base_model_path, "SD3.5 medium base model")
        pipe = DiffusionPipeline.from_pretrained(base, torch_dtype=dtype, local_files_only=self.local_files_only)
        if self.use_adapter:
            adapter = self._require_dir(self.adapter_path, "Flow-GRPO adapter")
            pipe.transformer = PeftModel.from_pretrained(
                pipe.transformer,
                adapter,
                torch_dtype=dtype,
                local_files_only=self.local_files_only,
            ).merge_and_unload()
        self._configure_pipe_memory(pipe, cuda_available=torch.cuda.is_available())
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def _configure_pipe_memory(self, pipe: Any, *, cuda_available: bool) -> None:
        if not cuda_available:
            raise RuntimeError("Flow-GRPO backend requires CUDA; CPU execution is unsupported")
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
                raise RuntimeError("Flow-GRPO low-memory mode requires diffusers/accelerate support for enable_model_cpu_offload")
            pipe.enable_model_cpu_offload()
            return
        if self.enable_sequential_cpu_offload:
            if not hasattr(pipe, "enable_sequential_cpu_offload"):
                raise RuntimeError("Flow-GRPO low-memory mode requires diffusers/accelerate support for enable_sequential_cpu_offload")
            pipe.enable_sequential_cpu_offload()
            return
        if hasattr(pipe, "to"):
            pipe.to("cuda")
        else:
            raise RuntimeError("Flow-GRPO pipeline does not expose .to('cuda'); CUDA execution is unavailable")

    def _prompt_for_program(self, task: GenerationTask, program: Dict[str, Any]) -> str:
        workflow = program.get("workflow", {})
        prompt = str(workflow.get("compiled_prompt") or "").strip()
        if not prompt:
            raise ValueError(f"gen_harness generation requires workflow.compiled_prompt for task {task.task_id}")
        return prompt

    def _sd3_prompt_kwargs(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        full_prompt: str,
        *,
        split_prompt_channels: bool | None = None,
        clip_max_tokens: int | None = None,
    ) -> Dict[str, str]:
        prompt = str(full_prompt or "").strip()
        if not prompt:
            raise ValueError(f"SD3 prompt kwargs require a non-empty compiled prompt for task {task.task_id}")
        workflow = program.get("workflow", {}) if isinstance(program, dict) else {}
        split_channels = self.split_prompt_channels if split_prompt_channels is None else bool(split_prompt_channels)
        if split_channels:
            clip_prompt = self._fit_compiled_clip_segments(
                task,
                workflow,
                clip_max_tokens=clip_max_tokens,
            )
            t5_prompt = str(workflow.get("t5_prompt") or "").strip()
            t5_prompt = self._fit_compiled_t5_segments(task, workflow, t5_prompt)
        else:
            clip_prompt = prompt
            t5_prompt = prompt
        if not clip_prompt or not t5_prompt:
            raise ValueError(f"SD3 prompt channels must be non-empty for task {task.task_id}")
        self._last_prompt_token_counts = self._validate_clip_prompt_budget(task, clip_prompt, clip_max_tokens=clip_max_tokens)
        return {
            "prompt": clip_prompt,
            "prompt_2": clip_prompt,
            "prompt_3": t5_prompt,
        }

    def _fit_compiled_t5_segments(
        self,
        task: GenerationTask,
        workflow: Dict[str, Any],
        prompt: str,
    ) -> str:
        """Fit whole semantic clauses using the pipeline's real T5 tokenizer."""
        tokenizer = getattr(self._pipe, "tokenizer_3", None) if self._pipe is not None else None
        if tokenizer is None:
            return prompt
        raw = workflow.get("t5_prompt_segments")
        segments = [str(value).strip() for value in raw] if isinstance(raw, list) else []
        segments = [value for value in segments if value]
        if not segments:
            segments = [prompt]
        raw_types = workflow.get("t5_prompt_segment_types")
        types = [
            str(raw_types[index]) if isinstance(raw_types, list) and index < len(raw_types) else "unspecified"
            for index in range(len(segments))
        ]
        priority = {"object": 0, "spatial": 0, "action": 0, "attribute": 0,
                    "repair_inventory": 0, "repair_spatial": 0, "repair_action": 0,
                    "repair_attribute": 0, "repair_layout": 1, "repair": 1,
                    "repair_feedback": 2,
                    "source": 2, "negative": 3, "reference": 3, "unspecified": 3, "presentation": 4}
        limit = min(self.t5_max_tokens, 256)
        selected: list[int] = []
        dropped: list[int] = []
        semantic_families = (
            "object", "spatial", "action", "attribute",
            "repair_inventory", "repair_spatial", "repair_action", "repair_attribute", "repair_layout",
        )
        representative_indexes = [
            next(index for index, segment_type in enumerate(types) if segment_type == family)
            for family in semantic_families
            if family in types
        ]
        remaining_indexes = [index for index in range(len(segments)) if index not in representative_indexes]
        candidate_order = [
            *representative_indexes,
            *sorted(remaining_indexes, key=lambda i: (priority.get(types[i], 3), i)),
        ]
        for index in candidate_order:
            indexes = sorted([*selected, index])
            candidate = " ".join(segments[i] for i in indexes)
            count = len(tokenizer.encode(candidate, add_special_tokens=True, truncation=False))
            if count <= limit:
                selected.append(index)
            else:
                dropped.append(index)
        if not selected:
            raise ValueError(f"no complete T5 clause fits the model context for task {task.task_id}")
        selected.sort()
        fitted = " ".join(segments[i] for i in selected)
        workflow["t5_prompt_unbudgeted"] = prompt
        workflow["t5_prompt"] = fitted
        workflow["t5_prompt_budget"] = {
            "mode": "tokenizer_aware_complete_clause_selection",
            "token_count": len(tokenizer.encode(fitted, add_special_tokens=True, truncation=False)),
            "token_budget": limit,
            "selected_segment_types": [types[i] for i in selected],
            "required_semantic_families": [family for family in semantic_families if family in types],
            "unrepresented_semantic_families": [
                family for family in semantic_families if family in types and not any(types[i] == family for i in selected)
            ],
            "dropped_segments": [segments[i] for i in sorted(dropped)],
        }
        return fitted

    def _fit_compiled_clip_segments(
        self,
        task: GenerationTask,
        workflow: Dict[str, Any],
        *,
        clip_max_tokens: int | None,
    ) -> str:
        """Select whole compiler-produced clauses using both real tokenizers."""

        raw_segments = workflow.get("clip_prompt_segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            self._last_clip_budget_trace = {"mode": "precompiled_prompt", "dropped_segments": []}
            return str(workflow.get("clip_prompt") or "").strip()
        raw_types = workflow.get("clip_prompt_segment_types")
        pairs = [
            (str(segment).strip(), str(raw_types[index]) if isinstance(raw_types, list) and index < len(raw_types) else "unspecified")
            for index, segment in enumerate(raw_segments)
            if str(segment).strip()
        ]
        segments = [segment for segment, _ in pairs]
        if not segments:
            raise ValueError(f"compiled CLIP segments are empty for task {task.task_id}")
        # Exact object inventory and semantic relations are more valuable than
        # a duplicated natural-language source prompt. Select whole clauses by
        # constraint family, then restore source order for a coherent prompt.
        priority = {
            "object": 0,
            "repair_inventory": 0,
            "spatial": 1,
            "action": 1,
            "attribute": 1,
            "repair_spatial": 1,
            "repair_action": 1,
            "repair_attribute": 1,
            "repair_layout": 2,
            "repair": 2,
            "repair_feedback": 3,
            "negative": 3,
            "reference": 3,
            "source": 4,
            "presentation": 5,
            "unspecified": 4,
        }
        selected_indexes: list[int] = []
        dropped_indexes: list[int] = []
        # Give every present semantic family one seat before spending spare
        # tokens on additional clauses from any one family. This prevents a
        # many-class count prompt from crowding out its relation or action.
        semantic_families = (
            "object", "spatial", "action", "attribute",
            "repair_inventory", "repair_spatial", "repair_action", "repair_attribute", "repair_layout",
        )
        representative_indexes = [
            next(index for index, (_, segment_type) in enumerate(pairs) if segment_type == family)
            for family in semantic_families
            if any(segment_type == family for _, segment_type in pairs)
        ]
        remaining_indexes = [index for index in range(len(pairs)) if index not in representative_indexes]
        candidate_order = [
            *representative_indexes,
            *sorted(remaining_indexes, key=lambda value: (priority.get(pairs[value][1], 4), value)),
        ]
        for index in candidate_order:
            candidate_indexes = sorted([*selected_indexes, index])
            candidate = " ".join(segments[value] for value in candidate_indexes)
            try:
                self._validate_clip_prompt_budget(task, candidate, clip_max_tokens=clip_max_tokens)
            except ValueError:
                dropped_indexes.append(index)
            else:
                selected_indexes.append(index)
        if not selected_indexes:
            # A policy extractor may legitimately emit one long, user-facing
            # clause (for example a preservation/style requirement) whose
            # complete sentence is longer than CLIP's context.  Dropping the
            # whole prompt makes the real backend fail before TRACE can record
            # evidence.  Keep the shortest compiler-produced clause and trim
            # it with the actual tokenizer as a last-resort transport step.
            shortest_index = min(
                range(len(segments)),
                key=lambda index: min(
                    len(getattr(self._pipe, name).encode(segments[index], add_special_tokens=True))
                    for name in ("tokenizer", "tokenizer_2")
                    if getattr(self._pipe, name, None) is not None
                ),
            )
            fitted = self._truncate_clip_clause(
                task,
                segments[shortest_index],
                clip_max_tokens=clip_max_tokens,
            )
            selected_indexes = [shortest_index]
            dropped_indexes = [index for index in range(len(segments)) if index != shortest_index]
        else:
            selected_indexes.sort()
            fitted = " ".join(segments[index] for index in selected_indexes)
        dropped = [segments[index] for index in sorted(dropped_indexes)]
        counts = self._validate_clip_prompt_budget(task, fitted, clip_max_tokens=clip_max_tokens)
        self._last_clip_budget_trace = {
            "mode": "tokenizer_aware_complete_clause_selection",
            "source_segment_count": len(segments),
            "selected_segment_count": len(selected_indexes),
            "selected_segment_types": [pairs[index][1] for index in selected_indexes],
            "dropped_segment_types": [pairs[index][1] for index in sorted(dropped_indexes)],
            "required_semantic_families": [family for family in semantic_families if any(segment_type == family for _, segment_type in pairs)],
            "unrepresented_semantic_families": [
                family
                for family in semantic_families
                if any(segment_type == family for _, segment_type in pairs)
                and not any(pairs[index][1] == family for index in selected_indexes)
            ],
            "dropped_segments": dropped,
            "token_counts": counts,
            "token_budget": min(int(clip_max_tokens or self.clip_max_tokens), 77),
        }
        original = str(workflow.get("clip_prompt") or "").strip()
        workflow["clip_prompt_unbudgeted"] = original
        workflow["clip_prompt"] = fitted
        workflow["clip_prompt_budget"] = dict(self._last_clip_budget_trace)
        provenance = workflow.get("prompt_provenance")
        if isinstance(provenance, dict):
            channels = provenance.get("backend_channels")
            if isinstance(channels, dict):
                channels["clip_prompt"] = fitted
            hashes = provenance.get("sha256")
            if isinstance(hashes, dict):
                hashes["clip_prompt"] = hashlib.sha256(fitted.encode("utf-8")).hexdigest()
            provenance["channels_equal"] = fitted == str(workflow.get("t5_prompt") or "").strip() == str(workflow.get("canonical_prompt") or "").strip()
        return fitted

    def _truncate_clip_clause(
        self,
        task: GenerationTask,
        clause: str,
        *,
        clip_max_tokens: int | None,
    ) -> str:
        """Transport one compiler clause without silently crashing a run."""

        budget = min(int(clip_max_tokens or self.clip_max_tokens), 77)
        for name in ("tokenizer", "tokenizer_2"):
            tokenizer = getattr(self._pipe, name, None)
            if tokenizer is None:
                continue
            token_ids = tokenizer.encode(clause, add_special_tokens=True, truncation=False)
            if len(token_ids) <= budget:
                try:
                    self._validate_clip_prompt_budget(task, clause, clip_max_tokens=budget)
                except ValueError:
                    pass
                else:
                    return clause
            # Decoding a prefix can change whitespace/tokenization. Search for
            # the longest prefix that is valid for both CLIP channels instead
            # of assuming that `decode(ids[:budget])` remains within budget.
            for end in range(min(len(token_ids), budget), 0, -1):
                candidate = tokenizer.decode(
                    token_ids[:end],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                if not candidate:
                    continue
                try:
                    self._validate_clip_prompt_budget(task, candidate, clip_max_tokens=budget)
                except ValueError:
                    continue
                return candidate
        raise ValueError(f"unable to fit any compiler clause into CLIP context for task {task.task_id}")

    def _tool_generation_parameters(self, program: Dict[str, Any]) -> Dict[str, Any]:
        tool = program.get("tool", {}) if isinstance(program, dict) else {}
        params = tool.get("generation_parameters", {}) if isinstance(tool, dict) else {}
        return dict(params) if isinstance(params, dict) else {}

    def _validate_clip_prompt_budget(self, task: GenerationTask, prompt: str, *, clip_max_tokens: int | None = None) -> Dict[str, int]:
        if self._pipe is None:
            return {}
        counts: Dict[str, int] = {}
        configured_budget = self.clip_max_tokens if clip_max_tokens is None else int(clip_max_tokens)
        for name in ("tokenizer", "tokenizer_2"):
            tokenizer = getattr(self._pipe, name, None)
            if tokenizer is None:
                continue
            model_limit = int(getattr(tokenizer, "model_max_length", configured_budget) or configured_budget)
            # Some tokenizers use enormous sentinels for "unknown".  CLIP's
            # actual context is 77, so never permit a config to exceed it.
            if model_limit <= 0 or model_limit > 10000:
                model_limit = 77
            budget = min(configured_budget, model_limit, 77)
            token_ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=False)
            count = len(token_ids)
            counts[name] = count
            if count > budget:
                raise ValueError(
                    f"CLIP prompt exceeds the model context for task {task.task_id}: "
                    f"{name}={count} tokens > {budget}"
                )
        return counts

    def _save_image(self, image: Any, task: GenerationTask, program: Dict[str, Any], seed: int) -> Path:
        path = self._image_path(task, program, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def _image_path(self, task: GenerationTask, program: Dict[str, Any], seed: int) -> Path:
        method = str(program.get("method", "gen_harness"))
        digest = hashlib.sha1(
            repr((task.task_id, method, seed, self._prompt_for_program(task, program))).encode()
        ).hexdigest()[:12]
        return self.image_cache_dir / "images" / method / f"{task.task_id}_seed{seed}_{digest}.png"

    def _trace_path(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.image_cache_dir / "images")
        return self.trace_cache_dir / relative.parent / f"{relative.name}.trace.json"

    def _save_trace(self, image_path: Path, task: GenerationTask, program: Dict[str, Any], result: Dict[str, Any], seed: int) -> Path:
        relative = image_path.relative_to(self.image_cache_dir / "images")
        trace_path = self.trace_cache_dir / relative.parent / f"{relative.name}.trace.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = {
            "task": public_task_row(task),
            "seed": seed,
            "method": str(program.get("method", "gen_harness")),
            "final_prompt": result.get("prompt"),
            "image_path": str(image_path),
            "generation": dict(result),
            "program": program,
            "backend_config": {
                "name": self.config.get("name", "flow_grpo"),
                "type": self.config.get("type", "flow_grpo"),
                "base_model_path": str(self.base_model_path),
                "adapter_path": str(self.adapter_path),
                "use_adapter": self.use_adapter,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "torch_dtype": self.torch_dtype,
                "min_free_cuda_memory_gb": self.min_free_cuda_memory_gb,
                "enable_model_cpu_offload": self.enable_model_cpu_offload,
                "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload,
                "enable_attention_slicing": self.enable_attention_slicing,
                "enable_vae_slicing": self.enable_vae_slicing,
                "enable_vae_tiling": self.enable_vae_tiling,
                "fast_import_package_distributions": self.fast_import_package_distributions,
            },
        }
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return trace_path

    def _require_dir(self, path: Path, label: str) -> str:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing local {label} directory: {path}")
        return str(path)

    def _preflight_cuda(self, torch_module: Any) -> None:
        if self.min_free_cuda_memory_gb is not None:
            nvidia_smi_memory = self._cuda_memory_info_from_nvidia_smi()
            if nvidia_smi_memory is not None:
                self._enforce_min_free_cuda_memory(nvidia_smi_memory)
                return
        if not torch_module.cuda.is_available():
            raise RuntimeError("Flow-GRPO backend requires CUDA; CPU execution is unsupported")
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
                f"Flow-GRPO {mode} backend needs at least {required_gb:.1f} GiB free CUDA memory before loading, "
                f"but only {free_gb:.1f}/{total_gb:.1f} GiB is free. Free the GPU or use configs/backends/flow_grpo_lowmem.json."
            )

    @contextmanager
    def _fast_package_distribution_map(self) -> Iterator[None]:
        if not self.fast_import_package_distributions:
            yield
            return

        import importlib.metadata as importlib_metadata
        import importlib.util

        original = importlib_metadata.packages_distributions
        package_map: Dict[str, list[str]] = defaultdict(list)
        try:
            for dist in importlib_metadata.distributions():
                dist_name = dist.metadata.get("Name")
                if not dist_name:
                    continue
                top_level = (dist.read_text("top_level.txt") or "").split()
                for package_name in top_level:
                    if package_name:
                        package_map[package_name].append(dist_name)
            known_distributions = {
                "accelerate": "accelerate",
                "bitsandbytes": "bitsandbytes",
                "diffusers": "diffusers",
                "huggingface_hub": "huggingface-hub",
                "numpy": "numpy",
                "peft": "peft",
                "PIL": "pillow",
                "safetensors": "safetensors",
                "sentencepiece": "sentencepiece",
                "torch": "torch",
                "torchvision": "torchvision",
                "transformers": "transformers",
            }
            for package_name, dist_name in known_distributions.items():
                if package_name not in package_map and importlib.util.find_spec(package_name) is not None:
                    package_map[package_name].append(dist_name)

            def fast_packages_distributions() -> Dict[str, list[str]]:
                return dict(package_map)

            importlib_metadata.packages_distributions = fast_packages_distributions
            yield
        finally:
            importlib_metadata.packages_distributions = original
