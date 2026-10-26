from __future__ import annotations

import base64
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..backend import VisualBackend
from ..fast_loop import FastLoopController, require_full_llm_fast_loop
from ..inspection import InspectionRunner
from ..evaluation.geneval import export_geneval_images
from ..io import append_jsonl, write_json, write_jsonl
from ..patcher import HarnessLocalizer, PatchProposer, build_harness_localizer, build_patch_proposer
from ..prompt_optimizer import PromptOptimizer
from ..repository import HarnessRepository
from ..schema import GenerationTask
from ..search import SearchBackend
from ..tasks import load_tasks
from ..telemetry import prompt_efficiency_observation
from ..tool_payloads import public_task_row
from ..trace_protocol import TRACE_DEFAULT_K


class GenEvalImageGenerationExperiment:
    """Generate prompt-only benchmark images for separate official evaluation.

    This path intentionally does not call an official evaluator and does not
    compute paper scores. GenEval retains its legacy directory export, while
    GenEval2 and WISE use their dedicated post-generation export adapters.
    """

    def __init__(
        self,
        harness: str | Path,
        backend: VisualBackend,
        search_backend: SearchBackend | None = None,
        prompt_optimizer: PromptOptimizer | None = None,
        inspector: InspectionRunner | None = None,
        fast_loop_attempts: int = TRACE_DEFAULT_K + 1,
        fast_loop_stochastic_k: int = TRACE_DEFAULT_K,
        policy_extractor_config: str | Path | None = None,
        harness_localizer_config: str | Path | None = None,
        patch_proposer_config: str | Path | None = None,
        harness_localizer: HarnessLocalizer | None = None,
        patch_proposer: PatchProposer | None = None,
        benchmark_name: str = "geneval",
        export_geneval_layout: bool = True,
    ):
        if policy_extractor_config is None:
            raise ValueError("policy_extractor_config is required for formal benchmark generation")
        self.harness = Path(harness)
        self.repo = HarnessRepository(
            self.harness,
            policy_extractor_config=policy_extractor_config,
        )
        self.backend = backend
        self.search_backend = search_backend
        self.prompt_optimizer = prompt_optimizer
        self.inspector = inspector
        self.fast_loop_attempts = max(1, int(fast_loop_attempts))
        self.fast_loop_stochastic_k = max(1, int(fast_loop_stochastic_k))
        self.harness_localizer = harness_localizer
        self.patch_proposer = patch_proposer
        if harness_localizer_config is not None:
            self.harness_localizer = build_harness_localizer(
                harness_localizer_config,
                repo=self.repo,
            )
        if patch_proposer_config is not None:
            self.patch_proposer = build_patch_proposer(
                patch_proposer_config,
                memory=self.repo.build_memory(),
                repo=self.repo,
            )
        self.benchmark_name = str(benchmark_name).strip().lower()
        self.export_geneval_layout = bool(export_geneval_layout)
        if self.benchmark_name not in {"geneval", "geneval2", "wise"}:
            raise ValueError(f"unsupported prompt-only benchmark: {self.benchmark_name}")
        require_full_llm_fast_loop(
            inspector=self.inspector,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            max_attempts=self.fast_loop_attempts,
            stochastic_evidence_k=self.fast_loop_stochastic_k,
            context="formal prompt benchmark generation",
        )

    def run(
        self,
        geneval_tasks: str | Path,
        output_dir: str | Path,
        *,
        methods: Sequence[str],
        seeds: Sequence[int],
        limit: int | None = None,
        continue_on_error: bool = False,
    ) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        source_tasks = load_tasks(geneval_tasks)
        if limit is not None:
            source_tasks = source_tasks[:limit]
        tasks = [self._prompt_only_task(task) for task in source_tasks]
        seed_list = list(seeds) or [0]
        method_list = [self._normalize_method(m) for m in methods if m.strip()]

        summaries: Dict[str, Any] = {}
        for method in method_list:
            self._clean_method_outputs(output, method)
            result_path = output / f"{method}_generations.jsonl"
            write_jsonl(result_path, [])
            try:
                rows = self._run_method(
                    method,
                    tasks,
                    source_tasks,
                    seed_list,
                    output,
                    result_path,
                    continue_on_error=continue_on_error,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                rows = []
                summaries[method] = {
                    "num_rows": 0,
                    "num_generated_images": 0,
                    "num_errors": len(source_tasks) * len(seed_list),
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "official_eval_status": "not_run",
                }
                continue
            export_rows = [row for row in rows if row.get("image_uri")]
            missing_images = len(rows) - len(export_rows)
            if missing_images and not continue_on_error:
                raise ValueError(f"{method} has {missing_images} rows without exportable images")
            export_input = result_path
            if missing_images:
                export_input = output / f"{method}_exportable_generations.jsonl"
                write_jsonl(export_input, export_rows)
            export_summary = None
            if self.export_geneval_layout:
                export_summary = export_geneval_images(export_input, output / "geneval_exports" / method, method=method) if export_rows else {"num_prompts": 0, "num_images": 0}
            summaries[method] = {
                "num_rows": len(rows),
                "num_generated_images": sum(1 for row in rows if row.get("image_uri")),
                "final_images_dir": str((output / "images" / method).resolve()),
                "num_errors": sum(1 for row in rows if row.get("error")),
                "num_internal_inspection_attempts": sum(len(row.get("fast_loop", {}).get("attempts", [])) for row in rows),
                "num_fast_loop_successes": sum(1 for row in rows if row.get("fast_loop", {}).get("passed")),
                "num_fast_loop_strictly_confirmed": sum(1 for row in rows if row.get("fast_loop", {}).get("strictly_confirmed")),
                "num_fast_loop_soft_abstention_accepts": sum(
                    1 for row in rows
                    if row.get("fast_loop", {}).get("passed")
                    and not row.get("fast_loop", {}).get("strictly_confirmed")
                ),
                "missing_images": missing_images,
                "export": export_summary,
                "official_eval_status": "pending" if not missing_images else "partial_pending",
            }

        latency_summary = self._latency_summary(method_list, output)
        write_json(output / "latency_summary.json", latency_summary)
        summary = {
            "benchmark": self.benchmark_name,
                "mode": "formal_gen_harness_fast_loop_generation",
            "num_tasks": len(source_tasks),
            "seeds": seed_list,
            "policy_extractor_manifest": self.repo.policy_extractor_manifest(),
            "methods": summaries,
            "latency_summary": str(output / "latency_summary.json"),
            "notes": [
                f"Official {self.benchmark_name} scoring is not called in this path.",
                "Generation is prompt-only: benchmark metadata is retained only after generation for export to the official evaluator.",
                "Every generated image passed through the serial LLM Fast Loop before freezing.",
                "Freeze generated images before invoking the separate official evaluator.",
            ],
        }
        write_json(output / f"{self.benchmark_name}_generation_summary.json", summary)
        return summary

    def _normalize_method(self, method: str) -> str:
        return method.strip()

    def _run_method(
        self,
        method: str,
        tasks: Sequence[GenerationTask],
        source_tasks: Sequence[GenerationTask],
        seeds: Sequence[int],
        output: Path,
        result_path: Path,
        *,
        continue_on_error: bool = False,
    ) -> List[Dict[str, Any]]:
        if len(tasks) != len(source_tasks):
            raise ValueError("prompt-only task list and source task list must have the same length")
        if method == "gen_harness":
            return self._run_gen_harness(
                tasks,
                source_tasks,
                seeds,
                output,
                result_path,
                method,
                continue_on_error=continue_on_error,
            )
        raise ValueError(f"unsupported official GenEval generation method: {method}")

    def _run_gen_harness(
        self,
        tasks: Sequence[GenerationTask],
        source_tasks: Sequence[GenerationTask],
        seeds: Sequence[int],
        output: Path,
        result_path: Path,
        method: str,
        *,
        continue_on_error: bool = False,
    ) -> List[Dict[str, Any]]:
        policy = self.repo.build_policy()
        middleware = self.repo.build_middleware()
        skills = self.repo.build_skills()
        memory = self.repo.build_memory()
        tool_metadata = self.repo.load_artifact("tools", "generation_tools.json")
        rows: List[Dict[str, Any]] = []
        for seed in seeds:
            for task, source_task in zip(tasks, source_tasks):
                execution_started = time.perf_counter()
                try:
                    program = self._build_program(
                        task,
                        method=method,
                        seed=seed,
                        policy=policy,
                        middleware=middleware,
                        skills=skills,
                        memory=memory,
                        tool_metadata=tool_metadata,
                    )
                    row = self._generate_row(method, task, source_task, seed, program, output)
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    row = self._compile_error_row(
                        method,
                        task,
                        source_task,
                        seed,
                        exc,
                        execution_started,
                    )
                wall_clock_seconds = time.perf_counter() - execution_started
                generation_seconds = float(row.get("latency", {}).get("row_seconds", 0.0) or 0.0)
                row.setdefault("latency", {})["harness_compile_seconds"] = round(
                    max(0.0, wall_clock_seconds - generation_seconds), 6
                )
                row["latency"]["wall_clock_seconds"] = round(wall_clock_seconds, 6)
                row_program = row.get("program", {}) if isinstance(row.get("program"), dict) else {}
                row_contract = row_program.get("contract", {}) if isinstance(row_program, dict) else {}
                row_workflow = row_program.get("workflow", {}) if isinstance(row_program, dict) else {}
                row["efficiency"] = prompt_efficiency_observation(
                    wall_clock_seconds=wall_clock_seconds,
                    fast_loop=row.get("fast_loop"),
                    policy_extractor=row_contract.get("policy_extractor"),
                    prompt_optimizer_used=bool(row_workflow.get("prompt_optimizer")),
                )
                rows.append(row)
                append_jsonl(result_path, [row])
        return rows

    def _build_program(
        self,
        task: GenerationTask,
        *,
        method: str,
        seed: int,
        policy: Any,
        middleware: Any,
        skills: Any,
        memory: Any,
        tool_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        contract = policy.compile_contract(task)
        evidence = None
        if contract.get("required_evidence"):
            if self.search_backend is None:
                raise ValueError("visual contract requires search/evidence, but no search backend was configured")
            evidence = self.search_backend.search(task, contract)
        skill_name = middleware.select_skill(task)
        bindings = middleware.bind_references(task, contract, evidence=evidence)
        tool = middleware.route_tool(task, contract, tool_metadata)
        tool = self._bind_backend_tool(tool, tool_metadata)
        memory_context = self._memory_context(task, memory, contract)
        workflow = skills.compile_workflow(task, skill_name, contract, bindings, memory_context=memory_context)
        program = {
            "method": method,
            "contract": contract,
            "workflow": workflow,
            "reference_bindings": bindings,
            "evidence": evidence,
            "tool": tool,
            "memory_context": memory_context,
            "middleware_decisions": middleware.trace_decisions(skill_name, bindings, tool),
            "component_sources": {
                component: str(self.repo.component_dir(component))
                for component in ("policy", "tools", "skills", "middleware", "memory")
            },
            "benchmark_generation_input": "prompt_only",
        }
        if self.prompt_optimizer is not None and workflow.get("skill_name") == "prompt_enhancement":
            workflow["compiled_prompt_before_optimizer"] = workflow.get("compiled_prompt")
            workflow["compiled_prompt"] = self.prompt_optimizer.optimize(task, program, seed=seed)
            workflow["prompt_optimizer"] = self.prompt_optimizer.describe()
        return program

    def _compile_error_row(
        self,
        method: str,
        task: GenerationTask,
        source_task: GenerationTask,
        seed: int,
        exc: Exception,
        started: float,
    ) -> Dict[str, Any]:
        return {
            "method": method,
            "task_id": task.task_id,
            "seed": seed,
            "image_uri": None,
            "generation": {},
            "program": {
                "method": method,
                "benchmark_generation_input": "prompt_only",
                "compile_error": f"{exc.__class__.__name__}: {exc}",
            },
            "component_decisions": {},
            "metadata": self._formal_row_metadata(task),
            "export_metadata": self._benchmark_export_metadata(source_task),
            "latency": {
                "row_seconds": round(time.perf_counter() - started, 6),
                "generation_seconds": 0.0,
                "inspection_seconds": 0.0,
                "fast_loop_attempts": 0,
                "inspection_by_source_seconds": {},
            },
            "error": f"{exc.__class__.__name__}: {exc}",
            "official_eval_status": "not_exportable",
        }

    def _bind_backend_tool(self, routed_tool: Dict[str, Any], tool_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Make the trace name the configured execution backend.

        Middleware routing remains visible in ``routing_diagnostics``.  The
        backend config is authoritative for the actual image-generation
        implementation, so a wrapper backend must not be reported as the
        generic base generator.
        """
        backend_name = str(getattr(self.backend, "tool_name", "") or "").strip()
        if not backend_name:
            return routed_tool
        configured = tool_metadata.get("tools", {}).get(backend_name)
        if not isinstance(configured, dict):
            raise ValueError(f"backend tool {backend_name!r} is missing from generation_tools.json")
        tool = dict(configured)
        tool["routing_diagnostics"] = dict(routed_tool.get("routing_diagnostics") or {})
        tool["routing_diagnostics"]["configured_backend_tool"] = backend_name
        tool["routing_diagnostics"]["middleware_routed_tool"] = routed_tool.get("name")
        if routed_tool.get("routing_warning"):
            tool["routing_warning"] = routed_tool["routing_warning"]
        return tool

    def _generate_row(
        self,
        method: str,
        task: GenerationTask,
        source_task: GenerationTask,
        seed: int,
        program: Dict[str, Any],
        output: Path,
    ) -> Dict[str, Any]:
        row_started = time.perf_counter()
        try:
            loop_result = FastLoopController(
                self.backend,
                self.inspector,
                max_attempts=self.fast_loop_attempts,
                harness_localizer=self.harness_localizer,
                patch_proposer=self.patch_proposer,
                stochastic_evidence_k=self.fast_loop_stochastic_k,
            ).run(task, program, seed=seed)
            generation = loop_result.generation
            fast_loop = loop_result.to_dict()
            image_path = self._save_image(generation, output, method, task.task_id, seed)
            image_uri = image_path or generation.get("image_path") or generation.get("image_uri")
            latency = self._row_latency(method, row_started, fast_loop=fast_loop, generation_latency=None)
            row = {
                "method": method,
                "task_id": task.task_id,
                "seed": seed,
                "image_uri": image_uri,
                "generation": generation,
                "program": program,
                "component_decisions": self._component_decisions(program),
                "metadata": self._formal_row_metadata(task),
                "export_metadata": self._benchmark_export_metadata(source_task),
                "latency": latency,
                "official_eval_status": "pending",
            }
            if fast_loop is not None:
                row["fast_loop"] = fast_loop
            trace_path = self._write_image_trace(image_uri, task, row)
            if trace_path:
                row["trace_path"] = trace_path
                generation.setdefault("trace_path", trace_path)
            return row
        except Exception as exc:
            return {
                "method": method,
                "task_id": task.task_id,
                "seed": seed,
                "image_uri": None,
                "generation": {},
                "program": program,
                "component_decisions": self._component_decisions(program),
                "metadata": self._formal_row_metadata(task),
                "export_metadata": self._benchmark_export_metadata(source_task),
                "latency": {
                    "row_seconds": round(time.perf_counter() - row_started, 6),
                    "generation_seconds": 0.0,
                    "inspection_seconds": 0.0,
                    "fast_loop_attempts": 0,
                    "inspection_by_source_seconds": {},
                },
                "error": f"{exc.__class__.__name__}: {exc}",
                "official_eval_status": "not_exportable",
            }

    def _component_decisions(self, program: Dict[str, Any]) -> Dict[str, Any]:
        contract = program.get("contract", {}) if isinstance(program, dict) else {}
        workflow = program.get("workflow", {}) if isinstance(program, dict) else {}
        memory = program.get("memory_context", {}) if isinstance(program, dict) else {}
        return {
            "policy": {
                "component": contract.get("component"),
                "constraint_ids": [row.get("constraint_id") for row in contract.get("constraints", []) if isinstance(row, dict)],
            },
            "skills": {
                "component": workflow.get("component"),
                "skill_name": workflow.get("skill_name"),
                "selected_prompt_family": workflow.get("selected_prompt_family"),
                "prompt_provenance": workflow.get("prompt_provenance"),
            },
            "tools": {"selected_tool": program.get("tool")},
            "memory": {
                "component": memory.get("component"),
                "retrieved_failure_count": len(memory.get("retrieved_failures", [])) if isinstance(memory.get("retrieved_failures"), list) else 0,
                "validated_patch_count": len(memory.get("validated_patches", [])) if isinstance(memory.get("validated_patches"), list) else 0,
            },
            "middleware": dict(program.get("middleware_decisions", {}) or {}),
            "component_sources": dict(program.get("component_sources", {}) or {}),
        }

    def _formal_row_metadata(self, task: GenerationTask) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"task_family": task.task_family}
        if isinstance(task.metadata, dict) and task.metadata.get("split"):
            metadata["split"] = task.metadata["split"]
        metadata["generation_input"] = "prompt_only_no_official_eval_outputs"
        return metadata

    def _benchmark_export_metadata(self, source_task: GenerationTask) -> Dict[str, Any]:
        metadata = source_task.metadata if isinstance(source_task.metadata, dict) else {}
        benchmark = str(metadata.get("benchmark") or self.benchmark_name).strip().lower()
        allowed = {
            "geneval": ("benchmark", "geneval_index", "geneval_tag", "geneval_metadata"),
            "geneval2": ("benchmark", "geneval2_index", "geneval2_prompt"),
            "wise": ("benchmark", "wise_index", "wise_id", "wise_domain"),
        }.get(benchmark)
        if allowed is None:
            raise ValueError(f"unsupported benchmark metadata: {benchmark!r}")
        return {
            key: metadata[key]
            for key in allowed
            if key in metadata
        }

    def _prompt_only_task(self, task: GenerationTask) -> GenerationTask:
        """Return the exact task visible to generation-time harness logic.

        Keep only the split used for ordinary harness bookkeeping. Benchmark
        metadata, evaluator labels, object tags, references, and authored
        constraints stay on the export-only source task.
        """

        return GenerationTask(
            task_id=task.task_id,
            prompt=task.prompt,
            task_family=task.task_family,
            constraints=[],
            references={},
            metadata={"split": task.metadata.get("split")} if isinstance(task.metadata, dict) and task.metadata.get("split") else {},
        )

    def _row_latency(
        self,
        method: str,
        row_started: float,
        *,
        fast_loop: Dict[str, Any] | None,
        generation_latency: float | None,
    ) -> Dict[str, Any]:
        if fast_loop is None:
            return {
                "row_seconds": round(time.perf_counter() - row_started, 6),
                "generation_seconds": round(float(generation_latency or 0.0), 6),
                "inspection_seconds": 0.0,
                "fast_loop_attempts": 0,
                "inspection_by_source_seconds": {},
            }
        attempts = fast_loop.get("attempts", []) if isinstance(fast_loop, dict) else []
        generation_seconds = 0.0
        inspection_seconds = 0.0
        inspection_by_source: Dict[str, float] = {}
        for attempt in attempts:
            latency = attempt.get("latency", {}) if isinstance(attempt, dict) else {}
            if not isinstance(latency, dict):
                continue
            generation_seconds += float(latency.get("generation_seconds", 0.0) or 0.0)
            inspection_seconds += float(latency.get("inspection_seconds", 0.0) or 0.0)
            by_source = latency.get("inspection_by_source_seconds", {})
            if isinstance(by_source, dict):
                for name, value in by_source.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    inspection_by_source[str(name)] = inspection_by_source.get(str(name), 0.0) + float(value)
        return {
            "row_seconds": round(time.perf_counter() - row_started, 6),
            "generation_seconds": round(generation_seconds, 6),
            "inspection_seconds": round(inspection_seconds, 6),
            "fast_loop_attempts": len(attempts),
            "inspection_by_source_seconds": {key: round(value, 6) for key, value in sorted(inspection_by_source.items())},
        }

    def _latency_summary(self, methods: Sequence[str], output: Path) -> Dict[str, Any]:
        method_summaries: Dict[str, Any] = {}
        all_rows: List[Dict[str, Any]] = []
        row_summaries: List[Dict[str, Any]] = []
        for method in methods:
            path = output / f"{method}_generations.jsonl"
            rows = self._read_rows(path)
            all_rows.extend(rows)
            row_summaries.extend(self._latency_row_summary(row) for row in rows)
            method_summaries[method] = self._summarize_latency_rows(rows)
        return {
            "schema": "gen_harness.latency_summary.v1",
            "methods": method_summaries,
            "overall": self._summarize_latency_rows(all_rows),
            "rows": row_summaries,
            "notes": [
                "generation_seconds measures backend.generate wall-clock time.",
                "inspection_seconds measures internal inspection wall-clock time across fast-loop attempts.",
                "inspection_by_source_seconds is populated by composite inspectors such as GroundingDINO, OpenCLIP, and OCR.",
                f"Official {getattr(self, 'benchmark_name', 'benchmark')} evaluator runtime is not included.",
            ],
        }

    def _read_rows(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    def _summarize_latency_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        latencies = [row.get("latency", {}) for row in rows if isinstance(row.get("latency"), dict)]
        generation_values = [float(lat.get("generation_seconds", 0.0) or 0.0) for lat in latencies]
        inspection_values = [float(lat.get("inspection_seconds", 0.0) or 0.0) for lat in latencies]
        row_values = [float(lat.get("row_seconds", 0.0) or 0.0) for lat in latencies]
        attempt_values = [int(lat.get("fast_loop_attempts", 0) or 0) for lat in latencies]
        efficiency_rows = [row.get("efficiency", {}) for row in rows if isinstance(row.get("efficiency"), dict)]
        wall_clock_values = [float(row.get("wall_clock_seconds", 0.0) or 0.0) for row in efficiency_rows]
        generator_call_values = [float(row.get("generator_calls", 0.0) or 0.0) for row in efficiency_rows]
        llm_vlm_call_values = [float(row.get("llm_vlm_calls", 0.0) or 0.0) for row in efficiency_rows]
        by_source_totals: Dict[str, float] = {}
        for lat in latencies:
            by_source = lat.get("inspection_by_source_seconds", {})
            if not isinstance(by_source, dict):
                continue
            for name, value in by_source.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                by_source_totals[str(name)] = by_source_totals.get(str(name), 0.0) + float(value)
        return {
            "num_rows": len(rows),
            "num_latency_rows": len(latencies),
            "total_row_seconds": round(sum(row_values), 6),
            "mean_row_seconds": self._mean(row_values),
            "total_generation_seconds": round(sum(generation_values), 6),
            "mean_generation_seconds": self._mean(generation_values),
            "total_inspection_seconds": round(sum(inspection_values), 6),
            "mean_inspection_seconds": self._mean(inspection_values),
            "total_fast_loop_attempts": sum(attempt_values),
            "mean_fast_loop_attempts": self._mean([float(v) for v in attempt_values]),
            "paper_efficiency": {
                "num_prompt_observations": len(efficiency_rows),
                "average_wall_clock_seconds_per_prompt": self._mean(wall_clock_values),
                "average_generator_calls_per_prompt": self._mean(generator_call_values),
                "average_llm_vlm_calls_per_prompt": self._mean(llm_vlm_call_values),
            },
            "inspection_by_source_seconds": {key: round(value, 6) for key, value in sorted(by_source_totals.items())},
            "mean_inspection_by_source_seconds": {
                key: round(value / len(latencies), 6) if latencies else 0.0
                for key, value in sorted(by_source_totals.items())
            },
        }

    def _latency_row_summary(self, row: Dict[str, Any]) -> Dict[str, Any]:
        latency = row.get("latency", {}) if isinstance(row.get("latency"), dict) else {}
        return {
            "method": row.get("method"),
            "task_id": row.get("task_id"),
            "seed": row.get("seed"),
            "image_uri": row.get("image_uri"),
            "error": row.get("error"),
            "efficiency": row.get("efficiency", {}),
            "row_seconds": float(latency.get("row_seconds", 0.0) or 0.0),
            "generation_seconds": float(latency.get("generation_seconds", 0.0) or 0.0),
            "inspection_seconds": float(latency.get("inspection_seconds", 0.0) or 0.0),
            "fast_loop_attempts": int(latency.get("fast_loop_attempts", 0) or 0),
            "inspection_by_source_seconds": dict(latency.get("inspection_by_source_seconds", {}) or {}),
        }

    def _mean(self, values: Sequence[float]) -> float:
        return round(statistics.fmean(values), 6) if values else 0.0

    def _memory_context(self, task: GenerationTask, memory: Any, contract: Dict[str, Any] | None = None) -> Dict[str, Any]:
        constraint_types = sorted({str(c.get("constraint_type")) for c in (contract or {}).get("constraints", [])} or {c.constraint_type for c in task.constraints})
        failures: List[Dict[str, Any]] = []
        for ctype in constraint_types:
            failures.extend(memory.retrieve_failures(constraint_type=ctype, limit=5))
        seen = set()
        unique_failures = []
        for row in failures:
            row_id = row.get("experience_id") or repr(row)
            if row_id not in seen:
                seen.add(row_id)
                unique_failures.append(row)
        return {
            "component": memory.describe(),
            "retrieved_failures": unique_failures[-10:],
            "validated_patches": memory.retrieve_validated_patches(constraint_types=constraint_types, limit=10),
            "regression_cases": memory.retrieve_regression_cases(constraint_types=constraint_types, limit=10),
            "prompt_guidelines": memory.retrieve_prompt_guidelines(task_family=task.task_family, constraint_types=constraint_types, limit=10),
        }

    def _save_image(self, generation: Dict[str, Any], output: Path, method: str, task_id: str, seed: int) -> str | None:
        if not isinstance(generation, dict):
            raise TypeError("generation must be a dictionary")
        image_uri = generation.get("image_uri")
        images_dir = output / "images" / method
        images_dir.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)
        if isinstance(image_uri, str) and image_uri.startswith("data:") and "," in image_uri:
            header, encoded = image_uri.split(",", 1)
            mime = header.split(";", 1)[0].replace("data:", "") or "image/png"
            ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp"}.get(mime)
            if ext is None:
                raise ValueError(f"unsupported generated image MIME type: {mime}")
            path = images_dir / f"{safe_task}_seed{seed}{ext}"
            path.write_bytes(base64.b64decode(encoded, validate=True))
            generation["mime_type"] = mime
        else:
            source_ref = generation.get("image_path") or image_uri
            if not isinstance(source_ref, str) or not source_ref:
                raise ValueError("generation did not return an image data URI or local image path")
            source = Path(source_ref)
            if not source.is_file():
                raise FileNotFoundError(f"generated image does not exist: {source}")
            ext = source.suffix.lower()
            if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise ValueError(f"unsupported generated image extension: {ext or '<none>'}")
            path = images_dir / f"{safe_task}_seed{seed}{ext}"
            shutil.copy2(source, path)
            generation["source_image_path"] = str(source.resolve())

        final_path = str(path.resolve())
        generation["image_path"] = final_path
        generation["image_uri"] = final_path
        return final_path

    def _write_image_trace(self, image_uri: str | None, task: GenerationTask, row: Dict[str, Any]) -> str | None:
        if not image_uri:
            return None
        path = Path(image_uri)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            return None
        try:
            trace_root = path.parents[2] / "traces" if len(path.parents) >= 3 else path.parent / "traces"
            trace_path = trace_root / path.parent.name / f"{path.name}.trace.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace: Dict[str, Any] = {}
            if trace_path.exists():
                try:
                    existing = json.loads(trace_path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict):
                        trace.update(existing)
                except (OSError, json.JSONDecodeError):
                    trace = {}
            trace.update({
                "trace_schema_version": "gen_harness.image_trace.v1",
                "task": public_task_row(task),
                "method": row.get("method"),
                "seed": row.get("seed"),
                "final_prompt": row.get("generation", {}).get("prompt") or row.get("program", {}).get("workflow", {}).get("compiled_prompt") or row.get("program", {}).get("prompt"),
                "image_path": str(path),
                "generation": row.get("generation"),
                "program": row.get("program"),
                "fast_loop": row.get("fast_loop"),
                "latency": row.get("latency"),
                "metadata": row.get("metadata"),
                "official_eval_status": row.get("official_eval_status"),
            })
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return str(trace_path)
        except OSError:
            return None

    def _clean_method_outputs(self, output: Path, method: str) -> None:
        for path in (output / "images" / method, output / "traces" / method, output / "geneval_exports" / method):
            if path.exists():
                shutil.rmtree(path)
        stale_files = [
            output / f"{method}_generations.jsonl",
            output / f"{method}_exportable_generations.jsonl",
            output / f"{method}_results.jsonl",
            output / f"{method}_official_geneval_results.jsonl",
            output / f"{method}_official_geneval_summary.txt",
        ]
        for path in stale_files:
            if path.exists():
                path.unlink()
