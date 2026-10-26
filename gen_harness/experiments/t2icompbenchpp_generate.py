from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..backend import VisualBackend
from ..fast_loop import FastLoopController, require_full_llm_fast_loop
from ..inspection import InspectionRunner
from ..io import append_jsonl, write_json, write_jsonl
from ..patcher import HarnessLocalizer, PatchProposer, build_harness_localizer, build_patch_proposer
from ..prompt_optimizer import PromptOptimizer
from ..repository import HarnessRepository
from ..schema import GenerationTask
from ..search import SearchBackend
from ..tasks import load_tasks
from ..telemetry import prompt_efficiency_observation


SAFE_T2ICOMPBENCHPP_REPORT_METADATA_KEYS = {
    "split",
    "benchmark",
    "category",
    "source_index",
    "complex_subtype",
    "generation_input",
}


class T2ICompBenchPPImageGenerationExperiment:
    """Generate T2ICompBench++ images through the full Gen-Harness path.

    This command is generation-only.  It never calls T2ICompBench++ scoring,
    never reads official answers, and never substitutes another image after a
    failed generation. Multiple attempts are the bounded Gen-Harness fast loop
    driven only by the configured internal inspector.
    """

    def __init__(
        self,
        harness: str | Path,
        backend: VisualBackend,
        search_backend: SearchBackend | None = None,
        prompt_optimizer: PromptOptimizer | None = None,
        inspector: InspectionRunner | None = None,
        fast_loop_attempts: int = 5,
        policy_extractor_config: str | Path | None = None,
        fast_loop_stochastic_k: int = 4,
        harness_localizer_config: str | Path | None = None,
        patch_proposer_config: str | Path | None = None,
        harness_localizer: HarnessLocalizer | None = None,
        patch_proposer: PatchProposer | None = None,
    ):
        if policy_extractor_config is None:
            raise ValueError("policy_extractor_config is required for formal T2ICompBench++ generation")
        self.harness = Path(harness)
        self.repo = HarnessRepository(
            self.harness,
            policy_extractor_config=policy_extractor_config,
        )
        self.backend = backend
        self.search_backend = search_backend
        self.prompt_optimizer = prompt_optimizer
        self.inspector = inspector
        self.fast_loop_attempts = int(fast_loop_attempts)
        self.fast_loop_stochastic_k = int(fast_loop_stochastic_k)
        self.harness_localizer = harness_localizer
        self.patch_proposer = patch_proposer
        if harness_localizer_config is not None:
            self.harness_localizer = build_harness_localizer(harness_localizer_config, repo=self.repo)
        if patch_proposer_config is not None:
            self.patch_proposer = build_patch_proposer(
                patch_proposer_config,
                memory=self.repo.build_memory(),
                repo=self.repo,
            )
        require_full_llm_fast_loop(
            inspector=self.inspector,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            max_attempts=self.fast_loop_attempts,
            stochastic_evidence_k=self.fast_loop_stochastic_k,
            context="formal T2ICompBench++ generation",
        )

    def run(
        self,
        tasks_path: str | Path,
        output_dir: str | Path,
        *,
        methods: Sequence[str],
        seeds: Sequence[int],
        limit: int | None = None,
        continue_on_error: bool = False,
    ) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        source_tasks = load_tasks(tasks_path)
        if limit is not None:
            source_tasks = source_tasks[:limit]
        tasks = [self._prompt_only_task(task) for task in source_tasks]
        seed_list = list(seeds) or [0]
        method_list = [self._normalize_method(method) for method in methods if method.strip()]

        summaries: Dict[str, Any] = {}
        for method in method_list:
            self._clean_method_outputs(output, method)
            result_path = output / f"{method}_generations.jsonl"
            write_jsonl(result_path, [])
            try:
                rows = self._run_method(method, tasks, source_tasks, seed_list, output, result_path)
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
            missing_images = sum(1 for row in rows if not row.get("image_uri"))
            if missing_images and not continue_on_error:
                raise ValueError(f"{method} has {missing_images} rows without exportable images")
            summaries[method] = {
                "num_rows": len(rows),
                "num_generated_images": sum(1 for row in rows if row.get("image_uri")),
                "final_images_dir": str((output / "images" / method).resolve()),
                "num_errors": sum(1 for row in rows if row.get("error")),
                "num_internal_inspection_attempts": sum(len(row.get("fast_loop", {}).get("attempts", [])) for row in rows),
                "num_fast_loop_successes": sum(1 for row in rows if row.get("fast_loop", {}).get("passed")),
                "num_fast_loop_strictly_confirmed": sum(1 for row in rows if row.get("fast_loop", {}).get("strictly_confirmed")),
                "missing_images": missing_images,
                "official_eval_status": "pending" if not missing_images else "partial_pending",
            }

        latency_summary = self._latency_summary(method_list, output)
        write_json(output / "latency_summary.json", latency_summary)
        summary = {
            "benchmark": "t2icompbenchpp",
            "mode": "official_compatible_generation_only",
            "num_tasks": len(source_tasks),
            "seeds": seed_list,
            "policy_extractor_manifest": self.repo.policy_extractor_manifest(),
            "methods": summaries,
            "latency_summary": str(output / "latency_summary.json"),
            "generation_integrity_guards": {
                "official_evaluator_used_for_generation_or_selection": False,
                "official_labels_used_for_prompt_compilation": False,
                "no_replacement_after_generation": True,
                "posthoc_candidate_selection_allowed": False,
                "internal_generated_candidate_selection_allowed": True,
                "internal_fast_loop_attempt_policy": "bounded_internal_inspection_only",
            },
            "notes": [
                "T2ICompBench++ official-compatible scoring is not called in this path.",
                "Generation receives only prompt text plus public prompt provenance metadata.",
                "Internal inspection is used only for bounded fast-loop candidate selection and repair.",
                "Run export-eval --dataset t2icompbenchpp followed by the official-compatible evaluator to report scores.",
            ],
        }
        write_json(output / "t2icompbenchpp_generation_summary.json", summary)
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
    ) -> List[Dict[str, Any]]:
        if len(tasks) != len(source_tasks):
            raise ValueError("prompt-only task list and source task list must have the same length")
        if method == "gen_harness":
            return self._run_gen_harness(tasks, source_tasks, seeds, output, result_path, method)
        raise ValueError(f"unsupported T2ICompBench++ generation method: {method}")

    def _run_gen_harness(
        self,
        tasks: Sequence[GenerationTask],
        source_tasks: Sequence[GenerationTask],
        seeds: Sequence[int],
        output: Path,
        result_path: Path,
        method: str,
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
                contract = policy.compile_contract(task)
                evidence = None
                if contract.get("required_evidence"):
                    if self.search_backend is None:
                        raise ValueError("visual contract requires search/evidence, but no search backend was configured")
                    evidence = self.search_backend.search(task, contract)
                skill_name = middleware.select_skill(task)
                bindings = middleware.bind_references(task, contract, evidence=evidence)
                tool = middleware.route_tool(task, contract, tool_metadata)
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
                    "t2icompbenchpp_generation_input": "prompt_only",
                }
                if self.prompt_optimizer is not None and workflow.get("skill_name") == "prompt_enhancement":
                    workflow["compiled_prompt_before_optimizer"] = workflow.get("compiled_prompt")
                    workflow["compiled_prompt"] = self.prompt_optimizer.optimize(task, program, seed=seed)
                    workflow["prompt_optimizer"] = self.prompt_optimizer.describe()
                row = self._generate_row(method, task, source_task, seed, program, output)
                wall_clock_seconds = time.perf_counter() - execution_started
                generation_seconds = float(row.get("latency", {}).get("row_seconds", 0.0) or 0.0)
                row.setdefault("latency", {})["harness_compile_seconds"] = round(
                    max(0.0, wall_clock_seconds - generation_seconds), 6
                )
                row["latency"]["wall_clock_seconds"] = round(wall_clock_seconds, 6)
                row["efficiency"] = prompt_efficiency_observation(
                    wall_clock_seconds=wall_clock_seconds,
                    fast_loop=row.get("fast_loop"),
                    policy_extractor=contract.get("policy_extractor"),
                    prompt_optimizer_used=bool(workflow.get("prompt_optimizer")),
                )
                rows.append(row)
                append_jsonl(result_path, [row])
        return rows

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
            generation.setdefault("image_origin", "fresh_generation_for_task")
            image_path = self._save_image(generation, output, method, task.task_id, seed)
            image_uri = image_path or generation.get("image_path") or generation.get("image_uri")
            latency = self._row_latency(
                method,
                row_started,
                fast_loop=fast_loop,
                generation_latency=None,
            )
            row = {
                "method": method,
                "task_id": task.task_id,
                "seed": seed,
                "image_uri": image_uri,
                "generation": generation,
                "program": program,
                "component_decisions": self._component_decisions(program),
                "metadata": self._formal_row_metadata(task),
                "export_metadata": self._safe_source_metadata(source_task),
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
                "export_metadata": self._safe_source_metadata(source_task),
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

    def _prompt_only_task(self, task: GenerationTask) -> GenerationTask:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        return GenerationTask(
            task_id=task.task_id,
            prompt=task.prompt,
            task_family=task.task_family,
            constraints=[],
            references={},
            metadata={"split": metadata.get("split")} if metadata.get("split") else {},
        )

    def _formal_row_metadata(self, task: GenerationTask) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"task_family": task.task_family}
        if isinstance(task.metadata, dict) and task.metadata.get("split"):
            metadata["split"] = task.metadata["split"]
        metadata["generation_input"] = "prompt_only_no_official_eval_outputs"
        return metadata

    def _safe_source_metadata(self, task: GenerationTask) -> Dict[str, Any]:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        safe = {
            key: metadata[key]
            for key in SAFE_T2ICOMPBENCHPP_REPORT_METADATA_KEYS
            if key in metadata
        }
        if safe.get("benchmark") not in {None, "t2icompbenchpp"}:
            safe.pop("benchmark", None)
        safe.setdefault("generation_input", "prompt_only_no_official_eval_outputs")
        return safe

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

    def _memory_context(self, task: GenerationTask, memory: Any, contract: Dict[str, Any] | None = None) -> Dict[str, Any]:
        constraint_types = sorted(
            {str(c.get("constraint_type")) for c in (contract or {}).get("constraints", [])}
            or {c.constraint_type for c in task.constraints}
        )
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
        from .geneval_generate import GenEvalImageGenerationExperiment

        helper = object.__new__(GenEvalImageGenerationExperiment)
        return helper._save_image(generation, output, method, task_id, seed)

    def _write_image_trace(self, image_uri: str | None, task: GenerationTask, row: Dict[str, Any]) -> str | None:
        from .geneval_generate import GenEvalImageGenerationExperiment

        helper = object.__new__(GenEvalImageGenerationExperiment)
        return helper._write_image_trace(image_uri, task, row)

    def _row_latency(
        self,
        method: str,
        row_started: float,
        *,
        fast_loop: Dict[str, Any] | None,
        generation_latency: float | None,
    ) -> Dict[str, Any]:
        from .geneval_generate import GenEvalImageGenerationExperiment

        helper = object.__new__(GenEvalImageGenerationExperiment)
        return helper._row_latency(method, row_started, fast_loop=fast_loop, generation_latency=generation_latency)

    def _latency_summary(self, methods: Sequence[str], output: Path) -> Dict[str, Any]:
        from .geneval_generate import GenEvalImageGenerationExperiment

        helper = object.__new__(GenEvalImageGenerationExperiment)
        return helper._latency_summary(methods, output)

    def _clean_method_outputs(self, output: Path, method: str) -> None:
        for path in (output / "images" / method, output / "traces" / method):
            if path.exists():
                shutil.rmtree(path)
        stale_files = [
            output / f"{method}_generations.jsonl",
            output / f"{method}_exportable_generations.jsonl",
            output / f"{method}_results.jsonl",
        ]
        for path in stale_files:
            if path.exists():
                path.unlink()
