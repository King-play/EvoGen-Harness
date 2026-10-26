from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .backends import VisualBackend
from .fast_loop import FastLoopController, require_full_llm_fast_loop
from .inspection import InspectionRunner
from .leakage_guard import sanitize_formal_trace_metadata
from .search import SearchBackend
from .prompt_optimizer import PromptOptimizer
from .patcher import HarnessLocalizer, PatchProposer
from .repository import HarnessRepository
from .schema import GenerationTask, ToolCall, VisualConstraint, VisualExperience
from .spatial_prompt import prompt_provenance
from .trace_protocol import TRACE_DEFAULT_K


HARNESS_COMPONENTS = ("policy", "tools", "skills", "middleware", "memory")


@dataclass
class RuntimeComponents:
    policy: Any
    middleware: Any
    skills: Any
    tools: Any
    memory: Any


class HarnessExecutor:
    """Executes and diagnoses a visual harness through file-backed components."""

    def __init__(
        self,
        repo: HarnessRepository,
        backend: VisualBackend,
        inspector: InspectionRunner,
        search_backend: SearchBackend | None = None,
        prompt_optimizer: PromptOptimizer | None = None,
        *,
        fast_loop_attempts: int = TRACE_DEFAULT_K + 1,
        harness_localizer: HarnessLocalizer | None = None,
        patch_proposer: PatchProposer | None = None,
        fast_loop_stochastic_k: int = TRACE_DEFAULT_K,
    ):
        self.repo = repo
        self.backend = backend
        self.inspector = inspector
        self.search_backend = search_backend
        self.prompt_optimizer = prompt_optimizer
        self.fast_loop_attempts = max(1, int(fast_loop_attempts))
        self.harness_localizer = harness_localizer
        self.patch_proposer = patch_proposer
        self.fast_loop_stochastic_k = max(1, int(fast_loop_stochastic_k))
        require_full_llm_fast_loop(
            inspector=self.inspector,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            max_attempts=self.fast_loop_attempts,
            stochastic_evidence_k=self.fast_loop_stochastic_k,
            context="HarnessExecutor",
        )
        self.runtime = self._build_runtime(repo)
        self.policy = self.runtime.policy
        self.middleware = self.runtime.middleware
        self.skills = self.runtime.skills
        self.tools = self.runtime.tools
        self.memory = self.runtime.memory
        # A contract is parsed once per public task for one exploration context.
        # Re-parsing it for every seed makes constraint-indexed K evidence
        # ill-defined when an LLM changes ordering or granularity.
        self._contract_cache: Dict[str, Dict[str, Any]] = {}

    def run_task(self, task: GenerationTask, seed: int = 0, write_memory: bool = False) -> List[VisualExperience]:
        fast_loop_result = None
        try:
            program, runtime = self._compile_program(task, seed=seed)
            fast_loop_result = FastLoopController(
                self.backend,
                self.inspector,
                max_attempts=self.fast_loop_attempts,
                harness_localizer=self.harness_localizer,
                patch_proposer=self.patch_proposer,
                stochastic_evidence_k=self.fast_loop_stochastic_k,
            ).run(task, program, seed=seed)
            result = fast_loop_result.generation
            inspection = fast_loop_result.inspection
            tool_result = {"fast_loop": fast_loop_result.to_dict(), "selected_generation": result}
            tool_call = ToolCall(tool_name=program["tool"]["name"], arguments={"program": program, "seed": seed}, result=tool_result).to_dict()
        except Exception as exc:
            experiences = self._exception_experiences(task, seed, exc)
            if write_memory:
                self.memory.write_experiences(experiences)
            return experiences

        experience_constraints = self._experience_constraints(task, program)
        attribution = self._llm_localizer_attribution(fast_loop_result)

        experiences: List[VisualExperience] = []
        for c in experience_constraints:
            c_inspection = inspection.get(c.constraint_id, {"passed": False, "symptom": "missing_inspection", "confidence": 0.0})
            passed = bool(c_inspection.get("passed", False))
            symptom = c_inspection.get("symptom", "unknown")
            verification_result = dict(c_inspection)
            verification_result["passed"] = passed
            verification_result["symptom"] = symptom
            verification_result["confidence"] = c_inspection.get("confidence", 1.0)
            attr = attribution.get(c.constraint_id, {})
            suspected = None if passed else self._localized_component(attr)
            selected_attempt_row = (
                fast_loop_result.attempts[fast_loop_result.selected_attempt]
                if fast_loop_result is not None
                and 0 <= fast_loop_result.selected_attempt < len(fast_loop_result.attempts)
                else {}
            )
            repair_attempt_row = next(
                (
                    attempt
                    for attempt in reversed(fast_loop_result.attempts)
                    if isinstance(attempt, dict) and attempt.get("repair_action") is not None
                ),
                {},
            ) if fast_loop_result is not None else {}
            recorded_attempt = repair_attempt_row or selected_attempt_row
            experiences.append(
                VisualExperience(
                    experience_id=self._experience_id(task.task_id, c.constraint_id, seed, program),
                    task_id=task.task_id,
                    constraint_id=c.constraint_id,
                    constraint_type=c.constraint_type,
                    constraint_text=c.text,
                    component_decisions=self._component_decisions(program),
                    tool_calls=[tool_call],
                    visual_result={"image_uri": result.get("image_uri"), "seed": seed, "raw_generation": result},
                    verification_result=verification_result,
                    failure_symptom=None if passed else symptom,
                    suspected_component=suspected,
                    repair_action=(
                        recorded_attempt.get("repair_action")
                    ),
                    repair_outcome=(
                        recorded_attempt.get("repair_outcome")
                    ),
                    metadata=_experience_metadata(
                        task,
                        attribution=attr,
                        fast_loop=fast_loop_result.to_dict() if fast_loop_result is not None else None,
                    ),
                )
            )
        if write_memory:
            self.memory.write_experiences(experiences)
        return experiences

    def _build_runtime(self, repo: HarnessRepository) -> RuntimeComponents:
        policy = repo.build_policy()
        middleware = repo.build_middleware()
        skills = repo.build_skills()
        tools = repo.build_tools(self.backend, self.inspector, self.search_backend)
        memory = repo.build_memory()
        return RuntimeComponents(policy=policy, middleware=middleware, skills=skills, tools=tools, memory=memory)

    def _compile_program(self, task: GenerationTask, seed: int = 0) -> tuple[Dict[str, Any], RuntimeComponents]:
        runtime = self._build_runtime(self.repo)
        cache_key = self._contract_cache_key(task)
        cached_contract = self._contract_cache.get(cache_key)
        if cached_contract is None:
            cached_contract = runtime.policy.compile_contract(task)
            self._contract_cache[cache_key] = copy.deepcopy(cached_contract)
        contract = copy.deepcopy(cached_contract)
        evidence = None
        if contract.get("required_evidence"):
            evidence = runtime.tools.execute_search(task, contract)

        skill_name = runtime.middleware.select_skill(task)
        bindings = runtime.middleware.bind_references(task, contract, evidence=evidence)
        tool = runtime.middleware.route_tool(task, contract, runtime.tools.artifact)
        contract_constraints = self._constraints_from_contract(contract)
        memory_context = self._memory_context(task, runtime.memory, constraints=contract_constraints)
        workflow = runtime.skills.compile_workflow(task, skill_name, contract, bindings, memory_context=memory_context)
        program = {
            "contract": contract,
            "workflow": workflow,
            "reference_bindings": bindings,
            "evidence": evidence,
            "tool": tool,
            "memory_context": memory_context,
            "component_sources": self._component_sources(self.repo),
        }
        if self.prompt_optimizer is not None and workflow.get("skill_name") == "prompt_enhancement":
            workflow["compiled_prompt_before_optimizer"] = workflow.get("compiled_prompt")
            workflow["compiled_prompt"] = self.prompt_optimizer.optimize(task, program, seed=seed)
            workflow["clip_prompt"] = workflow["compiled_prompt"]
            workflow["t5_prompt"] = workflow["compiled_prompt"]
            workflow["prompt_optimizer"] = self.prompt_optimizer.describe()
        prompt_provenance(workflow)
        decisions = runtime.middleware.trace_decisions(skill_name, bindings, tool)
        program["middleware_decisions"] = decisions
        return program, runtime

    @staticmethod
    def _contract_cache_key(task: GenerationTask) -> str:
        """Return a content key that is stable across stochastic executions."""
        payload = {
            "prompt": task.prompt,
            "task_family": task.task_family,
            "constraints": [constraint.to_dict() for constraint in task.constraints],
            "references": task.references,
            "metadata": task.metadata,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _experience_constraints(self, task: GenerationTask, program: Dict[str, Any]) -> List[VisualConstraint]:
        contract_constraints = self._constraints_from_contract(program.get("contract", {}))
        if contract_constraints:
            return contract_constraints
        return task.constraints

    def _constraints_from_contract(self, contract: Dict[str, Any]) -> List[VisualConstraint]:
        return [VisualConstraint.from_dict(c) for c in contract.get("constraints", [])]

    def _llm_localizer_attribution(self, fast_loop_result: Any) -> Dict[str, Dict[str, Any]]:
        localization: Dict[str, Any] | None = None
        attempts = getattr(fast_loop_result, "attempts", [])
        for attempt in reversed(attempts):
            if isinstance(attempt, dict) and isinstance(attempt.get("llm_localization"), dict):
                localization = attempt["llm_localization"]
                break
        if localization is None:
            row = {
                "mode": "llm_harness_localizer",
                "status": "not_invoked",
                "reason": "the K-level evidence window did not produce an actionable failure",
            }
        else:
            row = {
                "mode": "llm_harness_localizer",
                "status": "complete",
                "localization": localization,
            }
        return {
            constraint.constraint_id: copy.deepcopy(row)
            for constraint in self._constraints_from_contract(
                getattr(fast_loop_result, "attempts", [{}])[-1].get("program", {}).get("contract", {})
            )
        }

    @staticmethod
    def _localized_component(attribution: Dict[str, Any]) -> str | None:
        localization = attribution.get("localization", {}) if isinstance(attribution, dict) else {}
        frontier = localization.get("frontier", []) if isinstance(localization, dict) else []
        if isinstance(frontier, list) and frontier and str(frontier[0]) in HARNESS_COMPONENTS:
            return str(frontier[0])
        return None


    def _exception_experiences(self, task: GenerationTask, seed: int, exc: Exception) -> List[VisualExperience]:
        symptom = self._exception_symptom(exc)
        diagnostic_constraints: List[VisualConstraint] = []
        try:
            diagnostic_constraints = self.policy.extract_constraints(task)
        except Exception:
            diagnostic_constraints = []
        experiences: List[VisualExperience] = []
        decisions = {
            "execution_error": self._format_exception(exc),
            "policy": self.policy.describe(),
            "tools": {"generation_artifact": self.tools.artifact, "inspection_tool": self._safe_inspection_tool()},
            "skills": self.skills.describe(),
            "middleware": self.middleware.describe(),
            "memory": self.memory.describe(),
            "component_sources": self._component_sources(self.repo),
        }
        for c in diagnostic_constraints:
            attr = {
                "mode": "llm_harness_localizer",
                "status": "not_invoked",
                "reason": "generation failed before a complete Fast Loop evidence window",
            }
            experiences.append(
                VisualExperience(
                    experience_id=self._experience_id(task.task_id, c.constraint_id, seed, {"error": symptom}),
                    task_id=task.task_id,
                    constraint_id=c.constraint_id,
                    constraint_type=c.constraint_type,
                    constraint_text=c.text,
                    component_decisions=decisions,
                    tool_calls=[],
                    visual_result={"image_uri": None, "seed": seed, "raw_generation": None},
                    verification_result={"passed": False, "symptom": symptom, "confidence": 1.0},
                    failure_symptom=symptom,
                    suspected_component=None,
                    metadata=_experience_metadata(task, attribution=attr),
                )
            )
        return experiences

    def _memory_context(self, task: GenerationTask, memory: Any, constraints: List[VisualConstraint] | None = None) -> Dict[str, Any]:
        active_constraints = constraints if constraints is not None else task.constraints
        constraint_types = sorted({c.constraint_type for c in active_constraints})
        prompt_family = self._memory_prompt_family(task, active_constraints)
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
            "prompt_guidelines": memory.retrieve_prompt_guidelines(task_family=task.task_family, prompt_family=prompt_family, constraint_types=constraint_types, limit=10),
            "prompt_family": prompt_family,
        }

    def _memory_prompt_family(self, task: GenerationTask, constraints: List[VisualConstraint]) -> str:
        library = self.repo.build_skills()
        skill_name = self.repo.build_middleware().select_skill(task)
        skill = library.get_skill(skill_name)
        contract = {"constraints": [constraint.to_dict() for constraint in constraints]}
        return library._selected_prompt_family(task, skill, contract)

    def _component_decisions(self, program: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "policy": program["contract"],
            "tools": {
                "selected_tool": program["tool"],
                "inspection_tool": self._safe_inspection_tool(),
                "generation_artifact": self.tools.artifact,
            },
            "skills": program["workflow"],
            "middleware": program["middleware_decisions"],
            "memory": program["memory_context"],
            "component_sources": program.get("component_sources", {}),
        }

    def _component_sources(self, repo: HarnessRepository) -> Dict[str, str]:
        return {component: str(repo.component_dir(component)) for component in HARNESS_COMPONENTS}

    def _safe_inspection_tool(self) -> Dict[str, Any]:
        try:
            return self.tools.get_inspection_tool()
        except Exception as exc:
            return {"error": self._format_exception(exc)}

    def _exception_symptom(self, exc: Exception) -> str:
        msg = str(exc)
        if "constraint type not in ontology" in msg:
            return "constraint_not_in_contract"
        if "vision inspection request failed" in msg or "inspection request failed" in msg:
            return "inspection_error"
        if "missing required reference" in msg:
            return "missing_required_reference"
        if "unsupported reference binding mode" in msg:
            return "unsupported_reference_binding_mode"
        if "required capabilities" in msg or "routing_warning" in msg:
            return "missing_tool_capability"
        return "harness_execution_error"

    def _format_exception(self, exc: Exception) -> str:
        return f"{exc.__class__.__name__}: {exc}"

    def _experience_id(self, task_id: str, constraint_id: str, seed: int, program: Dict[str, Any]) -> str:
        h = hashlib.sha1(repr((task_id, constraint_id, seed, program)).encode()).hexdigest()[:12]
        return f"exp_{task_id}_{constraint_id}_{h}"

def _experience_metadata(task: GenerationTask, **extra: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"task_family": task.task_family}
    safe_task_metadata = sanitize_formal_trace_metadata(task.metadata)
    if isinstance(safe_task_metadata, dict):
        metadata.update(safe_task_metadata)
    metadata.update(extra)
    return metadata
