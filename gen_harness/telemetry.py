from __future__ import annotations

from typing import Any, Dict

from .backend import VisualBackend
from .inspection import InspectionRunner
from .schema import GenerationTask


class CountingVisualBackend(VisualBackend):
    """Transparent counter for actual generator invocations."""

    def __init__(self, delegate: VisualBackend):
        self.delegate = delegate
        self.call_count = 0

    def generate(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
        self.call_count += 1
        return self.delegate.generate(task, program, seed=seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class CountingInspectionRunner(InspectionRunner):
    """Transparent counter for actual internal visual-verifier invocations."""

    def __init__(self, delegate: InspectionRunner):
        self.delegate = delegate
        self.call_count = 0

    def inspect(
        self,
        task: GenerationTask,
        program: Dict[str, Any],
        generation: Dict[str, Any],
        seed: int = 0,
    ) -> Dict[str, Dict[str, Any]]:
        self.call_count += 1
        return self.delegate.inspect(task, program, generation, seed=seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def counting_backend(backend: VisualBackend) -> CountingVisualBackend:
    return backend if isinstance(backend, CountingVisualBackend) else CountingVisualBackend(backend)


def counting_inspector(inspector: InspectionRunner) -> CountingInspectionRunner:
    return inspector if isinstance(inspector, CountingInspectionRunner) else CountingInspectionRunner(inspector)


def prompt_efficiency_observation(
    *,
    wall_clock_seconds: float,
    fast_loop: Dict[str, Any] | None,
    policy_extractor: Dict[str, Any] | None,
    prompt_optimizer_used: bool = False,
) -> Dict[str, Any]:
    """Summarize attempt records with explicit component-call estimates.

    Generator calls and internal visual-verifier calls are counted from actual
    recorded attempts. The fixed LLM policy extractor and optional prompt
    optimizer each contribute one call when present. Detector-only operations
    are not counted as LLM/VLM calls.
    """
    attempts = fast_loop.get("attempts", []) if isinstance(fast_loop, dict) else []
    attempt_count = len(attempts) if isinstance(attempts, list) else 0
    generator_calls = attempt_count
    visual_verifier_calls = attempt_count
    extractor_name = str((policy_extractor or {}).get("component_name") or "")
    policy_llm_calls = int(extractor_name == "openai_chat_policy_extractor")
    optimizer_calls = int(bool(prompt_optimizer_used))
    return {
        "schema": "gen_harness.prompt_efficiency_observation.v1",
        "estimated": True,
        "call_count_source": "attempt_records_and_component_presence",
        "wall_clock_seconds": max(0.0, float(wall_clock_seconds)),
        "generator_calls": generator_calls,
        "llm_vlm_calls": policy_llm_calls + optimizer_calls + visual_verifier_calls,
        "call_breakdown": {
            "policy_llm_calls": policy_llm_calls,
            "prompt_optimizer_calls": optimizer_calls,
            "visual_verifier_calls": visual_verifier_calls,
            "detector_calls_excluded_from_llm_vlm": True,
        },
    }
