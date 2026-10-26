"""Request-driven rendering and evidence-preserving diagnostics."""
from pathlib import Path

import pytest

from gen_harness.components.skills import VisualSkillLibrary
from gen_harness.debugger import FailureDebugger
from gen_harness.executor import HarnessExecutor
from gen_harness.repository import HarnessRepository
from gen_harness.schema import GenerationTask, VisualExperience
from gen_harness.validator import PatchValidator


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["pizza", "lamp", "unlisted ceremonial object"])
@pytest.mark.parametrize("count", [1, 2, 6, 11])
def test_inventory_preserves_arbitrary_classes_and_counts(name, count):
    library = VisualSkillLibrary(ROOT / "examples/visual_harness/skills/skills.json")
    task = GenerationTask("request", f"{count} {name}", "compositional", [])
    contract = {"constraints": [{
        "constraint_id": "inventory", "constraint_type": "count", "text": task.prompt,
        "metadata": {"required_objects": [{"class": name, "count": count}]},
    }]}
    workflow = library.compile_workflow(task, "compositional_generation", contract, {})
    for channel in ("clip_prompt", "t5_prompt"):
        assert f"Exactly {count} visible instances of {name}." in workflow[channel]
        assert "Do not include a second pizza" not in workflow[channel]
        assert "2x3 grid" not in workflow[channel]


def test_unknown_and_shared_attributes_are_preserved():
    library = VisualSkillLibrary(ROOT / "examples/visual_harness/skills/skills.json")
    task = GenerationTask("attributes", "two iridescent objects", "compositional", [])
    contract = {"constraints": [{
        "constraint_id": "attributes", "constraint_type": "attribute", "text": task.prompt,
        "metadata": {"attribute_bindings": [
            {"object": "cup", "attribute": "iridescent"},
            {"object": "plate", "attribute": "iridescent"},
        ]},
    }]}
    workflow = library.compile_workflow(task, "compositional_generation", contract, {})
    assert "cup has the requested attribute iridescent" in workflow["t5_prompt"]
    assert "plate has the requested attribute iridescent" in workflow["t5_prompt"]
    assert "only on assigned objects" not in workflow["t5_prompt"]


def test_diagnostics_retain_actual_llm_attribution_instead_of_fixed_confidence():
    experience = VisualExperience(
        experience_id="experience", task_id="request", constraint_id="count",
        constraint_type="count", constraint_text="two cups", component_decisions={}, tool_calls=[],
        visual_result={}, verification_result={"passed": False},
        failure_symptom="object_count_mismatch", suspected_component=None,
        metadata={"attribution": {"mode": "llm_harness_localizer", "localization": {
            "frontier": ["policy"], "scores": {"policy": 0.63},
            "rationale": "The explicit count is missing from the requirement record.",
        }}},
    )
    diagnosis = FailureDebugger().diagnose(experience)
    assert diagnosis.suspected_component == "policy"
    assert diagnosis.confidence == 0.63
    assert diagnosis.root_cause == experience.metadata["attribution"]["localization"]["rationale"]
    experience.metadata.clear()
    assert FailureDebugger().diagnose(experience).confidence is None


def test_detector_queries_preserve_unlisted_and_distinct_class_names():
    from scripts.detector_utils import required_classes
    payload = {"task": {"constraints": [{"metadata": {"required_objects": [
        {"class": "mug"}, {"class": "cup"}, {"class": "unlisted ceremonial object"},
    ]}}]}}
    assert required_classes(payload) == ["mug", "cup", "unlisted ceremonial object"]


def test_memory_routing_uses_configured_skill_instead_of_object_keywords():
    executor = object.__new__(HarnessExecutor)
    executor.repo = HarnessRepository(ROOT / "examples/visual_harness")
    families = []
    for name in ("person", "apple", "unlisted ceremonial object"):
        task = GenerationTask.from_dict({
            "task_id": name, "prompt": f"two {name}", "task_family": "compositional",
            "constraints": [{"constraint_id": "count", "constraint_type": "count", "text": "two objects",
                             "metadata": {"required_objects": [{"class": name, "count": 2}]}}],
        })
        families.append(executor._memory_prompt_family(task, task.constraints))
    assert len(set(families)) == 1


def test_validation_without_evidence_cannot_report_a_perfect_score():
    validator = object.__new__(PatchValidator)
    validator._run_experiences = lambda tasks, seeds: iter(())
    task = GenerationTask("request", "two cups", "compositional", [])
    with pytest.raises(ValueError, match="no visual verification evidence"):
        validator._evaluate_tasks([task], [0])
    with pytest.raises(ValueError, match="non-empty task set"):
        validator._evaluate_tasks([], [0])
