from __future__ import annotations

from gen_harness.fast_loop import FastLoopController
from gen_harness.patcher import GenerationPatch, HarnessLocalization, PatchProposer
from gen_harness.schema import GenerationTask
from gen_harness.backend import VisualBackend
from gen_harness.inspection import InspectionRunner


class _Backend(VisualBackend):
    def __init__(self, events):
        self.events = events

    def generate(self, task, program, seed=0):
        self.events.append(("generate", seed, program["workflow"]["compiled_prompt"]))
        return {"image_uri": f"/tmp/llm_fast_loop_{seed}.png", "seed": seed}


class _Inspector(InspectionRunner):
    def __init__(self, events):
        self.events = events

    def inspect(self, task, program, generation, seed=0):
        self.events.append(("inspect", seed))
        if seed in {0, 1}:
            return {
                "objects": {
                    "passed": False,
                    "symptom": "missing_required_object",
                    "missing_required_objects": [{"class": "flower", "expected": 1, "found": 0}],
                }
            }
        return {"objects": {"passed": True, "symptom": "pass", "constraint_score": 1.0}}


class _Localizer:
    def __init__(self, events):
        self.events = events

    def localize(self, weakness, *, frontier_size=3):
        self.events.append(("localize", weakness.evidence["trace_evidence"]["schema"]))
        return HarnessLocalization(
            weakness_id=weakness.weakness_id,
            scores={
                "policy": 0.0,
                "tools": 0.0,
                "skills": 0.9,
                "middleware": 0.0,
                "memory": 0.0,
                "no_patch": 0.0,
            },
            frontier=["skills"],
            no_patch_preferred=False,
            rationale="The failed object constraint should be repaired in the reusable visual skill.",
        )


class _Proposer:
    def __init__(self, events):
        self.events = events

    def propose_generation_patch(self, weakness, localization, program):
        self.events.append(("propose", localization.frontier))
        return GenerationPatch(
            target_component="skills",
            operation="append_prompt_clause",
            path=["workflow", "t5_prompt"],
            value="Show one complete flower, centered and clearly recognizable.",
            predicted_improvement="Make the missing required object explicit to the image generator.",
            possible_regression="A longer prompt may reduce room for unrelated details.",
        )


def test_fast_loop_uses_serial_llm_localize_propose_apply_before_second_generation():
    events = []
    task = GenerationTask.from_dict(
        {
            "task_id": "llm_fast_loop_serial",
            "prompt": "a flower on a plain background",
            "task_family": "compositional",
            "constraints": [
                {
                    "constraint_id": "objects",
                    "text": "one flower",
                    "constraint_type": "count",
                    "metadata": {
                        "required_objects": [{"class": "flower", "count": 1}],
                    },
                }
            ],
        }
    )
    result = FastLoopController(
        _Backend(events),
        _Inspector(events),
        max_attempts=3,
        harness_localizer=_Localizer(events),
        patch_proposer=_Proposer(events),
        stochastic_evidence_k=2,
    ).run(
        task,
        {
            "workflow": {
                "compiled_prompt": "a flower on a plain background",
                "t5_prompt": "a flower on a plain background",
                "t5_prompt_segments": ["a flower on a plain background"],
                "t5_prompt_segment_types": ["source"],
            }
        },
        seed=0,
    )

    assert result.passed is True
    assert [event[0] for event in events] == [
        "generate",
        "inspect",
        "generate",
        "inspect",
        "localize",
        "propose",
        "generate",
        "inspect",
    ]
    assert result.attempts[0]["repair_decision"]["action"] == "collect_stochastic_evidence"
    assert result.attempts[1]["repair_decision"]["action"] == "retry_with_llm_repair"
    assert result.attempts[1]["llm_patch"]["target_component"] == "skills"
    assert result.attempts[2]["repair_action"]["source"] == "llm_localizer_proposer"
    assert "Show one complete flower" in result.attempts[2]["prompt_channels"]["t5_prompt"]


def test_generation_patch_schema_rejects_unallowed_path():
    proposer = PatchProposer(
        {
            "type": "openai_chat_patch_proposer",
            "base_url": "https://example.invalid/v1",
            "model": "test",
        }
    )
    localization = HarnessLocalization(
        weakness_id="w",
        scores={
            "policy": 0.0,
            "tools": 0.0,
            "skills": 1.0,
            "middleware": 0.0,
            "memory": 0.0,
            "no_patch": 0.0,
        },
        frontier=["skills"],
        no_patch_preferred=False,
    )
    row = {
        "target_component": "skills",
        "operation": "append_prompt_clause",
        "path": ["contract", "constraints"],
        "value": "invalid",
        "predicted_improvement": "reason",
        "possible_regression": "risk",
    }

    try:
        proposer._generation_patch_from_row(row, localization)
    except ValueError as exc:
        assert "T5 workflow prompt channel" in str(exc)
    else:
        raise AssertionError("unallowed generation patch path was accepted")


def test_generation_patch_updates_t5_segments_consumed_by_flow_backend():
    from gen_harness.fast_loop import FastLoopController

    controller = object.__new__(FastLoopController)
    program = {
        "workflow": {
            "compiled_prompt": "source contract",
            "t5_prompt": "source contract",
            "t5_prompt_segments": ["source contract"],
            "t5_prompt_segment_types": ["source"],
        }
    }
    patch = GenerationPatch(
        target_component="skills",
        operation="append_prompt_clause",
        path=["workflow", "t5_prompt"],
        value="Make the required object unmistakable.",
        predicted_improvement="preserve the contract while adding visual evidence",
        possible_regression="slightly longer T5 prompt",
    )

    patched = controller._apply_generation_patch(program, patch)

    assert patched["workflow"]["t5_prompt_segments"] == [
        "source contract",
        "Make the required object unmistakable.",
    ]
    assert patched["workflow"]["t5_prompt_segment_types"] == ["source", "llm_repair"]
