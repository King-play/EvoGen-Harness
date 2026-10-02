import json

import pytest

from gen_harness.miner import WeaknessMiner
from gen_harness.patcher import COMPONENT_ARTIFACTS, COMPONENT_STATE_ARTIFACTS, HarnessLocalizer
from gen_harness.schema import GenerationTask, VisualExperience
from gen_harness.trace_protocol import (
    TRACE_DEFAULT_LLM_MODEL_SNAPSHOT,
    TRACE_EVIDENCE_SCHEMA,
    validate_fixed_llm_config,
    validate_localizer_output,
    validate_paper_dataset,
    validate_trace_evidence,
)


def _experience(seed, *, passed):
    return VisualExperience(
        experience_id=f"exp_task_constraint_{seed}",
        task_id="task",
        constraint_id="constraint_0",
        constraint_type="count",
        constraint_text="two red cups",
        component_decisions={"skills": {"skill_name": "compositional_generation"}},
        tool_calls=[],
        visual_result={"seed": seed, "backend": "test"},
        verification_result={
            "passed": passed,
            "symptom": "pass" if passed else "object_count_mismatch",
            "confidence": 1.0,
        },
        failure_symptom=None if passed else "object_count_mismatch",
        suspected_component=None,
    )


def _valid_localizer_row():
    return {
        "scores": {
            "policy": 0.1,
            "tools": 0.2,
            "skills": 0.8,
            "middleware": 0.1,
            "memory": 0.0,
            "no_patch": 0.05,
        },
        "constraint_attributions": [
            {
                "constraint_id": "constraint_0",
                "scores": {
                    "policy": 0.1,
                    "tools": 0.2,
                    "skills": 0.8,
                    "middleware": 0.1,
                    "memory": 0.0,
                    "no_patch": 0.05,
                },
                "primary_component": "skills",
                "confidence": 0.8,
                "evidence_summary": "constraint_0 fails under one stochastic execution and points to reusable skill coverage.",
            }
        ],
        "rationale": "The repeated constraint-indexed evidence points primarily to the reusable skill procedure.",
    }


def test_constraint_indexed_evidence_contains_all_k_outcomes():
    weaknesses = WeaknessMiner(min_support=1, expected_stochastic_conditions=4).mine(
        [_experience(0, passed=False), _experience(1, passed=True), _experience(2, passed=True), _experience(3, passed=True)]
    )

    evidence = weaknesses[0].evidence["trace_evidence"]
    assert evidence["schema"] == TRACE_EVIDENCE_SCHEMA
    assert evidence["aggregation"]["exploration_complete_for_expected_k"] is True
    assert evidence["aggregation"]["failure_support_complete_for_all_k"] is False
    instance = evidence["constraint_instances"][0]
    assert instance["stochastic_conditions"] == [0, 1, 2, 3]
    assert [row["outcome"] for row in instance["executions"]] == ["failure", "pass", "pass", "pass"]
    validate_trace_evidence(evidence, expected_k=4)


def test_constraint_indexed_evidence_rejects_missing_stochastic_execution():
    with pytest.raises(ValueError, match="K executions"):
        WeaknessMiner(min_support=1, expected_stochastic_conditions=4).mine(
            [_experience(0, passed=False), _experience(1, passed=True), _experience(2, passed=True)]
        )


def test_localizer_output_is_exact_and_bounded():
    normalized = validate_localizer_output(_valid_localizer_row(), expected_constraint_ids=["constraint_0"])
    assert normalized["scores"]["skills"] == 0.8
    assert normalized["constraint_attributions"][0]["primary_component"] == "skills"

    extra = _valid_localizer_row()
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_localizer_output(extra)

    out_of_range = _valid_localizer_row()
    out_of_range["scores"]["skills"] = 1.1
    with pytest.raises(ValueError, match="within"):
        validate_localizer_output(out_of_range)

    missing_constraint = _valid_localizer_row()
    missing_constraint["constraint_attributions"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        validate_localizer_output(missing_constraint)

    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_localizer_output(_valid_localizer_row(), expected_constraint_ids=["other_constraint"])


def test_formal_trace_requires_the_pinned_gpt41_snapshot():
    valid = validate_fixed_llm_config(
        {
            "model": "gpt-4.1",
            "model_snapshot": TRACE_DEFAULT_LLM_MODEL_SNAPSHOT,
        },
        role="Harness Localizer",
    )
    assert valid["model_snapshot"] == "gpt-4.1-2025-04-14"
    with pytest.raises(ValueError, match="snapshot"):
        validate_fixed_llm_config(
            {"model": "gpt-4.1", "model_snapshot": "gpt-4.1-pinned-snapshot"},
            role="Harness Localizer",
        )


def test_strict_localizer_sends_the_pinned_snapshot_and_complete_evidence(monkeypatch):
    captured = []

    def fake_post(config, endpoint, body, timeout):
        captured.append(body)
        return {"choices": [{"message": {"content": json.dumps(_valid_localizer_row())}}]}

    monkeypatch.setattr("gen_harness.patcher._post_chat_json", fake_post)
    weakness = WeaknessMiner(min_support=1, expected_stochastic_conditions=4).mine(
        [_experience(0, passed=False), _experience(1, passed=True), _experience(2, passed=True), _experience(3, passed=True)]
    )[0]
    localizer = HarnessLocalizer(
        {
            "type": "openai_chat_harness_localizer",
            "base_url": "http://example.invalid",
            "model": "gpt-4.1",
            "model_snapshot": TRACE_DEFAULT_LLM_MODEL_SNAPSHOT,
        },
        strict_protocol=True,
    )

    result = localizer.localize(weakness)

    assert result.frontier == ["skills", "tools", "policy"]
    request = captured[0]
    assert request["model"] == TRACE_DEFAULT_LLM_MODEL_SNAPSHOT
    payload = json.loads(request["messages"][1]["content"])
    assert payload["evidence_schema"] == TRACE_EVIDENCE_SCHEMA
    assert payload["weakness"]["evidence"]["trace_evidence"]["aggregation"]["expected_k"] == 4


def test_paper_dataset_protocol_requires_exact_public_split_counts():
    task = GenerationTask(
        task_id="dev",
        prompt="a red cup",
        task_family="compositional",
        constraints=[],
        metadata={"split": "target"},
    )
    development = validate_paper_dataset([task], protocol="development")
    assert development["split_counts"] == {"target": 1}
    with pytest.raises(ValueError, match="exact target/heldout/preservation counts"):
        validate_paper_dataset([task], protocol="paper_p2")


def test_memory_edit_boundary_is_explicit():
    assert COMPONENT_ARTIFACTS["memory"] == {"prompt_guidelines.jsonl"}
    assert {
        "visual_experiences.jsonl",
        "failure_clusters.jsonl",
        "weaknesses.jsonl",
        "validated_patches.jsonl",
        "regression_cases.jsonl",
    } <= COMPONENT_STATE_ARTIFACTS["memory"]
