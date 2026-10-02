from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend
from gen_harness.repository import HarnessRepository
from gen_harness.schema import GenerationTask


def test_all_geneval2_prompts_fit_both_real_clip_tokenizers_by_whole_clause(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    tasks_path = root / "data/geneval2/geneval2_tasks_prompt_only_2026_09_04.jsonl"
    configured_root = os.environ.get("GENHARNESS_FLOW_GRPO_MODEL_ROOT")
    if not configured_root:
        pytest.skip("GENHARNESS_FLOW_GRPO_MODEL_ROOT is required for real SD3.5 tokenizer coverage")
    model_root = Path(configured_root)
    base_model = model_root / "stable-diffusion-3.5-medium"
    if not (base_model / "tokenizer").is_dir() or not (base_model / "tokenizer_2").is_dir():
        pytest.skip("local SD3.5 CLIP tokenizers are unavailable")

    transformers = pytest.importorskip("transformers")
    pipe = type("TokenizerPair", (), {})()
    pipe.tokenizer = transformers.AutoTokenizer.from_pretrained(base_model / "tokenizer", local_files_only=True)
    pipe.tokenizer_2 = transformers.AutoTokenizer.from_pretrained(base_model / "tokenizer_2", local_files_only=True)
    backend = FlowGRPOBackend({"clip_max_tokens": 77, "split_prompt_channels": True})
    backend._pipe = pipe
    from gen_harness.components.policy import LLMPolicyExtractor

    monkeypatch.setattr(
        LLMPolicyExtractor,
        "_request_constraints",
        lambda self, task, policy_artifact, correction=None: {
            "constraints": [constraint.to_dict() for constraint in task.constraints]
        },
    )
    policy_config = tmp_path / "policy_extractor.json"
    policy_config.write_text(
        json.dumps(
            {
                "type": "openai_chat_policy_extractor",
                "base_url": "https://llm.invalid/v1",
                "model": "test-policy-llm",
            }
        ),
        encoding="utf-8",
    )
    repo = HarnessRepository(
        root / "examples/visual_harness",
        policy_extractor_config=policy_config,
    )
    policy = repo.build_policy()
    skills = repo.build_skills()

    rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 800
    for row in rows:
        task = GenerationTask.from_dict(row)
        contract = policy.compile_contract(task)
        workflow = skills.compile_workflow(task, "compositional_generation", contract, {}, memory_context={})
        source_segments = list(workflow["clip_prompt_segments"])
        fitted = backend._fit_compiled_clip_segments(task, workflow, clip_max_tokens=77)
        selected = []
        remainder = fitted
        for segment in source_segments:
            if remainder == segment or remainder.startswith(segment + " "):
                selected.append(segment)
                remainder = remainder[len(segment) :].lstrip()
        assert remainder == ""
        assert fitted == " ".join(selected)
        counts = backend._validate_clip_prompt_budget(task, fitted, clip_max_tokens=77)
        assert counts and max(counts.values()) <= 77

        constraints = contract.get("constraints", [])
        required = {
            str(obj.get("class"))
            for constraint in constraints
            for obj in constraint.get("metadata", {}).get("required_objects", [])
            if isinstance(obj, dict) and obj.get("class")
        }
        for constraint in constraints:
            metadata = constraint.get("metadata", {})
            for relation in metadata.get("spatial_relations", []):
                assert relation["subject"] in required
                assert relation["object"] in required
            for binding in metadata.get("attribute_bindings", []):
                assert binding["object"] in required
