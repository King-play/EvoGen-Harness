import json
import re
import shutil
from pathlib import Path

from gen_harness.backend import VisualBackend
from gen_harness.evaluation.wise import build_wise_tasks, export_wise_manifest
from gen_harness.evaluation.trace_metrics import summarize_attribution, summarize_efficiency, summarize_paired_outcomes, summarize_responsibility_updates
from gen_harness.experiments.self_evolve_loop import SelfEvolveRunner
from gen_harness.experiments.slow_loop import estimate_slow_loop_budget
from gen_harness.faults import FaultInjector
from gen_harness.inspection import ObjectRequirementInspection
from gen_harness.patcher import HarnessLocalization, PatchProposer
from gen_harness.repository import HarnessRepository
from gen_harness.schema import GenerationTask, PatchManifest
from gen_harness.validator import ValidationReport
from gen_harness.cli import _parse_trace_seeds
from gen_harness.telemetry import prompt_efficiency_observation


def test_wise_adapter_is_prompt_only_and_exports_frozen_images(tmp_path):
    source = tmp_path / "wise.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "culture-1",
                "prompt": "A historically accurate public festival scene",
                "domain": "culture",
                "answer": "private evaluator label",
                "wiscore": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tasks_path = tmp_path / "tasks.jsonl"
    tasks = build_wise_tasks(source, tasks_path)
    serialized = json.dumps(tasks)
    assert "private evaluator label" not in serialized
    assert "wiscore" not in serialized.lower()
    assert tasks[0]["task_family"] == "world_knowledge"

    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    experiences = tmp_path / "experiences.jsonl"
    experiences.write_text(
        json.dumps(
            {
                "task_id": tasks[0]["task_id"],
                "visual_result": {"seed": 3, "image_uri": str(image)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "wise_manifest.jsonl"
    rows = export_wise_manifest(tasks_path, experiences, output, seed=3)
    assert rows[0]["id"] == "culture-1"
    audit = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert audit["num_mapped_prompts"] == 1
    assert audit["official_scores_used_for_selection"] is False


def test_patch_manifest_exposes_paper_edit_tuple():
    patch = PatchManifest(
        patch_id="p",
        target_component="policy",
        changed_artifact="visual_contract.json",
        changed_fields=["verification_rules.count"],
        operation="set_path",
        payload={"path": ["verification_rules", "count"], "value": "exact"},
        supporting_experience=["e"],
        predicted_improvement="count accuracy",
        possible_regression="none",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="policy/visual_contract.json",
        edit_operator="REPLACE",
        edit_key="verification_rules.count",
        old_value="soft",
        new_value="exact",
    )
    assert patch.to_dict()["canonical_edit"] == {
        "responsibility": "policy",
        "operator": "REPLACE",
        "key": "verification_rules.count",
        "old_value": "soft",
        "new_value": "exact",
    }
    assert PatchManifest.from_dict(patch.to_dict()).canonical_edit()["operator"] == "REPLACE"


def test_trace_report_records_cost_aware_utility():
    report = ValidationReport(
        patch_id="p",
        accepted=True,
        before={"target": 0.2, "heldout": 0.2, "preservation": 1.0},
        after={"target": 0.5, "heldout": 0.4, "preservation": 1.0},
        target_delta=0.3,
        heldout_delta=0.2,
        regression_delta=0.0,
        reason="accepted",
        seeds=[0, 1, 2, 3],
        execution_cost_delta=1.0,
        utility=0.25,
        utility_terms={
            "target_delta": 0.3,
            "preservation_weight": 0.5,
            "execution_cost_weight": 0.05,
        },
    )
    row = report.to_dict()
    assert row["utility"] == 0.25
    assert row["seeds"] == [0, 1, 2, 3]
    assert _parse_trace_seeds(None, 0) == [0, 1, 2, 3]
    assert _parse_trace_seeds("4,8", 0) == [4, 8]


def test_repository_restores_all_persistent_component_files(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "visual_harness"
    harness = tmp_path / "harness"
    shutil.copytree(source, harness)
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
        harness,
        policy_extractor_config=policy_config,
    )
    state = repo.snapshot_persistent_state()
    path = harness / "policy" / "visual_contract.json"
    row = json.loads(path.read_text())
    row["version"] = "mutated"
    path.write_text(json.dumps(row), encoding="utf-8")
    generated_memory = harness / "memory" / "new_round_state.jsonl"
    generated_memory.write_text('{"temporary": true}\n', encoding="utf-8")
    repo.restore_persistent_state(state)
    assert json.loads(path.read_text())["version"] != "mutated"
    assert not generated_memory.exists()


def test_trace_candidate_budget_is_global_per_responsibility(monkeypatch):
    proposer = PatchProposer(
        {
            "type": "openai_chat_patch_proposer",
            "base_url": "https://invalid.example/v1",
            "model": "test-model",
        }
    )

    def candidate_rows(_weakness, *, responsibility=None):
        assert responsibility == "policy"
        return [
            {
                "target_component": "policy",
                "changed_artifact": "visual_contract.json",
                "changed_fields": ["verification_rules"],
                "operation": "set_path",
                "payload": {"path": ["verification_rules", f"candidate_{index}"], "value": True},
                "predicted_improvement": "test candidate",
                "possible_regression": "none",
            }
            for index in range(5)
        ]

    monkeypatch.setattr(proposer, "_request_patch_rows", candidate_rows)
    from gen_harness.schema import Weakness

    weaknesses = [
        Weakness(
            weakness_id=f"w{index}",
            suspected_component="unknown",
            constraint_type="count",
            failure_symptom="object_count_mismatch",
            support_count=2,
            support_experience_ids=[f"e{index}a", f"e{index}b"],
            hypothesis="test",
            evidence={},
        )
        for index in range(2)
    ]
    patches = proposer.propose_many(
        weaknesses,
        frontier_by_weakness={row.weakness_id: ["policy"] for row in weaknesses},
        max_candidates_per_responsibility=3,
    )
    assert len(patches) == 3
    assert {patch.target_component for patch in patches} == {"policy"}


def test_budget_exposes_trace_r_times_m_upper_bound(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    rows = []
    for split in ("target", "heldout", "preservation"):
        rows.append(
            {
                "task_id": split,
                "prompt": "two cups",
                "task_family": "compositional",
                "constraints": [
                    {"constraint_id": f"{split}_count", "text": "two cups", "constraint_type": "count"}
                ],
                "metadata": {"split": split},
                "references": {},
            }
        )
    tasks_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    budget = estimate_slow_loop_budget(
        tasks_path,
        seeds=[0, 1],
        trace_frontier_size=3,
        trace_candidate_budget=3,
    )
    assert budget["max_candidate_edits_per_step"] == 9
    assert budget["max_validation_generation_calls_per_step"] == 540


def test_failed_complete_path_validation_rolls_back_every_persistent_file(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "examples" / "visual_harness"
    harness = tmp_path / "harness"
    shutil.copytree(source, harness)
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
        harness,
        policy_extractor_config=policy_config,
    )
    initial_state = repo.snapshot_persistent_state()
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": split,
                    "prompt": "two cups",
                    "task_family": "compositional",
                    "constraints": [
                        {"constraint_id": f"{split}_count", "text": "two cups", "constraint_type": "count"}
                    ],
                    "metadata": {"split": split},
                    "references": {},
                }
            )
            + "\n"
            for split in ("target", "heldout", "preservation")
        ),
        encoding="utf-8",
    )

    class UnusedBackend(VisualBackend):
        def generate(self, task, program, seed=0):
            raise AssertionError("stubbed slow loop must not invoke generation")

    def fake_round(slow_runner, *_args, **_kwargs):
        policy_path = slow_runner.repo.artifact_path("policy", "visual_contract.json")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["version"] = "mutated-by-promoted-path"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        (slow_runner.repo.component_dir("memory") / "path_only.jsonl").write_text(
            '{"temporary": true}\n', encoding="utf-8"
        )
        return {
            "num_experiences": 1,
            "natural_failures": {"num_failures": 1},
            "num_weaknesses": 1,
            "num_patches": 1,
            "num_accepted_patches": 1,
            "num_promoted_patches": 1,
            "self_evolve_gate_summary": {},
            "artifacts": {},
        }

    from gen_harness.experiments import self_evolve_loop

    monkeypatch.setattr(self_evolve_loop.SlowLoopEvolutionRunner, "run", fake_round)
    class NoPatchLocalizer:
        def localize(self, weakness, *, frontier_size=3):
            return HarnessLocalization(
                weakness_id=weakness.weakness_id,
                scores={
                    "policy": 0.0,
                    "tools": 0.0,
                    "skills": 0.0,
                    "middleware": 0.0,
                    "memory": 0.0,
                    "no_patch": 1.0,
                },
                frontier=[],
                no_patch_preferred=True,
                rationale="test no-patch localizer",
            )

    class NoPatchProposer:
        def propose_many(self, weaknesses, *, frontier_by_weakness=None, max_candidates_per_responsibility=3):
            return []

    runner = SelfEvolveRunner(
        repo,
        UnusedBackend(),
        ObjectRequirementInspection(),
        fast_loop_attempts=5,
        harness_localizer=NoPatchLocalizer(),
        patch_proposer=NoPatchProposer(),
    )
    monkeypatch.setattr(
        runner,
        "_validate_complete_path",
        lambda *_args, **_kwargs: {
            "schema": "gen_harness.trace_path_validation.v1",
            "accepted": False,
            "applicable": True,
            "rolled_back": False,
            "reason": "forced cumulative regression",
        },
    )
    summary = runner.run(tasks_path, tmp_path / "run", seeds=[0, 1], max_rounds=1)
    assert summary["stop_reason"] == "final_path_validation_failed_rolled_back"
    assert summary["final_path_validation"]["rolled_back"] is True
    assert repo.snapshot_persistent_state() == initial_state
    assert not (harness / "memory" / "path_only.jsonl").exists()


def test_fault_injector_builds_no_patch_and_all_compound_pairs(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "visual_harness"
    rows = FaultInjector(source, tmp_path / "faults").generate(include_compound=True)
    compound = [row for row in rows if row["fault_type"] == "compound_responsibility_fault"]
    no_patch = [row for row in rows if row["gold_component"] == "no_patch"]
    assert len(compound) == 10
    assert len(no_patch) == 1
    assert all(len(row["gold_components"]) == 2 for row in compound)
    assert {tuple(row["gold_components"]) for row in compound} == {
        ("policy", "tools"),
        ("policy", "skills"),
        ("policy", "middleware"),
        ("policy", "memory"),
        ("tools", "skills"),
        ("tools", "middleware"),
        ("tools", "memory"),
        ("skills", "middleware"),
        ("skills", "memory"),
        ("middleware", "memory"),
    }
    memory_fault = next(row for row in rows if row["gold_component"] == "memory")
    faulty_repo = HarnessRepository(memory_fault["harness_path"])
    guidelines = faulty_repo.build_memory().retrieve_prompt_guidelines(
        task_family="compositional",
        prompt_family="counting_small_objects",
        constraint_types=["count"],
    )
    harmful = "Treat exact quantities, attributes, relations, and actions as optional"
    assert any(harmful in str(row.get("guideline")) for row in guidelines)
    task = GenerationTask.from_dict(
        {
            "task_id": "memory_fault_probe",
            "prompt": "two cups",
            "task_family": "compositional",
            "constraints": [
                {"constraint_id": "count", "text": "exactly two cups", "constraint_type": "count"}
            ],
            "metadata": {"split": "target"},
            "references": {},
        }
    )
    workflow = faulty_repo.build_skills().compile_workflow(
        task,
        "compositional_generation",
        {"constraints": [constraint.to_dict() for constraint in task.constraints]},
        {},
        {"prompt_guidelines": guidelines},
    )
    assert harmful in workflow["compiled_prompt"]


def test_trace_metrics_match_appendix_definitions():
    cases = [
        {"ground_truth": name, "predicted": name}
        for name in ("policy", "tools", "skills", "middleware", "memory", "no_patch")
    ]
    cases.append(
        {
            "gold_components": ["policy", "tools"],
            "frontier": ["policy", "tools", "skills"],
            "recovered_components": ["policy", "tools"],
            "repair_success": True,
        }
    )
    metrics = summarize_attribution(cases)
    assert metrics["macro_attribution_f1"] == 1.0
    assert metrics["no_patch_accuracy"] == 1.0
    assert metrics["compound"]["cause_recall_at_frontier"] == 1.0
    assert metrics["compound"]["both_recovered"] == 1.0

    updates = summarize_responsibility_updates(
        [
            {"patch_id": "p1", "target_component": "policy"},
            {"patch_id": "p2", "target_component": "skills"},
            {"patch_id": "p3", "target_component": "policy"},
        ],
        [
            {"patch_id": "p1", "accepted": True, "heldout_delta": 0.2},
            {"patch_id": "p2", "accepted": False, "heldout_delta": 0.0},
            {"patch_id": "p3", "accepted": True, "heldout_delta": 0.4},
        ],
        [{"patch_id": "p1", "promoted_to_harness": True}],
    )
    assert updates["per_responsibility"]["policy"]["update_share"] == 1.0
    assert updates["per_responsibility"]["policy"]["commit_rate"] == 0.5
    assert updates["per_responsibility"]["policy"]["mean_heldout_gain"] == 0.2
    assert updates["per_responsibility"]["skills"]["commit_rate"] == 0.0

    paired = summarize_paired_outcomes(
        [
            {"split": "heldout", "before_score": 0.2, "after_score": 0.6},
            {"split": "preservation", "before_score": 1.0, "after_score": 0.0, "threshold": 0.5},
            {"split": "preservation", "before_score": 0.0, "after_score": 0.0, "threshold": 0.5},
        ]
    )
    assert abs(paired["heldout_gain"] - 0.4) < 1e-12
    assert paired["preservation_previously_successful"] == 1
    assert paired["regression_percent"] == 100.0

    efficiency = summarize_efficiency(
        [
            {"wall_clock_seconds": 10.0, "generator_calls": 1, "llm_vlm_calls": 2, "candidate_edits": 8},
            {"efficiency": {"wall_clock_seconds": 12.0, "generator_calls": 2, "llm_vlm_calls": 2}},
            {"evolution_efficiency": {"wall_clock_seconds": 8.0, "generator_calls": 4, "llm_vlm_calls": 6, "candidate_edits_per_evolution_task": 4}},
        ]
    )
    assert efficiency["average_wall_clock_seconds_per_prompt"] == 10.0
    assert efficiency["average_generator_calls_per_prompt"] == 7 / 3
    assert efficiency["average_llm_vlm_calls_per_prompt"] == 10 / 3
    assert efficiency["average_candidate_edits_per_task"] == 6.0


def test_prompt_efficiency_observation_counts_actual_attempts_and_fixed_llm():
    row = prompt_efficiency_observation(
        wall_clock_seconds=12.5,
        fast_loop={"attempts": [{"attempt": 0}, {"attempt": 1}]},
        policy_extractor={"component_name": "openai_chat_policy_extractor"},
        prompt_optimizer_used=False,
    )
    assert row["generator_calls"] == 2
    assert row["llm_vlm_calls"] == 3
    assert row["call_breakdown"] == {
        "policy_llm_calls": 1,
        "prompt_optimizer_calls": 0,
        "visual_verifier_calls": 2,
        "detector_calls_excluded_from_llm_vlm": True,
    }


def test_readme_formal_commands_reference_public_reproducible_inputs():
    readme = Path("README.md").read_text(encoding="utf-8")
    config_paths = re.findall(r"--[a-z-]+-config\s+(configs/[^\s\\]+)", readme)

    assert config_paths
    assert all(Path(path).is_file() for path in config_paths)
    assert "your_backend.json" not in readme
    assert "your_inspector.json" not in readme
    assert "compositional_suite_v2_action_self_evolve_2026_09_05" not in readme
    assert "python -m gen_harness.cli build-tasks" in readme
    assert "python -m gen_harness.cli generate" in readme
    assert "python -m gen_harness.cli export-eval" in readme
    assert "--dataset geneval2" in readme
    assert "--dataset t2icompbenchpp" in readme
    assert "--dataset wise" in readme
    assert "Do not use `--limit` for a full benchmark run" in readme


def test_public_env_example_covers_paper_default_stack():
    example = Path(".env.example").read_text(encoding="utf-8")
    for name in (
        "GENHARNESS_LLM_BASE_URL",
        "GENHARNESS_LLM_MODEL",
        "GENHARNESS_LLM_API_KEY",
        "GENHARNESS_FLUX_ENDPOINT",
        "GENHARNESS_OWLV2_MODEL",
        "GENHARNESS_NVILA_MODEL",
    ):
        assert name in example
    configured_variables = {
        value
        for path in Path("configs").rglob("*.json")
        for value in re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", path.read_text(encoding="utf-8"))
    }
    documented_variables = set(re.findall(r"\b([A-Z][A-Z0-9_]+)=", example))
    assert configured_variables <= documented_variables
