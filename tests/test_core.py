import base64
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from gen_harness.backend import VisualBackend
from gen_harness.component_audit import ComponentBoundaryAuditor, GoalReadinessAuditor, RunArtifactAuditor
from gen_harness.debugger import FailureDebugger
from gen_harness.executor import HarnessExecutor
from gen_harness.fast_loop import FastLoopController
from gen_harness.inspection import InspectionRunner, LightweightAlignmentInspection, ObjectRequirementInspection
from gen_harness.miner import WeaknessMiner
from gen_harness.patcher import HarnessLocalization, PatchProposer, sanitize_trace_weakness_for_llm
from gen_harness.repository import HarnessRepository
from gen_harness.experiments.collector import NaturalFailureCollector
from gen_harness.experiments.geneval_generate import GenEvalImageGenerationExperiment
from gen_harness.experiments.t2icompbenchpp_generate import T2ICompBenchPPImageGenerationExperiment
from gen_harness.experiments.quality import QualityExperiment
from gen_harness.experiments.self_evolve_loop import SelfEvolveRunner
from gen_harness.experiments.slow_loop import SlowLoopEvolutionRunner
from gen_harness.experiments.suite_eval import SuiteEvaluationExperiment
from gen_harness.evaluation import fetch_geneval_dataset
from gen_harness.evaluation.geneval import build_geneval_tasks
from gen_harness.evaluation.geneval2 import build_geneval2_tasks, export_geneval2_image_map
from gen_harness.datasets.evolution import build_partiprompts_evolution_split
from gen_harness.io import expand_env_vars, write_json, write_jsonl
from gen_harness.schema import GenerationTask, PatchManifest, VisualConstraint, VisualExperience, Weakness
from gen_harness.validator import PatchValidator, ValidationReport
from gen_harness.components.policy import PolicyExtractor


ROOT = Path(__file__).resolve().parents[1]


class RecordingBackend(VisualBackend):
    def generate(self, task, program, seed=0):
        return {"image_uri": f"/tmp/gen_harness_test_outputs/{task.task_id}_{seed}.png", "program": program, "seed": seed}


class DataUriBackend(VisualBackend):
    def generate(self, task, program, seed=0):
        png_1x1 = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00"
            b"\x05\xfe\x02\xfeA\x89\x81\x81\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode("ascii")
        return {
            "image_uri": "data:image/png;base64," + png_1x1,
            "object_counts": {"bench": 1},
            "program": program,
            "seed": seed,
        }


class HarnessAwareInspector(InspectionRunner):
    def inspect(self, task, program, generation, seed=0):
        out = {}
        bindings = program.get("reference_bindings", {})
        tool = program.get("tool", {})
        caps = tool.get("capabilities", {})
        workflow = program.get("workflow", {})
        missing_steps = set(workflow.get("missing_steps", []))
        negative_constraints = set(program.get("contract", {}).get("negative_constraints", []))
        is_harness = True
        image_metrics = {
            "overall_quality": 0.82 if is_harness else 0.42,
            "aesthetic_quality": 0.80 if is_harness else 0.40,
            "prompt_alignment": 0.84 if is_harness else 0.55,
            "identity_consistency": 0.75 if is_harness else 0.50,
            "style_consistency": 0.75 if is_harness else 0.50,
            "artifact_free": 0.88 if is_harness else 0.60,
            "text_logo_absence": 0.90 if is_harness else 0.70,
        }
        for c in task.constraints:
            passed = True
            symptom = "pass"
            if c.constraint_type == "identity" and not caps.get("identity_conditioning", False):
                passed = False
                symptom = "unsupported_identity_conditioning"
            elif c.constraint_type == "style" and not caps.get("style_conditioning", False):
                passed = False
                symptom = "missing_tool_capability"
            elif c.constraint_type in missing_steps:
                passed = False
                symptom = f"{c.constraint_type}_step_missing"
            elif c.required_reference_role:
                ref_id = bindings.get(c.required_reference_role)
                ref = task.references.get(ref_id, {}) if ref_id else {}
                if ref.get("role") != c.required_reference_role:
                    passed = False
                    symptom = "identity_drift" if c.constraint_type == "identity" else "style_mismatch"
            elif c.constraint_type == "negative" and c.target and c.target not in negative_constraints:
                passed = False
                symptom = "negative_constraint_missing"
            out[c.constraint_id] = {"passed": passed, "symptom": symptom, "confidence": 1.0, "image_metrics": image_metrics, **image_metrics}
        return out


class _TestTaskPolicyExtractor(PolicyExtractor):
    """Explicit test double; production never uses task-embedded constraints."""

    name = "test_task_policy_extractor"

    def extract(self, task, policy_artifact):
        return [copy.deepcopy(constraint) for constraint in task.constraints]

    def describe(self):
        return {
            "component_name": self.name,
            "component_type": "policy_extractor",
            "test_double": True,
            "fallback_used": False,
        }


class _TestHarnessRepository(HarnessRepository):
    """Keep component tests explicit without weakening the production contract."""

    def __init__(self, root, *args, **kwargs):
        if kwargs.get("policy_extractor_config") is None and kwargs.get("policy_extractor") is None:
            kwargs["policy_extractor"] = _TestTaskPolicyExtractor()
        super().__init__(root, *args, **kwargs)


ProductionHarnessRepository = HarnessRepository
HarnessRepository = _TestHarnessRepository


@pytest.fixture(autouse=True)
def _stub_formal_policy_llm(monkeypatch):
    """Use a deterministic local response for tests of the formal LLM path."""

    monkeypatch.setattr(
        "gen_harness.components.policy.LLMPolicyExtractor._request_constraints",
        lambda self, task, policy_artifact, correction=None: {
            "constraints": [constraint.to_dict() for constraint in task.constraints]
        },
    )


def make_harness(tmp_path):
    dst = tmp_path / "harness"
    shutil.copytree(ROOT / "examples" / "visual_harness", dst)
    return dst


def make_task():
    return GenerationTask.from_dict(json.loads((ROOT / "examples" / "tasks" / "task_schema.example.json").read_text()))


def make_policy_config(tmp_path):
    path = tmp_path / "policy_extractor.json"
    path.write_text(
        json.dumps(
            {
                "type": "openai_chat_policy_extractor",
                "base_url": "https://llm.invalid/v1",
                "model": "test-policy-llm",
            }
        ),
        encoding="utf-8",
    )
    return path


def clean_reference_frameworks():
    return [
        {
            "source": "agentic-harness-engineering",
            "used_as": "conceptual_reference_only",
            "adopted_invariants": ["failure evidence", "validation gate"],
            "copied_runtime": False,
            "local_path_recorded": False,
            "official_benchmark_reference_trajectories_allowed": False,
        },
        {
            "source": "JIT",
            "used_as": "conceptual_reference_only",
            "adopted_invariants": ["fixed harness component protocol", "selection trace before evaluation"],
            "copied_runtime": False,
            "local_path_recorded": False,
            "official_benchmark_reference_trajectories_allowed": False,
        },
    ]


def clean_trace_certificate():
    return {
        "schema": "gen_harness.trace_certificate.v1",
        "flow": [
            "constraint_indexed_visual_experience",
            "llm_harness_localization_with_no_patch",
            "responsibility_scoped_patch_proposal",
            "target_heldout_preservation_validation",
            "regression_safe_promotion_only",
        ],
        "localization": {
            "num_localizations": 1,
            "num_no_patch": 0,
            "num_active_frontiers": 1,
            "no_patch_supported": True,
            "frontier_source": "llm_harness_localizer",
            "diagnostic_priors_non_binding": True,
            "responsibilities": ["policy", "tools", "skills", "middleware", "memory"],
        },
        "patch_proposals": {
            "num_patches": 1,
            "all_component_scoped": True,
            "proposal_source": "llm_patch_proposer",
        },
        "validation": {
            "num_reports": 1,
            "target_heldout_preservation_present": True,
            "official_evaluator_used": False,
            "official_scores_used": False,
            "fallback_used": False,
            "scope_rows": [
                {
                    "patch_id": "patch",
                    "target_task_ids": ["target"],
                    "heldout_task_ids": ["heldout"],
                    "preservation_task_ids": ["preservation"],
                    "seeds": [0, 1],
                    "accepted": True,
                    "target_delta": 1.0,
                    "heldout_delta": 0.0,
                    "regression_delta": 0.0,
                }
            ],
        },
        "promotion": {
            "num_promoted": 1,
            "promoted_patch_ids": ["patch"],
            "accepted_patch_ids": ["patch"],
            "non_accepted_promotions": [],
            "accepted_only": True,
        },
        "leakage_and_fallback_guards": {
            "official_labels_used_for_prompt_compilation": False,
            "official_evaluator_used_for_generation_or_selection": False,
            "fallback_image_substitution_allowed": False,
        },
    }


def nested_keys(value):
    keys = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(nested_keys(child))
    return keys


def make_reference_task():
    return GenerationTask.from_dict(
        {
            "task_id": "test_multiref",
            "prompt": "Create a portrait of identity_ref in the style of style_ref.",
            "task_family": "multi_reference",
            "references": {
                "identity_ref": {"role": "identity", "uri": "https://example.com/identity.png"},
                "style_ref": {"role": "style", "uri": "https://example.com/style.png"},
            },
            "constraints": [
                {"constraint_id": "identity_1", "text": "match identity_ref", "constraint_type": "identity", "required_reference_role": "identity"},
                {"constraint_id": "style_1", "text": "match style_ref", "constraint_type": "style", "required_reference_role": "style"},
            ],
            "metadata": {"split": "target"},
        }
    )


def test_cli_imports_without_circular_adapter_imports():
    from gen_harness.cli import build_parser

    parser = build_parser()
    assert parser.prog


def test_middleware_reroutes_to_capable_tool_when_preferred_route_is_unsupported(tmp_path):
    harness = make_harness(tmp_path)
    router_path = harness / "middleware" / "tool_router.json"
    router = json.loads(router_path.read_text())
    router["routes"] = {"identity_conditioning+style_conditioning+text_to_image": "base_generator"}
    router["strict_preferred_route"] = False
    router_path.write_text(json.dumps(router, indent=2) + "\n")

    repo = HarnessRepository(harness)
    middleware = repo.build_middleware()
    tools = repo.load_artifact("tools", "generation_tools.json")
    selected = middleware.route_tool(make_reference_task(), {}, tools)

    assert selected["name"] == "identity_generator"
    assert selected["routing_diagnostics"]["preferred_tool"] == "base_generator"
    assert selected["routing_diagnostics"]["rerouted_from_preferred"] is True


def test_middleware_can_keep_strict_wrong_route_for_controlled_faults(tmp_path):
    harness = make_harness(tmp_path)
    router_path = harness / "middleware" / "tool_router.json"
    router = json.loads(router_path.read_text())
    router["routes"] = {"identity_conditioning+style_conditioning+text_to_image": "base_generator"}
    router["strict_preferred_route"] = True
    router_path.write_text(json.dumps(router, indent=2) + "\n")

    repo = HarnessRepository(harness)
    middleware = repo.build_middleware()
    tools = repo.load_artifact("tools", "generation_tools.json")
    selected = middleware.route_tool(make_reference_task(), {}, tools)

    assert selected["name"] == "base_generator"
    assert selected["routing_warning"]["missing_capabilities"] == ["identity_conditioning"]
    assert selected["routing_diagnostics"]["strict_preferred_route"] is True


def test_quality_experiment_reports_gen_harness_only(tmp_path):
    harness = make_harness(tmp_path)
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(tasks_path, [make_task().to_dict()])
    summary = QualityExperiment(
        HarnessRepository(harness),
        RecordingBackend(),
        HarnessAwareInspector(),
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    ).run(tasks_path, tmp_path / "quality", seeds=[0])

    assert summary["num_gen_harness_rows"] == 4
    assert "policy_extractor_manifest" in summary
    assert "policy_extractor" in summary["policy_extractor_manifest"]
    assert "image_metrics" in summary
    assert "overall_quality" in summary["image_metrics"]["gen_harness"]


def test_build_backend_rejects_removed_image_generator_backend(tmp_path):
    from gen_harness.backends import build_backend

    config = tmp_path / "backend.json"
    config.write_text(json.dumps({"type": "image_llm"}), encoding="utf-8")

    with pytest.raises(ValueError, match="formal image generation"):
        build_backend(config)


def test_paper_backend_validation_accepts_frozen_open_source_http_models(tmp_path):
    from gen_harness.backends import validate_formal_backend_config

    config = tmp_path / "backend.json"
    config.write_text(json.dumps({
        "type": "http",
        "endpoint": "https://generator.invalid/generate",
        "request_metadata": {"model": "qwen-image", "frozen": True},
    }), encoding="utf-8")

    assert validate_formal_backend_config(config)["request_metadata"]["model"] == "qwen-image"


def test_paper_backend_validation_rejects_unfrozen_http_model(tmp_path):
    from gen_harness.backends import validate_formal_backend_config

    config = tmp_path / "backend.json"
    config.write_text(json.dumps({
        "type": "http",
        "endpoint": "https://generator.invalid/generate",
        "request_metadata": {"model": "qwen-image"},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen=true"):
        validate_formal_backend_config(config)


def test_paper_backend_validation_requires_frozen_flow_grpo_identity(tmp_path):
    from gen_harness.backends import validate_formal_backend_config

    config = tmp_path / "backend.json"
    config.write_text(json.dumps({
        "type": "flow_grpo",
        "model_family": "SD3.5 + Flow-GRPO",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen=true"):
        validate_formal_backend_config(config)


def test_paper_backend_validation_accepts_frozen_flux_diffusers(tmp_path):
    from gen_harness.backends import build_backend, validate_formal_backend_config
    from gen_harness.adapters.flux_diffusers_backend import FluxDiffusersBackend

    model_dir = tmp_path / "flux"
    model_dir.mkdir()
    config = tmp_path / "backend.json"
    config.write_text(json.dumps({
        "type": "flux_diffusers",
        "name": "flux_test",
        "model_family": "FLUX.1-dev",
        "frozen": True,
        "model_path": str(model_dir),
    }), encoding="utf-8")

    assert validate_formal_backend_config(config)["model_family"] == "FLUX.1-dev"
    backend = build_backend(config)
    assert isinstance(backend, FluxDiffusersBackend)
    assert backend.model_path == model_dir


def test_paper_backend_validation_requires_flux_model_path(tmp_path):
    from gen_harness.backends import validate_formal_backend_config

    config = tmp_path / "backend.json"
    config.write_text(json.dumps({
        "type": "flux_diffusers",
        "model_family": "FLUX.1-dev",
        "frozen": True,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="model_path"):
        validate_formal_backend_config(config)


def test_flux_diffusers_requires_compiled_prompt_and_uses_portable_default(monkeypatch):
    from gen_harness.adapters.flux_diffusers_backend import FluxDiffusersBackend

    monkeypatch.delenv("GENHARNESS_FLUX_MODEL_ROOT", raising=False)
    backend = FluxDiffusersBackend({})
    task = make_task()

    assert str(backend.model_path) == "models/FLUX.1-dev"
    assert backend._prompt_for_program(task, {"workflow": {"compiled_prompt": "a precise prompt"}}) == "a precise prompt"
    with pytest.raises(ValueError, match="compiled_prompt"):
        backend._prompt_for_program(task, {"workflow": {}})


def test_flux_diffusers_writes_deterministic_cache_paths(tmp_path):
    from gen_harness.adapters.flux_diffusers_backend import FluxDiffusersBackend

    task = make_task()
    program = {"method": "gen_harness", "workflow": {"compiled_prompt": "a red cube on a table"}}
    backend = FluxDiffusersBackend({
        "image_cache_dir": str(tmp_path / "images"),
        "trace_cache_dir": str(tmp_path / "traces"),
    })

    first = backend._image_path(task, program, 7)
    second = backend._image_path(task, program, 7)

    assert first == second
    assert first.name.startswith(f"{task.task_id}_seed7_")
    assert backend._trace_path(first).name.endswith(".trace.json")


def test_paper_inspection_validation_requires_owlv2_and_nvila(tmp_path):
    from gen_harness.inspection import validate_formal_inspection_config

    config = tmp_path / "inspection.json"
    config.write_text(json.dumps({
        "type": "openai_vision_judge",
        "model": "vision-model",
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="OWLv2/NVILA"):
        validate_formal_inspection_config(config)


def test_cli_run_writes_policy_extractor_summary_for_artifact_audit(tmp_path, monkeypatch):
    from gen_harness import cli

    harness = make_harness(tmp_path)
    task = make_task()
    tasks_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "run" / "experiences.jsonl"
    backend_config = tmp_path / "backend.json"
    inspection_config = tmp_path / "inspection.json"
    policy_config = tmp_path / "policy_extractor.json"
    write_jsonl(tasks_path, [task.to_dict()])
    backend_config.write_text(json.dumps({"type": "recording"}), encoding="utf-8")
    inspection_config.write_text(json.dumps({"type": "harness_aware"}), encoding="utf-8")
    policy_config.write_text(
        json.dumps({
            "type": "openai_chat_policy_extractor",
            "base_url": "https://llm.invalid/v1",
            "model": "test-policy-llm",
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "build_backend", lambda _path: RecordingBackend())
    monkeypatch.setattr(cli, "build_inspector", lambda _path: HarnessAwareInspector())
    monkeypatch.setattr(cli, "build_harness_localizer", lambda *args, **kwargs: _StaticHarnessLocalizer(no_patch=True))
    monkeypatch.setattr(cli, "build_patch_proposer", lambda *args, **kwargs: _StaticPatchProposer([]))
    monkeypatch.setattr(
        "gen_harness.components.policy.LLMPolicyExtractor._request_constraints",
        lambda self, task, policy_artifact, correction=None: {
            "constraints": [constraint.to_dict() for constraint in task.constraints]
        },
    )

    args = cli.build_parser().parse_args([
        "run",
        "--harness", str(harness),
        "--tasks", str(tasks_path),
        "--output", str(output_path),
        "--backend-config", str(backend_config),
        "--inspection-config", str(inspection_config),
        "--policy-extractor-config", str(policy_config),
        "--harness-localizer-config", str(policy_config),
        "--patch-proposer-config", str(policy_config),
    ])

    cli.cmd_run(args)

    summary = json.loads((output_path.parent / "run_summary.json").read_text(encoding="utf-8"))
    report = RunArtifactAuditor().audit(experiences_path=output_path)

    assert summary["schema"] == "gen_harness.run_summary.v1"
    assert summary["policy_extractor_manifest"]["formal_llm_policy_extractor"] is True
    assert report["checks"]["formal_policy_extractor_provenance_present"] is True
    assert report["checks"]["formal_policy_extractor_is_llm"] is True



class PolicySensitiveInspector(InspectionRunner):
    def inspect(self, task, program, generation, seed=0):
        rules = program.get("contract", {}).get("prompt_governance", {}).get("default_quality_profile", [])
        boosted = any("coherence boost" in rule for rule in rules)
        score = 0.9 if boosted else 0.4
        return {
            c.constraint_id: {
                "passed": score >= 0.5,
                "symptom": "pass" if score >= 0.5 else "low_overall_quality",
                "confidence": 1.0,
                "image_metrics": {
                    "overall_quality": score,
                    "aesthetic_quality": score,
                    "prompt_alignment": score,
                    "artifact_free": score,
                    "text_logo_absence": score,
                },
                "overall_quality": score,
                "aesthetic_quality": score,
                "prompt_alignment": score,
                "artifact_free": score,
                "text_logo_absence": score,
            }
            for c in task.constraints
        }


def test_repository_applies_memory_jsonl_patch(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    patch = PatchManifest(
        patch_id="patch_memory_guideline",
        target_component="memory",
        changed_artifact="prompt_guidelines.jsonl",
        changed_fields=["prompt_guidelines.evolved_test"],
        operation="append_jsonl_unique",
        payload={
            "unique_key": "memory_id",
            "row": {"memory_id": "evolved_test", "task_family": "vague_prompt", "source": "test", "guideline": "test guideline"},
        },
        supporting_experience=[],
        predicted_improvement="test",
        possible_regression="test",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="memory/prompt_guidelines.jsonl",
    )
    before, after = repo.apply_patch(patch)
    assert len(after) == len(before) + 1
    assert any(row.get("memory_id") == "evolved_test" for row in after)
    with repo.temporary_patch(patch):
        rows = json.loads('{"ok": true}')
        assert rows["ok"]


def test_validator_can_accept_patch_using_image_metrics(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = make_task()
    patch = PatchManifest(
        patch_id="patch_quality_rule",
        target_component="policy",
        changed_artifact="visual_contract.json",
        changed_fields=["prompt_governance.default_quality_profile"],
        operation="append_unique",
        payload={"path": ["prompt_governance", "default_quality_profile"], "value": "coherence boost rule"},
        supporting_experience=[],
        predicted_improvement="boost image metrics",
        possible_regression="none",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="policy/visual_contract.json",
    )
    report = PatchValidator(
        repo,
        RecordingBackend(),
        PolicySensitiveInspector(),
        metric_mode="image_metrics",
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    ).validate(patch, [task], [task], [task], seed=0, seeds=[0])
    assert report.accepted
    assert report.target_delta > 0
    assert report.heldout_delta > 0
    assert "preservation" in report.before
    row = report.to_dict()
    assert row["validation_scope"]["target_heldout_preservation_present"] is True
    assert row["validation_guards"]["target_improved"] is True
    assert row["validation_guards"]["heldout_non_regressing"] is True
    assert row["validation_guards"]["preservation_regression_bounded"] is True
    assert row["validation_guards"]["official_evaluator_used"] is False
    assert row["validation_guards"]["official_scores_used"] is False
    assert row["validation_guards"]["fallback_used"] is False

def test_natural_failure_collector_reports_failures_by_seed(tmp_path):
    task = make_task()
    exps = [
        VisualExperience(
            experience_id="exp_seed0_pass",
            task_id=task.task_id,
            constraint_id="quality_1",
            constraint_type="quality",
            constraint_text="quality",
            component_decisions={},
            tool_calls=[],
            visual_result={"image_uri": "/tmp/0.png", "seed": 0},
            verification_result={"passed": True},
            failure_symptom=None,
            suspected_component=None,
        ),
        VisualExperience(
            experience_id="exp_seed1_fail",
            task_id=task.task_id,
            constraint_id="quality_1",
            constraint_type="quality",
            constraint_text="quality",
            component_decisions={},
            tool_calls=[],
            visual_result={"image_uri": "/tmp/1.png", "seed": 1},
            verification_result={"passed": False},
            failure_symptom="low_overall_quality",
            suspected_component="policy",
        ),
    ]

    summary = NaturalFailureCollector().collect(exps, tmp_path)

    assert summary["by_seed"]["0"]["failure_rate"] == 0.0
    assert summary["by_seed"]["1"]["failure_rate"] == 1.0



def test_fetch_geneval_dataset_from_file_source(tmp_path):
    source = tmp_path / "source"
    prompts = source / "prompts"
    prompts.mkdir(parents=True)
    metadata = prompts / "evaluation_metadata.jsonl"
    metadata.write_text(
        json.dumps({
            "tag": "counting",
            "include": [{"class": "cat", "count": 2}],
            "prompt": "two cats on a chair",
        }) + "\n",
        encoding="utf-8",
    )

    manifest = fetch_geneval_dataset(tmp_path / "out", source_base=source.as_uri())

    assert manifest["num_tasks"] == 1
    assert Path(manifest["metadata_path"]).exists()
    rows = [json.loads(line) for line in Path(manifest["tasks_path"]).read_text().splitlines()]
    assert rows[0]["metadata"]["benchmark"] == "geneval"
    assert rows[0]["metadata"]["geneval_tag"] == "counting"
    assert rows[0]["constraints"][0]["constraint_type"] == "count"


def test_build_compositional_suite_is_non_geneval_and_split_balanced(tmp_path):
    from gen_harness.datasets import build_compositional_suite

    output = tmp_path / "synthetic_compositional.jsonl"
    manifest = build_compositional_suite(
        output,
        seed=7,
        target_per_family=2,
        heldout_per_family=2,
        preservation_per_family=1,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert manifest["num_tasks"] == 15
    assert manifest["splits"] == {"target": 6, "heldout": 6, "preservation": 3}
    assert manifest["families"] == {"count": 5, "spatial": 5, "attribute": 5}
    assert Path(manifest["manifest_path"]).exists()
    assert all("geneval" not in json.dumps(row).lower() for row in rows)
    assert all(row["metadata"]["suite"] == "synthetic_compositional_contracts" for row in rows)


def test_build_compositional_suite_can_include_non_benchmark_action_tasks(tmp_path):
    from gen_harness.datasets import build_compositional_suite

    output = tmp_path / "synthetic_compositional_action.jsonl"
    manifest = build_compositional_suite(
        output,
        seed=11,
        target_per_family=2,
        heldout_per_family=2,
        preservation_per_family=1,
        include_action=True,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    action_rows = [row for row in rows if row["metadata"]["family"] == "action"]

    assert manifest["num_tasks"] == 20
    assert manifest["splits"] == {"target": 8, "heldout": 8, "preservation": 4}
    assert manifest["families"] == {"count": 5, "spatial": 5, "attribute": 5, "action": 5}
    assert action_rows
    assert all(row["constraints"][0]["constraint_type"] == "action" for row in action_rows)
    assert all("action_relations" in row["constraints"][0]["metadata"] for row in action_rows)
    assert all("benchmark" not in row["metadata"] for row in rows)
    assert all("geneval" not in json.dumps(row).lower() for row in rows)


def test_compositional_suite_covers_depth_mixed_counts_and_visual_materials(tmp_path):
    from gen_harness.datasets import build_compositional_suite

    output = tmp_path / "balanced_compositional.jsonl"
    build_compositional_suite(
        output,
        seed=13,
        target_per_family=6,
        heldout_per_family=6,
        preservation_per_family=6,
        include_action=True,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    target_rows = [row for row in rows if row["metadata"]["split"] == "target"]
    spatial = [row for row in target_rows if row["metadata"]["family"] == "spatial"]
    counts = [row for row in target_rows if row["metadata"]["family"] == "count"]
    attributes = [row for row in rows if row["metadata"]["family"] == "attribute"]

    assert {row["constraints"][0]["metadata"]["spatial_relations"][0]["relation"] for row in spatial} == {
        "left of", "right of", "above", "below", "in front of", "behind"
    }
    assert {row["constraints"][0]["metadata"]["required_objects"][0]["count"] for row in counts} == set(range(2, 8))
    assert any(len(row["constraints"][0]["metadata"]["required_objects"]) > 1 for row in counts)
    assert any(
        binding["attribute"] in {"wooden", "metal", "plastic", "stone", "spotted", "sparkling"}
        for row in attributes
        for binding in row["constraints"][0]["metadata"]["attribute_bindings"]
    )


def test_build_geneval2_tasks_keeps_vqa_labels_out_of_generation_metadata(tmp_path):
    source = tmp_path / "geneval2_data.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": "a green backpack and a pig",
                "atom_count": 3,
                "vqa_list": [["Is the backpack green?", "Yes"]],
                "question_answers": [{"question": "Is there a pig?", "answer": "yes"}],
                "skills": ["attribute"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "geneval2_tasks.jsonl"

    tasks = build_geneval2_tasks(source, output)

    assert len(tasks) == 1
    row = json.loads(output.read_text().strip())
    assert row["task_id"] == "geneval2_0000"
    assert row["prompt"] == "a green backpack and a pig"
    assert row["constraints"] == []
    serialized_row = json.dumps(row)
    assert "atom_count" not in serialized_row
    assert "vqa_list" not in serialized_row
    assert "question_answers" not in serialized_row
    assert "skills" not in serialized_row
    serialized_metadata = json.dumps(row["metadata"])
    assert "vqa_list" not in serialized_metadata
    assert "Is the backpack green?" not in serialized_metadata
    assert "Yes" not in serialized_metadata
    assert row["metadata"]["generation_input"] == "prompt_only_no_vqa_labels"


def test_geneval_generate_prompt_only_task_keeps_contract_metadata_minimal(tmp_path):
    exp = GenEvalImageGenerationExperiment(
        make_harness(tmp_path),
        RecordingBackend(),
        inspector=HarnessAwareInspector(),
        policy_extractor_config=make_policy_config(tmp_path),
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    )
    task = GenerationTask.from_dict({
        "task_id": "geneval2_0000",
        "prompt": "a green backpack and a pig",
        "task_family": "compositional",
        "references": {},
        "constraints": [],
        "metadata": {
            "split": "heldout",
            "benchmark": "geneval2",
            "geneval2_index": 0,
            "generation_input": "prompt_only_no_vqa_labels",
        },
    })

    prompt_only = exp._prompt_only_task(task)

    assert prompt_only.metadata["split"] == "heldout"
    assert "benchmark" not in prompt_only.metadata
    assert "generation_input" not in prompt_only.metadata




def test_generate_cli_uses_dataset_parameter_for_geneval2_and_wise():
    from gen_harness import cli

    for dataset in ("geneval2", "wise"):
        args = cli.build_parser().parse_args([
            "generate",
            "--dataset", dataset,
            "--harness", "examples/visual_harness",
            "--tasks", "tasks.jsonl",
            "--output-dir", "out",
            "--backend-config", "backend.json",
            "--inspection-config", "inspection.json",
            "--policy-extractor-config", "policy.json",
            "--harness-localizer-config", "localizer.json",
            "--patch-proposer-config", "proposer.json",
        ])
        assert args.dataset == dataset
        assert args.func is cli.cmd_generate


def test_t2icompbenchpp_generation_task_strips_benchmark_provenance(tmp_path):
    exp = T2ICompBenchPPImageGenerationExperiment(
        make_harness(tmp_path),
        RecordingBackend(),
        inspector=HarnessAwareInspector(),
        policy_extractor_config=make_policy_config(tmp_path),
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    )
    task = GenerationTask.from_dict({
        "task_id": "t2icompbenchpp_color_val_0007",
        "prompt": "a green backpack and a pig",
        "task_family": "compositional",
        "references": {},
        "constraints": [],
        "metadata": {
            "split": "val",
            "benchmark": "t2icompbenchpp",
            "category": "color",
            "source_index": 7,
            "complex_subtype": "none",
            "generation_input": "prompt_only_no_official_eval_outputs",
        },
    })

    prompt_only = exp._prompt_only_task(task)

    assert prompt_only.metadata == {"split": "val"}
    assert prompt_only.constraints == []
    assert prompt_only.references == {}


def test_generation_experiments_require_formal_policy_extractor_by_default(tmp_path):
    harness = make_harness(tmp_path)

    with pytest.raises(ValueError, match="policy_extractor_config is required"):
        GenEvalImageGenerationExperiment(harness, RecordingBackend())
    with pytest.raises(ValueError, match="policy_extractor_config is required"):
        T2ICompBenchPPImageGenerationExperiment(harness, RecordingBackend())
    with pytest.raises(ValueError, match="policy_extractor_config is required"):
        SuiteEvaluationExperiment(
            harness,
            backend_config=ROOT / "configs" / "backends" / "flow_grpo.json",
            inspection_config=ROOT / "configs" / "inspection" / "object_requirements.example.json",
        )


def test_repository_always_requires_formal_policy_extractor(tmp_path, monkeypatch):
    harness = make_harness(tmp_path)
    monkeypatch.delenv("GENHARNESS_DIAGNOSTIC_ALLOW_LEGACY_POLICY_EXTRACTOR", raising=False)

    with pytest.raises(ValueError, match="policy_extractor_config is required"):
        ProductionHarnessRepository(harness).build_policy()


def test_formal_policy_module_has_no_rule_based_parser():
    source = Path("gen_harness/components/policy.py").read_text(encoding="utf-8")
    lines = [line.strip() for line in source.splitlines()]

    assert "import re" not in lines
    assert "re.compile" not in source
    assert "re.match" not in source
    assert "re.search" not in source
    assert "class RuleBasedPolicyExtractor" not in source
    assert "t2icompbenchpp" not in source
    assert 'metadata.get("benchmark"' not in source


def test_export_geneval2_image_map_deduplicates_constraint_experiences(tmp_path):
    benchmark = tmp_path / "geneval2_data.jsonl"
    benchmark.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "one red cup", "atom_count": 3, "vqa_list": [], "skills": []}),
                json.dumps({"prompt": "two blue bowls", "atom_count": 3, "vqa_list": [], "skills": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    image0 = tmp_path / "0.png"
    image1 = tmp_path / "1.png"
    image0.write_bytes(b"png0")
    image1.write_bytes(b"png1")
    experiences = tmp_path / "experiences.jsonl"
    write_jsonl(
        experiences,
        [
            {
                "task_id": "geneval2_0000",
                "visual_result": {"seed": 0, "raw_generation": {"image_path": str(image0)}},
                "metadata": {"benchmark": "geneval2", "geneval2_index": 0},
            },
            {
                "task_id": "geneval2_0000",
                "visual_result": {"seed": 0, "raw_generation": {"image_path": str(image0)}},
                "metadata": {"benchmark": "geneval2", "geneval2_index": 0},
            },
            {
                "task_id": "geneval2_0001",
                "visual_result": {"seed": 0, "raw_generation": {"image_uri": str(image1)}},
                "metadata": {"split": "heldout", "generation_input": "prompt_only_no_official_eval_outputs"},
                "export_metadata": {"benchmark": "geneval2", "geneval2_index": 1},
            },
        ],
    )
    output = tmp_path / "image_map.json"

    image_map = export_geneval2_image_map(experiences, benchmark, output, seed=0)

    assert image_map == {
        "one red cup": str(image0.resolve()),
        "two blue bowls": str(image1.resolve()),
    }
    manifest = json.loads(output.with_suffix(".json.manifest.json").read_text())
    assert manifest["num_mapped_prompts"] == 2
    assert manifest["missing_indices"] == []
    serialized_manifest = json.dumps(manifest)
    assert "atom_count" not in serialized_manifest
    assert "vqa_list" not in serialized_manifest
    assert "skills" not in serialized_manifest


def test_export_geneval2_image_map_requires_prompt_subset_only(tmp_path):
    benchmark = tmp_path / "geneval2_data.jsonl"
    benchmark.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "one red cup", "atom_count": 3, "vqa_list": [], "skills": []}),
                json.dumps({"prompt": "two blue bowls", "atom_count": 3, "vqa_list": [], "skills": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subset = tmp_path / "subset_tasks.jsonl"
    write_jsonl(subset, [
        {
            "task_id": "geneval2_0001",
            "prompt": "two blue bowls",
            "task_family": "compositional",
            "constraints": [],
            "metadata": {"benchmark": "geneval2", "generation_input": "prompt_only_no_vqa_labels"},
        }
    ])
    image1 = tmp_path / "1.png"
    image1.write_bytes(b"png1")
    experiences = tmp_path / "experiences.jsonl"
    write_jsonl(
        experiences,
        [
            {
                "task_id": "geneval2_0001",
                "visual_result": {"seed": 0, "raw_generation": {"image_path": str(image1)}},
                "metadata": {"benchmark": "geneval2", "geneval2_index": 1},
            },
        ],
    )
    output = tmp_path / "subset_image_map.json"

    image_map = export_geneval2_image_map(experiences, benchmark, output, seed=0, prompts_file=subset)

    assert image_map == {"two blue bowls": str(image1.resolve())}
    manifest = json.loads(output.with_suffix(".json.manifest.json").read_text())
    assert manifest["require_all"] is True
    assert manifest["prompt_source"] == str(subset)
    assert manifest["full_benchmark_prompts"] == 2
    assert manifest["num_benchmark_prompts"] == 1
    assert manifest["num_mapped_prompts"] == 1
    assert manifest["missing_indices"] == []


def test_cli_build_compositional_suite_command(tmp_path, capsys):
    from gen_harness.cli import main

    output = tmp_path / "suite.jsonl"
    main([
        "build-compositional-suite",
        "--output", str(output),
        "--seed", "3",
        "--target-per-family", "1",
        "--heldout-per-family", "1",
        "--preservation-per-family", "1",
    ])

    captured = capsys.readouterr()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert "'num_tasks': 9" in captured.out
    assert len(rows) == 9




def test_suite_evaluation_reports_internal_scores_without_geneval_export(tmp_path, monkeypatch):
    from gen_harness.datasets import build_compositional_suite
    from gen_harness.experiments import SuiteEvaluationExperiment

    class ConstraintSatisfyingBackend(VisualBackend):
        def generate(self, task, program, seed=0):
            counts = {}
            for constraint in task.constraints:
                required = constraint.metadata.get("required_objects", [])
                for obj in required:
                    if isinstance(obj, dict) and obj.get("class"):
                        counts[str(obj["class"])] = int(obj.get("count", 1) or 1)
            return {"image_uri": f"/tmp/suite_eval_{task.task_id}_{seed}.png", "object_counts": counts, "seed": seed}

    harness = make_harness(tmp_path)
    tasks_path = tmp_path / "suite.jsonl"
    build_compositional_suite(
        tasks_path,
        seed=5,
        target_per_family=1,
        heldout_per_family=1,
        preservation_per_family=1,
    )

    monkeypatch.setenv("GENHARNESS_FLOW_GRPO_MODEL_ROOT", str(tmp_path / "models"))
    summary = SuiteEvaluationExperiment(
        harness,
        backend_config=ROOT / "configs" / "backends" / "flow_grpo.json",
        inspection_config=ROOT / "configs" / "inspection" / "object_requirements.example.json",
        policy_extractor_config=make_policy_config(tmp_path),
        fast_loop_attempts=5,
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    )
    summary.backend = ConstraintSatisfyingBackend()
    summary.inspector = ObjectRequirementInspection()
    result = summary.run(tasks_path, tmp_path / "suite_eval", seeds=[0], continue_on_error=True)

    assert result["suite"] == "generic_task_internal_inspection"
    assert result["num_tasks"] == 9
    assert result["overall"]["pass_rate"] > 0
    assert set(result["by_split"]) == {"heldout", "preservation", "target"}
    assert not (tmp_path / "suite_eval" / "geneval_exports").exists()
    assert (tmp_path / "suite_eval" / "suite_evaluation_summary.json").exists()


def test_http_generation_backend_sends_public_prompt_payload_without_task_provenance(monkeypatch):
    from gen_harness.adapters.http_backend import HTTPImageGenerationBackend

    captured = {}
    backend = HTTPImageGenerationBackend({
        "endpoint": "https://example.invalid/generate",
        "request_metadata": {"model": "frozen-generator"},
    })
    task = GenerationTask.from_dict({
        "task_id": "t2icompbenchpp_color_val_0007",
        "prompt": "a green backpack and a pig",
        "task_family": "compositional",
        "constraints": [],
        "metadata": {"benchmark": "t2icompbenchpp", "category": "color", "source_index": 7},
    })
    program = {
        "workflow": {"compiled_prompt": "a green backpack and a pig on a plain background"},
        "contract": {
            "task_metadata": {"benchmark": "t2icompbenchpp", "category": "color", "source_index": 7},
            "constraints": [
                {
                    "constraint_id": "t2icompbenchpp_color_val_0007_attr",
                    "constraint_type": "attribute",
                    "text": "green backpack",
                    "metadata": {"category": "color", "official_score": 1.0, "attribute": "green"},
                }
            ],
        },
    }

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"image_uri": "https://example.invalid/out.png"}).encode("utf-8")

        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = backend.generate(task, program, seed=4)

    assert result["image_uri"] == "https://example.invalid/out.png"
    assert captured["body"]["input"]["generation_prompt"] == "a green backpack and a pig on a plain background"
    serialized = json.dumps(captured["body"], ensure_ascii=False)
    assert "green backpack" in serialized
    for forbidden in ("task", "program", "task_id", "t2icompbenchpp", "category", "source_index", "official_score"):
        assert forbidden not in serialized


def test_http_inspection_backend_sends_public_constraints_and_remaps_ids(monkeypatch):
    from gen_harness.adapters.http_inspection import HTTPInspectionRunner

    captured = {}
    inspector = HTTPInspectionRunner({"endpoint": "https://example.invalid/inspect"})
    task = GenerationTask.from_dict({
        "task_id": "t2icompbenchpp_color_val_0007",
        "prompt": "a green backpack and a pig",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "t2icompbenchpp_color_val_0007_attr",
                "constraint_type": "attribute",
                "text": "the backpack must be green",
                "metadata": {
                    "category": "color",
                    "source_index": 7,
                    "official_score": 1.0,
                    "attribute_bindings": [{"object": "backpack", "attribute": "green"}],
                },
            }
        ],
        "metadata": {"benchmark": "t2icompbenchpp", "category": "color"},
    })
    program = {"workflow": {"compiled_prompt": "a green backpack and a pig"}}

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "constraint_results": {
                        "constraint_0": {"passed": True, "symptom": "pass", "confidence": 0.9}
                    }
                }).encode("utf-8")

        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = inspector.inspect(task, program, {"image_uri": "data:image/png;base64,AAAA"}, seed=4)

    assert result["t2icompbenchpp_color_val_0007_attr"]["passed"] is True
    serialized = json.dumps(captured["body"], ensure_ascii=False)
    assert "constraint_0" in serialized
    assert "the backpack must be green" in serialized
    for forbidden in ("task", "program", "task_id", "t2icompbenchpp", "category", "source_index", "official_score"):
        assert forbidden not in serialized


def test_external_command_generation_payload_excludes_full_task_and_program(tmp_path):
    from gen_harness.backends import ExternalCommandBackend

    capture = tmp_path / "payload.json"
    runner = tmp_path / "backend.py"
    runner.write_text(
        "import json, shutil, sys\n"
        "payload=json.load(open(sys.argv[sys.argv.index('--input')+1]))\n"
        "shutil.copyfile(sys.argv[sys.argv.index('--input')+1], sys.argv[sys.argv.index('--capture')+1])\n"
        "json.dump({'image_uri':'https://example.invalid/out.png'}, open(sys.argv[sys.argv.index('--output')+1], 'w'))\n",
        encoding="utf-8",
    )
    backend = ExternalCommandBackend({
        "command": [
            "{python}",
            str(runner),
            "--input",
            "{input_json}",
            "--output",
            "{output_json}",
            "--capture",
            str(capture),
        ]
    })
    task = GenerationTask.from_dict({
        "task_id": "geneval2_0007",
        "prompt": "a green backpack",
        "constraints": [],
        "metadata": {"benchmark": "geneval2", "official_score": 1.0},
    })
    program = {
        "workflow": {"compiled_prompt": "a green backpack on a plain background"},
        "contract": {"constraints": [{"constraint_id": "geneval2_0007_attr", "constraint_type": "attribute", "text": "green backpack"}]},
    }

    result = backend.generate(task, program, seed=5)
    payload = json.loads(capture.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert result["image_uri"] == "https://example.invalid/out.png"
    assert payload["input"]["generation_prompt"] == "a green backpack on a plain background"
    assert "output_json" in payload
    for forbidden in ("task", "program", "task_id", "geneval2_0007", "geneval2", "official_score"):
        assert forbidden not in serialized


def test_external_search_payload_excludes_task_provenance(tmp_path):
    from gen_harness.search import ExternalCommandSearch

    capture = tmp_path / "payload.json"
    runner = tmp_path / "search.py"
    runner.write_text(
        "import json, shutil, sys\n"
        "shutil.copyfile(sys.argv[sys.argv.index('--input')+1], sys.argv[sys.argv.index('--capture')+1])\n"
        "json.dump({'evidence': [], 'references': []}, open(sys.argv[sys.argv.index('--output')+1], 'w'))\n",
        encoding="utf-8",
    )
    search = ExternalCommandSearch({
        "command": [
            "{python}",
            str(runner),
            "--input",
            "{input_json}",
            "--output",
            "{output_json}",
            "--capture",
            str(capture),
        ]
    })
    task = GenerationTask.from_dict({
        "task_id": "geneval2_0007",
        "prompt": "a green backpack",
        "constraints": [],
        "metadata": {"benchmark": "geneval2", "source_index": 7},
    })
    contract = {
        "task_metadata": {"benchmark": "geneval2", "source_index": 7},
        "constraints": [{"constraint_id": "geneval2_0007_attr", "constraint_type": "attribute", "text": "green backpack"}],
    }

    search.search(task, contract)
    payload = json.loads(capture.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["input"]["prompt"] == "a green backpack"
    assert "output_json" in payload
    assert payload["input"]["contract"]["constraints"][0]["text"] == "green backpack"
    for forbidden in ("task", "task_id", "geneval2_0007", "geneval2", "source_index"):
        assert forbidden not in serialized


def test_external_command_inspection_payload_excludes_full_task_and_remaps_ids(tmp_path):
    from gen_harness.inspection import ExternalCommandInspection

    capture = tmp_path / "payload.json"
    runner = tmp_path / "inspect.py"
    runner.write_text(
        "import json, shutil, sys\n"
        "payload=json.load(open(sys.argv[sys.argv.index('--input')+1]))\n"
        "shutil.copyfile(sys.argv[sys.argv.index('--input')+1], sys.argv[sys.argv.index('--capture')+1])\n"
        "json.dump({'constraint_results': {'constraint_0': {'passed': True, 'symptom': 'pass', 'confidence': 0.8}}}, open(sys.argv[sys.argv.index('--output')+1], 'w'))\n",
        encoding="utf-8",
    )
    inspector = ExternalCommandInspection({
        "command": [
            "{python}",
            str(runner),
            "--input",
            "{input_json}",
            "--output",
            "{output_json}",
            "--capture",
            str(capture),
        ]
    })
    task = GenerationTask.from_dict({
        "task_id": "geneval2_0007",
        "prompt": "a green backpack",
        "constraints": [
            {
                "constraint_id": "geneval2_0007_attr",
                "constraint_type": "attribute",
                "text": "green backpack",
                "metadata": {"source_index": 7, "attribute_bindings": [{"object": "backpack", "attribute": "green"}]},
            }
        ],
        "metadata": {"benchmark": "geneval2"},
    })
    program = {"workflow": {"compiled_prompt": "a green backpack on a plain background"}}

    result = inspector.inspect(task, program, {"image_uri": "/tmp/image.png"}, seed=2)
    payload = json.loads(capture.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert result["geneval2_0007_attr"]["passed"] is True
    assert "constraint_0" in serialized
    assert "green backpack" in serialized
    assert payload["generation"]["image_path"] == "/tmp/image.png"
    assert payload["generation"]["image_uri"] == "/tmp/image.png"
    for forbidden in ("task", "program", "task_id", "geneval2_0007", "geneval2", "source_index"):
        assert forbidden not in serialized


def test_external_inspection_prefers_sanitized_contract_constraints(tmp_path):
    from gen_harness.inspection import ExternalCommandInspection

    capture = tmp_path / "payload.json"
    runner = tmp_path / "inspect.py"
    runner.write_text(
        "import json, shutil, sys\n"
        "payload=json.load(open(sys.argv[sys.argv.index('--input')+1]))\n"
        "shutil.copyfile(sys.argv[sys.argv.index('--input')+1], sys.argv[sys.argv.index('--capture')+1])\n"
        "json.dump({'constraint_results': {'constraint_0': {'passed': True, 'symptom': 'pass', 'confidence': 0.9}}}, open(sys.argv[sys.argv.index('--output')+1], 'w'))\n",
        encoding="utf-8",
    )
    inspector = ExternalCommandInspection({
        "command": [
            "{python}",
            str(runner),
            "--input",
            "{input_json}",
            "--output",
            "{output_json}",
            "--capture",
            str(capture),
        ]
    })
    task = GenerationTask.from_dict({
        "task_id": "geneval2_0007",
        "prompt": "a green backpack",
        "constraints": [
            {
                "constraint_id": "geneval2_0007_attr",
                "constraint_type": "attribute",
                "text": "green backpack",
                "metadata": {"source_index": 7, "attribute_bindings": [{"object": "backpack", "attribute": "green"}]},
            }
        ],
        "metadata": {"benchmark": "geneval2"},
    })
    program = {
        "workflow": {"compiled_prompt": "a green backpack on a plain background"},
        "contract": {
            "constraints": [
                {
                    "constraint_id": "safe_attr",
                    "constraint_type": "attribute",
                    "text": "green backpack",
                    "metadata": {"attribute_bindings": [{"object": "backpack", "attribute": "green"}]},
                }
            ]
        },
    }

    result = inspector.inspect(task, program, {"image_uri": "/tmp/image.png"}, seed=2)
    payload = json.loads(capture.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert result["safe_attr"]["passed"] is True
    assert payload["constraints"][0]["constraint_id"] == "constraint_0"
    assert payload["constraints"][0]["metadata"] == {"attribute_bindings": [{"object": "backpack", "attribute": "green"}]}
    for forbidden in ("geneval2_0007_attr", "source_index", "geneval2"):
        assert forbidden not in serialized


def test_public_task_row_sanitizes_constraint_metadata_provenance():
    from gen_harness.tool_payloads import public_task_row

    task = GenerationTask.from_dict({
        "task_id": "geneval2_0007",
        "prompt": "a green backpack",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "constraint_attr",
                "constraint_type": "attribute",
                "text": "green backpack",
                "metadata": {
                    "constraint_schema": "attribute_bindings.v1",
                    "attribute_bindings": [
                        {"object": "backpack", "attribute": "green", "source_index": 7}
                    ],
                    "benchmark": "geneval2",
                    "source_index": 7,
                    "official_score": 1.0,
                },
            }
        ],
        "references": {
            "identity_ref": {
                "uri": "https://example.invalid/ref.png",
                "role": "identity",
                "source_index": 7,
                "official_score": 1.0,
                "metadata": {
                    "benchmark": "geneval2",
                    "caption": "green backpack reference",
                },
            }
        },
        "metadata": {
            "split": "target",
            "benchmark": "geneval2",
            "source_index": 7,
        },
    })

    row = public_task_row(task)
    serialized = json.dumps(row, ensure_ascii=False)

    assert row["metadata"] == {"split": "target"}
    assert row["constraints"] == []
    assert row["constraint_source"] == "compiled_visual_contract"
    assert row["references"] == {
        "identity_ref": {
            "uri": "https://example.invalid/ref.png",
            "role": "identity",
            "metadata": {"caption": "green backpack reference"},
        }
    }
    for forbidden in ("source_index", "official_score", "benchmark"):
        assert forbidden not in serialized


def test_internal_object_inspection_prefers_sanitized_contract_constraints():
    task = GenerationTask.from_dict({
        "task_id": "geneval2_0007",
        "prompt": "a green backpack",
        "constraints": [
            {
                "constraint_id": "geneval2_0007_attr",
                "constraint_type": "attribute",
                "text": "green backpack",
                "metadata": {
                    "source_index": 7,
                    "required_objects": [{"class": "backpack", "count": 2}],
                },
            }
        ],
        "metadata": {"benchmark": "geneval2"},
    })
    program = {
        "contract": {
            "constraints": [
                {
                    "constraint_id": "safe_attr",
                    "constraint_type": "attribute",
                    "text": "green backpack",
                    "metadata": {"required_objects": [{"class": "backpack", "count": 1}]},
                }
            ]
        }
    }

    result = ObjectRequirementInspection().inspect(task, program, {"object_counts": {"backpack": 1}})
    serialized = json.dumps(result, ensure_ascii=False)

    assert set(result) == {"safe_attr"}
    assert result["safe_attr"]["passed"] is True
    assert "geneval2_0007_attr" not in serialized
    assert "source_index" not in serialized


def test_lightweight_alignment_prefers_sanitized_contract_constraints():
    task = GenerationTask.from_dict({
        "task_id": "geneval2_0007",
        "prompt": "a green backpack",
        "constraints": [
            {
                "constraint_id": "geneval2_0007_attr",
                "constraint_type": "attribute",
                "text": "green backpack",
                "metadata": {
                    "source_index": 7,
                    "attribute_bindings": [{"object": "backpack", "attribute": "green"}],
                },
            }
        ],
        "metadata": {"benchmark": "geneval2"},
    })
    program = {
        "contract": {
            "constraints": [
                {
                    "constraint_id": "safe_attr",
                    "constraint_type": "attribute",
                    "text": "green backpack",
                    "metadata": {"attribute_bindings": [{"object": "backpack", "attribute": "green"}]},
                }
            ]
        }
    }

    result = LightweightAlignmentInspection({"threshold": 0.30}).inspect(
        task,
        program,
        {"alignment_scores": {"safe_attr": 0.75}},
    )
    serialized = json.dumps(result, ensure_ascii=False)

    assert set(result) == {"safe_attr"}
    assert result["safe_attr"]["passed"] is True
    assert "geneval2_0007_attr" not in serialized
    assert "source_index" not in serialized


def test_composite_inspection_indexes_sanitized_contract_constraints():
    from gen_harness.inspection import CompositeInspectionRunner

    class Detector(InspectionRunner):
        config = {"name": "detector"}

        def inspect(self, task, program, generation, seed=0):
            return {
                "safe_attr": {
                    "passed": True,
                    "symptom": "pass",
                    "hard_evidence_required": True,
                    "evidence_status": "confirmed",
                    "attribute_color_scores": [{"object": "backpack", "attribute": "green", "score": 0.8}],
                }
            }

    class Semantic(InspectionRunner):
        config = {"name": "semantic"}

        def inspect(self, task, program, generation, seed=0):
            return {
                "safe_attr": {
                    "passed": False,
                    "symptom": "attribute_binding_mismatch",
                    "hard_evidence_required": True,
                    "evidence_status": "mismatch",
                }
            }

    task = GenerationTask.from_dict({
        "task_id": "geneval2_0007",
        "prompt": "a green backpack",
        "constraints": [
            {
                "constraint_id": "geneval2_0007_attr",
                "constraint_type": "count",
                "text": "green backpack",
                "metadata": {"source_index": 7},
            }
        ],
        "metadata": {"benchmark": "geneval2"},
    })
    program = {
        "contract": {
            "constraints": [
                {
                    "constraint_id": "safe_attr",
                    "constraint_type": "attribute",
                    "text": "green backpack",
                    "metadata": {"attribute_bindings": [{"object": "backpack", "attribute": "green"}]},
                }
            ]
        }
    }

    result = CompositeInspectionRunner(
        [Detector(), Semantic()],
        config={
            "merge_strategy": "capability_partitioned",
            "capability_owners": {
                "attribute_detector": "detector",
                "attribute": "semantic",
                "count": "semantic",
            },
        },
    ).inspect(task, program, {}, seed=0)
    serialized = json.dumps(result, ensure_ascii=False)

    assert set(result) == {"safe_attr"}
    assert result["safe_attr"]["verifier_conflict"] is True
    assert result["safe_attr"]["evidence_status"] == "unknown"
    assert "geneval2_0007_attr" not in serialized
    assert "source_index" not in serialized


def test_lightweight_alignment_external_payload_uses_public_constraint_ids(tmp_path):
    capture = tmp_path / "payload.json"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, shutil, sys\n"
        "payload=json.load(open(sys.argv[sys.argv.index('--input')+1]))\n"
        "shutil.copyfile(sys.argv[sys.argv.index('--input')+1], sys.argv[sys.argv.index('--capture')+1])\n"
        "json.dump({'scores': {'constraint_0': 0.77}, 'scorer': 'fake'}, open(sys.argv[sys.argv.index('--output')+1], 'w'))\n",
        encoding="utf-8",
    )
    inspector = LightweightAlignmentInspection({
        "constraint_types": ["attribute"],
        "threshold": 0.5,
        "command": [
            "{python}",
            str(worker),
            "--input",
            "{input_json}",
            "--output",
            "{output_json}",
            "--capture",
            str(capture),
        ],
    })
    task = GenerationTask.from_dict({
        "task_id": "t2icompbenchpp_color_val_0007",
        "prompt": "a green backpack",
        "constraints": [
            {
                "constraint_id": "t2icompbenchpp_color_val_0007_attr",
                "constraint_type": "attribute",
                "text": "green backpack",
                "metadata": {"category": "color"},
            }
        ],
    })
    program = {"workflow": {"compiled_prompt": "a green backpack"}}

    result = inspector.inspect(task, program, {"image_uri": "/tmp/image.png"}, seed=2)
    payload = json.loads(capture.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert result["t2icompbenchpp_color_val_0007_attr"]["passed"] is True
    assert "constraint_0" in serialized
    for forbidden in ("task", "program", "task_id", "t2icompbenchpp", "category"):
        assert forbidden not in serialized


def test_http_request_metadata_rejects_forbidden_provenance():
    from gen_harness.adapters.http_backend import HTTPImageGenerationBackend

    task = GenerationTask.from_dict({"task_id": "task", "prompt": "a cup", "constraints": []})
    program = {"workflow": {"compiled_prompt": "a cup"}}

    with pytest.raises(ValueError, match="request_metadata"):
        HTTPImageGenerationBackend({
            "endpoint": "https://example.invalid/generate",
            "request_metadata": {"task_id": "geneval2_0001"},
        }).generate(task, program, seed=0)


def test_json_object_extractor_handles_fenced_and_embedded_json():
    from gen_harness.json_extract import parse_json_object_from_text

    assert parse_json_object_from_text('{"ok": true}') == {"ok": True}
    assert parse_json_object_from_text('```json\n{"ok": {"nested": 1}}\n```') == {"ok": {"nested": 1}}
    assert parse_json_object_from_text('notes before {"value": "brace } in string", "n": 2} notes after') == {
        "value": "brace } in string",
        "n": 2,
    }


def test_expand_env_vars_uses_braced_names_without_regex(monkeypatch):
    monkeypatch.setenv("GENHARNESS_TEST_ROOT", "/tmp/genharness")

    expanded = expand_env_vars({
        "path": "${GENHARNESS_TEST_ROOT}/models",
        "nested": ["keep-${1INVALID}", "${GENHARNESS_TEST_ROOT}"],
    })

    assert expanded == {
        "path": "/tmp/genharness/models",
        "nested": ["keep-${1INVALID}", "/tmp/genharness"],
    }
    with pytest.raises(ValueError, match="environment variable is not set"):
        expand_env_vars("${GENHARNESS_TEST_MISSING}")



def test_geneval_tasks_preserve_structured_object_constraints(tmp_path):
    metadata = tmp_path / "evaluation_metadata.jsonl"
    tasks_path = tmp_path / "geneval_tasks.jsonl"
    metadata.write_text(
        json.dumps({
            "tag": "color_attr",
            "include": [
                {"class": "hair drier", "count": 1, "color": "purple"},
                {"class": "cake", "count": 1, "color": "blue", "position": ["right of", 0]},
            ],
            "prompt": "a photo of a purple hair drier and a blue cake",
        }) + "\n",
        encoding="utf-8",
    )

    rows = build_geneval_tasks(metadata, tasks_path, split="target")

    constraint = rows[0]["constraints"][0]
    assert constraint["constraint_type"] == "attribute"
    assert constraint["metadata"]["include"][0] == {"class": "hair drier", "count": 1}
    assert constraint["metadata"]["attribute_bindings"] == [
        {"object": "hair drier", "attribute": "purple", "attribute_type": "color"},
        {"object": "cake", "attribute": "blue", "attribute_type": "color"},
    ]
    assert constraint["metadata"]["spatial_relations"] == [
        {"subject": "cake", "relation": "right of", "object": "hair drier"}
    ]
    assert "required visible objects" in constraint["text"]


def test_compositional_prompt_uses_structured_object_requirements(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = GenerationTask.from_dict({
        "task_id": "geneval_00086",
        "prompt": "a photo of a hair drier and a cake",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "geneval_two_object_0",
            "text": "a photo of a hair drier and a cake; required visible objects: one hair drier, one cake",
            "constraint_type": "count",
            "target": "two_object",
            "verification_rule": "official GenEval object-focused evaluator",
            "metadata": {
                "benchmark": "geneval",
                "tag": "two_object",
                "include": [{"class": "hair drier", "count": 1}, {"class": "cake", "count": 1}],
                "exclude": [],
                "prompt": "a photo of a hair drier and a cake",
            },
        }],
        "metadata": {"split": "target", "benchmark": "geneval"},
    })
    contract = repo.build_policy().compile_contract(task)
    workflow = repo.build_skills().compile_workflow(task, "compositional_generation", contract, {}, memory_context={})
    prompt = workflow["compiled_prompt"]

    assert workflow["selected_prompt_family"] == "clutter_and_occlusion_control"
    assert "exactly one visible hair drier" in prompt
    assert "exactly one visible cake" in prompt
    assert "[{\"class\"" not in prompt
    assert "exactly one visible cake" in prompt
    assert "Skill rules:" not in prompt
    assert len(prompt.split()) < 80


def test_visual_harness_memory_has_no_evaluation_history():
    memory_dir = ROOT / "examples" / "visual_harness" / "memory"
    assert memory_dir.exists()
    history_files = [
        "visual_experiences.jsonl",
        "failure_clusters.jsonl",
        "validated_patches.jsonl",
        "regression_cases.jsonl",
        "successful_repairs.jsonl",
        "weaknesses.jsonl",
    ]
    for name in history_files:
        path = memory_dir / name
        assert path.exists()
        assert path.read_text(encoding="utf-8") == ""
    guidelines = (memory_dir / "prompt_guidelines.jsonl").read_text(encoding="utf-8")
    assert "geneval" not in guidelines.lower()


def test_geneval_position_prompt_family_adds_layout_program(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = GenerationTask.from_dict({
        "task_id": "geneval_00353",
        "prompt": "a photo of a dog right of a teddy bear",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "geneval_position_0",
            "text": "a photo of a dog right of a teddy bear; required visible objects: one teddy bear, one dog",
            "constraint_type": "spatial",
            "target": "position",
            "metadata": {
                "constraint_schema": "object_requirements.v1",
                "required_objects": [{"class": "teddy bear", "count": 1}, {"class": "dog", "count": 1}],
                "spatial_relations": [{"subject": "dog", "relation": "right of", "object": "teddy bear"}],
            },
        }],
        "metadata": {
            "benchmark": "geneval",
            "geneval_tag": "position",
            "geneval_metadata": {
                "tag": "position",
                "include": [
                    {"class": "teddy bear", "count": 1},
                    {"class": "dog", "count": 1, "position": ["right of", 0]},
                ],
                "prompt": "a photo of a dog right of a teddy bear",
            },
        },
    })

    contract = repo.build_policy().compile_contract(task)
    workflow = repo.build_skills().compile_workflow(task, "compositional_generation", contract, {}, memory_context={})
    prompt = workflow["compiled_prompt"]

    assert "benchmark_tag" not in contract
    assert workflow["selected_prompt_family"] == "object_relations"
    assert "Canvas layout program:" not in prompt
    assert "SPATIAL ANCHOR (right of)" in prompt
    assert "dog in the right edge band" in prompt
    assert "teddy bear in the left edge band" in prompt
    assert "no overlap" in prompt
    assert len(prompt.split()) < 120








def test_executor_formal_trace_metadata_removes_benchmark_provenance(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = GenerationTask.from_dict({
        "task_id": "trace_metadata_sanitized",
        "prompt": "a red cup",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "count_0",
                "text": "one red cup",
                "constraint_type": "count",
                "metadata": {"constraint_schema": "object_requirements.v1", "required_objects": [{"class": "cup", "count": 1}]},
            }
        ],
        "metadata": {
            "split": "heldout",
            "benchmark": "t2icompbenchpp",
            "category": "color",
            "source_index": 7,
            "geneval_metadata": {"include": [{"class": "cup", "count": 1}]},
        },
    })

    experiences = HarnessExecutor(
        repo,
        RecordingBackend(),
        HarnessAwareInspector(),
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    ).run_task(task, seed=0)

    assert experiences
    metadata = experiences[0].metadata
    assert metadata["task_family"] == "compositional"
    assert metadata["split"] == "heldout"
    assert "benchmark" not in metadata
    assert "category" not in metadata
    assert "source_index" not in metadata
    assert "geneval_metadata" not in metadata
    assert experiences[0].component_decisions["policy"]["task_metadata"] == {"split": "heldout"}


def test_policy_contract_sanitizes_constraint_metadata_provenance(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = GenerationTask.from_dict({
        "task_id": "constraint_metadata_sanitized",
        "prompt": "a red cup",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "count_0",
                "text": "one red cup",
                "constraint_type": "count",
                "metadata": {
                    "constraint_schema": "object_requirements.v1",
                    "required_objects": [{"class": "cup", "count": 1, "source_index": 7}],
                    "benchmark": "geneval",
                    "source_index": 7,
                    "official_score": 1.0,
                },
            }
        ],
        "metadata": {"split": "target"},
    })

    contract = repo.build_policy().compile_contract(task)
    metadata = contract["constraints"][0]["metadata"]

    assert metadata["constraint_schema"] == "object_requirements.v1"
    assert metadata["required_objects"] == [{"class": "cup", "count": 1}]
    assert "benchmark" not in metadata
    assert "source_index" not in metadata
    assert "official_score" not in metadata










def test_flow_grpo_passes_full_prompt_to_all_sd3_channels():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "sd3_full_prompt_passthrough",
        "prompt": "a photo of a dog right of a teddy bear",
        "task_family": "compositional",
        "constraints": [],
    })
    sentinel = "SENTINEL_FULL_PROMPT_AT_END"
    full_prompt = (
        "Create one image from this structured generation brief. "
        "Show a white dog clearly right of a brown teddy bear. "
        + " ".join(["context"] * 120)
        + " "
        + sentinel
    )
    program = {"method": "gen_harness", "workflow": {"compiled_prompt": full_prompt}}

    prompts = FlowGRPOBackend({})._sd3_prompt_kwargs(task, program, full_prompt)

    assert prompts == {"prompt": full_prompt, "prompt_2": full_prompt, "prompt_3": full_prompt}
    assert sentinel in prompts["prompt"]


def test_flow_grpo_default_model_root_is_open_source_safe(monkeypatch):
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    monkeypatch.delenv("GENHARNESS_FLOW_GRPO_MODEL_ROOT", raising=False)
    backend = FlowGRPOBackend({})

    assert str(backend.model_root) == "models"
    local_root = "/" + "/".join(["data", "luojiabin"])
    assert local_root not in str(backend.model_root)


def test_flow_grpo_adapter_use_is_explicit_and_defaults_on():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    assert FlowGRPOBackend({}).use_adapter is True
    assert FlowGRPOBackend({"use_adapter": False}).use_adapter is False


def test_inspection_command_python_placeholder_is_portable(monkeypatch):
    from gen_harness.inspection import format_inspection_command

    monkeypatch.setenv("GENHARNESS_PYTHON", "/opt/portable-python")
    monkeypatch.setenv("GENHARNESS_TEST_MODEL", "org/model")

    command = format_inspection_command(
        ["{python}", "tool.py", "--model", "${GENHARNESS_TEST_MODEL}", "--input", "{input_json}", "--output", "{output_json}"],
        input_json="/tmp/input.json",
        output_json="/tmp/output.json",
    )

    assert command == [
        "/opt/portable-python",
        "tool.py",
        "--model",
        "org/model",
        "--input",
        "/tmp/input.json",
        "--output",
        "/tmp/output.json",
    ]


def test_inspection_configs_do_not_hardcode_local_python_path():
    local_python = "/" + "/".join(["data", "luojiabin", "envs", "genharness", "bin", "python"])
    for path in (ROOT / "configs" / "inspection").glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert local_python not in text


def test_build_inspector_rejects_metadata_only_inspection_without_diagnostic_opt_in(tmp_path, monkeypatch):
    from gen_harness.inspection import ObjectRequirementInspection, build_inspector

    monkeypatch.delenv("GENHARNESS_DIAGNOSTIC_ALLOW_METADATA_INSPECTION", raising=False)
    config = tmp_path / "inspection.json"
    config.write_text(json.dumps({"type": "object_requirements"}), encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostic-only"):
        build_inspector(config)

    config.write_text(
        json.dumps({"type": "object_requirements", "allow_diagnostic_inspection_metadata": True}),
        encoding="utf-8",
    )
    assert isinstance(build_inspector(config), ObjectRequirementInspection)


def test_build_inspector_rejects_detector_passthrough_without_real_command(tmp_path, monkeypatch):
    from gen_harness.inspection import build_inspector

    monkeypatch.delenv("GENHARNESS_DIAGNOSTIC_ALLOW_METADATA_INSPECTION", raising=False)
    config = tmp_path / "inspection.json"
    config.write_text(json.dumps({"type": "detector_object_requirements"}), encoding="utf-8")

    with pytest.raises(ValueError, match="passthrough_generation_fields"):
        build_inspector(config)


def test_build_inspector_rejects_lightweight_metadata_scores_without_real_scorer(tmp_path, monkeypatch):
    from gen_harness.inspection import build_inspector

    monkeypatch.delenv("GENHARNESS_DIAGNOSTIC_ALLOW_METADATA_INSPECTION", raising=False)
    config = tmp_path / "inspection.json"
    config.write_text(json.dumps({"type": "lightweight_alignment"}), encoding="utf-8")

    with pytest.raises(ValueError, match="backend-provided scores"):
        build_inspector(config)


def test_flow_grpo_ignores_short_clip_prompt_for_full_prompt_passthrough():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "sd3_clip_prompt",
        "prompt": "a photo of a dog right of a teddy bear",
        "task_family": "compositional",
        "constraints": [],
    })
    full_prompt = "full structured prompt with all contract details and long trace guidance"
    clip_prompt = "short natural caption for CLIP"
    program = {"method": "gen_harness", "workflow": {"compiled_prompt": full_prompt, "clip_prompt": clip_prompt}}

    prompts = FlowGRPOBackend({})._sd3_prompt_kwargs(task, program, full_prompt)

    assert prompts["prompt"] == full_prompt
    assert prompts["prompt_2"] == full_prompt
    assert prompts["prompt_3"] == full_prompt


def test_flow_grpo_uses_generation_parameters_from_selected_tool():
    from types import SimpleNamespace
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    class FakePipe:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[SimpleNamespace(save=lambda path: None)])

    class FakeGenerator:
        def __init__(self, device=None):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    task = GenerationTask.from_dict({
        "task_id": "tool_override_case",
        "prompt": "a simple photo of a lamp left of a pizza on a plain background",
        "task_family": "compositional",
        "constraints": [],
        "metadata": {},
    })
    pipe = FakePipe()
    backend = FlowGRPOBackend({
        "split_prompt_channels": True,
        "num_inference_steps": 12,
        "guidance_scale": 2.0,
    })
    backend._load_pipe = lambda: pipe
    backend._torch = SimpleNamespace(Generator=FakeGenerator)
    backend._save_image = lambda image, task, program, seed: Path("/tmp/tool_override_case.png")
    backend._save_trace = lambda *args, **kwargs: Path("/tmp/tool_override_case.trace.json")
    program = {
        "method": "gen_harness",
        "workflow": {
            "compiled_prompt": "full structured prompt",
            "clip_prompt": "short clip prompt",
            "t5_prompt": "full structured prompt",
        },
        "tool": {
            "name": "base_generator",
            "generation_parameters": {
                "split_prompt_channels": False,
                "num_inference_steps": 33,
                "guidance_scale": 6.5,
            },
        },
    }

    result = backend.generate(task, program, seed=7)

    assert pipe.kwargs["num_inference_steps"] == 33
    assert pipe.kwargs["guidance_scale"] == 6.5
    assert pipe.kwargs["prompt"] == "full structured prompt"
    assert pipe.kwargs["prompt_3"] == "full structured prompt"
    assert result["num_inference_steps"] == 33
    assert result["guidance_scale"] == 6.5
    assert result["generation_parameters"]["split_prompt_channels"] is False


def test_flow_grpo_writes_trace_outside_image_cache(tmp_path):
    from PIL import Image
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "trace_layout_case",
        "prompt": "a photo of a bench",
        "task_family": "compositional",
        "constraints": [],
        "metadata": {
            "split": "heldout",
            "benchmark": "geneval",
            "geneval_index": 3,
            "category": "counting",
            "source_index": 9,
            "generation_input": "prompt_only_no_official_eval_outputs",
        },
    })
    backend = FlowGRPOBackend({
        "image_cache_dir": str(tmp_path / "images"),
        "trace_cache_dir": str(tmp_path / "traces"),
    })
    image = Image.new("RGB", (1, 1), color="white")
    program = {"method": "gen_harness", "workflow": {"compiled_prompt": "a photo of a bench"}}
    image_path = backend._save_image(image, task, program, seed=0)
    trace_path = backend._save_trace(image_path, task, program, {"prompt": "a photo of a bench"}, seed=0)

    assert image_path.parent.parent.name == "images"
    assert trace_path.parent.parent.name == "traces"
    assert image_path.name in trace_path.name
    assert not str(trace_path).startswith(str(image_path.parent))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["task"]["metadata"] == {
        "split": "heldout",
        "generation_input": "prompt_only_no_official_eval_outputs",
    }


def test_flow_grpo_requires_compiled_prompt_for_gen_harness():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "missing_compiled_prompt",
        "prompt": "a photo of a bench",
        "task_family": "compositional",
        "constraints": [],
    })
    program = {"method": "gen_harness", "workflow": {}}

    try:
        FlowGRPOBackend({})._prompt_for_program(task, program)
    except ValueError as exc:
        assert "requires workflow.compiled_prompt" in str(exc)
    else:
        raise AssertionError("expected missing compiled prompt to fail instead of falling back")


def test_flow_grpo_does_not_build_prompt_from_contract_metadata():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "generic_spatial_prompt",
        "prompt": "a simple photo of a pizza left of a toothbrush",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_relation_0",
            "text": "pizza must be clearly left of toothbrush",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "required_objects": [{"class": "toothbrush", "count": 1}, {"class": "pizza", "count": 1}],
                "spatial_relations": [{"subject": "pizza", "relation": "left of", "object": "toothbrush"}],
            },
        }],
        "metadata": {},
    })
    program = {
        "method": "gen_harness",
        "contract": {"constraints": [c.to_dict() for c in task.constraints]},
        "workflow": {"compiled_prompt": "full structured prompt"},
    }

    prompts = FlowGRPOBackend({})._sd3_prompt_kwargs(task, program, "full structured prompt")

    assert prompts == {
        "prompt": "full structured prompt",
        "prompt_2": "full structured prompt",
        "prompt_3": "full structured prompt",
    }
    assert "pizza clearly left of toothbrush" not in prompts["prompt"]
    assert "Place pizza entirely in the left half" not in prompts["prompt_3"]


def test_flow_grpo_does_not_inject_detector_disambiguation_rules():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "clip_detector_rules",
        "prompt": "a photo of a toothbrush below a pizza",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "geneval_position_0",
            "text": "required visible objects: one pizza, one toothbrush",
            "constraint_type": "spatial",
            "metadata": {
                "required_objects": [{"class": "pizza", "count": 1}, {"class": "toothbrush", "count": 1}],
                "spatial_relations": [{"subject": "toothbrush", "relation": "below", "object": "pizza"}],
            },
        }],
        "metadata": {
            "geneval_metadata": {
                "tag": "position",
                "include": [
                    {"class": "pizza", "count": 1},
                    {"class": "toothbrush", "count": 1, "position": ["below", 0]},
                ],
            }
        },
    })
    program = {
        "method": "gen_harness",
        "contract": {"constraints": [c.to_dict() for c in task.constraints]},
        "workflow": {"compiled_prompt": "full structured prompt"},
    }

    prompts = FlowGRPOBackend({})._sd3_prompt_kwargs(task, program, "full structured prompt")

    assert "toothbrush with bristles and handle" not in prompts["prompt"]
    assert "toothbrush not knife" not in prompts["prompt"]
    assert "one whole pizza only" not in prompts["prompt"]
    assert prompts["prompt"] == "full structured prompt"
    assert prompts["prompt_3"] == "full structured prompt"


def test_flow_grpo_keeps_single_object_antiduplicate_when_compiled_prompt_has_it():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "sd3_passthrough_single_antidup",
        "prompt": "a simple photo of a chair left of a bench on a plain background",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_relation_0",
            "text": "chair must be clearly left of bench",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "count_mode": "exact",
                "required_objects": [{"class": "bench", "count": 1}, {"class": "chair", "count": 1}],
                "spatial_relations": [{"subject": "chair", "relation": "left of", "object": "bench"}],
            },
        }],
    })
    program = {
        "method": "gen_harness",
        "contract": {"constraints": [c.to_dict() for c in task.constraints]},
        "workflow": {
            "compiled_prompt": (
                "a simple photo of a chair left of a bench on a plain background. "
                "Include exactly one bench, no duplicate. Include exactly one chair, no duplicate."
            )
        },
    }

    full_prompt = program["workflow"]["compiled_prompt"]
    clip_prompt = FlowGRPOBackend({})._sd3_prompt_kwargs(task, program, full_prompt)["prompt"]

    assert "Include exactly one bench, no duplicate" in clip_prompt
    assert "Include exactly one chair, no duplicate" in clip_prompt


def test_flow_grpo_uses_split_prompt_channels_from_config():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "sd3_split_prompt_config",
        "prompt": "a photo of a lamp left of a pizza",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_0",
            "text": "lamp left of pizza",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "spatial_relations": [{"subject": "lamp", "relation": "left of", "object": "pizza"}],
                "required_objects": [{"class": "lamp", "count": 1}, {"class": "pizza", "count": 1}],
            },
        }],
    })
    backend = FlowGRPOBackend({"split_prompt_channels": True})
    program = {
        "method": "gen_harness",
        "workflow": {
            "compiled_prompt": "full structured prompt",
            "clip_prompt": "short split prompt",
            "t5_prompt": "long split prompt",
        },
    }

    kwargs = backend._sd3_prompt_kwargs(task, program, program["workflow"]["compiled_prompt"])

    assert kwargs["prompt"] == "short split prompt"
    assert kwargs["prompt_3"] == "long split prompt"


def test_compositional_prompt_frontloads_attribute_binding_program(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = GenerationTask.from_dict({
        "task_id": "attribute_prompt_order",
        "prompt": "a simple photo of a red cup and a blue book",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "attr_1",
            "text": "red cup and blue book",
            "constraint_type": "attribute",
            "metadata": {
                "constraint_schema": "attribute_bindings.v1",
                "required_objects": [{"class": "cup", "count": 1}, {"class": "book", "count": 1}],
                "attribute_bindings": [
                    {"object": "cup", "attribute": "red"},
                    {"object": "book", "attribute": "blue"},
                ],
            },
        }],
    })

    contract = repo.build_policy().compile_contract(task)
    workflow = repo.build_skills().compile_workflow(task, "compositional_generation", contract, {}, memory_context={})
    prompt = workflow["compiled_prompt"]

    assert "exact colors: red cup, blue book" in prompt.lower()
    assert "Recognizable cup, visibly red" in prompt
    assert "Recognizable book, visibly blue" in prompt
    assert "Recognizable cup" in prompt
    assert "Skill rules:" not in prompt
    assert "Exact colors: red cup, blue book" in workflow["clip_prompt"]
    assert "Recognizable cup, visibly red" in workflow["clip_prompt"]
    assert "Recognizable book, visibly blue" in workflow["clip_prompt"]
    assert workflow["clip_prompt"] != prompt
    assert workflow["t5_prompt"] == prompt
    assert len(prompt.split()) < 150


def test_compositional_prompt_is_compact_without_manual_truncation(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = GenerationTask.from_dict({
        "task_id": "compact_not_truncated",
        "prompt": "a simple photo of a red cup left of a blue book",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "spatial_1",
                "text": "cup left of book",
                "constraint_type": "spatial",
                "metadata": {
                    "constraint_schema": "spatial_relations.v1",
                    "required_objects": [{"class": "cup", "count": 1}, {"class": "book", "count": 1}],
                    "spatial_relations": [{"subject": "cup", "relation": "left of", "object": "book"}],
                },
            },
            {
                "constraint_id": "attr_1",
                "text": "red cup and blue book",
                "constraint_type": "attribute",
                "metadata": {
                    "constraint_schema": "attribute_bindings.v1",
                    "attribute_bindings": [
                        {"object": "cup", "attribute": "red"},
                        {"object": "book", "attribute": "blue"},
                    ],
                },
            },
        ],
    })

    contract = repo.build_policy().compile_contract(task)
    prompt = repo.build_skills().compile_workflow(task, "compositional_generation", contract, {}, memory_context={})["compiled_prompt"]

    assert "red cup" in prompt
    assert "blue book" in prompt
    assert "SPATIAL ANCHOR (left of)" in prompt
    assert "cup in the left edge band" in prompt
    assert "book in the right edge band" in prompt
    assert "Skill rules:" not in prompt
    assert "Memory guidance:" not in prompt
    assert len(prompt.split()) < 170




def test_generation_path_does_not_use_geneval_metadata_when_contract_is_empty():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "no_annotation_shortcut",
        "prompt": "a photo of a dog right of a teddy bear",
        "task_family": "compositional",
        "constraints": [],
        "metadata": {
            "benchmark": "geneval",
            "geneval_tag": "position",
            "geneval_metadata": {
                "tag": "position",
                "include": [
                    {"class": "teddy bear", "count": 1},
                    {"class": "dog", "count": 1, "position": ["right of", 0], "color": "white"},
                ],
            },
        },
    })
    program = {"method": "gen_harness", "contract": {"constraints": []}, "workflow": {"compiled_prompt": "full structured prompt"}}

    prompts = FlowGRPOBackend({})._sd3_prompt_kwargs(task, program, "full structured prompt")

    assert "dog entirely right half" not in prompts["prompt"]
    assert "white dog" not in prompts["prompt"]
    assert "Layout: dog must be clearly right of teddy bear." not in prompts["prompt_3"]
    assert prompts["prompt"] == "full structured prompt"
    assert prompts["prompt_3"] == "full structured prompt"


def test_flow_grpo_count_prompt_uses_compiled_prompt_without_rewriting():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "count_prompt_slots",
        "prompt": "a clean studio photo of 5 books on a plain surface",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "count_0",
            "text": "include exactly 5 books",
            "constraint_type": "count",
            "metadata": {
                "constraint_schema": "object_requirements.v1",
                "count_mode": "exact",
                "required_objects": [{"class": "book", "count": 5}],
            },
        }],
    })
    program = {
        "method": "gen_harness",
        "contract": {"constraints": [c.to_dict() for c in task.constraints]},
        "workflow": {
            "compiled_prompt": (
                "full structured prompt. "
                "Include exactly 5 complete separate books. "
                "Arrange the 5 books in 5 distinct slots. "
                "Use one whole object per slot."
            )
        },
    }

    full_prompt = program["workflow"]["compiled_prompt"]
    prompts = FlowGRPOBackend({})._sd3_prompt_kwargs(task, program, full_prompt)

    assert prompts["prompt"] == full_prompt
    assert prompts["prompt_2"] == full_prompt
    assert prompts["prompt_3"] == full_prompt
    assert "Include exactly 5 complete separate books" in prompts["prompt_3"]
    assert "5 distinct slots" in prompts["prompt_3"]
    assert "Use one whole object per slot" in prompts["prompt_3"]


def test_compositional_skill_count_prompt_uses_exact_separate_plural(tmp_path):
    harness = make_harness(tmp_path)
    task = GenerationTask.from_dict({
        "task_id": "skill_count_prompt_slots",
        "prompt": "a clean studio photo of 5 books on a plain surface",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "count_0",
            "text": "include exactly 5 books",
            "constraint_type": "count",
            "metadata": {
                "constraint_schema": "object_requirements.v1",
                "count_mode": "exact",
                "required_objects": [{"class": "book", "count": 5}],
            },
        }],
    })
    repo = HarnessRepository(harness)
    contract = repo.build_policy().compile_contract(task)
    workflow = repo.build_skills().compile_workflow(task, "compositional_generation", contract, {})

    assert "Exactly five complete books" in workflow["compiled_prompt"]
    assert "three in the back row and two in the front row" in workflow["compiled_prompt"]
    assert "5 instances of book" not in workflow["compiled_prompt"]








def test_t2icompbenchpp_policy_contract_is_invariant_to_category_and_source_index(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    prompt = "The smooth white marble statue stood in front of the rough grey wall."
    contracts = []
    for category, source_index in [("complex", 1), ("color", 999), ("non_spatial", 42)]:
        task = GenerationTask.from_dict({
            "task_id": f"t2icompbenchpp_category_invariant_{category}",
            "prompt": prompt,
            "task_family": "compositional",
            "constraints": [],
            "metadata": {
                "benchmark": "t2icompbenchpp",
                "category": category,
                "source_index": source_index,
                "generation_input": "prompt_only_no_official_eval_outputs",
            },
        })
        contract = repo.build_policy().compile_contract(task)
        contracts.append(contract["constraints"])

    assert contracts[0] == contracts[1] == contracts[2]
    serialized = json.dumps(contracts[0])
    assert "source_index" not in serialized
    assert "benchmark_category_used" not in serialized






























def test_flow_grpo_lowmem_config_uses_cpu_offload_instead_of_full_cuda_move():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    class FakePipe:
        def __init__(self):
            self.calls = []

        def enable_attention_slicing(self):
            self.calls.append("attention_slicing")

        def enable_vae_slicing(self):
            self.calls.append("vae_slicing")

        def enable_vae_tiling(self):
            self.calls.append("vae_tiling")

        def enable_model_cpu_offload(self):
            self.calls.append("model_cpu_offload")

        def to(self, device):
            self.calls.append(f"to:{device}")

    pipe = FakePipe()
    backend = FlowGRPOBackend({
        "enable_model_cpu_offload": True,
        "enable_attention_slicing": True,
        "enable_vae_slicing": True,
        "enable_vae_tiling": True,
    })

    backend._configure_pipe_memory(pipe, cuda_available=True)

    assert pipe.calls == ["attention_slicing", "vae_slicing", "vae_tiling", "model_cpu_offload"]


def test_flow_grpo_rejects_conflicting_lowmem_modes():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    backend = FlowGRPOBackend({
        "enable_model_cpu_offload": True,
        "enable_sequential_cpu_offload": True,
    })

    with pytest.raises(ValueError, match="mutually exclusive"):
        backend._configure_pipe_memory(object(), cuda_available=True)


def test_flow_grpo_fast_import_package_distribution_map_is_scoped():
    import importlib.metadata as importlib_metadata
    pytest.importorskip("torch")
    pytest.importorskip("transformers", reason="optional Flow-GRPO distribution-map integration test")

    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    backend = FlowGRPOBackend({})
    original = importlib_metadata.packages_distributions

    with backend._fast_package_distribution_map():
        patched = importlib_metadata.packages_distributions
        package_map = patched()
        assert patched is not original
        assert package_map.get("torch")
        assert package_map.get("transformers")

    assert importlib_metadata.packages_distributions is original


def test_flow_grpo_preflight_reports_insufficient_cuda_memory():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def mem_get_info():
            return 20 * 1024**3, 80 * 1024**3

    class FakeTorch:
        cuda = FakeCuda()

    backend = FlowGRPOBackend({"min_free_cuda_memory_gb": 32})
    backend._cuda_memory_info_from_nvidia_smi = lambda: (20.0, 80.0)

    with pytest.raises(RuntimeError, match="only 20.0/80.0 GiB is free"):
        backend._preflight_cuda(FakeTorch)


def test_inspection_config_command_scripts_exist():
    config_dir = ROOT / "configs" / "inspection"
    script_refs = []
    for path in config_dir.glob("*.json"):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        inspectors = cfg.get("inspectors", [cfg]) if isinstance(cfg, dict) else []
        for item in inspectors:
            if not isinstance(item, dict):
                continue
            for key in ("command", "persistent_command"):
                command = item.get(key, [])
                if not isinstance(command, list):
                    continue
                for part in command:
                    text = str(part)
                    if text.startswith("scripts/") and text.endswith(".py"):
                        script_refs.append((path.name, text))

    assert script_refs
    missing = [(cfg_name, ref) for cfg_name, ref in script_refs if not (ROOT / ref).is_file()]
    assert missing == []




def test_flow_grpo_keeps_compiled_prompt_order_before_repair_hints():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "sd3_passthrough_repair_priority",
        "prompt": "a photo of a toothbrush below a pizza",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "generic_position_0",
            "text": "toothbrush below pizza",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "spatial_relations": [{"subject": "toothbrush", "relation": "below", "object": "pizza"}],
            },
        }],
        "metadata": {
            "geneval_metadata": {
                "tag": "position",
                "include": [
                    {"class": "pizza", "count": 1},
                    {"class": "toothbrush", "count": 1, "position": ["below", 0]},
                ],
            }
        },
    })
    program = {
        "method": "gen_harness",
        "contract": {"constraints": [c.to_dict() for c in task.constraints]},
        "workflow": {
            "compiled_prompt": (
                "a photo of a toothbrush below a pizza. "
                "toothbrush bottom, pizza top, vertical gap, not side-by-side. "
                "The previous image placed the toothbrush left of the pizza; regenerate with the toothbrush clearly below the pizza."
            )
        },
        "fast_loop_repair": {
            "hints": [
                "The previous image placed the toothbrush left of the pizza; regenerate with the toothbrush clearly below the pizza.",
                "Place the toothbrush in the lower half and the pizza in the upper half; do not swap them.",
            ]
        },
    }

    full_prompt = program["workflow"]["compiled_prompt"]
    clip_prompt = FlowGRPOBackend({})._sd3_prompt_kwargs(task, program, full_prompt)["prompt"]

    assert clip_prompt.startswith("a photo of a toothbrush below a pizza")
    assert "regenerate with the toothbrush clearly below the pizza" in clip_prompt
    assert "toothbrush bottom, pizza top, vertical gap, not side-by-side" in clip_prompt


def test_detector_inspection_checks_spatial_relation_with_bboxes(tmp_path):
    detector = tmp_path / "detector.py"
    detector.write_text(
        "import json, sys\n"
        "out=sys.argv[sys.argv.index('--output')+1]\n"
        "json.dump({'object_counts': {'dog': 1, 'teddy bear': 1}, "
        "'detected_objects': ["
        "{'class': 'dog', 'score': 0.9, 'bbox': [10, 10, 40, 40]}, "
        "{'class': 'teddy bear', 'score': 0.9, 'bbox': [80, 10, 120, 40]}]}, open(out, 'w'))\n",
        encoding="utf-8",
    )
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "relation_fail",
        "prompt": "a photo of a dog right of a teddy bear",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "geneval_position_0",
            "text": "dog right of teddy bear",
            "constraint_type": "spatial",
            "metadata": {
                "required_objects": [{"class": "dog", "count": 1}, {"class": "teddy bear", "count": 1}],
                "spatial_relations": [{"subject": "dog", "relation": "right of", "object": "teddy bear"}],
            },
        }],
        "metadata": {
            "geneval_metadata": {
                "tag": "position",
                "include": [
                    {"class": "teddy bear", "count": 1},
                    {"class": "dog", "count": 1, "position": ["right of", 0]},
                ],
            }
        },
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "command": ["{python}", str(detector), "--input", "{input_json}", "--output", "{output_json}"],
    })

    result = inspector.inspect(task, {}, {"image_uri": "/tmp/image.png"}, seed=0)

    assert result["geneval_position_0"]["passed"] is False
    assert result["geneval_position_0"]["symptom"] == "spatial_relation_mismatch"
    assert result["geneval_position_0"]["spatial_relation_mismatches"][0]["actual_relation"] == "left of"


def test_detector_spatial_relation_exposes_continuous_constraint_score():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "spatial_soft_score",
        "prompt": "a photo of a dog right of a teddy bear",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_1",
            "text": "dog right of teddy bear",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "required_objects": [{"class": "dog", "count": 1}, {"class": "teddy bear", "count": 1}],
                "spatial_relations": [{"subject": "dog", "relation": "right of", "object": "teddy bear"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({"type": "detector_object_requirements", "passthrough_generation_fields": True})

    wrong = inspector.inspect(task, {}, {
        "detected_objects": [
            {"class": "dog", "score": 0.9, "bbox": [10, 10, 40, 40]},
            {"class": "teddy bear", "score": 0.9, "bbox": [80, 10, 120, 40]},
        ]
    })
    right = inspector.inspect(task, {}, {
        "detected_objects": [
            {"class": "dog", "score": 0.9, "bbox": [90, 10, 130, 40]},
            {"class": "teddy bear", "score": 0.9, "bbox": [10, 10, 40, 40]},
        ]
    })

    assert wrong["spatial_1"]["passed"] is False
    assert 0.0 <= wrong["spatial_1"]["constraint_score"] < 1.0
    assert right["spatial_1"]["passed"] is True
    assert right["spatial_1"]["constraint_score"] == 1.0


def test_detector_optional_unrequested_object_gate_is_generic():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "generic_extra_object_gate",
        "prompt": "a photo of an orange microwave and a black spoon",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "objects",
            "text": "one microwave and one spoon",
            "constraint_type": "count",
            "metadata": {
                "count_mode": "exact",
                "required_objects": [
                    {"class": "microwave", "count": 1},
                    {"class": "spoon", "count": 1},
                ],
            },
        }],
    })
    generation = {
        "object_counts": {"microwave": 1, "spoon": 1},
        "detected_objects": [
            {"class": "microwave", "score": 0.95, "bbox": [0, 0, 20, 20]},
            {"class": "spoon", "score": 0.94, "bbox": [30, 0, 40, 20]},
        ],
        "all_detected_objects": [
            {"class": "microwave", "score": 0.95, "bbox": [0, 0, 20, 20]},
            {"class": "spoon", "score": 0.94, "bbox": [30, 0, 40, 20]},
            {"class": "bowl", "score": 0.91, "bbox": [45, 0, 60, 20]},
        ],
        "classes": ["microwave", "spoon"],
    }
    gated = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
        "reject_unrequested_detected_objects": True,
        "unexpected_object_threshold": 0.80,
        "unexpected_object_overlap_suppression": 0.90,
    }).inspect(task, {}, generation)
    ungated = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
    }).inspect(task, {}, generation)

    assert gated["objects"]["passed"] is False
    assert gated["objects"]["evidence_status"] == "mismatch"
    assert gated["objects"]["unexpected_detected_objects"][0]["class"] == "bowl"
    assert ungated["objects"]["passed"] is True


def test_detector_unrequested_object_gate_suppresses_same_region_label_alias():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "generic_overlapping_label_alias",
        "prompt": "a photo of a potted plant",
        "constraints": [{
            "constraint_id": "objects",
            "text": "one potted plant",
            "constraint_type": "object",
            "metadata": {"required_objects": [{"class": "potted plant", "count": 1}]},
        }],
    })
    generation = {
        "object_counts": {"potted plant": 1},
        "detected_objects": [
            {"class": "potted plant", "score": 0.93, "bbox": [10, 10, 70, 90]},
        ],
        "all_detected_objects": [
            {"class": "potted plant", "score": 0.93, "bbox": [10, 10, 70, 90]},
            {"class": "vase", "score": 0.88, "bbox": [14, 48, 66, 88]},
        ],
        "classes": ["potted plant"],
    }

    result = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
        "reject_unrequested_detected_objects": True,
        "unexpected_object_threshold": 0.80,
        "unexpected_object_overlap_suppression": 0.90,
    }).inspect(task, {}, generation)

    assert result["objects"]["passed"] is True
    assert result["objects"]["suppressed_overlapping_detections"][0]["class"] == "vase"
    assert result["objects"]["suppressed_overlapping_detections"][0]["intersection_over_smaller"] >= 0.90


def test_detector_spatial_relation_uses_best_matching_bbox_pair():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "spatial_best_pair",
        "prompt": "a photo of a dog right of a teddy bear",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_1",
            "text": "dog right of teddy bear",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "spatial_relations": [{"subject": "dog", "relation": "right of", "object": "teddy bear"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({"type": "detector_object_requirements", "passthrough_generation_fields": True})

    result = inspector.inspect(task, {}, {
        "detected_objects": [
            {"class": "dog", "score": 0.99, "bbox": [10, 10, 40, 40]},
            {"class": "dog", "score": 0.75, "bbox": [120, 10, 160, 40]},
            {"class": "teddy bear", "score": 0.90, "bbox": [60, 10, 100, 40]},
        ]
    })

    row = result["spatial_1"]
    assert row["passed"] is True
    assert row["spatial_relations_passed"][0]["subject_bbox"] == [120, 10, 160, 40]


def test_detector_spatial_relation_can_use_top_confidence_pair_for_stricter_selection():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "spatial_top_pair",
        "prompt": "a photo of a dog right of a teddy bear",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_1",
            "text": "dog right of teddy bear",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "spatial_relations": [{"subject": "dog", "relation": "right of", "object": "teddy bear"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
        "spatial_match_mode": "top_confidence",
    })

    result = inspector.inspect(task, {}, {
        "detected_objects": [
            {"class": "dog", "score": 0.99, "bbox": [10, 10, 40, 40]},
            {"class": "dog", "score": 0.75, "bbox": [120, 10, 160, 40]},
            {"class": "teddy bear", "score": 0.90, "bbox": [60, 10, 100, 40]},
        ]
    })

    row = result["spatial_1"]
    assert row["passed"] is False
    assert row["spatial_relation_mismatches"][0]["subject_bbox"] == [10, 10, 40, 40]


def test_detector_spatial_relation_rejects_duplicate_exact_bound_objects():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "spatial_duplicate_exact",
        "prompt": "a photo of a sandwich below a knife",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_1",
            "text": "sandwich below knife",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "count_mode": "exact",
                "required_objects": [{"class": "sandwich", "count": 1}, {"class": "knife", "count": 1}],
                "spatial_relations": [{"subject": "sandwich", "relation": "below", "object": "knife"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({"type": "detector_object_requirements", "passthrough_generation_fields": True})

    result = inspector.inspect(task, {}, {
        "object_counts": {"sandwich": 2, "knife": 1},
        "detected_objects": [
            {"class": "sandwich", "score": 0.98, "bbox": [20, 220, 120, 270]},
            {"class": "sandwich", "score": 0.97, "bbox": [20, 40, 120, 90]},
            {"class": "knife", "score": 0.95, "bbox": [160, 10, 210, 300]},
        ],
    })

    row = result["spatial_1"]
    assert row["passed"] is False
    assert row["symptom"] == "spatial_relation_mismatch"
    assert row["spatial_relation_mismatches"][0]["reason"] == "ambiguous_duplicate_object_for_relation"


def test_detector_spatial_relation_requires_primary_axis_separation():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "spatial_axis_dominance",
        "prompt": "a photo of a dog below a teddy bear",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_1",
            "text": "dog below teddy bear",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "spatial_relations": [{"subject": "dog", "relation": "below", "object": "teddy bear"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({"type": "detector_object_requirements", "passthrough_generation_fields": True})

    result = inspector.inspect(task, {}, {
        "detected_objects": [
            {"class": "dog", "score": 0.9, "bbox": [120, 60, 160, 100]},
            {"class": "teddy bear", "score": 0.9, "bbox": [10, 30, 50, 70]},
        ]
    })

    row = result["spatial_1"]
    assert row["passed"] is False
    assert row["spatial_relation_mismatches"][0]["actual_relation"] == "right of"


def test_detector_spatial_relation_can_use_geneval_size_aware_geometry():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "spatial_size_aware",
        "prompt": "a photo of a dog right of a teddy bear",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_1",
            "text": "dog right of teddy bear",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "spatial_relations": [{"subject": "dog", "relation": "right of", "object": "teddy bear"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
        "spatial_geometry_mode": "geneval_size_aware_v1",
        "spatial_size_threshold": 0.1,
    })

    too_close = inspector.inspect(task, {}, {
        "detected_objects": [
            {"class": "dog", "score": 0.9, "bbox": [45, 0, 65, 100]},
            {"class": "teddy bear", "score": 0.9, "bbox": [0, 0, 100, 100]},
        ]
    })
    separated = inspector.inspect(task, {}, {
        "detected_objects": [
            {"class": "dog", "score": 0.9, "bbox": [80, 0, 100, 100]},
            {"class": "teddy bear", "score": 0.9, "bbox": [0, 0, 40, 100]},
        ]
    })

    assert too_close["spatial_1"]["passed"] is False
    assert too_close["spatial_1"]["spatial_relation_mismatches"][0]["actual_relation"] == "none"
    assert separated["spatial_1"]["passed"] is True


def test_detector_spatial_merge_preserves_object_count_failure():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "spatial_count_merge",
        "prompt": "a photo of a lamp right of a chair",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_1",
            "text": "lamp right of chair",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "count_mode": "exact",
                "required_objects": [{"class": "lamp", "count": 1}, {"class": "chair", "count": 1}],
                "spatial_relations": [{"subject": "lamp", "relation": "right of", "object": "chair"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({"type": "detector_object_requirements", "passthrough_generation_fields": True})

    result = inspector.inspect(task, {}, {
        "object_counts": {"lamp": 1, "chair": 2},
        "detected_objects": [
            {"class": "chair", "score": 0.9, "bbox": [10, 10, 40, 40]},
            {"class": "lamp", "score": 0.9, "bbox": [90, 10, 120, 40]},
        ],
    })

    row = result["spatial_1"]
    assert row["passed"] is False
    assert row["symptom"] == "object_count_mismatch"
    assert row["constraint_score"] == 0.5


def test_detector_attribute_color_verification_uses_object_crops(tmp_path):
    from PIL import Image, ImageDraw

    from gen_harness.inspection import DetectorObjectRequirementInspection

    image_path = tmp_path / "red_cup_blue_book.png"
    image = Image.new("RGB", (120, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 50, 50], fill=(230, 20, 20))
    draw.rectangle([70, 10, 110, 50], fill=(20, 40, 230))
    image.save(image_path)

    task = GenerationTask.from_dict({
        "task_id": "attribute_color_crop",
        "prompt": "a red cup and a blue book",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "attr_1",
            "text": "red cup and blue book",
            "constraint_type": "attribute",
            "metadata": {
                "constraint_schema": "attribute_bindings.v1",
                "required_objects": [{"class": "cup", "count": 1}, {"class": "book", "count": 1}],
                "attribute_bindings": [
                    {"object": "cup", "attribute": "red"},
                    {"object": "book", "attribute": "blue"},
                ],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({"type": "detector_object_requirements", "passthrough_generation_fields": True})

    result = inspector.inspect(task, {}, {
        "image_path": str(image_path),
        "object_counts": {"cup": 1, "book": 1},
        "detected_objects": [
            {"class": "cup", "score": 0.9, "bbox": [10, 10, 50, 50]},
            {"class": "book", "score": 0.9, "bbox": [70, 10, 110, 50]},
        ],
    })

    row = result["attr_1"]
    assert row["passed"] is True
    assert row["attribute_color_scores"][0]["positive_color_score"] > 0.8
    assert row["attribute_color_scores"][1]["positive_color_score"] > 0.8


def test_detector_color_rules_separate_brown_from_orange():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    inspector = DetectorObjectRequirementInspection({})
    # Same orange/brown hue family, separated by HSV value.
    import colorsys
    brown_hsv = colorsys.rgb_to_hsv(120 / 255.0, 60 / 255.0, 20 / 255.0)
    orange_hsv = colorsys.rgb_to_hsv(255 / 255.0, 128 / 255.0, 0.0)

    assert inspector._pixel_matches_color(*brown_hsv, "brown") is True
    assert inspector._pixel_matches_color(*brown_hsv, "orange") is False
    assert inspector._pixel_matches_color(*orange_hsv, "orange") is True
    assert inspector._pixel_matches_color(*orange_hsv, "brown") is False


def test_chromatic_crop_ignores_achromatic_bbox_background(tmp_path):
    from PIL import Image, ImageDraw
    from gen_harness.inspection import DetectorObjectRequirementInspection

    image_path = tmp_path / "blue_object_white_bbox.png"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle([30, 30, 69, 69], fill=(0, 70, 255))
    image.save(image_path)
    task = GenerationTask.from_dict({
        "task_id": "mask_like_color",
        "prompt": "a blue kite",
        "constraints": [{
            "constraint_id": "attr",
            "constraint_type": "attribute",
            "text": "kite is blue",
            "metadata": {
                "required_objects": [{"class": "kite", "count": 1}],
                "attribute_bindings": [{"object": "kite", "attribute": "blue"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({
        "passthrough_generation_fields": True,
        "attribute_color_verification": True,
        "attribute_color_compare_all_colors": True,
        "attribute_ignore_achromatic_competitors_for_chromatic": True,
        "attribute_color_threshold": 0.045,
        "attribute_color_margin": 0.01,
    })

    row = inspector.inspect(task, {}, {
        "image_path": str(image_path),
        "object_counts": {"kite": 1},
        "detected_objects": [{"class": "kite", "score": 0.9, "bbox": [0, 0, 100, 100]}],
    })["attr"]

    assert row["passed"] is True
    assert row["attribute_color_scores"][0]["positive_color_score"] > 0.10


def test_detector_attribute_color_verification_catches_swapped_colors(tmp_path):
    from PIL import Image, ImageDraw

    from gen_harness.inspection import DetectorObjectRequirementInspection

    image_path = tmp_path / "swapped_cup_book.png"
    image = Image.new("RGB", (120, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 50, 50], fill=(20, 40, 230))
    draw.rectangle([70, 10, 110, 50], fill=(230, 20, 20))
    image.save(image_path)

    task = GenerationTask.from_dict({
        "task_id": "attribute_color_swap",
        "prompt": "a red cup and a blue book",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "attr_1",
            "text": "red cup and blue book",
            "constraint_type": "attribute",
            "metadata": {
                "constraint_schema": "attribute_bindings.v1",
                "required_objects": [{"class": "cup", "count": 1}, {"class": "book", "count": 1}],
                "attribute_bindings": [
                    {"object": "cup", "attribute": "red"},
                    {"object": "book", "attribute": "blue"},
                ],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({"type": "detector_object_requirements", "passthrough_generation_fields": True})

    result = inspector.inspect(task, {}, {
        "image_path": str(image_path),
        "object_counts": {"cup": 1, "book": 1},
        "detected_objects": [
            {"class": "cup", "score": 0.9, "bbox": [10, 10, 50, 50]},
            {"class": "book", "score": 0.9, "bbox": [70, 10, 110, 50]},
        ],
    })

    row = result["attr_1"]
    assert row["passed"] is False
    assert row["symptom"] == "attribute_binding_mismatch"
    assert {m["object"] for m in row["attribute_binding_mismatches"]} == {"cup", "book"}


def test_detector_attribute_color_penalizes_duplicate_exact_bound_object(tmp_path):
    from PIL import Image, ImageDraw

    from gen_harness.inspection import DetectorObjectRequirementInspection

    image_path = tmp_path / "duplicate_red_cup_blue_book.png"
    image = Image.new("RGB", (160, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 50, 50], fill=(230, 20, 20))
    draw.rectangle([55, 10, 75, 50], fill=(230, 20, 20))
    draw.rectangle([100, 10, 140, 50], fill=(20, 40, 230))
    image.save(image_path)

    task = GenerationTask.from_dict({
        "task_id": "attribute_color_duplicate",
        "prompt": "a red cup and a blue book",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "attr_1",
            "text": "red cup and blue book",
            "constraint_type": "attribute",
            "metadata": {
                "constraint_schema": "attribute_bindings.v1",
                "required_objects": [{"class": "cup", "count": 1}, {"class": "book", "count": 1}],
                "attribute_bindings": [
                    {"object": "cup", "attribute": "red"},
                    {"object": "book", "attribute": "blue"},
                ],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({"type": "detector_object_requirements", "passthrough_generation_fields": True})

    result = inspector.inspect(task, {}, {
        "image_path": str(image_path),
        "object_counts": {"cup": 2, "book": 1},
        "detected_objects": [
            {"class": "cup", "score": 0.9, "bbox": [10, 10, 50, 50]},
            {"class": "cup", "score": 0.8, "bbox": [55, 10, 75, 50]},
            {"class": "book", "score": 0.9, "bbox": [100, 10, 140, 50]},
        ],
    })

    row = result["attr_1"]
    assert row["passed"] is True
    assert row["attribute_color_scores"][0]["ambiguous_duplicate_count"] == 2
    assert row["attribute_color_scores"][0]["selection_score"] < 1.0


def test_detector_attribute_color_can_use_top_confidence_crop_for_stricter_selection(tmp_path):
    from PIL import Image, ImageDraw

    from gen_harness.inspection import DetectorObjectRequirementInspection

    image_path = tmp_path / "two_cups_top_wrong.png"
    image = Image.new("RGB", (160, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 50, 50], fill=(20, 40, 230))
    draw.rectangle([80, 10, 120, 50], fill=(230, 20, 20))
    image.save(image_path)

    task = GenerationTask.from_dict({
        "task_id": "attribute_top_confidence",
        "prompt": "a red cup",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "attr_1",
            "text": "red cup",
            "constraint_type": "attribute",
            "metadata": {
                "constraint_schema": "attribute_bindings.v1",
                "required_objects": [{"class": "cup", "count": 1}],
                "attribute_bindings": [{"object": "cup", "attribute": "red"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
        "attribute_match_mode": "top_confidence",
    })

    result = inspector.inspect(task, {}, {
        "image_path": str(image_path),
        "object_counts": {"cup": 2},
        "detected_objects": [
            {"class": "cup", "score": 0.99, "bbox": [10, 10, 50, 50]},
            {"class": "cup", "score": 0.80, "bbox": [80, 10, 120, 50]},
        ],
    })

    row = result["attr_1"]
    assert row["passed"] is False
    assert row["attribute_color_scores"][0]["bbox"] == [10, 10, 50, 50]


def test_detector_object_requirement_skips_external_command_when_no_detector_constraints():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "quality_only_detector_skip",
        "prompt": "a cinematic landscape",
        "task_family": "quality",
        "constraints": [
            {
                "constraint_id": "quality_1",
                "text": "coherent and sharp",
                "constraint_type": "quality",
                "metadata": {"constraint_schema": "quality.v1"},
            }
        ],
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "command": ["python", "-c", "raise SystemExit(17)"],
    })

    result = inspector.inspect(task, {}, {"image_uri": "/tmp/x.png"}, seed=0)

    assert result["quality_1"]["passed"] is True
    assert result["quality_1"]["not_applicable"] is True


def test_detector_marks_unsupported_vocab_constraint_not_applicable():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "open_vocab_count",
        "prompt": "seven green croissants",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "count_1",
            "text": "seven croissants",
            "constraint_type": "count",
            "metadata": {
                "constraint_schema": "object_requirements.v1",
                "count_mode": "exact",
                "required_objects": [{"class": "croissant", "count": 7}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
        "skip_unsupported_classes": True,
    })

    result = inspector.inspect(task, {}, {
        "object_counts": {"croissant": 0},
        "unsupported_classes": ["croissant"],
    })

    assert result["count_1"]["passed"] is True
    assert result["count_1"]["not_applicable"] is True
    assert result["count_1"]["unsupported_classes"] == ["croissant"]


def test_detector_keeps_supported_class_evidence_in_mixed_open_vocab_constraint():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "mixed_vocab_count",
        "prompt": "one dog and six bagels",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "count_1",
            "text": "one dog and six bagels",
            "constraint_type": "count",
            "metadata": {
                "constraint_schema": "object_requirements.v1",
                "count_mode": "exact",
                "required_objects": [{"class": "dog", "count": 1}, {"class": "bagel", "count": 6}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
        "skip_unsupported_classes": True,
    })

    result = inspector.inspect(task, {}, {
        "object_counts": {"dog": 1, "bagel": 0},
        "unsupported_classes": ["bagel"],
    })

    row = result["count_1"]
    assert row["passed"] is True
    assert row["partially_applicable"] is True
    assert row.get("not_applicable") is not True
    assert row["object_counts"] == {"dog": 1}
    assert row["supported_classes"] == ["dog"]
    assert row["unsupported_classes"] == ["bagel"]


def test_inspection_canonicalizes_common_irregular_object_aliases():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "sheeps_alias",
        "prompt": "four sheeps",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "count_1",
            "text": "four sheep",
            "constraint_type": "count",
            "metadata": {
                "constraint_schema": "object_requirements.v1",
                "count_mode": "exact",
                "required_objects": [{"class": "sheeps", "count": 4}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
    })

    result = inspector.inspect(task, {}, {"object_counts": {"sheep": 4}})

    assert result["count_1"]["passed"] is True
    assert result["count_1"]["object_counts"] == {"sheep": 4}


def test_detector_spatial_inspection_maps_mug_to_coco_cup_alias():
    from gen_harness.inspection import DetectorObjectRequirementInspection

    task = GenerationTask.from_dict({
        "task_id": "mug_alias_spatial",
        "prompt": "a pizza right of a mug",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "spatial_1",
            "text": "pizza right of mug",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "required_objects": [{"class": "pizza", "count": 1}, {"class": "mug", "count": 1}],
                "spatial_relations": [{"subject": "pizza", "relation": "right_of", "object": "mug"}],
            },
        }],
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "passthrough_generation_fields": True,
        "spatial_geometry_mode": "geneval_size_aware_v1",
        "hard_evidence_required": True,
    })

    result = inspector.inspect(task, {}, {
        "object_counts": {"pizza": 1, "cup": 1},
        "detected_objects": [
            {"class": "cup", "score": 0.91, "bbox": [80, 30, 180, 160]},
            {"class": "pizza", "score": 0.93, "bbox": [300, 40, 460, 170]},
        ],
    })

    row = result["spatial_1"]
    assert row["passed"] is True
    assert row["evidence_status"] == "confirmed"
    assert row["spatial_relations_passed"][0]["object"] == "mug"


def test_detector_object_requirement_uses_persistent_command(tmp_path):
    from gen_harness.inspection import DetectorObjectRequirementInspection

    worker = tmp_path / "detector_worker.py"
    worker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    payload=json.loads(line)\n"
        "    assert payload['generation']['image_path'].endswith('.png')\n"
        "    print(json.dumps({'object_counts': {'cup': 1}, 'detected_objects': [{'class': 'cup', 'count': 1, 'score': 0.9, 'bbox': [0,0,10,10]}]}), flush=True)\n",
        encoding="utf-8",
    )
    task = GenerationTask.from_dict({
        "task_id": "persistent_detector",
        "prompt": "a photo of one cup",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "count_1",
                "text": "one cup",
                "constraint_type": "count",
                "metadata": {
                    "constraint_schema": "object_requirements.v1",
                    "required_objects": [{"class": "cup", "count": 1}],
                },
            }
        ],
    })
    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "persistent_command": ["{python}", str(worker)],
    })

    first = inspector.inspect(task, {}, {"image_uri": "/tmp/x.png"}, seed=0)
    second = inspector.inspect(task, {}, {"image_uri": "/tmp/y.png"}, seed=1)

    assert first["count_1"]["passed"] is True
    assert second["count_1"]["object_counts"] == {"cup": 1}


def test_composite_inspection_preserves_prior_failure_symptom():
    from gen_harness.inspection import CompositeInspectionRunner

    class FailingInspector(InspectionRunner):
        config = {"name": "spatial_bbox"}

        def inspect(self, task, program, generation, seed=0):
            return {"spatial_1": {"passed": False, "symptom": "spatial_relation_mismatch", "confidence": 1.0}}

    class PassingInspector(InspectionRunner):
        config = {"name": "clip_alignment"}

        def inspect(self, task, program, generation, seed=0):
            return {"spatial_1": {"passed": True, "symptom": "pass", "confidence": 1.0}}

    task = GenerationTask.from_dict({"task_id": "merge_symptom", "prompt": "a dog right of a teddy bear", "task_family": "compositional", "constraints": []})
    result = CompositeInspectionRunner([FailingInspector(), PassingInspector()]).inspect(task, {}, {"image_uri": "/tmp/x.png"})

    assert result["spatial_1"]["passed"] is False
    assert result["spatial_1"]["symptom"] == "spatial_relation_mismatch"


def test_composite_inspection_preserves_detector_constraint_score():
    from gen_harness.inspection import CompositeInspectionRunner

    class SpatialScoreInspector(InspectionRunner):
        config = {"name": "spatial_bbox"}

        def inspect(self, task, program, generation, seed=0):
            return {"spatial_1": {"passed": False, "symptom": "spatial_relation_mismatch", "confidence": 0.25, "constraint_score": 0.25}}

    class NotApplicableInspector(InspectionRunner):
        config = {"name": "alignment"}

        def inspect(self, task, program, generation, seed=0):
            return {"spatial_1": {"passed": True, "symptom": "pass", "confidence": 1.0, "not_applicable": True}}

    task = GenerationTask.from_dict({
        "task_id": "composite_score",
        "prompt": "dog right of bear",
        "task_family": "compositional",
        "constraints": [{"constraint_id": "spatial_1", "text": "dog right of bear", "constraint_type": "spatial"}],
    })
    result = CompositeInspectionRunner([SpatialScoreInspector(), NotApplicableInspector()]).inspect(task, {}, {}, seed=0)

    assert result["spatial_1"]["passed"] is False
    assert result["spatial_1"]["constraint_score"] == 0.25
    assert result["spatial_1"]["confidence"] == 0.25


def test_composite_inspection_selection_score_uses_all_applicable_inspectors():
    from gen_harness.inspection import CompositeInspectionRunner

    class ObjectInspector(InspectionRunner):
        config = {"name": "object_detector"}

        def inspect(self, task, program, generation, seed=0):
            return {
                "attr_1": {
                    "passed": True,
                    "symptom": "pass",
                    "confidence": 1.0,
                    "constraint_score": 1.0,
                }
            }

    class AttributeInspector(InspectionRunner):
        config = {"name": "attribute_alignment"}

        def inspect(self, task, program, generation, seed=0):
            return {
                "attr_1": {
                    "passed": False,
                    "symptom": "attribute_binding_mismatch",
                    "confidence": 0.2,
                    "alignment_score": 0.2,
                }
            }

    task = GenerationTask.from_dict({
        "task_id": "composite_selection_score",
        "prompt": "a red cup and a blue book",
        "task_family": "compositional",
        "constraints": [{"constraint_id": "attr_1", "text": "red cup and blue book", "constraint_type": "attribute"}],
    })
    result = CompositeInspectionRunner([ObjectInspector(), AttributeInspector()]).inspect(task, {}, {}, seed=0)

    assert result["attr_1"]["passed"] is False
    assert result["attr_1"]["symptom"] == "attribute_binding_mismatch"
    assert result["attr_1"]["selection_score"] == 0.2
    assert result["attr_1"]["constraint_score"] == 1.0
    assert result["attr_1"]["alignment_score"] == 0.2


def test_composite_inspection_structured_attribute_result_overrides_lightweight_veto():
    from gen_harness.inspection import CompositeInspectionRunner

    class StructuredAttributeInspector(InspectionRunner):
        config = {"name": "groundingdino_object_requirements"}

        def inspect(self, task, program, generation, seed=0):
            return {
                "attr_1": {
                    "passed": True,
                    "symptom": "pass",
                    "confidence": 0.91,
                    "constraint_score": 0.91,
                    "selection_score": 0.91,
                    "attribute_color_scores": [
                        {"object": "cup", "attribute": "red", "selection_score": 0.91},
                    ],
                    "inspection_source": "groundingdino_object_requirements",
                }
            }

    class LightweightAttributeInspector(InspectionRunner):
        config = {"name": "openclip_lightweight_alignment"}

        def inspect(self, task, program, generation, seed=0):
            return {
                "attr_1": {
                    "passed": False,
                    "symptom": "attribute_binding_mismatch",
                    "confidence": 0.15,
                    "alignment_score": 0.15,
                    "inspection_source": "lightweight_alignment",
                }
            }

    task = GenerationTask.from_dict({
        "task_id": "structured_attr_merge",
        "prompt": "a red cup",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "attr_1",
                "text": "red cup",
                "constraint_type": "attribute",
                "metadata": {
                    "constraint_schema": "attribute_bindings.v1",
                    "attribute_bindings": [{"object": "cup", "attribute": "red"}],
                },
            }
        ],
    })
    result = CompositeInspectionRunner([StructuredAttributeInspector(), LightweightAttributeInspector()]).inspect(task, {}, {}, seed=0)

    row = result["attr_1"]
    assert row["passed"] is True
    assert row["symptom"] == "pass"
    assert row["selection_score"] == 0.91
    assert row["ignored_for_merge_sources"] == ["openclip_lightweight_alignment"]
    assert row["inspection_subresults"]["openclip_lightweight_alignment"]["passed"] is False


def test_composite_inspection_structured_spatial_result_overrides_lightweight_veto():
    from gen_harness.inspection import CompositeInspectionRunner

    class StructuredSpatialInspector(InspectionRunner):
        config = {"name": "groundingdino_object_requirements"}

        def inspect(self, task, program, generation, seed=0):
            return {
                "spatial_1": {
                    "passed": True,
                    "symptom": "pass",
                    "confidence": 0.88,
                    "constraint_score": 0.88,
                    "relation_scores": [0.88],
                    "spatial_relations_passed": [{"subject": "dog", "relation": "left of", "object": "bear"}],
                    "inspection_source": "groundingdino_object_requirements",
                }
            }

    class LightweightSpatialInspector(InspectionRunner):
        config = {"name": "openclip_lightweight_alignment"}

        def inspect(self, task, program, generation, seed=0):
            return {
                "spatial_1": {
                    "passed": False,
                    "symptom": "spatial_relation_mismatch",
                    "confidence": 0.10,
                    "alignment_score": 0.10,
                    "inspection_source": "lightweight_alignment",
                }
            }

    task = GenerationTask.from_dict({
        "task_id": "structured_spatial_merge",
        "prompt": "a dog left of a bear",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "spatial_1",
                "text": "dog left of bear",
                "constraint_type": "spatial",
                "metadata": {"constraint_schema": "spatial_relations.v1"},
            }
        ],
    })
    result = CompositeInspectionRunner([StructuredSpatialInspector(), LightweightSpatialInspector()]).inspect(task, {}, {}, seed=0)

    row = result["spatial_1"]
    assert row["passed"] is True
    assert row["symptom"] == "pass"
    assert row["selection_score"] == 0.88
    assert row["ignored_for_merge_sources"] == ["openclip_lightweight_alignment"]
    assert row["inspection_subresults"]["openclip_lightweight_alignment"]["passed"] is False


def test_compositional_prompt_preserves_full_rules_and_memory(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = GenerationTask.from_dict({
        "task_id": "prompt_preserve_full_context",
        "prompt": "a photo of a bench",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "objects_1",
            "text": "required visible objects: one bench; non-object clause that must survive intact at the end SENTINEL_CONSTRAINT_TEXT",
            "constraint_type": "count",
            "metadata": {
                "constraint_schema": "object_requirements.v1",
                "required_objects": [{"class": "bench", "count": 1}],
            },
        }, {
            "constraint_id": "layout_1",
            "text": "place the bench in the left foreground with a clear empty path on the right SENTINEL_LAYOUT_TEXT",
            "constraint_type": "spatial",
        }],
    })
    skills = repo.build_skills()
    skill = skills.artifact["skills"]["compositional_generation"]
    long_rule = "keep every semantic clause in this validated rule including SENTINEL_LONG_RULE_AT_END"
    skill["prompt_rules"] = [long_rule]
    contract = repo.build_policy().compile_contract(task)
    memory_context = {
        "prompt_guidelines": [{"guideline": "preserve this complete memory guideline including SENTINEL_MEMORY_AT_END"}],
        "validated_patches": [{
            "target_component": "skills",
            "changed_artifact": "skills.json",
            "payload": {
                "path": ["skills", "compositional_generation", "prompt_families", "clutter_and_occlusion_control", "prompt_rules"],
                "value": "reuse this validated patch rule including SENTINEL_PATCH_AT_END",
            },
        }],
    }

    workflow = skills.compile_workflow(task, "compositional_generation", contract, {}, memory_context=memory_context)
    prompt = workflow["compiled_prompt"]

    assert "SENTINEL_LAYOUT_TEXT" in prompt
    assert "SENTINEL_LONG_RULE_AT_END" in prompt
    assert "SENTINEL_MEMORY_AT_END" in prompt
    assert "SENTINEL_PATCH_AT_END" in prompt


def test_flow_grpo_uses_full_prompt_for_all_sd3_prompt_channels():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    task = GenerationTask.from_dict({
        "task_id": "flow_prompt_channels",
        "prompt": "a photo of a bench and a cow",
        "task_family": "compositional",
        "constraints": [],
    })
    sentinel = "SENTINEL_FULL_STRUCTURED_PROMPT_AT_END"
    full_prompt = "Create one image from this structured generation brief. " + " ".join(["context"] * 120) + " " + sentinel
    program = {
        "method": "gen_harness",
        "workflow": {"compiled_prompt": full_prompt},
        "contract": {"constraints": [{
            "constraint_type": "count",
            "metadata": {"required_objects": [{"class": "bench", "count": 1}, {"class": "cow", "count": 1}]},
        }]},
    }

    prompts = FlowGRPOBackend({})._sd3_prompt_kwargs(task, program, full_prompt)

    assert sentinel in prompts["prompt_3"]
    assert sentinel in prompts["prompt"]
    assert prompts["prompt"] == prompts["prompt_2"]
    assert prompts["prompt"] == prompts["prompt_3"]
    assert prompts["prompt"] == full_prompt






def test_object_inspection_treats_forbidden_count_as_overcount_threshold():
    task = GenerationTask.from_dict({
        "task_id": "exact_count_task",
        "prompt": "a photo of two clocks",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "count_0",
            "text": "required visible objects: 2 clocks; must not contain: 3 clocks",
            "constraint_type": "count",
            "metadata": {
                "constraint_schema": "object_requirements.v1",
                "required_objects": [{"class": "clock", "count": 2}],
                "forbidden_objects": [{"class": "clock", "count": 3}],
            },
        }],
    })

    inspector = ObjectRequirementInspection()
    exact = inspector.inspect(task, {}, {"object_counts": {"clock": 2}})["count_0"]
    over = inspector.inspect(task, {}, {"object_counts": {"clock": 3}})["count_0"]

    assert exact["passed"] is True
    assert exact["forbidden_objects_present"] == []
    assert over["passed"] is False
    assert over["symptom"] == "object_count_mismatch"
    assert over["count_mismatches"] == [{"class": "clock", "expected": 2, "found": 3, "direction": "over"}]
    assert over["forbidden_objects_present"] == []


def test_object_inspection_exact_count_mode_flags_required_over_count():
    task = GenerationTask.from_dict({
        "task_id": "exact_count_mode_task",
        "prompt": "a photo of one cup and one mug",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "objects_0",
            "text": "required visible objects: one cup, one mug",
            "constraint_type": "spatial",
            "metadata": {
                "constraint_schema": "spatial_relations.v1",
                "count_mode": "exact",
                "required_objects": [{"class": "cup", "count": 1}, {"class": "mug", "count": 1}],
            },
        }],
    })

    inspector = ObjectRequirementInspection()
    exact = inspector.inspect(task, {}, {"object_counts": {"cup": 1, "mug": 1}})["objects_0"]
    over = inspector.inspect(task, {}, {"object_counts": {"cup": 2, "mug": 1}})["objects_0"]

    assert exact["passed"] is True
    assert over["passed"] is False
    assert over["symptom"] == "object_count_mismatch"
    assert over["count_mismatches"] == [{"class": "cup", "expected": 1, "found": 2, "direction": "over"}]


def test_object_inspection_requires_objects_implied_by_attribute_bindings():
    task = GenerationTask.from_dict({
        "task_id": "attr_implied_objects",
        "prompt": "a red cup and a blue book",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "attr_1",
            "text": "red cup and blue book",
            "constraint_type": "attribute",
            "metadata": {
                "constraint_schema": "attribute_bindings.v1",
                "attribute_bindings": [
                    {"object": "cup", "attribute": "red"},
                    {"object": "book", "attribute": "blue"},
                ],
            },
        }],
    })

    result = ObjectRequirementInspection().inspect(task, {}, {"object_counts": {"cup": 1}})

    assert result["attr_1"]["passed"] is False
    assert result["attr_1"]["missing_required_objects"] == [{"class": "book", "expected": 1, "found": 0}]





def test_llm_policy_extractor_compiles_prompt_constraints(tmp_path, monkeypatch):
    from gen_harness.components.policy import LLMPolicyExtractor

    harness = make_harness(tmp_path)
    config_path = tmp_path / "policy_extractor.json"
    write_json(config_path, {
        "type": "openai_chat_policy_extractor",
        "base_url": "https://llm.invalid/v1",
        "model": "test-policy-llm",
    })

    def fake_request(self, task, policy_artifact):
        return {
            "constraints": [
                {
                    "constraint_id": "policy_llm_count_0",
                    "text": "the image must contain one cup",
                    "constraint_type": "count",
                    "target": "object_requirements",
                    "metadata": {
                        "constraint_schema": "object_requirements.v1",
                        "required_objects": [{"class": "cup", "count": 1}],
                        "forbidden_objects": [],
                    },
                }
            ]
        }

    monkeypatch.setattr(LLMPolicyExtractor, "_request_constraints", fake_request)
    repo = HarnessRepository(harness, policy_extractor_config=config_path)
    task = GenerationTask.from_dict({
        "task_id": "llm_policy_task",
        "prompt": "a clean image of one cup",
        "task_family": "compositional",
        "constraints": [],
        "metadata": {"split": "target"},
    })

    contract = repo.build_policy().compile_contract(task)

    assert contract["policy_extractor"]["component_name"] == "openai_chat_policy_extractor"
    assert contract["policy_extractor"]["model"] == "test-policy-llm"
    assert contract["policy_extractor"]["official_feedback_used"] is False
    assert contract["constraint_source"] == "policy_extractor"
    assert contract["constraints"][0]["metadata"]["source"] == "policy_extractor.openai_chat"
    assert "official_feedback_used" not in contract["constraints"][0]["metadata"]
    assert contract["constraints"][0]["metadata"]["required_objects"] == [{"class": "cup", "count": 1}]


def test_llm_policy_extractor_repairs_noncanonical_relation_entities(tmp_path, monkeypatch):
    from gen_harness.components.policy import LLMPolicyExtractor

    config = {
        "type": "openai_chat_policy_extractor",
        "base_url": "https://llm.invalid/v1",
        "model": "test-policy-llm",
        "max_schema_retries": 1,
    }
    extractor = LLMPolicyExtractor(config)
    task = GenerationTask.from_dict({
        "task_id": "llm_policy_entity_repair",
        "prompt": "three birds on top of a rabbit",
        "task_family": "compositional",
        "constraints": [],
    })
    calls = []

    def fake_request(self, task, policy_artifact, correction=None):
        calls.append(correction)
        if correction is None:
            return {
                "constraints": [
                    {
                        "constraint_id": "count",
                        "text": "three birds and one rabbit",
                        "constraint_type": "count",
                        "metadata": {
                            "required_objects": [
                                {"class": "bird", "count": 3},
                                {"class": "rabbit", "count": 1},
                            ]
                        },
                    },
                    {
                        "constraint_id": "spatial",
                        "text": "birds above rabbit",
                        "constraint_type": "spatial",
                        "metadata": {
                            "required_objects": [
                                {"class": "bird", "count": 3},
                                {"class": "rabbit", "count": 1},
                            ],
                            "spatial_relations": [
                                {"subject": "birds", "relation": "above", "object": "rabbit"}
                            ],
                        },
                    },
                ]
            }
        return {
            "constraints": [
                {
                    "constraint_id": "count",
                    "text": "three birds and one rabbit",
                    "constraint_type": "count",
                    "metadata": {
                        "required_objects": [
                            {"class": "bird", "count": 3},
                            {"class": "rabbit", "count": 1},
                        ]
                    },
                },
                {
                    "constraint_id": "spatial",
                    "text": "birds above rabbit",
                    "constraint_type": "spatial",
                    "metadata": {
                        "required_objects": [
                            {"class": "bird", "count": 3},
                            {"class": "rabbit", "count": 1},
                        ],
                        "spatial_relations": [
                            {"subject": "bird", "relation": "above", "object": "rabbit"}
                        ],
                    },
                },
            ]
        }

    monkeypatch.setattr(LLMPolicyExtractor, "_request_constraints", fake_request)
    constraints = extractor.extract(task, {"constraint_ontology": ["count", "spatial"]})

    assert len(calls) == 2
    assert calls[0] is None
    assert "exact canonical entity identifier" in calls[1]
    assert constraints[1].metadata["spatial_relations"] == [
        {"subject": "bird", "relation": "above", "object": "rabbit"}
    ]


def test_geneval2_export_prefers_exact_prompt_for_reindexed_subsets(tmp_path):
    from gen_harness.evaluation.geneval2 import export_geneval2_image_map

    benchmark = tmp_path / "geneval2_data.jsonl"
    benchmark.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "first prompt"}),
                json.dumps({"prompt": "selected prompt"}),
            ]
        )
        + "\n"
    )
    tasks = tmp_path / "selected_tasks.jsonl"
    tasks.write_text(json.dumps({"prompt": "selected prompt"}) + "\n")
    image = tmp_path / "selected.png"
    image.write_bytes(b"png")
    experiences = tmp_path / "experiences.jsonl"
    experiences.write_text(
        json.dumps(
            {
                "task_id": "reindexed_0000",
                "seed": 0,
                "image_uri": str(image),
                "export_metadata": {
                    "benchmark": "geneval2",
                    "geneval2_index": 0,
                    "geneval2_prompt": "selected prompt",
                },
            }
        )
        + "\n"
    )

    output = tmp_path / "image_map.json"
    result = export_geneval2_image_map(
        experiences,
        benchmark,
        output,
        seed=0,
        prompts_file=tasks,
    )

    assert result == {"selected prompt": str(image.resolve())}


def test_formal_llm_policy_extractor_ignores_task_embedded_constraints(tmp_path, monkeypatch):
    from gen_harness.components.policy import LLMPolicyExtractor

    harness = make_harness(tmp_path)
    config_path = tmp_path / "policy_extractor.json"
    write_json(config_path, {
        "type": "openai_chat_policy_extractor",
        "base_url": "https://llm.invalid/v1",
        "model": "test-policy-llm",
    })

    def fake_request(self, task, policy_artifact):
        return {
            "constraints": [
                {
                    "constraint_id": "policy_llm_count_0",
                    "text": "the image must contain one cup",
                    "constraint_type": "count",
                    "target": "object_requirements",
                    "metadata": {
                        "constraint_schema": "object_requirements.v1",
                        "required_objects": [{"class": "cup", "count": 1}],
                        "forbidden_objects": [],
                    },
                }
            ]
        }

    monkeypatch.setattr(LLMPolicyExtractor, "_request_constraints", fake_request)
    repo = HarnessRepository(harness, policy_extractor_config=config_path)
    task = GenerationTask.from_dict({
        "task_id": "llm_policy_ignores_embedded_constraints",
        "prompt": "a clean image of one cup",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "embedded_count",
                "text": "hidden embedded constraint that formal policy must not trust",
                "constraint_type": "count",
                "metadata": {
                    "constraint_schema": "object_requirements.v1",
                    "required_objects": [{"class": "plate", "count": 9}],
                    "source": "dataset_embedded_constraint",
                },
            }
        ],
        "metadata": {"split": "target", "benchmark": "geneval2", "score": 1.0},
    })

    contract = repo.build_policy().compile_contract(task)

    assert contract["constraint_source"] == "policy_extractor"
    assert [row["constraint_id"] for row in contract["constraints"]] == ["policy_llm_count_0"]
    assert contract["constraints"][0]["metadata"]["required_objects"] == [{"class": "cup", "count": 1}]
    assert "benchmark" not in contract["task_metadata"]
    assert "score" not in contract["task_metadata"]




def test_formal_llm_policy_extractor_is_not_overridden_by_t2icompbenchpp_metadata(tmp_path, monkeypatch):
    from gen_harness.components.policy import LLMPolicyExtractor

    harness = make_harness(tmp_path)
    config_path = tmp_path / "policy_extractor.json"
    write_json(config_path, {
        "type": "openai_chat_policy_extractor",
        "base_url": "https://llm.invalid/v1",
        "model": "formal-policy-model",
    })

    def fake_request(self, task, policy_artifact):
        return {
            "constraints": [
                {
                    "constraint_id": "policy_llm_count_0",
                    "text": "the image must contain one backpack and one pig",
                    "constraint_type": "count",
                    "target": "object_requirements",
                    "metadata": {
                        "constraint_schema": "object_requirements.v1",
                        "required_objects": [{"class": "backpack", "count": 1}, {"class": "pig", "count": 1}],
                        "forbidden_objects": [],
                    },
                }
            ]
        }

    monkeypatch.setattr(LLMPolicyExtractor, "_request_constraints", fake_request)
    repo = HarnessRepository(harness, policy_extractor_config=config_path)
    task = GenerationTask.from_dict({
        "task_id": "formal_t2i_policy_task",
        "prompt": "a green backpack and a pig",
        "task_family": "compositional",
        "constraints": [],
        "metadata": {"benchmark": "t2icompbenchpp", "category": "color"},
    })

    contract = repo.build_policy().compile_contract(task)

    assert contract["policy_extractor"]["component_name"] == "openai_chat_policy_extractor"
    assert contract["policy_extractor"]["model"] == "formal-policy-model"
    assert contract["constraints"][0]["metadata"]["source"] == "policy_extractor.openai_chat"




def test_llm_policy_extractor_rejects_official_feedback_tokens(tmp_path, monkeypatch):
    from gen_harness.components.policy import LLMPolicyExtractor

    config = {
        "type": "openai_chat_policy_extractor",
        "base_url": "https://llm.invalid/v1",
        "model": "test-policy-llm",
    }
    extractor = LLMPolicyExtractor(config)
    task = GenerationTask.from_dict({
        "task_id": "llm_policy_forbidden",
        "prompt": "a clean image of one cup",
        "task_family": "compositional",
        "constraints": [],
        "metadata": {"split": "target"},
    })

    monkeypatch.setattr(
        extractor,
        "_request_constraints",
        lambda task, policy_artifact: {
            "constraints": [
                {
                    "constraint_id": "leaky",
                    "text": "use the official evaluator result",
                    "constraint_type": "count",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="forbidden token"):
        extractor.extract(task, {"constraint_ontology": ["count"]})


def test_llm_policy_extractor_rejects_structured_benchmark_provenance(monkeypatch):
    from gen_harness.components.policy import LLMPolicyExtractor

    extractor = LLMPolicyExtractor({
        "type": "openai_chat_policy_extractor",
        "base_url": "https://llm.invalid/v1",
        "model": "test-policy-llm",
    })
    task = GenerationTask.from_dict({
        "task_id": "llm_policy_structured_leak",
        "prompt": "a clean image of one cup",
        "task_family": "compositional",
        "constraints": [],
    })
    monkeypatch.setattr(
        extractor,
        "_request_constraints",
        lambda task, policy_artifact: {
            "constraints": [
                {
                    "constraint_id": "policy_llm_count_0",
                    "text": "one cup",
                    "constraint_type": "count",
                    "metadata": {
                        "constraint_schema": "object_requirements.v1",
                        "required_objects": [{"class": "cup", "count": 1}],
                        "source_index": 7,
                    },
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="provenance"):
        extractor.extract(task, {"constraint_ontology": ["count"]})




def test_openai_prompt_optimizer_filters_benchmark_provenance_from_request(monkeypatch):
    from gen_harness.prompt_optimizer import OpenAIChatPromptOptimizer

    captured = {}
    optimizer = OpenAIChatPromptOptimizer({
        "type": "openai_chat_prompt_optimizer",
        "base_url": "https://llm.invalid/v1",
        "model": "test-prompt-llm",
    })
    task = GenerationTask.from_dict({
        "task_id": "t2icompbenchpp_color_val_0007",
        "prompt": "a green backpack and a pig",
        "task_family": "compositional",
        "constraints": [],
        "metadata": {"benchmark": "t2icompbenchpp", "category": "color", "source_index": 7},
    })
    program = {
        "workflow": {
            "compiled_prompt": "a green backpack and a pig in a clean studio scene",
            "skill_name": "prompt_enhancement",
        },
        "contract": {
            "task_metadata": {"benchmark": "t2icompbenchpp", "category": "color", "source_index": 7},
            "constraints": [
                {
                    "constraint_id": "t2icompbenchpp_color_val_0007_attr",
                    "constraint_type": "attribute",
                    "text": "green backpack",
                    "metadata": {"category": "color", "official_score": 1.0, "attribute": "green"},
                }
            ],
        },
        "memory_context": {
            "retrieved_failures": [
                {"task_id": "t2icompbenchpp_color_val_0001", "official_score": 0.0, "lesson": "bind colors"}
            ],
            "prompt_guidelines": [{"guideline": "keep colors visibly bound"}],
        },
    }

    def fake_post(endpoint, body):
        captured["body"] = body
        return {"choices": [{"message": {"content": json.dumps({"enhanced_prompt": "a vivid green backpack beside a pig"})}}]}

    monkeypatch.setattr(optimizer, "_post_json", fake_post)
    assert optimizer.optimize(task, program, seed=12) == "a vivid green backpack beside a pig"

    request_text = captured["body"]["messages"][1]["content"]
    assert "green backpack" in request_text
    assert "bind colors" in request_text
    for forbidden in (
        "task_id",
        "t2icompbenchpp_color_val_0007",
        "t2icompbenchpp",
        "category",
        "source_index",
        "official_score",
        "seed",
    ):
        assert forbidden not in request_text


def test_openai_vision_inspection_filters_provenance_and_maps_public_constraint_ids(monkeypatch):
    from gen_harness.adapters.openai_vision_inspection import OpenAIVisionInspectionRunner

    captured = {}
    inspector = OpenAIVisionInspectionRunner({
        "type": "openai_vision_judge",
        "base_url": "https://llm.invalid/v1",
        "model": "test-vision-llm",
    })
    task = GenerationTask.from_dict({
        "task_id": "t2icompbenchpp_color_val_0007",
        "prompt": "a green backpack and a pig",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "t2icompbenchpp_color_val_0007_attr",
                "constraint_type": "attribute",
                "text": "the backpack must be green",
                "metadata": {
                    "category": "color",
                    "source_index": 7,
                    "official_score": 1.0,
                    "attribute_bindings": [{"object": "backpack", "attribute": "green"}],
                },
            }
        ],
        "metadata": {"benchmark": "t2icompbenchpp", "category": "color"},
    })
    program = {"workflow": {"compiled_prompt": "a green backpack and a pig in a clean studio scene"}}

    def fake_post(endpoint, body):
        captured["body"] = body
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "image_metrics": {"overall_quality": 0.9, "prompt_alignment": 0.8},
                            "constraint_results": {
                                "constraint_0": {"passed": True, "symptom": "pass", "confidence": 0.75}
                            },
                        })
                    }
                }
            ]
        }

    monkeypatch.setattr(inspector, "_post_json", fake_post)
    result = inspector.inspect(task, program, {"image_uri": "data:image/png;base64,AAAA"}, seed=3)

    assert result["t2icompbenchpp_color_val_0007_attr"]["passed"] is True
    assert result["t2icompbenchpp_color_val_0007_attr"]["confidence"] == 0.75
    request_text = captured["body"]["messages"][1]["content"][0]["text"]
    assert "constraint_0" in request_text
    assert "the backpack must be green" in request_text
    assert "attribute_bindings" in request_text
    for forbidden in (
        "task_id",
        "t2icompbenchpp_color_val_0007",
        "t2icompbenchpp",
        "category",
        "source_index",
        "official_score",
    ):
        assert forbidden not in request_text



















def test_validated_skill_patch_is_reused_from_memory_context(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    memory = repo.build_memory()
    patch = PatchManifest(
        patch_id="patch_validated_rule",
        target_component="skills",
        changed_artifact="skills.json",
        changed_fields=["skills.compositional_generation.prompt_families.clutter_and_occlusion_control.prompt_rules"],
        operation="append_unique",
        payload={"path": ["skills", "compositional_generation", "prompt_families", "clutter_and_occlusion_control", "prompt_rules"], "value": "validated object visibility rule"},
        supporting_experience=[],
        predicted_improvement="test",
        possible_regression="test",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="skills/skills.json",
        evidence_summary={"constraint_type": "count", "failure_symptom": "missing_required_object"},
    )
    memory.write_patch_result(patch, {
        "accepted": True,
        "before": {"target": 0.0, "heldout": 0.0, "preservation": 1.0},
        "after": {"target": 1.0, "heldout": 1.0, "preservation": 1.0},
        "target_delta": 1.0,
        "heldout_delta": 1.0,
        "regression_delta": 0.0,
        "utility": 1.0,
        "seeds": [0],
        "validation_scope": {"target_heldout_preservation_present": True},
        "validation_guards": {
            "target_improved": True,
            "heldout_improved": True,
            "heldout_non_regressing": True,
            "preservation_regression_bounded": True,
            "positive_utility": True,
            "matched_stochastic_conditions": True,
            "official_evaluator_used": False,
            "official_scores_used": False,
            "fallback_used": False,
            "internal_verifier_only": True,
        },
    })
    task = GenerationTask.from_dict({
        "task_id": "reuse_task",
        "prompt": "a photo of a hair drier and a cake",
        "task_family": "compositional",
        "constraints": [],
    })
    contract = repo.build_policy().compile_contract(task)
    memory_context = {
        "validated_patches": memory.retrieve_validated_patches(constraint_types=["count"]),
        "prompt_guidelines": [],
        "retrieved_failures": [],
    }

    workflow = repo.build_skills().compile_workflow(task, "compositional_generation", contract, {}, memory_context=memory_context)

    assert "validated object visibility rule" in workflow["compiled_prompt"]
    assert workflow["memory_used"] is True


def test_visual_memory_does_not_validate_patch_without_gate_certificate(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    memory = repo.build_memory()
    patch = PatchManifest(
        patch_id="patch_uncertified",
        target_component="skills",
        changed_artifact="skills.json",
        changed_fields=["skills.compositional_generation.prompt_rules"],
        operation="append_unique",
        payload={"path": ["skills", "compositional_generation", "prompt_rules"], "value": "uncertified rule"},
        supporting_experience=[],
        predicted_improvement="test",
        possible_regression="test",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="skills/skills.json",
        evidence_summary={"constraint_type": "count", "failure_symptom": "missing_required_object"},
    )

    memory.write_patch_result(patch, {"accepted": True})

    assert memory.retrieve_validated_patches(constraint_types=["count"]) == []
    regression_cases = memory.retrieve_regression_cases(constraint_types=["count"])
    assert len(regression_cases) == 1
    assert regression_cases[0]["patch_id"] == "patch_uncertified"


def test_patch_proposer_blocks_known_regression_case(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    memory = repo.build_memory()
    task = make_task()
    exp = VisualExperience(
        experience_id="exp_missing_object",
        task_id=task.task_id,
        constraint_id="count_1",
        constraint_type="count",
        constraint_text="required object",
        component_decisions={"skills": {"skill_name": "compositional_generation"}},
        tool_calls=[],
        visual_result={"image_uri": "/tmp/x.png", "seed": 0},
        verification_result={"passed": False, "symptom": "missing_required_object", "confidence": 1.0},
        failure_symptom="missing_required_object",
        suspected_component="skills",
        metadata={"task_family": "compositional"},
    )
    weakness = WeaknessMiner(min_support=1).mine([exp])[0]
    blocked_patch = PatchManifest(
        patch_id="old_regressive_patch",
        target_component="skills",
        changed_artifact="skills.json",
        changed_fields=["skills.compositional_generation.prompt_rules"],
        operation="append_unique",
        payload={"path": ["skills", "compositional_generation", "prompt_rules"], "value": "old bad rule"},
        supporting_experience=weakness.support_experience_ids,
        predicted_improvement="old",
        possible_regression="known",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="skills/skills.json",
        evidence_summary={"constraint_type": "count", "failure_symptom": "missing_required_object"},
        regression_risk={},
    )
    memory.write_patch_result(blocked_patch, {"accepted": False, "regression_delta": 1.0})

    proposer = PatchProposer({
        "type": "openai_chat_patch_proposer",
        "base_url": "http://example.invalid",
        "model": "test-llm",
    }, memory=memory)
    proposer._request_patch_rows = lambda weakness, **kwargs: [{
        "target_component": "skills",
        "changed_artifact": "skills.json",
        "changed_fields": ["skills.compositional_generation.prompt_rules"],
        "operation": "append_unique",
        "payload": {"path": ["skills", "compositional_generation", "prompt_rules"], "value": "old bad rule"},
        "predicted_improvement": "same",
        "possible_regression": "same",
    }]

    assert proposer.propose(weakness) is None


def test_trace_llm_requests_sanitize_task_ids_and_benchmark_provenance(monkeypatch, tmp_path):
    import gen_harness.patcher as patcher_module
    from gen_harness.patcher import HarnessLocalizer

    weakness = Weakness(
        weakness_id="weak_skills_count_missing_2",
        suspected_component="skills",
        constraint_type="count",
        failure_symptom="missing_required_object",
        support_count=2,
        support_experience_ids=["exp_t2icompbenchpp_color_val_0001", "exp_t2icompbenchpp_color_val_0002"],
        hypothesis="Repeated missing object failures suggest skill prompt coverage is incomplete.",
        evidence={
            "task_ids": ["t2icompbenchpp_color_val_0001"],
            "examples": [
                {
                    "experience_id": "exp_t2icompbenchpp_color_val_0001",
                    "task_id": "t2icompbenchpp_color_val_0001",
                    "constraint_text": "one green backpack",
                    "metadata": {
                        "benchmark": "t2icompbenchpp",
                        "category": "color",
                        "source_index": 7,
                        "official_score": 0.0,
                    },
                    "verification_result": {
                        "passed": False,
                        "symptom": "missing_required_object",
                        "score": 0.0,
                    },
                }
            ],
        },
    )
    sanitized = sanitize_trace_weakness_for_llm(weakness)
    serialized_sanitized = json.dumps(sanitized, ensure_ascii=False)
    assert "constraint_text" in serialized_sanitized
    for forbidden in (
        "weak_skills_count_missing_2",
        "exp_t2icompbenchpp",
        "t2icompbenchpp",
        "category",
        "source_index",
        "official_score",
        "score",
        "task_id",
    ):
        assert forbidden not in serialized_sanitized

    captured = []

    def fake_post(config, endpoint, body, timeout):
        captured.append(body)
        if "scores" in body["messages"][1]["content"]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "scores": {
                                    "policy": 0.0,
                                    "tools": 0.0,
                                    "skills": 1.0,
                                    "middleware": 0.0,
                                    "memory": 0.0,
                                    "no_patch": 0.0,
                                },
                                "constraint_attributions": [
                                    {
                                        "constraint_id": "constraint_0",
                                        "scores": {
                                            "policy": 0.0,
                                            "tools": 0.0,
                                            "skills": 1.0,
                                            "middleware": 0.0,
                                            "memory": 0.0,
                                            "no_patch": 0.0,
                                        },
                                        "primary_component": "skills",
                                        "confidence": 0.9,
                                        "evidence_summary": "constraint_0 lacks object coverage in reusable skill prompting.",
                                    }
                                ],
                                "rationale": "skills prompt omits object coverage",
                            })
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "patches": [
                                {
                                    "target_component": "skills",
                                    "changed_artifact": "skills.json",
                                    "changed_fields": ["skills.compositional_generation.prompt_rules"],
                                    "operation": "append_unique",
                                    "payload": {
                                        "path": ["skills", "compositional_generation", "prompt_rules"],
                                        "value": "Make each requested object independently visible.",
                                    },
                                    "predicted_improvement": "Improves object visibility.",
                                    "possible_regression": "May constrain style.",
                                }
                            ]
                        })
                    }
                }
            ]
        }

    monkeypatch.setattr(patcher_module, "_post_chat_json", fake_post)
    repo = HarnessRepository(make_harness(tmp_path))
    localizer = HarnessLocalizer({
        "type": "openai_chat_harness_localizer",
        "base_url": "http://example.invalid",
        "model": "test-localizer",
    }, repo=repo)
    proposer = PatchProposer({
        "type": "openai_chat_patch_proposer",
        "base_url": "http://example.invalid",
        "model": "test-proposer",
    }, repo=repo)

    localization = localizer.localize(weakness)
    patch = proposer.propose(weakness, responsibilities=["skills"])

    assert localization.frontier == ["skills"]
    assert patch is not None
    localization_system_prompt = captured[0]["messages"][0]["content"]
    assert "visual decision responsibilities" in localization_system_prompt
    assert "attribution component, not a repair component" in localization_system_prompt
    assert "The harness is decomposed into five independent" in localization_system_prompt
    assert "Perform attribution using the following reasoning procedure" in localization_system_prompt
    assert "NO_PATCH" in localization_system_prompt
    localization_payload = json.loads(captured[0]["messages"][1]["content"])
    assert "responsibility_rubric" not in localization_payload
    assert "scoring_guidance" not in localization_payload
    localization_state = localization_payload["current_harness_state"]
    assert localization_state["available"] is True
    assert set(localization_state["components"]) == {"policy", "tools", "skills", "middleware", "memory"}
    proposal_payload = json.loads(captured[1]["messages"][1]["content"])
    current_state = proposal_payload["current_harness_state"]
    assert current_state["available"] is True
    assert current_state["scope"] == "responsibility_conditioned"
    assert set(current_state["components"]) == {"skills"}
    assert "skills.json" in current_state["components"]["skills"]
    for body in captured:
        request_payload = json.loads(body["messages"][1]["content"])
        request_text = json.dumps(request_payload["weakness"], ensure_ascii=False)
        assert "constraint_text" in request_text
        for forbidden in (
            "weak_skills_count_missing_2",
            "exp_t2icompbenchpp",
            "t2icompbenchpp",
            "category",
            "source_index",
            "official_score",
            "score",
            "task_id",
        ):
            assert forbidden not in request_text


def test_llm_patch_proposer_validates_component_boundary_and_leakage():
    proposer = PatchProposer({
        "type": "openai_chat_patch_proposer",
        "base_url": "http://example.invalid",
        "model": "test-llm",
    })
    weakness = Weakness(
        weakness_id="weak_skills_count_missing_2",
        suspected_component="skills",
        constraint_type="count",
        failure_symptom="missing_required_object",
        support_count=2,
        support_experience_ids=["exp_1", "exp_2"],
        hypothesis="Missing object failures require better skill prompting.",
        evidence={"examples": []},
    )
    proposer._request_patch_rows = lambda weakness, **kwargs: [{
        "target_component": "skills",
        "changed_artifact": "skills.json",
        "changed_fields": ["skills.compositional_generation.prompt_rules"],
        "operation": "append_unique",
        "payload": {"path": ["skills", "compositional_generation", "prompt_rules"], "value": "Make each requested object independently visible."},
        "predicted_improvement": "Improves object visibility.",
        "possible_regression": "May constrain style.",
    }]

    patch = proposer.propose(weakness)

    assert patch is not None
    assert patch.target_component == "skills"
    assert patch.changed_artifact == "skills.json"
    assert patch.rollback_boundary == "skills/skills.json"
    assert patch.evidence_summary["proposal_source"]["type"] == "openai_chat_patch_proposer"
    assert patch.payload["proposal_rationale"]["official_feedback_used"] is False

    proposer._request_patch_rows = lambda weakness, **kwargs: [{
        "target_component": "skills",
        "changed_artifact": "visual_contract.json",
        "changed_fields": ["x"],
        "operation": "set_path",
        "payload": {"path": ["x"], "value": "y"},
    }]
    with pytest.raises(ValueError, match="not owned"):
        proposer.propose(weakness)

    proposer._request_patch_rows = lambda weakness, **kwargs: [{
        "target_component": "skills",
        "changed_artifact": "skills.json",
        "changed_fields": ["x"],
        "operation": "set_path",
        "payload": {"path": ["x"], "value": "uses official_score"},
    }]
    with pytest.raises(ValueError, match="forbidden"):
        proposer.propose(weakness)


def test_llm_patch_proposer_rejects_task_specific_identifiers():
    proposer = PatchProposer({
        "type": "openai_chat_patch_proposer",
        "base_url": "http://example.invalid",
        "model": "test-llm",
    })
    weakness = Weakness(
        weakness_id="weak_skills_count_missing_2",
        suspected_component="skills",
        constraint_type="count",
        failure_symptom="missing_required_object",
        support_count=2,
        support_experience_ids=["exp_1", "exp_2"],
        hypothesis="Missing object failures require better skill prompting.",
        evidence={"examples": []},
    )
    proposer._request_patch_rows = lambda weakness, **kwargs: [{
        "target_component": "skills",
        "changed_artifact": "skills.json",
        "changed_fields": ["skills.compositional_generation.prompt_rules"],
        "operation": "append_unique",
        "payload": {
            "path": ["skills", "compositional_generation", "prompt_rules"],
            "value": "Special case geneval_00086 with extra object visibility.",
        },
        "predicted_improvement": "Improves that task.",
        "possible_regression": "May constrain style.",
    }]

    with pytest.raises(ValueError, match="task-specific identifier"):
        proposer.propose(weakness)


def test_llm_patch_proposer_rejects_structured_provenance_keys():
    proposer = PatchProposer({
        "type": "openai_chat_patch_proposer",
        "base_url": "http://example.invalid",
        "model": "test-llm",
    })
    weakness = Weakness(
        weakness_id="weak_skills_count_missing_2",
        suspected_component="skills",
        constraint_type="count",
        failure_symptom="missing_required_object",
        support_count=2,
        support_experience_ids=["exp_1", "exp_2"],
        hypothesis="Missing object failures require better skill prompting.",
        evidence={"examples": []},
    )
    proposer._request_patch_rows = lambda weakness, **kwargs: [{
        "target_component": "skills",
        "changed_artifact": "skills.json",
        "changed_fields": ["skills.compositional_generation.prompt_rules"],
        "operation": "append_unique",
        "payload": {
            "path": ["skills", "compositional_generation", "prompt_rules"],
            "value": {
                "rule": "Make each requested object independently visible.",
                "source_index": 7,
            },
        },
        "predicted_improvement": "Improves object visibility.",
        "possible_regression": "May constrain style.",
    }]

    with pytest.raises(ValueError, match="provenance"):
        proposer.propose(weakness)


def test_harness_localizer_rejects_structured_provenance_keys():
    from gen_harness.patcher import HarnessLocalizer

    localizer = HarnessLocalizer({
        "type": "openai_chat_harness_localizer",
        "base_url": "http://example.invalid",
        "model": "test-llm",
    })
    weakness = Weakness(
        weakness_id="weak_skills_count_missing_2",
        suspected_component="skills",
        constraint_type="count",
        failure_symptom="missing_required_object",
        support_count=2,
        support_experience_ids=["exp_1", "exp_2"],
        hypothesis="Missing object failures require better skill prompting.",
        evidence={"examples": []},
    )
    with pytest.raises(ValueError, match="provenance"):
        localizer._localization_from_row(
            weakness,
            {
                "scores": {
                    "policy": 0.0,
                    "tools": 0.0,
                    "skills": 1.0,
                    "middleware": 0.0,
                    "memory": 0.0,
                    "no_patch": 0.0,
                },
                "rationale": {"reason": "skills issue", "source_index": 7},
            },
            frontier_size=3,
        )


def test_llm_patch_proposer_rejects_cross_boundary_payload_path():
    proposer = PatchProposer({
        "type": "openai_chat_patch_proposer",
        "base_url": "http://example.invalid",
        "model": "test-llm",
    })
    weakness = Weakness(
        weakness_id="weak_skills_count_missing_2",
        suspected_component="skills",
        constraint_type="count",
        failure_symptom="missing_required_object",
        support_count=2,
        support_experience_ids=["exp_1", "exp_2"],
        hypothesis="Missing object failures require better skill prompting.",
        evidence={"examples": []},
    )
    proposer._request_patch_rows = lambda weakness, **kwargs: [{
        "target_component": "skills",
        "changed_artifact": "skills.json",
        "changed_fields": ["skills.compositional_generation.prompt_rules"],
        "operation": "set_path",
        "payload": {"path": ["tools", "base_generator", "capabilities"], "value": {"text_to_image": True}},
        "predicted_improvement": "Escalate by changing another component.",
        "possible_regression": "Breaks component ownership.",
    }]

    with pytest.raises(ValueError, match="not owned"):
        proposer.propose(weakness)


def test_llm_patch_proposer_rejects_dataset_specific_rules():
    proposer = PatchProposer({
        "type": "openai_chat_patch_proposer",
        "base_url": "http://example.invalid",
        "model": "test-llm",
    })
    weakness = Weakness(
        weakness_id="weak_skills_count_missing_2",
        suspected_component="skills",
        constraint_type="count",
        failure_symptom="missing_required_object",
        support_count=2,
        support_experience_ids=["exp_1", "exp_2"],
        hypothesis="Missing object failures require better skill prompting.",
        evidence={"examples": []},
    )
    proposer._request_patch_rows = lambda weakness, **kwargs: [{
        "target_component": "skills",
        "changed_artifact": "skills.json",
        "changed_fields": ["skills.compositional_generation.prompt_rules"],
        "operation": "append_unique",
        "payload": {
            "path": ["skills", "compositional_generation", "prompt_rules"],
            "value": "Special-case GeneVal2 task_id geneval2_0007.",
        },
        "predicted_improvement": "Improves this benchmark item.",
        "possible_regression": "None.",
    }]

    with pytest.raises(ValueError, match="forbidden token|benchmark-specific"):
        proposer.propose(weakness)




def test_patch_proposer_requires_llm_config():
    from gen_harness.patcher import build_harness_localizer, build_patch_proposer

    with pytest.raises(ValueError, match="no rule-based patch fallback"):
        build_patch_proposer(None)
    with pytest.raises(ValueError, match="no rule-based attribution fallback"):
        build_harness_localizer(None)




def test_validator_reports_constraint_breakdown(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = make_task()
    patch = PatchManifest(
        patch_id="patch_quality_rule_breakdown",
        target_component="policy",
        changed_artifact="visual_contract.json",
        changed_fields=["prompt_governance.default_quality_profile"],
        operation="append_unique",
        payload={"path": ["prompt_governance", "default_quality_profile"], "value": "coherence boost rule"},
        supporting_experience=[],
        predicted_improvement="boost image metrics",
        possible_regression="none",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="policy/visual_contract.json",
    )

    report = PatchValidator(
        repo,
        RecordingBackend(),
        PolicySensitiveInspector(),
        metric_mode="image_metrics",
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    ).validate(
        patch, [task], [task], [task], seed=0, seeds=[0]
    )

    assert report.constraint_breakdown["before"]["target"]
    assert report.constraint_breakdown["after"]["heldout"]
    assert "target" in report.constraint_breakdown["delta"]




def test_validator_rejects_empty_preservation_split(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    task = make_task()
    patch = PatchManifest(
        patch_id="patch_missing_preservation_gate",
        target_component="policy",
        changed_artifact="visual_contract.json",
        changed_fields=["prompt_governance.default_quality_profile"],
        operation="append_unique",
        payload={"path": ["prompt_governance", "default_quality_profile"], "value": "coherence boost rule"},
        supporting_experience=[],
        predicted_improvement="boost image metrics",
        possible_regression="none",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="policy/visual_contract.json",
    )

    report = PatchValidator(
        repo,
        RecordingBackend(),
        PolicySensitiveInspector(),
        metric_mode="image_metrics",
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    ).validate(
        patch, [task], [task], [], seed=0, seeds=[0]
    )

    assert report.accepted is False
    assert report.reason == "rejected: preservation validation split is empty"


def test_validator_refuses_promotion_without_complete_three_split_gate(tmp_path):
    harness = make_harness(tmp_path)
    repo = HarnessRepository(harness)
    patch = PatchManifest(
        patch_id="patch_forged_acceptance",
        target_component="policy",
        changed_artifact="visual_contract.json",
        changed_fields=["prompt_governance.default_quality_profile"],
        operation="append_unique",
        payload={"path": ["prompt_governance", "default_quality_profile"], "value": "forged promotion rule"},
        supporting_experience=[],
        predicted_improvement="boost image metrics",
        possible_regression="none",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="policy/visual_contract.json",
    )
    forged = ValidationReport(
        patch_id=patch.patch_id,
        accepted=True,
        before={"target": 0.0, "heldout": 1.0},
        after={"target": 1.0, "heldout": 1.0},
        target_delta=1.0,
        heldout_delta=0.0,
        regression_delta=0.0,
        reason="accepted",
        seeds=[0],
    )
    validator = PatchValidator(
        repo,
        RecordingBackend(),
        PolicySensitiveInspector(),
        metric_mode="image_metrics",
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    )
    before = repo.load_artifact("policy", "visual_contract.json")

    with pytest.raises(ValueError, match="cannot be promoted"):
        validator.promote_if_accepted(patch, forged)

    assert repo.load_artifact("policy", "visual_contract.json") == before


def test_cli_has_no_external_evaluator_experience_converter():
    from gen_harness.cli import build_parser

    help_text = build_parser().format_help()
    assert "geneval-experiences" not in help_text



def test_detector_object_requirement_inspection_normalizes_detector_output(tmp_path):
    detector = tmp_path / "detector.py"
    detector.write_text(
        "import json, sys\n"
        "payload=json.load(open(sys.argv[sys.argv.index('--input')+1]))\n"
        "out=sys.argv[sys.argv.index('--output')+1]\n"
        "json.dump({'object_counts': {'bench': 1}}, open(out, 'w'))\n",
        encoding="utf-8",
    )
    from gen_harness.inspection import DetectorObjectRequirementInspection

    inspector = DetectorObjectRequirementInspection({
        "type": "detector_object_requirements",
        "command": ["{python}", str(detector), "--input", "{input_json}", "--output", "{output_json}"],
    })
    task = GenerationTask.from_dict({
        "task_id": "detector_task",
        "prompt": "a photo of a bench",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "objects_1",
            "text": "required visible objects: one bench",
            "constraint_type": "count",
            "metadata": {"constraint_schema": "object_requirements.v1", "required_objects": [{"class": "bench", "count": 1}]},
        }],
    })

    result = inspector.inspect(task, {}, {"image_uri": "/tmp/image.png"}, seed=0)

    assert result["objects_1"]["passed"] is True
    assert result["objects_1"]["object_counts"] == {"bench": 1}


def test_generate_exposes_optional_fast_loop_without_evaluator_coupling():
    from gen_harness.cli import build_parser

    help_text = build_parser().format_help()
    assert "geneval-experiences" not in help_text
    parsed = build_parser().parse_args([
        "generate",
        "--dataset", "geneval2",
        "--harness", "examples/visual_harness",
        "--tasks", "data/geneval2/geneval2_tasks_prompt_only_smoke32_2026_09_04.jsonl",
        "--output-dir", "/tmp/out",
        "--backend-config", "configs/backends/flow_grpo.json",
        "--inspection-config", "configs/inspection/object_requirements.example.json",
        "--policy-extractor-config", "configs/policy_extractors/openai_chat_policy_extractor.json",
        "--harness-localizer-config", "configs/patch_proposers/openai_chat_harness_localizer.json",
        "--patch-proposer-config", "configs/patch_proposers/openai_chat_patch_proposer.json",
        "--dry-run-budget",
    ])
    assert parsed.fast_loop_attempts == 5






def test_fast_loop_soft_abstention_prefers_better_weakest_constraint_over_mean():
    from gen_harness.fast_loop import FastLoopController

    controller = object.__new__(FastLoopController)
    high_mean_low_phrase = {
        "objects": {"passed": True, "selection_score": 1.0},
        "attribute": {"passed": True, "selection_score": 1.0},
        "prompt_alignment": {
            "passed": False,
            "symptom": "missing_inspection",
            "evidence_status": "unknown",
            "hard_evidence_required": False,
            "selection_score": 0.10,
        },
    }
    lower_mean_better_phrase = {
        "objects": {"passed": True, "selection_score": 0.80},
        "attribute": {"passed": True, "selection_score": 0.80},
        "prompt_alignment": {
            "passed": False,
            "symptom": "missing_inspection",
            "evidence_status": "unknown",
            "hard_evidence_required": False,
            "selection_score": 0.40,
        },
    }

    assert controller._passed(high_mean_low_phrase) is True
    assert controller._passed(lower_mean_better_phrase) is True
    assert controller._score(high_mean_low_phrase) > controller._score(lower_mean_better_phrase)
    assert controller._attempt_is_better(
        True,
        controller._score(lower_mean_better_phrase),
        lower_mean_better_phrase,
        controller._score(high_mean_low_phrase),
        high_mean_low_phrase,
    )










def test_removed_invalid_h100_mmdet3_geneval_script():
    assert not (ROOT / "scripts" / "run_geneval_100_h100_mmdet3_eval.sh").exists()
    assert not (ROOT / "scripts" / "run_geneval_official_eval.sh").exists()






















def test_lightweight_alignment_inspection_uses_scores_and_ocr_text():
    task = GenerationTask.from_dict({
        "task_id": "light_align_task",
        "prompt": "a red cup with text 'OPEN'",
        "task_family": "compositional",
        "constraints": [
            {"constraint_id": "attr_1", "text": "red cup", "constraint_type": "attribute", "metadata": {"constraint_schema": "attribute_bindings.v1"}},
            {"constraint_id": "text_1", "text": "text OPEN", "constraint_type": "text", "metadata": {"constraint_schema": "rendered_text.v1", "required_text": [{"text": "OPEN", "match": "contains"}]}},
        ],
    })
    inspector = LightweightAlignmentInspection({"threshold": 0.30})
    result = inspector.inspect(task, {}, {"alignment_scores": {"attr_1": 0.42}, "ocr_text": "open daily"})

    assert result["attr_1"]["passed"] is True
    assert result["text_1"]["passed"] is True


def test_lightweight_alignment_skips_external_command_when_no_applicable_constraints():
    task = GenerationTask.from_dict({
        "task_id": "count_only_light_align_skip",
        "prompt": "a photo of two cups",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "count_1",
                "text": "two cups",
                "constraint_type": "count",
                "metadata": {"constraint_schema": "object_requirements.v1"},
            }
        ],
    })
    inspector = LightweightAlignmentInspection({
        "constraint_types": ["attribute"],
        "command": ["python", "-c", "raise SystemExit(17)"],
    })

    result = inspector.inspect(task, {}, {"image_uri": "/tmp/x.png"})

    assert result["count_1"]["passed"] is True
    assert result["count_1"]["not_applicable"] is True


def test_lightweight_alignment_uses_persistent_command(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    payload=json.loads(line)\n"
        "    scores={c['constraint_id']: 0.77 for c in payload.get('constraints', [])}\n"
        "    print(json.dumps({'scores': scores, 'scorer': 'fake'}), flush=True)\n",
        encoding="utf-8",
    )
    task = GenerationTask.from_dict({
        "task_id": "persistent_align",
        "prompt": "a red cup",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "attr_1",
                "text": "red cup",
                "constraint_type": "attribute",
                "metadata": {"constraint_schema": "attribute_bindings.v1"},
            }
        ],
    })
    inspector = LightweightAlignmentInspection({
        "constraint_types": ["attribute"],
        "threshold": 0.5,
        "persistent_command": ["{python}", str(worker)],
    })

    first = inspector.inspect(task, {}, {"image_uri": "/tmp/x.png"})
    second = inspector.inspect(task, {}, {"image_uri": "/tmp/y.png"})

    assert first["attr_1"]["passed"] is True
    assert second["attr_1"]["alignment_score"] == 0.77


def test_lightweight_alignment_attribute_binding_contrastive_scores():
    task = GenerationTask.from_dict({
        "task_id": "attr_contrastive",
        "prompt": "a red cup and a blue book",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "attr_1",
                "text": "red cup and blue book",
                "constraint_type": "attribute",
                "metadata": {
                    "constraint_schema": "attribute_bindings.v1",
                    "attribute_bindings": [
                        {"object": "cup", "attribute": "red"},
                        {"object": "book", "attribute": "blue"},
                    ],
                },
            }
        ],
    })
    inspector = LightweightAlignmentInspection({"threshold": 0.30})
    result = inspector.inspect(task, {}, {
        "alignment_scores": {
            "attr_1": 0.70,
            "attr_1:binding:0:positive": 0.62,
            "attr_1:binding:0:negative:1": 0.50,
            "attr_1:binding:1:positive": 0.64,
            "attr_1:binding:1:negative:0": 0.30,
        }
    })

    row = result["attr_1"]
    assert row["passed"] is True
    assert len(row["attribute_binding_scores"]) == 2
    assert row["selection_score"] < row["alignment_score"]


def test_lightweight_alignment_attribute_binding_fails_swapped_color_scores():
    task = GenerationTask.from_dict({
        "task_id": "attr_contrastive_fail",
        "prompt": "a red cup and a blue book",
        "task_family": "compositional",
        "constraints": [
            {
                "constraint_id": "attr_1",
                "text": "red cup and blue book",
                "constraint_type": "attribute",
                "metadata": {
                    "constraint_schema": "attribute_bindings.v1",
                    "attribute_bindings": [
                        {"object": "cup", "attribute": "red"},
                        {"object": "book", "attribute": "blue"},
                    ],
                },
            }
        ],
    })
    inspector = LightweightAlignmentInspection({"threshold": 0.30})
    result = inspector.inspect(task, {}, {
        "alignment_scores": {
            "attr_1": 0.70,
            "attr_1:binding:0:positive": 0.45,
            "attr_1:binding:0:negative:1": 0.61,
            "attr_1:binding:1:positive": 0.64,
            "attr_1:binding:1:negative:0": 0.30,
        }
    })

    row = result["attr_1"]
    assert row["passed"] is False
    assert row["symptom"] == "attribute_binding_mismatch"
    assert row["selection_score"] < 0.70


def test_detector_utils_required_classes_include_relation_and_attribute_objects():
    module_path = ROOT / "scripts" / "detector_utils.py"
    spec = importlib.util.spec_from_file_location("detector_utils", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required_classes = module.required_classes

    payload = {
        "task": {
            "constraints": [
                {
                    "constraint_id": "spatial_1",
                    "constraint_type": "spatial",
                    "metadata": {"spatial_relations": [{"subject": "dog", "relation": "below", "object": "teddy bear"}]},
                },
                {
                    "constraint_id": "attr_1",
                    "constraint_type": "attribute",
                    "metadata": {
                        "attribute_bindings": [
                            {"object": "cup", "attribute": "red"},
                            {"object": "mug", "attribute": "white"},
                        ]
                    },
                },
            ]
        }
    }

    assert required_classes(payload) == ["dog", "teddy bear", "cup", "mug"]
    assert required_classes({
        "task": {
            "constraints": [
                {
                    "constraint_id": "spatial_mug",
                    "constraint_type": "spatial",
                    "metadata": {"spatial_relations": [{"subject": "pizza", "relation": "right_of", "object": "mug"}]},
                }
            ]
        }
    }) == ["pizza", "cup"]


def test_detector_utils_filters_bear_when_teddy_bear_confounder_overlaps():
    module_path = ROOT / "scripts" / "detector_utils.py"
    spec = importlib.util.spec_from_file_location("detector_utils", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    detections = [
        {"class": "bear", "score": 0.81, "bbox": [10, 10, 100, 100]},
        {"class": "teddy bear", "score": 0.92, "bbox": [12, 12, 98, 98]},
        {"class": "bear", "score": 0.88, "bbox": [200, 200, 300, 300]},
    ]
    detections_by_class = {
        "bear": [detections[0], detections[2]],
        "teddy bear": [detections[1]],
    }

    filtered = module.filter_confounded_detections(detections, detections_by_class)

    assert [row["class"] for row in filtered] == ["bear"]
    assert filtered[0]["bbox"] == [200, 200, 300, 300]




def test_patch_validation_budget_filters_to_patch_constraint_type(monkeypatch):
    from gen_harness.experiments.slow_loop import _validation_tasks_for_patch

    monkeypatch.setenv("GENHARNESS_SELF_EVOLVE_VALIDATION_TARGET_MAX", "2")
    monkeypatch.setenv("GENHARNESS_SELF_EVOLVE_VALIDATION_HELDOUT_MAX", "1")
    monkeypatch.setenv("GENHARNESS_SELF_EVOLVE_VALIDATION_PRESERVATION_MAX", "1")
    weakness = Weakness(
        weakness_id="weak_skills_spatial_spatial_relation_mismatch_2",
        suspected_component="skills",
        constraint_type="spatial",
        failure_symptom="spatial_relation_mismatch",
        support_count=2,
        support_experience_ids=["exp_spatial_1", "exp_spatial_2"],
        hypothesis="Repeated spatial failures need stronger visible layout prompting.",
        evidence={"examples": [{"component_decisions": {"skills": {"skill_name": "compositional_generation"}}}]},
    )
    patch = _llm_count_visibility_patch()
    patch.evidence_summary["constraint_type"] = "spatial"

    def task(task_id, split, constraint_type):
        return GenerationTask.from_dict({
            "task_id": task_id,
            "prompt": "a test prompt",
            "task_family": "compositional",
            "constraints": [{
                "constraint_id": f"{constraint_type}_0",
                "text": constraint_type,
                "constraint_type": constraint_type,
            }],
            "metadata": {"split": split},
        })

    splits = {
        "target": [task("target_count", "target", "count"), task("target_spatial_1", "target", "spatial"), task("target_spatial_2", "target", "spatial")],
        "heldout": [task("heldout_spatial", "heldout", "spatial"), task("heldout_action", "heldout", "action")],
        "preservation": [task("pres_spatial", "preservation", "spatial"), task("pres_count", "preservation", "count")],
    }

    selected = _validation_tasks_for_patch(patch, splits)

    assert [row.task_id for row in selected["target"]] == ["target_spatial_1", "target_spatial_2"]
    assert [row.task_id for row in selected["heldout"]] == ["heldout_spatial"]
    assert [row.task_id for row in selected["preservation"]] == ["pres_spatial"]
    assert selected["budget"]["constraint_type"] == "spatial"
    assert selected["budget"]["official_benchmark_score_used"] is False




def test_validated_prompt_rules_accepts_list_payloads():
    from gen_harness.components.skills import VisualSkillLibrary

    lib = VisualSkillLibrary(Path("examples/visual_harness/skills/skills.json"))
    rules = lib._validated_prompt_rules(
        [
            {
                "target_component": "skills",
                "changed_artifact": "skills.json",
                "payload": {
                    "path": ["skills", "compositional_generation", "prompt_families", "object_relations", "prompt_rules"],
                    "value": ["rule a", "rule b"],
                },
            }
        ],
        "compositional_generation",
        "object_relations",
    )

    assert rules == ["rule a", "rule b"]


def test_visual_harness_object_relations_has_no_duplicate_prompt_templates():
    skills_data = json.loads(Path("examples/visual_harness/skills/skills.json").read_text())
    object_relations = skills_data["skills"]["compositional_generation"]["prompt_families"]["object_relations"]
    assert object_relations == {}
    assert "benchmark_prompt_only" not in skills_data["skills"]["compositional_generation"]["prompt_families"]






def test_object_inspection_explicit_inventory_count_overrides_implied_attribute_count():
    inspector = ObjectRequirementInspection()
    required = inspector._required_objects_for_constraint({
        "required_objects": [{"class": "croissant", "count": 7}],
        "attribute_bindings": [{"object": "croissant", "attribute": "spotted"}],
    })

    assert required == [{"class": "croissant", "count": 7}]


def test_clip_prompt_keeps_complete_exact_count_layout_without_fragments(tmp_path):
    repo = HarnessRepository(make_harness(tmp_path))
    task = GenerationTask.from_dict({
        "task_id": "count_grid",
        "prompt": "six cars and a kangaroo",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "count_0",
            "text": "six cars and one kangaroo",
            "constraint_type": "count",
            "metadata": {
                "count_mode": "exact",
                "required_objects": [{"class": "car", "count": 6}, {"class": "kangaroo", "count": 1}],
            },
        }],
    })
    contract = repo.build_policy().compile_contract(task)
    workflow = repo.build_skills().compile_workflow(task, "compositional_generation", contract, {}, memory_context={})

    assert "Exactly six complete cars in a clean 2x3 grid" in workflow["clip_prompt"]
    assert "Exactly Plain" not in workflow["clip_prompt"]






def test_lamp_realizability_clip_phrases_support_detector_identity():
    from gen_harness.visual_realizability import object_realizability_clip_phrases

    phrases = object_realizability_clip_phrases("lamp")

    assert "lamp with shade stand and base" in phrases
    assert "single table lamp silhouette" in phrases





def test_visual_harness_inspection_tool_declares_action_capability():
    tools = json.loads((ROOT / "examples" / "visual_harness" / "tools" / "inspection_tools.json").read_text())

    assert "action" in tools["tools"]["constraint_inspector"]["capabilities"]




def _slow_loop_task(task_id, split):
    return GenerationTask.from_dict({
        "task_id": task_id,
        "prompt": "a photo of two cups on a plain table",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": f"{task_id}_count",
            "text": "required visible objects: two cups",
            "constraint_type": "count",
            "metadata": {
                "constraint_schema": "object_requirements.v1",
                "required_objects": [{"class": "cup", "count": 2}],
            },
        }],
        "metadata": {"split": split},
    })


def _llm_count_visibility_patch():
    return PatchManifest(
        patch_id="llm_test_count_visibility_patch",
        target_component="skills",
        changed_artifact="skills.json",
        changed_fields=["skills.compositional_generation.prompt_families.counting_small_objects.prompt_rules"],
        operation="append_unique",
        payload={
            "path": ["skills", "compositional_generation", "prompt_families", "counting_small_objects", "prompt_rules"],
            "value": "no extra same-class object is allowed",
            "schema_validation": {
                "operation": "append_unique",
                "target_component": "skills",
                "changed_artifact": "skills.json",
                "required_payload_fields": ["path", "value"],
                "rollback_boundary": "skills/skills.json",
            },
            "proposal_rationale": {
                "minimality": "Test stub simulates one LLM-proposed component-scoped patch.",
                "official_feedback_used": False,
            },
        },
        supporting_experience=["exp_test"],
        predicted_improvement="Improves exact-count visibility.",
        possible_regression="May constrain compositional prompts.",
        target_validation_set="target",
        preservation_set="preservation",
        held_out_set="heldout",
        rollback_boundary="skills/skills.json",
        evidence_summary={
            "proposal_source": {"type": "openai_chat_patch_proposer", "model": "test-llm"},
            "suspected_component": "skills",
            "constraint_type": "count",
            "failure_symptom": "object_count_mismatch",
            "support_count": 1,
            "support_experience_ids": ["exp_test"],
            "hypothesis": "Count failures require clearer visibility instructions.",
        },
        regression_risk={
            "affected_constraint_families": ["count"],
            "requires_preservation_validation": True,
            "requires_heldout_validation": True,
            "known_regression_match": False,
        },
    )


class _StaticPatchProposer:
    def __init__(self, patches):
        self.patches = list(patches)

    def propose_many(self, weaknesses, *, frontier_by_weakness=None, max_candidates_per_responsibility=3):
        if not weaknesses:
            return []
        support_ids = list(dict.fromkeys(
            exp_id
            for weakness in weaknesses
            for exp_id in getattr(weakness, "support_experience_ids", [])
        ))
        result = []
        for patch in self.patches:
            row = patch.to_dict()
            if support_ids:
                row["supporting_experience"] = support_ids
                row.setdefault("evidence_summary", {})["support_experience_ids"] = support_ids
            result.append(PatchManifest.from_dict(row))
        return result


class _RecordingPatchProposer(_StaticPatchProposer):
    def __init__(self, patches):
        super().__init__(patches)
        self.frontier_by_weakness = None

    def propose_many(self, weaknesses, *, frontier_by_weakness=None, max_candidates_per_responsibility=3):
        self.frontier_by_weakness = frontier_by_weakness
        return super().propose_many(
            weaknesses,
            frontier_by_weakness=frontier_by_weakness,
            max_candidates_per_responsibility=max_candidates_per_responsibility,
        )


class _StaticHarnessLocalizer:
    def __init__(self, frontier=None, *, no_patch=False):
        self.frontier = list(frontier if frontier is not None else ["skills"])
        self.no_patch = bool(no_patch)

    def localize(self, weakness, *, frontier_size=3):
        frontier = [] if self.no_patch else self.frontier[: max(1, int(frontier_size))]
        scores = {name: 0.0 for name in ("policy", "tools", "skills", "middleware", "memory", "no_patch")}
        if self.no_patch:
            scores["no_patch"] = 1.0
        for index, name in enumerate(frontier):
            scores[name] = float(len(frontier) - index)
        return HarnessLocalization(
            weakness_id=weakness.weakness_id,
            scores=scores,
            frontier=frontier,
            no_patch_preferred=self.no_patch,
            rationale="test LLM localizer stub",
        )


def test_component_boundary_audit_passes_reference_harness_and_prompt_only_tasks(tmp_path):
    harness = make_harness(tmp_path)
    tasks_path = tmp_path / "prompt_only_tasks.jsonl"
    write_jsonl(tasks_path, [
        {
            "task_id": "prompt_only",
            "prompt": "a red cup next to a blue bowl",
            "task_family": "compositional",
            "constraints": [],
            "metadata": {"source": "dev_prompt_only"},
            "references": {},
        }
    ])

    report = ComponentBoundaryAuditor().audit(harness, tasks_path=tasks_path)

    assert report["schema"] == "gen_harness.component_boundary_audit.v1"
    assert report["passed"] is True
    assert report["checks"]["component_artifacts_exist"] is True
    assert report["checks"]["component_names_match"] is True
    assert report["checks"]["component_types_match"] is True
    assert report["checks"]["boundary_contracts_present"] is True
    assert report["checks"]["responsibilities_non_overlapping"] is True
    assert report["checks"]["benchmark_labels_absent"] is True
    assert set(report["components"]) == {"policy", "skills", "tools", "memory", "middleware"}
    assert "constraint ontology" in report["components"]["policy"]["responsibilities"]
    assert "task workflow recipes" in report["components"]["skills"]["responsibilities"]
    assert "generation tool capability declarations" in report["components"]["tools"]["responsibilities"]
    assert "experience records" in report["components"]["memory"]["responsibilities"]
    assert "memory_contract.json" in report["components"]["memory"]["required_artifacts"]
    assert "tool routing" in report["components"]["middleware"]["responsibilities"]


def test_component_boundary_audit_requires_component_type(tmp_path):
    harness = make_harness(tmp_path)
    path = harness / "skills" / "skills.json"
    data = json.loads(path.read_text())
    data.pop("component_type", None)
    path.write_text(json.dumps(data), encoding="utf-8")

    report = ComponentBoundaryAuditor().audit(harness)

    assert report["passed"] is False
    assert report["checks"]["component_types_match"] is False
    assert any(issue["code"] == "wrong_component_type" for issue in report["issues"])


def test_component_boundary_audit_rejects_benchmark_specific_formal_interfaces(tmp_path):
    harness = make_harness(tmp_path)
    path = harness / "skills" / "skills.json"
    data = json.loads(path.read_text())
    families = data["skills"]["compositional_generation"].setdefault("prompt_families", {})
    families["t2icompbenchpp_scene_templates"] = {"family_templates": {}}
    path.write_text(json.dumps(data), encoding="utf-8")

    report = ComponentBoundaryAuditor().audit(harness)

    assert report["passed"] is False
    assert any(issue["code"] == "benchmark_specific_formal_interface" for issue in report["issues"])


def test_component_boundary_audit_requires_explicit_boundary_contract(tmp_path):
    harness = make_harness(tmp_path)
    path = harness / "tools" / "generation_tools.json"
    data = json.loads(path.read_text())
    data.pop("boundary_contract", None)
    path.write_text(json.dumps(data), encoding="utf-8")

    report = ComponentBoundaryAuditor().audit(harness)

    assert report["passed"] is False
    assert report["checks"]["boundary_contracts_present"] is False
    assert any(issue["code"] == "boundary_contract_incomplete" for issue in report["issues"])




def test_component_boundary_audit_blocks_geneval2_label_leakage(tmp_path):
    harness = make_harness(tmp_path)
    tasks_path = tmp_path / "leaky_geneval2_tasks.jsonl"
    write_jsonl(tasks_path, [
        {
            "task_id": "leaky",
            "prompt": "a red cup",
            "task_family": "compositional",
            "constraints": [],
            "metadata": {"benchmark": "geneval2"},
            "vqa_list": [{"question": "is the cup red?", "answer": "yes"}],
        }
    ])

    report = ComponentBoundaryAuditor().audit(harness, tasks_path=tasks_path)

    assert report["passed"] is False
    assert report["checks"]["benchmark_labels_absent"] is False
    assert any(issue["code"] == "benchmark_label_leak" for issue in report["issues"])


def test_run_artifact_audit_passes_clean_internal_trace(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    experiences_path = tmp_path / "experiences.jsonl"
    inspection_config = tmp_path / "inspection.json"
    backend_config = tmp_path / "backend.json"
    write_jsonl(tasks_path, [{"task_id": "t", "prompt": "a red cup", "task_family": "compositional", "constraints": []}])
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t",
            "constraint_id": "c",
            "constraint_type": "attribute",
            "constraint_text": "red cup",
            "component_decisions": {
                "policy": {},
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "examples/visual_harness/policy",
                    "skills": "examples/visual_harness/skills",
                    "tools": "examples/visual_harness/tools",
                    "middleware": "examples/visual_harness/middleware",
                    "memory": "examples/visual_harness/memory",
                },
            },
            "tool_calls": [{"tool_name": "base_generator"}],
            "visual_result": {"image_uri": "/tmp/generated.png"},
            "verification_result": {"passed": True, "inspection_source": "nvila_lite_2b_probability_verifier", "raw_result_summary": {"raw_answer_prefix": "cup=red"}},
            "failure_symptom": None,
            "suspected_component": None,
            "metadata": {},
        }
    ])
    inspection_config.write_text(json.dumps({"type": "external_command", "persistent_command": ["python", "scripts/nvila_semantic_verifier.py", "--serve"]}), encoding="utf-8")
    backend_config.write_text(json.dumps({
        "type": "flow_grpo",
        "model_family": "SD3.5 + Flow-GRPO",
        "frozen": True,
    }), encoding="utf-8")
    (tmp_path / "suite_evaluation_summary.json").write_text(json.dumps({
        "policy_extractor_manifest": {
            "policy_extractor": {"component_name": "openai_chat_policy_extractor", "model": "test-policy-llm"},
            "policy_extractor_configured": True,
            "legacy_policy_extractor_allowed": False,
            "formal_llm_policy_extractor": True,
        }
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(
        tasks_path=tasks_path,
        experiences_path=experiences_path,
        inspection_config_path=inspection_config,
        backend_config_path=backend_config,
    )

    assert report["schema"] == "gen_harness.run_artifact_audit.v1"
    assert report["passed"] is True
    assert report["checks"]["benchmark_labels_absent"] is True
    assert report["checks"]["fallback_absent"] is True
    assert report["checks"]["official_evaluator_absent_from_internal_loop"] is True
    assert report["checks"]["benchmark_conditioning_absent"] is True
    assert report["checks"]["runtime_component_provenance_complete"] is True
    assert report["checks"]["formal_policy_extractor_provenance_present"] is True
    assert report["checks"]["formal_policy_extractor_is_llm"] is True


def test_run_artifact_audit_requires_complete_runtime_component_provenance(tmp_path):
    experiences_path = tmp_path / "missing_provenance_experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t",
            "constraint_id": "c",
            "constraint_type": "attribute",
            "constraint_text": "red cup",
            "component_decisions": {"policy": {}, "skills": {}, "tools": {}, "middleware": {}, "memory": {}},
            "tool_calls": [{"tool_name": "base_generator"}],
            "visual_result": {"image_uri": "/tmp/generated.png"},
            "verification_result": {"passed": True},
            "metadata": {},
        }
    ])

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["checks"]["runtime_component_provenance_complete"] is False
    issue_codes = {issue["code"] for issue in report["issues"]}
    assert "runtime_component_provenance_incomplete" in issue_codes


def test_run_artifact_audit_rejects_oversized_experience_trace_without_loading(tmp_path, monkeypatch):
    import gen_harness.component_audit as component_audit

    experiences_path = tmp_path / "oversized_experiences.jsonl"
    experiences_path.write_text("{}\n{}\n{}\n", encoding="utf-8")
    monkeypatch.setattr(component_audit, "MAX_RUN_EXPERIENCES_BYTES", 2)

    report = component_audit.RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["checks"]["experience_trace_size_bounded"] is False
    assert report["experiences"]["checked"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "experience_trace_too_large",
        "formal_policy_extractor_provenance_missing",
    }


def test_run_artifact_audit_allows_internal_semantic_probe_answer_only():
    from gen_harness.component_audit import RunArtifactAuditor

    auditor = RunArtifactAuditor()
    assert auditor._run_benchmark_label_keys({
        "fast_loop": {"inspection": {"c0": {"semantic_checks": [{"answer": "yes"}]}}}
    }) == []
    assert auditor._run_benchmark_label_keys({"answer": "yes"}) == ["answer"]
    assert auditor._run_benchmark_label_keys({"official_answer": "yes"}) == ["official_answer"]


def test_run_artifact_audit_blocks_legacy_policy_extractor_trace(tmp_path):
    experiences_path = tmp_path / "legacy_policy_experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t",
            "constraint_id": "c",
            "constraint_type": "count",
            "constraint_text": "two cups",
            "component_decisions": {
                "policy": {
                    "policy_extractor": {
                        "component_name": "rule_based_policy_extractor",
                    },
                    "constraints": [
                        {
                            "metadata": {
                                "source": "policy_extractor.rule_based",
                            }
                        }
                    ],
                },
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "h/policy",
                    "skills": "h/skills",
                    "tools": "h/tools",
                    "middleware": "h/middleware",
                    "memory": "h/memory",
                },
            },
            "tool_calls": [],
            "visual_result": {"image_uri": "/tmp/generated.png"},
            "verification_result": {"passed": True},
            "metadata": {},
        }
    ])

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert "legacy_policy_extractor_in_formal_trace" in {issue["code"] for issue in report["issues"]}


def test_run_artifact_audit_requires_llm_policy_extractor_summary(tmp_path):
    experiences_path = tmp_path / "experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "method": "gen_harness",
            "task_id": "t",
            "seed": 0,
            "image_uri": "/tmp/generated.png",
            "generation": {"image_uri": "/tmp/generated.png"},
            "program": {},
            "component_decisions": {
                "policy": {"policy_extractor": {"component_name": "openai_chat_policy_extractor"}},
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "h/policy",
                    "skills": "h/skills",
                    "tools": "h/tools",
                    "middleware": "h/middleware",
                    "memory": "h/memory",
                },
            },
            "tool_calls": [],
            "visual_result": {"image_uri": "/tmp/generated.png"},
            "verification_result": {"passed": True},
            "metadata": {},
        }
    ])
    (tmp_path / "suite_evaluation_summary.json").write_text(json.dumps({
        "policy_extractor_manifest": {
            "policy_extractor": {"component_name": "rule_based_policy_extractor"},
            "policy_extractor_configured": False,
            "legacy_policy_extractor_allowed": True,
            "formal_llm_policy_extractor": False,
        }
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["checks"]["formal_policy_extractor_is_llm"] is False
    assert "formal_policy_extractor_not_llm" in {issue["code"] for issue in report["issues"]}


def test_run_artifact_audit_requires_policy_extractor_summary_presence(tmp_path):
    experiences_path = tmp_path / "experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "method": "gen_harness",
            "task_id": "t",
            "seed": 0,
            "image_uri": "/tmp/generated.png",
            "generation": {"image_uri": "/tmp/generated.png"},
            "program": {},
            "component_decisions": {
                "policy": {"policy_extractor": {"component_name": "openai_chat_policy_extractor"}},
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "h/policy",
                    "skills": "h/skills",
                    "tools": "h/tools",
                    "middleware": "h/middleware",
                    "memory": "h/memory",
                },
            },
            "tool_calls": [],
            "visual_result": {"image_uri": "/tmp/generated.png"},
            "verification_result": {"passed": True},
            "metadata": {},
        }
    ])

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["summaries"]["num_summaries"] == 0
    assert report["checks"]["formal_policy_extractor_provenance_present"] is False
    assert "formal_policy_extractor_provenance_missing" in {issue["code"] for issue in report["issues"]}


def test_run_artifact_audit_requires_existing_experience_trace(tmp_path):
    experiences_path = tmp_path / "missing_experiences.jsonl"
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "policy_extractor_manifest": {
            "policy_extractor": {"component_name": "openai_chat_policy_extractor", "model": "test-policy-llm"},
            "policy_extractor_configured": True,
            "legacy_policy_extractor_allowed": False,
            "formal_llm_policy_extractor": True,
        }
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["experiences"]["reason"] == "experience_trace_missing"
    assert "experience_trace_missing" in {issue["code"] for issue in report["issues"]}


def test_run_artifact_audit_blocks_leakage_fallback_and_official_internal_eval(tmp_path):
    experiences_path = tmp_path / "leaky_experiences.jsonl"
    inspection_config = tmp_path / "inspection.json"
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t",
            "constraint_id": "c",
            "constraint_type": "attribute",
            "verification_result": {"passed": False},
            "metadata": {"vqa_list": [{"question": "q", "answer": "yes"}]},
            "visual_result": {"fallback_image": "/tmp/reference.png"},
        }
    ])
    inspection_config.write_text(json.dumps({"type": "external_command", "persistent_command": ["python", "evaluate_geneval2_soft_tifa_batched.py"]}), encoding="utf-8")

    report = RunArtifactAuditor().audit(
        experiences_path=experiences_path,
        inspection_config_path=inspection_config,
    )

    assert report["passed"] is False
    assert report["checks"]["benchmark_labels_absent"] is False
    assert report["checks"]["fallback_absent"] is False
    assert report["checks"]["official_evaluator_absent_from_internal_loop"] is False
    assert {issue["code"] for issue in report["issues"]} >= {
        "benchmark_label_leak",
        "fallback_detected",
        "official_evaluator_in_internal_loop",
    }


def test_run_artifact_audit_blocks_structured_leakage_flags(tmp_path):
    experiences_path = tmp_path / "flagged_experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t",
            "constraint_id": "c",
            "constraint_type": "attribute",
            "constraint_text": "red cup",
            "component_decisions": {
                "policy": {"policy_extractor": {"component_name": "openai_chat_policy_extractor"}},
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "h/policy",
                    "skills": "h/skills",
                    "tools": "h/tools",
                    "middleware": "h/middleware",
                    "memory": "h/memory",
                },
            },
            "tool_calls": [],
            "visual_result": {"image_uri": "/tmp/generated.png", "fallback_used": True},
            "verification_result": {"passed": False},
            "metadata": {"routing": {"uses_official_labels": True}},
        }
    ])
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "policy_extractor_manifest": {
            "policy_extractor": {"component_name": "openai_chat_policy_extractor", "model": "test-policy-llm"},
            "policy_extractor_configured": True,
            "legacy_policy_extractor_allowed": False,
            "formal_llm_policy_extractor": True,
        }
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["checks"]["fallback_absent"] is False
    assert report["checks"]["official_evaluator_absent_from_internal_loop"] is False
    assert {issue["code"] for issue in report["issues"]} >= {
        "fallback_detected",
        "official_evaluator_in_internal_loop",
    }


def test_run_artifact_audit_accepts_explicit_false_fallback_capability(tmp_path):
    experiences_path = tmp_path / "clean_experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t",
            "constraint_id": "c",
            "constraint_type": "object",
            "component_decisions": {
                "policy": {},
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "h/policy",
                    "skills": "h/skills",
                    "tools": "h/tools",
                    "middleware": "h/middleware",
                    "memory": "h/memory",
                },
            },
            "tool_calls": [],
            "program": {
                "tool": {
                    "actual_capabilities": {
                        "uses_fallback_images": False,
                    }
                }
            },
            "visual_result": {"image_uri": "/tmp/generated.png"},
            "verification_result": {"passed": True},
        }
    ])
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "policy_extractor_manifest": {
            "policy_extractor": {
                "component_name": "openai_chat_policy_extractor",
                "model": "test-policy-llm",
            },
            "policy_extractor_configured": True,
            "legacy_policy_extractor_allowed": False,
            "formal_llm_policy_extractor": True,
        }
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert "fallback_detected" not in {issue["code"] for issue in report["issues"]}


def test_run_artifact_audit_blocks_benchmark_conditioned_generation_tokens(tmp_path):
    experiences_path = tmp_path / "benchmark_conditioned_experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t",
            "constraint_id": "action",
            "constraint_type": "action",
            "generation_strategy": {"name": "raw_geneval2_action_prompt"},
            "metadata": {"benchmark": "geneval2", "geneval2_index": 7},
        }
    ])

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["checks"]["benchmark_conditioning_absent"] is False
    assert any(issue["code"] == "benchmark_conditioned_generation" for issue in report["issues"])


def test_run_artifact_audit_blocks_benchmark_specific_internal_trace_strings(tmp_path):
    experiences_path = tmp_path / "benchmark_specific_trace.jsonl"
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t2icompbenchpp_color_val_0007",
            "constraint_id": "constraint_0",
            "constraint_type": "attribute",
            "constraint_text": "green backpack",
            "component_decisions": {
                "policy": {
                    "policy_extractor": {"component_name": "openai_chat_policy_extractor"},
                    "constraints": [
                        {
                            "constraint_id": "constraint_0",
                            "constraint_type": "attribute",
                            "text": "green backpack",
                            "metadata": {"source": "policy_extractor.t2icompbenchpp_prompt_only"},
                        }
                    ],
                },
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "h/policy",
                    "skills": "h/skills",
                    "tools": "h/tools",
                    "middleware": "h/middleware",
                    "memory": "h/memory",
                },
            },
            "tool_calls": [],
            "visual_result": {"image_uri": "/tmp/generated.png"},
            "verification_result": {"passed": True},
            "metadata": {"split": "heldout"},
        }
    ])
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "policy_extractor_manifest": {
            "policy_extractor": {"component_name": "openai_chat_policy_extractor", "model": "test-policy-llm"},
            "policy_extractor_configured": True,
            "legacy_policy_extractor_allowed": False,
            "formal_llm_policy_extractor": True,
        }
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["checks"]["benchmark_conditioning_absent"] is False
    assert any(issue["code"] == "benchmark_conditioned_generation" for issue in report["issues"])


def test_run_artifact_audit_allows_export_only_benchmark_metadata(tmp_path):
    experiences_path = tmp_path / "experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "method": "gen_harness",
            "task_id": "t",
            "seed": 0,
            "image_uri": "/tmp/generated.png",
            "generation": {"image_uri": "/tmp/generated.png"},
            "program": {},
            "component_decisions": {
                "policy": {"policy_extractor": {"component_name": "openai_chat_policy_extractor"}},
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "h/policy",
                    "skills": "h/skills",
                    "tools": "h/tools",
                    "middleware": "h/middleware",
                    "memory": "h/memory",
                },
            },
            "metadata": {"split": "heldout", "generation_input": "prompt_only_no_official_eval_outputs"},
            "export_metadata": {
                "benchmark": "geneval",
                "geneval_index": 0,
                "geneval_metadata": {"prompt": "two cups", "include": [{"class": "cup", "count": 2}]},
            },
            "official_eval_status": "pending",
        }
    ])
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "policy_extractor_manifest": {
            "policy_extractor": {"component_name": "openai_chat_policy_extractor", "model": "test-policy-llm"},
            "policy_extractor_configured": True,
            "legacy_policy_extractor_allowed": False,
            "formal_llm_policy_extractor": True,
        }
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is True
    assert report["checks"]["benchmark_labels_absent"] is True


def test_run_artifact_audit_rejects_export_metadata_on_visual_experience_rows(tmp_path):
    experiences_path = tmp_path / "experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t",
            "constraint_id": "c",
            "constraint_type": "count",
            "constraint_text": "two cups",
            "component_decisions": {
                "policy": {"policy_extractor": {"component_name": "openai_chat_policy_extractor"}},
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "h/policy",
                    "skills": "h/skills",
                    "tools": "h/tools",
                    "middleware": "h/middleware",
                    "memory": "h/memory",
                },
            },
            "tool_calls": [],
            "visual_result": {"image_uri": "/tmp/generated.png"},
            "verification_result": {"passed": True},
            "metadata": {"split": "heldout"},
            "export_metadata": {"benchmark": "geneval", "geneval_index": 0},
        }
    ])
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "policy_extractor_manifest": {
            "policy_extractor": {"component_name": "openai_chat_policy_extractor", "model": "test-policy-llm"},
            "policy_extractor_configured": True,
            "legacy_policy_extractor_allowed": False,
            "formal_llm_policy_extractor": True,
        }
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["checks"]["benchmark_labels_absent"] is False
    assert any(issue["code"] == "benchmark_label_leak" for issue in report["issues"])


def test_run_artifact_audit_rejects_generation_time_benchmark_metadata(tmp_path):
    experiences_path = tmp_path / "experiences.jsonl"
    write_jsonl(experiences_path, [
        {
            "experience_id": "exp",
            "task_id": "t",
            "constraint_id": "c",
            "constraint_type": "count",
            "constraint_text": "two cups",
            "component_decisions": {
                "policy": {"policy_extractor": {"component_name": "openai_chat_policy_extractor"}},
                "skills": {},
                "tools": {},
                "middleware": {},
                "memory": {},
                "component_sources": {
                    "policy": "h/policy",
                    "skills": "h/skills",
                    "tools": "h/tools",
                    "middleware": "h/middleware",
                    "memory": "h/memory",
                },
            },
            "tool_calls": [],
            "visual_result": {"image_uri": "/tmp/generated.png"},
            "verification_result": {"passed": True},
            "metadata": {"benchmark": "geneval"},
        }
    ])
    (tmp_path / "run_summary.json").write_text(json.dumps({
        "policy_extractor_manifest": {
            "policy_extractor": {"component_name": "openai_chat_policy_extractor", "model": "test-policy-llm"},
            "policy_extractor_configured": True,
            "legacy_policy_extractor_allowed": False,
            "formal_llm_policy_extractor": True,
        }
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(experiences_path=experiences_path)

    assert report["passed"] is False
    assert report["checks"]["benchmark_labels_absent"] is False
    assert "benchmark_label_leak" in {issue["code"] for issue in report["issues"]}


def test_run_artifact_audit_rejects_non_diffusion_backend_config(tmp_path):
    backend_config = tmp_path / "backend.json"
    backend_config.write_text(json.dumps({
        "type": "image_llm",
        "name": "removed_image_llm",
        "uses_scores": True,
        "fallback_used": True,
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(backend_config_path=backend_config)

    assert report["passed"] is False
    assert report["checks"]["formal_backend_uses_real_generator"] is False
    assert any(issue["code"] == "unsupported_formal_generator_backend" for issue in report["issues"])
    assert "fallback_detected" in {issue["code"] for issue in report["issues"]}
    assert "official_evaluator_in_internal_loop" in {issue["code"] for issue in report["issues"]}


def test_run_artifact_audit_rejects_metadata_only_inspection_config(tmp_path):
    inspection_config = tmp_path / "inspection.json"
    inspection_config.write_text(json.dumps({
        "type": "composite",
        "inspectors": [
            {
                "type": "object_requirements",
                "allow_diagnostic_inspection_metadata": True,
            },
            {
                "type": "lightweight_alignment",
            },
            {
                "type": "detector_object_requirements",
                "passthrough_generation_fields": True,
            },
        ],
    }), encoding="utf-8")

    report = RunArtifactAuditor().audit(inspection_config_path=inspection_config)

    assert report["passed"] is False
    assert report["checks"]["formal_inspection_uses_real_visual_evidence"] is False
    assert {
        issue["code"]
        for issue in report["issues"]
    } == {"formal_inspection_uses_metadata_only_diagnostic"}




def test_goal_readiness_audit_fails_until_full_score_exists(tmp_path):
    root = tmp_path / "run"
    (root / "self_evolve_dev").mkdir(parents=True)

    report = GoalReadinessAuditor().audit(root)

    assert report["passed"] is False
    assert any(issue["code"] == "missing_required_artifact" for issue in report["issues"])
    assert report["checks"]["full_passed"] is False


def test_goal_readiness_rejects_incomplete_full_image_map(tmp_path):
    root = tmp_path / "run"
    full = root / "geneval2_full800"
    full.mkdir(parents=True)
    clean_component = {"passed": True, "issues": []}
    write_json(full / "component_audit.json", clean_component)
    write_json(full / "run_artifact_audit.json", {"passed": True, "issues": []})
    write_json(full / "geneval2_image_map.json", {"prompt": "/tmp/image.png"})
    write_json(full / "geneval2_image_map.json.manifest.json", {
        "schema": "gen_harness.geneval2_image_map.v1",
        "require_all": False,
        "num_benchmark_prompts": 800,
        "num_mapped_prompts": 574,
        "missing_indices": [100, 102],
    })
    write_json(full / "official_soft_tifa_batched_full800.json", {
        "soft_tifa_am": 91.0,
        "soft_tifa_gm": 73.0,
    })
    issues = []

    report = GoalReadinessAuditor()._audit_eval_stage(
        full,
        score_name="official_soft_tifa_batched_full800.json",
        issues=issues,
        min_gm=72.0,
        require_full_score=True,
    )

    assert report["passed"] is False
    assert report["image_map"]["passed"] is False
    codes = {issue.code for issue in issues}
    assert "image_map_not_require_all" in codes
    assert "image_map_incomplete" in codes


def test_goal_readiness_runtime_strategy_requires_semantic_self_evolve_env(tmp_path):
    path = tmp_path / "runtime_strategy_env.json"
    write_json(path, {
        "schema": "gen_harness.runtime_strategy_env.v1",
        "strategy_env": {"GENHARNESS_BENCHMARK_CONDITIONED_ROUTING": "1"},
        "benchmark_conditioned_env_absent": True,
    })
    issues = []

    report = GoalReadinessAuditor()._audit_runtime_strategy(path, issues)

    assert report["passed"] is False
    codes = {issue.code for issue in issues}
    assert "self_evolve_strategy_unproven" in codes


def test_goal_readiness_rejects_cross_component_patch_scope():
    change = {
        "component": "skills",
        "artifact": "tool_router.json",
        "changed_fields": ["routes.count"],
        "operation": "set_path",
        "targeted_fix": {
            "payload": {
                "path": ["routes", "count"],
                "value": "base_generator",
                "schema_validation": {
                    "operation": "set_path",
                    "target_component": "middleware",
                    "changed_artifact": "tool_router.json",
                },
            },
            "rollback_boundary": "middleware/tool_router.json",
        },
    }

    missing = GoalReadinessAuditor()._missing_change_component_scope_fields(change)

    assert "artifact_owned_by_component" in missing
    assert "targeted_fix.payload.schema_validation.target_component" in missing
    assert "targeted_fix.rollback_boundary" in missing


def test_goal_readiness_rejects_incomplete_self_evolve_stop_reason(tmp_path):
    summary = {
        "stop_reason": "no_accepted_patches",
        "promote_accepted": True,
    }
    manifest = {
        "observability": {"num_failed_experiences": 3, "num_patch_proposals": 2},
        "gate_summary": {"num_accepted": 0, "num_promoted": 0},
    }
    issues = []

    passed = GoalReadinessAuditor()._self_evolve_completion_is_clean(summary, manifest, issues, tmp_path)

    assert passed is False
    assert {issue.code for issue in issues} == {"self_evolve_stop_reason_incomplete"}


def test_goal_readiness_rejects_self_evolve_without_effective_patch(tmp_path):
    summary = {
        "stop_reason": "max_rounds_reached",
        "promote_accepted": True,
    }
    manifest = {
        "observability": {"num_failed_experiences": 3, "num_patch_proposals": 2},
        "gate_summary": {"num_accepted": 0, "num_promoted": 0},
    }
    issues = []

    passed = GoalReadinessAuditor()._self_evolve_completion_is_clean(summary, manifest, issues, tmp_path)

    assert passed is False
    assert {issue.code for issue in issues} == {"self_evolve_no_effective_patch"}


def test_goal_readiness_rejects_trace_certificate_with_unsafe_promotion():
    manifest = {
        "schema": "gen_harness.self_evolve_manifest.v1",
        "reference_model": {"copied_runtime": False, "local_path_recorded": False},
        "reference_frameworks": clean_reference_frameworks(),
        "reference_trajectory_policy": {
            "official_benchmark_reference_trajectories_allowed": False,
            "allowed_sources": ["non_benchmark_dev_tasks", "internal_baseline_traces"],
            "official_labels_used_for_generation_or_selection": False,
            "official_evaluator_used_for_generation_or_selection": False,
            "benchmark_tasks_allowed": False,
            "leakage_safe": True,
        },
        "failure_attribution": {"official_labels_used": False, "fallback_used": False, "leakage_safe": True},
        "observability": {"num_failed_experiences": 1, "num_patch_proposals": 1, "num_failure_diagnoses": 1},
        "trace_certificate": {
            **clean_trace_certificate(),
            "promotion": {
                "num_promoted": 1,
                "promoted_patch_ids": ["patch"],
                "accepted_patch_ids": [],
                "non_accepted_promotions": ["patch"],
                "accepted_only": False,
            },
        },
        "changes": [
            {
                "component": "skills",
                "artifact": "skills.json",
                "changed_fields": ["skills.compositional_generation.prompt_rules"],
                "operation": "append_unique",
                "failure_evidence": {
                    "supporting_experience": ["exp1"],
                    "failure_diagnoses": [{"suspected_component": "skills"}],
                },
                "root_cause": "skills prompt omitted count specificity",
                "targeted_fix": {
                    "payload": {
                        "path": ["skills", "compositional_generation", "prompt_rules"],
                        "value": "rule",
                        "schema_validation": {
                            "operation": "append_unique",
                            "target_component": "skills",
                            "changed_artifact": "skills.json",
                        },
                    },
                    "rollback_boundary": "skills/skills.json",
                },
                "predicted_impact": {"improvement": "improves target failures"},
                "gate": {"accepted": True, "promoted_to_harness": True},
            }
        ],
    }
    issues = []

    passed = GoalReadinessAuditor()._manifest_proves_self_evolve_chain(manifest, issues, Path("/tmp/self_evolve"))

    assert passed is False
    assert "trace_certificate_unproven" in {issue.code for issue in issues}


def test_goal_readiness_rejects_unsafe_reference_trajectory_policy():
    auditor = GoalReadinessAuditor()

    assert auditor._reference_trajectory_policy_is_clean({
        "official_benchmark_reference_trajectories_allowed": False,
        "allowed_sources": ["non_benchmark_dev_tasks", "internal_baseline_traces"],
        "official_labels_used_for_generation_or_selection": False,
        "official_evaluator_used_for_generation_or_selection": False,
        "benchmark_tasks_allowed": False,
        "leakage_safe": True,
    }) is True
    assert auditor._reference_trajectory_policy_is_clean({
        "official_benchmark_reference_trajectories_allowed": True,
        "allowed_sources": ["official_geneval2_reference_trajectories"],
        "official_labels_used_for_generation_or_selection": True,
        "official_evaluator_used_for_generation_or_selection": True,
        "benchmark_tasks_allowed": True,
        "leakage_safe": False,
    }) is False


def test_goal_readiness_rejects_unproven_reference_frameworks():
    auditor = GoalReadinessAuditor()

    assert auditor._reference_frameworks_are_clean(clean_reference_frameworks()) is True
    assert auditor._reference_frameworks_are_clean([
        {
            "source": "agentic-harness-engineering",
            "used_as": "conceptual_reference_only",
            "adopted_invariants": ["failure evidence"],
            "copied_runtime": False,
            "local_path_recorded": False,
            "official_benchmark_reference_trajectories_allowed": False,
        }
    ]) is False
    assert auditor._reference_frameworks_are_clean([
        {
            "source": "agentic-harness-engineering",
            "used_as": "conceptual_reference_only",
            "adopted_invariants": ["failure evidence"],
            "copied_runtime": False,
            "local_path_recorded": False,
            "official_benchmark_reference_trajectories_allowed": False,
        },
        {
            "source": "JIT",
            "used_as": "runtime_dependency",
            "adopted_invariants": ["fixed harness component protocol"],
            "copied_runtime": False,
            "local_path_recorded": True,
            "local_path": "LOCAL_REFERENCE_PATH",
            "official_benchmark_reference_trajectories_allowed": False,
        },
    ]) is False


def test_goal_readiness_audit_requires_self_evolve_chain_evidence(tmp_path):
    root = tmp_path / "run"
    self_evolve = root / "self_evolve_dev"
    (self_evolve / "output").mkdir(parents=True)
    clean_component = {"passed": True, "issues": []}
    write_json(root / "runtime_strategy_env.json", {
        "schema": "gen_harness.runtime_strategy_env.v1",
        "strategy_env": {"GENHARNESS_SELF_EVOLVE_ACTION_RAW_FIRST": "1"},
        "benchmark_conditioned_env_absent": True,
    })
    write_json(self_evolve / "component_audit_before.json", clean_component)
    write_json(self_evolve / "component_audit_after.json", clean_component)
    write_json(self_evolve / "output" / "self_evolve_run_summary.json", {
        "schema": "gen_harness.self_evolve_run.v1",
        "fast_loop_attempts": 1,
        "rounds": [{"round": 1}],
        "leakage_and_fallback_guards": {
            "allow_benchmark_tasks": False,
            "official_evaluator_used_for_generation_or_selection": False,
            "official_labels_used_for_prompt_compilation": False,
            "fallback_image_substitution_allowed": False,
        },
    })
    write_json(self_evolve / "self_evolve_manifest.json", {
        "schema": "gen_harness.self_evolve_manifest.v1",
        "reference_model": {"copied_runtime": False, "local_path_recorded": False},
        "reference_frameworks": clean_reference_frameworks(),
        "leakage_and_fallback_guards": {"fallback_used": False},
        "failure_attribution": {"official_labels_used": False, "fallback_used": False, "leakage_safe": True},
        "observability": {"num_failed_experiences": 1, "num_patch_proposals": 1, "num_failure_diagnoses": 0},
        "changes": [{"failure_evidence": {}, "targeted_fix": {}, "predicted_impact": {}, "gate": {}}],
    })

    report = GoalReadinessAuditor().audit(root)

    assert report["checks"]["self_evolve_completed"] is False
    codes = {issue["code"] for issue in report["issues"]}
    assert "fast_loop_not_enabled" in codes
    assert "self_evolve_round_manifest_missing" in codes
    assert "failure_diagnosis_missing" in codes
    assert "trace_certificate_unproven" in codes
    assert "self_evolve_chain_incomplete" in codes


def test_visual_experience_compact_dict_limits_nested_trace_size():
    huge_prompt = "very detailed prompt " * 1000
    exp = VisualExperience(
        experience_id="exp",
        task_id="task",
        constraint_id="c",
        constraint_type="attribute",
        constraint_text="red cup",
        component_decisions={
            "skills": {"compiled_prompt": huge_prompt, "nested": {"program": {"again": huge_prompt}}},
        },
        tool_calls=[
            {
                "tool_name": "base_generator",
                "arguments": {"program": {"workflow": {"compiled_prompt": huge_prompt}}},
                "result": {"image_uri": "/tmp/x.png", "prompt": huge_prompt},
            }
        ],
        visual_result={"image_uri": "/tmp/x.png", "raw_generation": {"prompt": huge_prompt, "trace": {"prompt": huge_prompt}}},
        verification_result={"passed": False, "symptom": "attribute_binding_mismatch", "reason": huge_prompt},
        failure_symptom="attribute_binding_mismatch",
        suspected_component="skills",
        metadata={"fast_loop": {"attempts": [{"prompt": huge_prompt, "program": {"huge": huge_prompt}, "passed": False} for _ in range(20)]}},
    )

    compact = exp.to_compact_dict()
    serialized = json.dumps(compact)

    assert len(serialized) < 20000
    assert "attempts_truncated" in serialized
    assert "truncated" in serialized


def test_visual_memory_writes_compact_experiences(tmp_path):
    from gen_harness.components.memory import VisualMemory

    huge_prompt = "prompt " * 5000
    exp = VisualExperience(
        experience_id="exp",
        task_id="task",
        constraint_id="c",
        constraint_type="count",
        constraint_text="two cups",
        component_decisions={"skills": {"compiled_prompt": huge_prompt}},
        tool_calls=[],
        visual_result={"image_uri": "/tmp/x.png", "prompt": huge_prompt},
        verification_result={"passed": False, "symptom": "object_count_mismatch", "reason": huge_prompt},
        failure_symptom="object_count_mismatch",
        suspected_component="skills",
        metadata={"fast_loop": {"attempts": [{"prompt": huge_prompt} for _ in range(10)]}},
    )

    memory = VisualMemory(tmp_path)
    memory.write_experiences([exp])

    assert (tmp_path / "visual_experiences.jsonl").stat().st_size < 20000
    assert (tmp_path / "failure_clusters.jsonl").stat().st_size < 20000


def test_visual_memory_enforces_hard_row_size_limit(tmp_path, monkeypatch):
    import gen_harness.components.memory as memory_module
    from gen_harness.components.memory import VisualMemory

    monkeypatch.setattr(memory_module, "MAX_MEMORY_ROW_BYTES", 1000)
    huge_blob = "x" * 10000
    exp = VisualExperience(
        experience_id="exp_huge",
        task_id="task",
        constraint_id="c",
        constraint_type="count",
        constraint_text="two cups",
        component_decisions={"tools": {"generation_artifact": {"tools": {"base": {"capabilities": {"text_to_image": True}, "actual_capabilities": {"text_to_image": True}, "note": huge_blob}}}}},
        tool_calls=[],
        visual_result={"image_uri": "/tmp/x.png", "prompt": huge_blob},
        verification_result={"passed": False, "symptom": "object_count_mismatch", "reason": huge_blob},
        failure_symptom="object_count_mismatch",
        suspected_component="skills",
        metadata={"split": "target", "huge": huge_blob},
    )

    memory = VisualMemory(tmp_path)
    memory.write_experiences([exp])
    row = json.loads((tmp_path / "visual_experiences.jsonl").read_text())

    assert (tmp_path / "visual_experiences.jsonl").stat().st_size < 1000
    assert row["memory_compaction"]["reason"] == "compact_row_exceeded_max_bytes"
    assert row["visual_result"]["image_uri"] == "/tmp/x.png"


def test_failure_debugger_assigns_component_scoped_root_cause():
    exp = VisualExperience(
        experience_id="exp_attr",
        task_id="task_attr",
        constraint_id="attr_0",
        constraint_type="attribute",
        constraint_text="red cup and blue bowl",
        component_decisions={"skills": {"missing_steps": []}, "middleware": {}},
        tool_calls=[{"tool_name": "base_generator"}],
        visual_result={"image_uri": "/tmp/x.png", "seed": 0},
        verification_result={
            "passed": False,
            "symptom": "attribute_binding_mismatch",
            "confidence": 0.8,
            "attribute_mismatches": [{"object": "cup", "expected": "red", "observed": "blue"}],
        },
        failure_symptom="attribute_binding_mismatch",
        suspected_component=None,
        metadata={"task_family": "compositional"},
    )

    diagnosis = FailureDebugger().diagnose(exp).to_dict()

    assert diagnosis["schema"] == "gen_harness.failure_diagnosis.v1"
    assert diagnosis["suspected_component"] == "skills"
    assert diagnosis["leakage_safe"] is True
    assert diagnosis["fallback_used"] is False
    assert diagnosis["next_action"]["type"] == "patch_skill_prompt_family"
    assert "attribute" in diagnosis["root_cause"].lower()


def test_failure_debugger_assigns_prompt_budget_attempt_errors_to_skills():
    exp = VisualExperience(
        experience_id="exp_budget",
        task_id="task_budget",
        constraint_id="attr_0",
        constraint_type="attribute",
        constraint_text="red cup and blue bowl",
        component_decisions={"skills": {"missing_steps": []}, "middleware": {}},
        tool_calls=[{"tool_name": "base_generator"}],
        visual_result={"image_uri": None, "seed": 0, "raw_generation": {}},
        verification_result={"passed": False, "symptom": "missing_inspection", "confidence": 0.0},
        failure_symptom="missing_inspection",
        suspected_component=None,
        metadata={
            "task_family": "compositional",
            "fast_loop": {
                "passed": False,
                "selected_attempt": 1,
                "attempts": [
                    {
                        "error": "ValueError: CLIP prompt exceeds v23 budget for task task_budget: tokenizer=96 tokens > 72",
                    }
                ],
            },
        },
    )

    diagnosis = FailureDebugger().diagnose(exp).to_dict()

    assert diagnosis["suspected_component"] == "skills"
    assert diagnosis["next_action"]["type"] == "patch_skill_prompt_family"
    assert "prompt budget" in diagnosis["root_cause"].lower()
    assert diagnosis["evidence"]["attempt_error_count"] == 1


def test_slow_loop_attaches_failure_diagnosis_before_weakness_mining(tmp_path):
    harness = make_harness(tmp_path)
    exp = VisualExperience(
        experience_id="exp_spatial",
        task_id="task_spatial",
        constraint_id="spatial_0",
        constraint_type="spatial",
        constraint_text="cup left of bowl",
        component_decisions={"skills": {}, "middleware": {}},
        tool_calls=[{"tool_name": "base_generator"}],
        visual_result={"image_uri": "/tmp/x.png", "seed": 0},
        verification_result={"passed": False, "symptom": "spatial_relation_mismatch", "confidence": 0.7},
        failure_symptom="spatial_relation_mismatch",
        suspected_component=None,
        metadata={},
    )
    runner = SlowLoopEvolutionRunner(
        HarnessRepository(harness),
        DataUriBackend(),
        ObjectRequirementInspection(),
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    )

    runner._attach_failure_diagnoses([exp])

    assert exp.suspected_component is None
    assert exp.metadata["failure_diagnosis"]["suspected_component"] == "skills"
    assert exp.metadata["failure_diagnosis"]["leakage_safe"] is True








def test_slow_loop_evolution_can_promote_validated_patch(tmp_path):
    harness = make_harness(tmp_path)
    tasks_path = tmp_path / "slow_loop_tasks.jsonl"
    write_jsonl(tasks_path, [
        _slow_loop_task("slow_target", "target").to_dict(),
        _slow_loop_task("slow_heldout", "heldout").to_dict(),
        _slow_loop_task("slow_preservation", "preservation").to_dict(),
    ])

    class BeforeAfterBackend(VisualBackend):
        def generate(self, task, program, seed=0):
            prompt = program.get("workflow", {}).get("compiled_prompt", "")
            count = 2 if "no extra same-class object is allowed" in prompt else 1
            return {"image_uri": f"/tmp/{task.task_id}_{seed}.png", "object_counts": {"cup": count}, "prompt": prompt, "seed": seed}

    runner = SlowLoopEvolutionRunner(
        HarnessRepository(harness),
        BeforeAfterBackend(),
        ObjectRequirementInspection(),
        harness_localizer=_StaticHarnessLocalizer(["skills"]),
        patch_proposer=_StaticPatchProposer([_llm_count_visibility_patch()]),
    )
    summary = runner.run(tasks_path, tmp_path / "slow_loop_promote", seeds=[0], min_support=1, promote=True)

    skills_text = (Path(harness) / "skills" / "skills.json").read_text(encoding="utf-8")
    assert summary["num_promoted_patches"] == 1
    assert "no extra same-class object is allowed" in skills_text
    assert (Path(harness) / "memory" / "validated_patches.jsonl").read_text(encoding="utf-8").strip()




def test_self_evolve_run_refuses_leaky_tasks_before_exploration(tmp_path):
    harness = make_harness(tmp_path)
    tasks_path = tmp_path / "leaky_self_evolve_tasks.jsonl"
    task = _slow_loop_task("leaky_target", "target").to_dict()
    task["vqa_list"] = [{"question": "is there a cup?", "answer": "yes"}]
    write_jsonl(tasks_path, [
        task,
        _slow_loop_task("leaky_heldout", "heldout").to_dict(),
        _slow_loop_task("leaky_preservation", "preservation").to_dict(),
    ])

    runner = SelfEvolveRunner(
        HarnessRepository(harness),
        DataUriBackend(),
        ObjectRequirementInspection(),
        fast_loop_attempts=1,
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
    )

    with pytest.raises(ValueError, match="initial component audit failed"):
        runner.run(tasks_path, tmp_path / "leaky_self_evolve_out", seeds=[0], max_rounds=1, min_support=1)

    summary = json.loads((tmp_path / "leaky_self_evolve_out" / "self_evolve_run_summary.json").read_text())
    assert summary["stop_reason"] == "initial_component_audit_failed"
    assert summary["leakage_and_fallback_guards"]["initial_component_audit_passed"] is False
    assert summary["rounds"] == []










def test_slow_loop_refuses_benchmark_tasks_by_default(tmp_path):
    harness = make_harness(tmp_path)
    tasks_path = tmp_path / "benchmark_tasks.jsonl"
    task = _slow_loop_task("geneval_leak_guard", "target")
    task.metadata["benchmark"] = "geneval"
    write_jsonl(tasks_path, [task.to_dict(), _slow_loop_task("h", "heldout").to_dict(), _slow_loop_task("p", "preservation").to_dict()])

    runner = SlowLoopEvolutionRunner(
        HarnessRepository(harness),
        DataUriBackend(),
        ObjectRequirementInspection(),
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    )

    try:
        runner.run(tasks_path, tmp_path / "blocked", seeds=[0], min_support=1)
    except ValueError as exc:
        assert "refuses benchmark tasks by default" in str(exc)
    else:
        raise AssertionError("slow-loop evolution accepted benchmark task metadata")


def test_slow_loop_allow_benchmark_tasks_requires_diagnostic_env(tmp_path, monkeypatch):
    harness = make_harness(tmp_path)
    tasks_path = tmp_path / "benchmark_tasks.jsonl"
    task = _slow_loop_task("geneval_leak_guard", "target")
    task.metadata["benchmark"] = "geneval"
    write_jsonl(tasks_path, [task.to_dict(), _slow_loop_task("h", "heldout").to_dict(), _slow_loop_task("p", "preservation").to_dict()])
    runner = SlowLoopEvolutionRunner(
        HarnessRepository(harness),
        DataUriBackend(),
        ObjectRequirementInspection(),
        harness_localizer=_StaticHarnessLocalizer(no_patch=True),
        patch_proposer=_StaticPatchProposer([]),
    )

    monkeypatch.delenv("GENHARNESS_DIAGNOSTIC_ALLOW_BENCHMARK_EVOLVE", raising=False)
    with pytest.raises(ValueError, match="GENHARNESS_DIAGNOSTIC_ALLOW_BENCHMARK_EVOLVE"):
        runner.run(tasks_path, tmp_path / "blocked", seeds=[0], min_support=1, allow_benchmark_tasks=True)

    monkeypatch.setenv("GENHARNESS_DIAGNOSTIC_ALLOW_BENCHMARK_EVOLVE", "1")
    summary = runner.run(tasks_path, tmp_path / "diagnostic", seeds=[0], min_support=1, allow_benchmark_tasks=True)

    assert summary["allow_benchmark_tasks"] is True
def test_formal_runtime_paths_do_not_import_or_call_regex():
    formal_paths = [
        ROOT / "gen_harness" / "components" / "policy.py",
        ROOT / "gen_harness" / "prompt_optimizer.py",
        ROOT / "gen_harness" / "adapters" / "openai_vision_inspection.py",
        ROOT / "gen_harness" / "io.py",
        ROOT / "gen_harness" / "json_extract.py",
        ROOT / "gen_harness" / "safety.py",
    ]

    for path in formal_paths:
        source = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in source.splitlines()]
        assert "import re" not in lines, path
        assert "from re import" not in source, path
        for token in ("re.compile", "re.search", "re.match", "re.findall", "re.sub", "re.split", "re.fullmatch"):
            assert token not in source, path


def _load_detector_utils_module():
    module_path = ROOT / "scripts" / "detector_utils.py"
    spec = importlib.util.spec_from_file_location("detector_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_detector_utils_keeps_explicitly_requested_confounder_class():
    module = _load_detector_utils_module()
    teddy = {"class": "teddy bear", "score": 0.91, "bbox": [10, 10, 50, 50]}

    kept = module.filter_confounded_detections(
        [teddy],
        {"teddy bear": [teddy]},
        required_classes=["dog", "teddy bear"],
    )

    assert kept == [teddy]


def test_detector_utils_classwise_nms_removes_only_overlapping_duplicates():
    module = _load_detector_utils_module()
    detections = [
        {"class": "wine glass", "score": 0.95, "bbox": [10, 10, 110, 210]},
        {"class": "wine glass", "score": 0.90, "bbox": [12, 12, 108, 208]},
        {"class": "wine glass", "score": 0.88, "bbox": [180, 10, 280, 210]},
        {"class": "kite", "score": 0.92, "bbox": [12, 12, 108, 208]},
    ]

    kept = module.classwise_nms(detections, iou_threshold=0.65)

    assert [(row["class"], row["score"]) for row in kept] == [
        ("wine glass", 0.95),
        ("wine glass", 0.88),
        ("kite", 0.92),
    ]
def test_capability_partitioned_count_conflict_is_unknown():
    from gen_harness.inspection import CompositeInspectionRunner

    class Detector(InspectionRunner):
        config = {"name": "detector"}

        def inspect(self, task, program, generation, seed=0):
            return {"count_1": {
                "passed": True,
                "symptom": "pass",
                "hard_evidence_required": True,
                "evidence_status": "confirmed",
            }}

    class Semantic(InspectionRunner):
        config = {"name": "semantic"}

        def inspect(self, task, program, generation, seed=0):
            return {"count_1": {
                "passed": False,
                "symptom": "object_count_mismatch",
                "hard_evidence_required": True,
                "evidence_status": "mismatch",
            }}

    task = GenerationTask.from_dict({
        "task_id": "dual_count",
        "prompt": "a cup",
        "task_family": "compositional",
        "constraints": [{
            "constraint_id": "count_1",
            "text": "one cup",
            "constraint_type": "count",
            "metadata": {"required_objects": [{"class": "cup", "count": 1}]},
        }],
    })
    row = CompositeInspectionRunner(
        [Detector(), Semantic()],
        config={
            "merge_strategy": "capability_partitioned",
            "capability_owners": {"count": "semantic", "count_detector": "detector"},
        },
    ).inspect(task, {}, {})["count_1"]

    assert row["passed"] is False
    assert row["evidence_status"] == "unknown"
    assert row["verifier_conflict"] is True
    assert row["num_independent_evidence_sources"] == 2
    assert row["single_verifier_evidence"] is False


@pytest.mark.parametrize("relation", ["under", "on top of", "beside"])
def test_capability_partition_routes_planar_relations_to_geometry_owner(relation):
    from gen_harness.inspection import CompositeInspectionRunner

    runner = CompositeInspectionRunner([HarnessAwareInspector()], config={
        "merge_strategy": "capability_partitioned",
        "capability_owners": {
            "spatial_2d_detector": "geometry",
            "spatial_2d_open_vocab": "geometry",
            "spatial_depth": "depth",
        },
    })
    constraint = VisualConstraint.from_dict({
        "constraint_id": "spatial_1",
        "text": f"cup {relation} plate",
        "constraint_type": "spatial",
        "metadata": {"spatial_relations": [{"subject": "cup", "relation": relation, "object": "plate"}]},
    })
    rows = runner._capability_partition_rows({
        "geometry": {"passed": True, "symptom": "pass"},
        "depth": {"not_applicable": True},
    }, constraint)

    assert list(rows) == ["geometry"]


def test_capability_partition_routes_depth_relation_to_semantic_owner():
    from gen_harness.inspection import CompositeInspectionRunner

    runner = CompositeInspectionRunner([HarnessAwareInspector()], config={
        "merge_strategy": "capability_partitioned",
        "capability_owners": {
            "spatial_2d_detector": "geometry",
            "spatial_2d_open_vocab": "geometry",
            "spatial_depth": "depth",
        },
    })
    constraint = VisualConstraint.from_dict({
        "constraint_id": "spatial_1",
        "text": "cup behind plate",
        "constraint_type": "spatial",
        "metadata": {"spatial_relations": [{"subject": "cup", "relation": "behind", "object": "plate"}]},
    })
    rows = runner._capability_partition_rows({
        "geometry": {"passed": True, "symptom": "pass"},
        "depth": {"passed": False, "symptom": "spatial_relation_mismatch"},
    }, constraint)

    assert list(rows) == ["depth"]


def test_weakness_miner_rejects_unknown_and_single_verifier_failures():
    def experience(experience_id, verification_result):
        return VisualExperience(
            experience_id=experience_id,
            task_id=experience_id,
            constraint_id="count_1",
            constraint_type="count",
            constraint_text="count objects",
            component_decisions={},
            tool_calls=[],
            visual_result={},
            verification_result=verification_result,
            failure_symptom="object_count_mismatch",
            suspected_component="skills",
        )

    rows = [
        experience("unknown", {"passed": False, "evidence_status": "unknown"}),
        experience("single", {"passed": False, "evidence_status": "mismatch", "single_verifier_evidence": True}),
    ]

    assert WeaknessMiner(min_support=1).mine(rows) == []




def test_patch_validation_expands_single_seed_without_changing_given_seeds():
    from gen_harness.experiments.slow_loop import _ensure_multiple_validation_seeds

    assert _ensure_multiple_validation_seeds([7]) == [7, 8]
    assert _ensure_multiple_validation_seeds([7, 11]) == [7, 11]
def test_composite_inspection_passes_upstream_rows_to_later_inspector():
    from gen_harness.inspection import CompositeInspectionRunner, InspectionRunner

    class Detector(InspectionRunner):
        config = {"name": "detector"}

        def inspect(self, task, program, generation, seed=0):
            return {"attr": {"passed": True, "attribute_color_scores": [{"object": "cup", "bbox": [1, 2, 3, 4]}]}}

    class Consumer(InspectionRunner):
        config = {"name": "consumer"}

        def inspect(self, task, program, generation, seed=0):
            assert generation["upstream_inspection_results"]["detector"]["attr"]["attribute_color_scores"][0]["bbox"] == [1, 2, 3, 4]
            return {"attr": {"passed": True}}

    runner = CompositeInspectionRunner([Detector(), Consumer()])
    task = GenerationTask.from_dict({"task_id": "context", "prompt": "a cup", "constraints": []})

    result = runner.inspect(task, {"contract": {"constraints": []}}, {"image_uri": "unused.png"})

    assert result["attr"]["passed"] is True
def test_flow_grpo_splits_prompt_channels_and_enforces_real_token_budget():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    class Tokenizer:
        def encode(self, text, add_special_tokens=True, truncation=False):
            return list(range(len(text.split()) + 2))

    class Pipe:
        tokenizer = Tokenizer()
        tokenizer_2 = Tokenizer()

    task = GenerationTask.from_dict({"task_id": "prompt_channels", "prompt": "a red cup"})
    backend = FlowGRPOBackend({"clip_max_tokens": 8, "split_prompt_channels": True})
    backend._pipe = Pipe()
    program = {"workflow": {"compiled_prompt": "long t5 semantic detail", "clip_prompt": "red cup", "t5_prompt": "long t5 semantic detail"}}

    prompts = backend._sd3_prompt_kwargs(task, program, program["workflow"]["compiled_prompt"])

    assert prompts == {"prompt": "red cup", "prompt_2": "red cup", "prompt_3": "long t5 semantic detail"}
    assert backend._last_prompt_token_counts == {"tokenizer": 4, "tokenizer_2": 4}
    program["workflow"]["clip_prompt"] = "one two three four five six seven"
    with pytest.raises(ValueError, match="exceeds the model context"):
        backend._sd3_prompt_kwargs(task, program, program["workflow"]["compiled_prompt"])


def test_flow_grpo_budgeting_uses_both_tokenizers_and_never_splits_clauses():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    class WordTokenizer:
        model_max_length = 77

        def encode(self, text, add_special_tokens=True, truncation=False):
            return list(range(len(text.split()) + 2))

    class CharacterTokenizer:
        model_max_length = 77

        def encode(self, text, add_special_tokens=True, truncation=False):
            return list(range(len(text) // 4 + 2))

    class Pipe:
        tokenizer = WordTokenizer()
        tokenizer_2 = CharacterTokenizer()

    task = GenerationTask.from_dict({"task_id": "whole_clause_budget", "prompt": "a flamingo and a turtle"})
    backend = FlowGRPOBackend({"clip_max_tokens": 12, "split_prompt_channels": True})
    backend._pipe = Pipe()
    segments = [
        "a flamingo and a turtle.",
        "Exactly one flamingo and one turtle.",
        "This complete low priority clause cannot fit.",
    ]
    workflow = {
        "compiled_prompt": "full semantic prompt",
        "canonical_prompt": "full semantic prompt",
        "clip_prompt": " ".join(segments),
        "clip_prompt_segments": segments,
        "t5_prompt": "full semantic prompt",
        "prompt_provenance": {"backend_channels": {}, "sha256": {}},
    }

    prompts = backend._sd3_prompt_kwargs(task, {"workflow": workflow}, workflow["compiled_prompt"], clip_max_tokens=12)

    assert prompts["prompt"] == segments[0]
    assert workflow["clip_prompt"] == segments[0]
    assert workflow["clip_prompt_budget"]["dropped_segments"] == segments[1:]
    assert all(count <= 12 for count in backend._last_prompt_token_counts.values())
    assert "Exactly one flamingo" not in prompts["prompt"]


def test_flow_grpo_budgeting_prioritizes_typed_constraints_over_source_prose():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    class WordTokenizer:
        model_max_length = 77

        def encode(self, text, add_special_tokens=True, truncation=False):
            return list(range(len(text.split()) + 2))

    class Pipe:
        tokenizer = WordTokenizer()
        tokenizer_2 = WordTokenizer()

    task = GenerationTask.from_dict({"task_id": "typed_budget", "prompt": "source prose"})
    backend = FlowGRPOBackend({"clip_max_tokens": 18, "split_prompt_channels": True})
    backend._pipe = Pipe()
    segments = [
        "Long decorative source prose that consumes nearly the entire available context window.",
        "Exactly three complete flamingos.",
        "Flamingos behind turtle.",
        "Action: turtle chasing giraffe.",
    ]
    workflow = {
        "clip_prompt": " ".join(segments),
        "clip_prompt_segments": segments,
        "clip_prompt_segment_types": ["source", "object", "spatial", "action"],
    }

    fitted = backend._fit_compiled_clip_segments(task, workflow, clip_max_tokens=18)

    assert fitted == " ".join(segments[1:])
    assert workflow["clip_prompt_budget"]["selected_segment_types"] == ["object", "spatial", "action"]
    assert workflow["clip_prompt_budget"]["dropped_segment_types"] == ["source"]


def test_flow_grpo_t5_budgeting_drops_only_complete_clauses():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    class Tokenizer:
        def encode(self, text, add_special_tokens=True, truncation=False):
            return list(range(len(text.split()) + 2))

    class Pipe:
        tokenizer_3 = Tokenizer()

    backend = FlowGRPOBackend({"t5_max_tokens": 10})
    backend._pipe = Pipe()
    task = GenerationTask.from_dict({"task_id": "t5_whole_clause", "prompt": "three flamingos"})
    workflow = {
        "t5_prompt": "three flamingos. Keep every flamingo complete. Decorative low priority words here.",
        "t5_prompt_segments": [
            "three flamingos.",
            "Keep every flamingo complete.",
            "Decorative low priority words here.",
        ],
        "t5_prompt_segment_types": ["object", "repair", "presentation"],
    }

    fitted = backend._fit_compiled_t5_segments(task, workflow, workflow["t5_prompt"])

    assert fitted == "three flamingos. Keep every flamingo complete."
    assert workflow["t5_prompt_budget"]["token_count"] <= 10
    assert workflow["t5_prompt_budget"]["dropped_segments"] == ["Decorative low priority words here."]
    assert "flamingos" in fitted


def test_flow_grpo_t5_budget_reserves_each_constraint_family_before_extra_object_clauses():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    class Tokenizer:
        def encode(self, text, add_special_tokens=True, truncation=False):
            return list(range(len(text.split()) + 2))

    class Pipe:
        tokenizer_3 = Tokenizer()

    backend = FlowGRPOBackend({"t5_max_tokens": 24})
    backend._pipe = Pipe()
    task = GenerationTask.from_dict({"task_id": "t5_family_seats", "prompt": "many objects acting"})
    workflow = {
        "t5_prompt": "unbudgeted",
        "t5_prompt_segments": [
            "Exact inventory has three object groups.",
            "Extra long object layout consumes many remaining words here.",
            "Subjects clearly behind targets.",
            "Subjects actively chasing targets.",
            "Targets have striped surfaces.",
        ],
        "t5_prompt_segment_types": ["object", "object", "spatial", "action", "attribute"],
    }

    backend._fit_compiled_t5_segments(task, workflow, workflow["t5_prompt"])

    selected = set(workflow["t5_prompt_budget"]["selected_segment_types"])
    assert {"object", "spatial", "action", "attribute"} <= selected
    assert workflow["t5_prompt_budget"]["unrepresented_semantic_families"] == []


def test_flow_grpo_clip_budget_reserves_concrete_repair_layout():
    from gen_harness.adapters.flow_grpo_backend import FlowGRPOBackend

    class WordTokenizer:
        model_max_length = 77

        def encode(self, text, add_special_tokens=True, truncation=False):
            return list(range(len(text.split()) + 2))

    class Pipe:
        tokenizer = WordTokenizer()
        tokenizer_2 = WordTokenizer()

    backend = FlowGRPOBackend({"clip_max_tokens": 26, "split_prompt_channels": True})
    backend._pipe = Pipe()
    task = GenerationTask.from_dict({"task_id": "repair_layout_seat", "prompt": "seven green pastries"})
    workflow = {
        "clip_prompt": "unbudgeted",
        "clip_prompt_segments": [
            "seven green pastries from the source prompt.",
            "Show exactly seven complete pastries with no extras.",
            "Make every pastry surface visibly solid green.",
            "Arrange seven pastries in two separated rows with gaps.",
        ],
        "clip_prompt_segment_types": ["source", "repair_inventory", "repair_attribute", "repair_layout"],
    }

    backend._fit_compiled_clip_segments(task, workflow, clip_max_tokens=26)

    selected = set(workflow["clip_prompt_budget"]["selected_segment_types"])
    assert {"repair_inventory", "repair_attribute", "repair_layout"} <= selected
    assert "source" not in selected


def test_fast_loop_failed_candidate_selection_protects_weakest_constraint():
    from gen_harness.fast_loop import FastLoopController

    controller = object.__new__(FastLoopController)
    balanced = {
        "count": {"passed": False, "selection_score": 0.45},
        "attribute": {"passed": False, "selection_score": 0.55},
    }
    biased = {
        "count": {"passed": False, "selection_score": 0.05},
        "attribute": {"passed": False, "selection_score": 1.0},
    }

    assert not controller._attempt_is_better(False, 0.525, biased, 0.50, balanced)
    assert controller._attempt_is_better(False, 0.50, balanced, 0.525, biased)


def test_fast_loop_missing_category_is_worse_than_soft_low_score():
    from gen_harness.fast_loop import FastLoopController

    controller = object.__new__(FastLoopController)
    complete_but_imperfect = {
        "objects": {"passed": False, "selection_score": 0.70, "count_mismatches": [{"class": "toy"}]},
        "attribute": {"passed": False, "selection_score": 0.19},
        "spatial": {"passed": False, "selection_score": 0.50},
    }
    missing_category = {
        "objects": {
            "passed": False,
            "selection_score": 0.25,
            "missing_required_objects": [{"class": "toy"}],
        },
        "attribute": {"passed": False, "selection_score": 0.23},
        "spatial": {
            "passed": False,
            "selection_score": 0.25,
            "missing_required_objects": [{"class": "toy"}],
        },
    }

    assert controller._hard_omission_count(missing_category) == 1
    assert controller._hard_omission_count(complete_but_imperfect) == 0
    assert not controller._attempt_is_better(False, 0.24, missing_category, 0.46, complete_but_imperfect)


def test_fast_loop_first_inspected_failure_establishes_baseline_even_with_omission():
    from gen_harness.fast_loop import FastLoopController

    controller = object.__new__(FastLoopController)
    first = {
        "objects": {
            "passed": False,
            "selection_score": 0.0,
            "missing_required_objects": [{"class": "bagel", "expected": 3, "found": 0}],
        }
    }

    assert controller._attempt_is_better(False, 0.0, first, -1.0, {})
    assert not controller._attempt_is_better(False, -1.0, {}, -1.0, {})


def test_fast_loop_hard_inventory_precedes_soft_semantic_minimax():
    from gen_harness.fast_loop import FastLoopController

    controller = object.__new__(FastLoopController)
    nearly_exact = {
        "objects": {
            "passed": False,
            "selection_score": 0.94,
            "object_counts": {"rabbit": 3, "bagel": 5, "chair": 1},
            "count_mismatches": [{"class": "bagel", "expected": 6, "found": 5}],
        },
        "spatial": {"passed": False, "selection_score": 0.07},
    }
    count_regression = {
        "objects": {
            "passed": False,
            "selection_score": 0.61,
            "object_counts": {"rabbit": 6, "bagel": 5, "chair": 1},
            "count_mismatches": [
                {"class": "rabbit", "expected": 3, "found": 6},
                {"class": "bagel", "expected": 6, "found": 5},
            ],
        },
        "spatial": {"passed": False, "selection_score": 0.18},
    }

    assert controller._attempt_is_better(False, 0.44, nearly_exact, 0.38, count_regression)
    assert not controller._attempt_is_better(False, 0.38, count_regression, 0.44, nearly_exact)


def test_fast_loop_inventory_distance_penalizes_over_and_under_counts_monotonically():
    from gen_harness.fast_loop import FastLoopController

    controller = object.__new__(FastLoopController)
    closer = {
        "objects": {
            "object_counts": {"guitar": 2, "turtle": 5, "zebra": 1},
            "count_mismatches": [
                {"class": "guitar", "expected": 1, "found": 2},
                {"class": "turtle", "expected": 7, "found": 5},
                {"class": "zebra", "expected": 2, "found": 1},
            ],
        }
    }
    farther = {
        "objects": {
            "object_counts": {"guitar": 1, "turtle": 12, "zebra": 1},
            "count_mismatches": [
                {"class": "turtle", "expected": 7, "found": 12},
                {"class": "zebra", "expected": 2, "found": 1},
            ],
        }
    }

    assert controller._hard_inventory_error(closer) == 4
    assert controller._hard_inventory_error(farther) == 6


def test_fast_loop_combined_score_penalizes_failed_attribute_subresult():
    from gen_harness.fast_loop import FastLoopController

    controller = object.__new__(FastLoopController)
    merged_attribute_failure = {
        "passed": False,
        "object_counts": {"counter top": 1, "sink": 1},
        "selection_score": 0.9,
        "inspection_subresults": {
            "detector": {
                "passed": False,
                "symptom": "attribute_binding_mismatch",
                "selection_score": 0.76,
                "attribute_binding_mismatches": [{"object": "counter top", "attribute": "grey"}],
            },
            "semantic": {"passed": False, "symptom": "missing_inspection", "selection_score": 1.0},
        },
    }
    weaker_attribute_failure = {
        "passed": False,
        "object_counts": {"counter top": 1, "sink": 1},
        "selection_score": 0.55,
        "inspection_subresults": {
            "detector": {
                "passed": False,
                "symptom": "attribute_binding_mismatch",
                "selection_score": 0.55,
                "attribute_binding_mismatches": [{"object": "sink", "attribute": "black"}],
            }
        },
    }

    assert controller._combined_constraint_score(merged_attribute_failure) == pytest.approx(0.76)
    assert controller._weakest_constraint_score({"attribute": merged_attribute_failure}) == pytest.approx(0.76)
    assert controller._score({"attribute": merged_attribute_failure}) == pytest.approx(0.76)
    assert not controller._attempt_is_better(
        False,
        1.0,
        {"attribute": weaker_attribute_failure},
        0.1,
        {"attribute": merged_attribute_failure},
    )


@pytest.mark.parametrize(
    ("source", "canonical"),
    [("under", "below"), ("underneath", "below"), ("beneath", "below"),
     ("on top of", "above"), ("atop", "above"), ("over", "above")],
)
def test_spatial_relation_aliases_compile_to_executable_axes(source, canonical):
    from gen_harness.spatial_prompt import normalize_relation, spatial_repair_clause

    assert normalize_relation(source) == canonical
    clause = spatial_repair_clause("bear", source, "toy", subject_count=4, target_count=6)
    assert f"({canonical})" in clause
    assert "all 4 bear" in clause and "all 6 toy" in clause
    assert ("below" in clause and "above" in clause) if canonical == "below" else ("above" in clause and "below" in clause)














def test_open_source_safety_scan_prunes_excluded_directories(tmp_path):
    from gen_harness.safety import scan_open_source_safety

    (tmp_path / "runs").mkdir()
    excluded_secret_text = "api_" + "key = 'SHOULD_NOT_BE_SCANNED_1234567890'"
    (tmp_path / "runs" / "secret.txt").write_text(excluded_secret_text, encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "safe.py").write_text("VALUE = 'ok'\n", encoding="utf-8")

    findings = scan_open_source_safety(tmp_path, include_ignored_local=False)

    assert findings == []


def test_open_source_safety_scan_flags_local_absolute_paths(tmp_path):
    from gen_harness.safety import scan_open_source_safety

    (tmp_path / "src").mkdir()
    local_path = "/" + "/".join(["data", "luojiabin", "envs", "genharness", "bin", "python"])
    (tmp_path / "src" / "config.py").write_text(f"PYTHON = '{local_path}'\n", encoding="utf-8")

    findings = scan_open_source_safety(tmp_path, include_ignored_local=False)

    assert [(f.path, f.kind) for f in findings] == [("src/config.py", "local_absolute_path")]


def test_open_source_safety_scan_flags_common_secret_shapes_without_regex(tmp_path):
    from gen_harness.safety import scan_open_source_safety

    (tmp_path / "src").mkdir()
    image_payload = "A" * 520
    auth_header = "Authori" + "zation: "
    bearer_prefix = "Bea" + "rer "
    (tmp_path / "src" / "unsafe.txt").write_text(
        "\n".join(
            [
                "OPENAI = 'sk-" + "a" * 48 + "'",
                "GOOGLE = 'AIza" + "b" * 36 + "'",
                auth_header + bearer_prefix + "MIXEDCaseValue_1234567890",
                "api_" + "key = 'plain_value_1234567890'",
                "image='data:image/png;base64," + image_payload + "'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    findings = scan_open_source_safety(tmp_path, include_ignored_local=False)
    kinds = [finding.kind for finding in findings]

    assert kinds.count("possible_secret") == 4
    assert "inline_image_data" in kinds


def test_t2icompbenchpp_inspection_config_enables_internal_color_verification():
    config = json.loads(Path("configs/inspection/composite_owlv2_nvila_t2icompbenchpp.json").read_text(encoding="utf-8"))
    detector = config["inspectors"][0]
    config_text = json.dumps(config).lower()

    assert detector["attribute_color_verification"] is True
    assert config["capability_owners"]["attribute"] == detector["name"]
    assert "official_score" not in config_text
    assert "selected_score" not in config_text
    assert "fallback_image" not in config_text


def test_nvila_prompt_alignment_checks_prompt_noun_phrase_coverage():
    spec = importlib.util.spec_from_file_location(
        "nvila_semantic_verifier",
        ROOT / "scripts" / "nvila_semantic_verifier.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    checks = module.semantic_checks(
        {"task": {"prompt": "A cylindrical bottle with a square lid."}},
        {
            "constraint_type": "prompt_alignment",
            "text": "A cylindrical bottle with a square lid.",
            "metadata": {
                "prompt_noun_phrases": ["cylindrical bottle", "square lid", "square lid"],
            },
        },
    )

    coverage = [row for row in checks if row["kind"] == "prompt_noun_phrase_coverage"]
    assert [row["target"] for row in coverage] == ["cylindrical bottle", "square lid"]
    assert "large enough to inspect" in coverage[0]["question"]
    assert "official" not in json.dumps(checks).lower()








def test_t2icompbenchpp_builds_prompt_only_tasks(tmp_path):
    from gen_harness.evaluation import build_t2icompbenchpp_tasks

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "color_val.txt").write_text("a red cup and a blue bowl\n", encoding="utf-8")
    (dataset / "complex_val.txt").write_text("The red hat was on top of the brown rack.\n", encoding="utf-8")
    (dataset / "complex_val_spatial.txt").write_text("The red hat was on top of the brown rack.\n", encoding="utf-8")
    output = tmp_path / "tasks.jsonl"

    tasks = build_t2icompbenchpp_tasks(
        dataset,
        output,
        categories=["color", "complex"],
    )

    assert [row["task_id"] for row in tasks] == [
        "t2icompbenchpp_color_val_0000",
        "t2icompbenchpp_complex_val_0000",
    ]
    assert tasks[0]["metadata"] == {
        "split": "val",
        "benchmark": "t2icompbenchpp",
        "category": "color",
        "source_index": 0,
        "generation_input": "prompt_only_no_official_eval_outputs",
    }
    assert tasks[1]["metadata"]["complex_subtype"] == "spatial"
    assert tasks[0]["constraints"] == []
    assert "score" not in str(tasks).lower()


def test_t2icompbenchpp_build_tasks_preserves_duplicate_prompts_by_source_index(tmp_path):
    from gen_harness.evaluation import build_t2icompbenchpp_tasks

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "color_val.txt").write_text(
        "a red cup and a blue bowl\n"
        "a red cup and a blue bowl\n",
        encoding="utf-8",
    )
    output = tmp_path / "tasks.jsonl"

    tasks = build_t2icompbenchpp_tasks(
        dataset,
        output,
        categories=["color"],
    )

    assert [row["metadata"]["source_index"] for row in tasks] == [0, 1]
    assert [row["task_id"] for row in tasks] == [
        "t2icompbenchpp_color_val_0000",
        "t2icompbenchpp_color_val_0001",
    ]
    assert tasks[0]["prompt"] == tasks[1]["prompt"]


def test_t2icompbenchpp_prepare_eval_manifest_links_images(tmp_path):
    from gen_harness.evaluation import prepare_t2icompbenchpp_eval_manifest
    from gen_harness.io import write_jsonl

    image = tmp_path / "image.png"
    image.write_bytes(b"not-a-real-png")
    tasks = tmp_path / "tasks.jsonl"
    experiences = tmp_path / "experiences.jsonl"
    write_jsonl(tasks, [{
        "task_id": "t2icompbenchpp_color_val_0000",
        "prompt": "a red cup and a blue bowl",
        "task_family": "compositional",
        "references": {},
        "constraints": [],
        "metadata": {
            "benchmark": "t2icompbenchpp",
            "category": "color",
            "source_index": 7,
        },
    }])
    write_jsonl(experiences, [{
        "method": "gen_harness",
        "task_id": "t2icompbenchpp_color_val_0000",
        "image_uri": str(image),
        "generation": {"image_path": str(image), "fallback_used": False},
    }])

    summary = prepare_t2icompbenchpp_eval_manifest(
        tasks,
        experiences,
        tmp_path / "manifests",
        tmp_path / "samples",
        symlink=False,
    )

    assert summary["total_rows"] == 1
    assert summary["category_counts"]["color"] == 1
    rows = (tmp_path / "manifests" / "color.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert "official scores are not used" not in rows[0].lower()
    assert (tmp_path / "samples" / "color" / "0000.png").exists()


def test_t2icompbenchpp_smoke_builder_is_prompt_only_and_keeps_hard_prompts(tmp_path):
    from gen_harness.evaluation import build_t2icompbenchpp_smoke
    from gen_harness.io import write_jsonl

    tasks = []
    categories = ("color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex")
    for category in categories:
        for index in range(6):
            prompt = f"three red wooden cups next to two blue bowls marker {category} {index}"
            if category == "complex" and index == 0:
                prompt = "The gentle, rolling hills of the countryside were a peaceful escape from the hustle and bustle of the city."
            if category == "complex" and index == 1:
                prompt = "The red hat was on top of the brown coat rack."
            tasks.append({
                "task_id": f"t2icompbenchpp_{category}_val_{index:04d}",
                "prompt": prompt,
                "task_family": "compositional",
                "references": {},
                "constraints": [],
                "metadata": {
                    "benchmark": "t2icompbenchpp",
                    "category": category,
                    "source_index": index,
                    "complex_subtype": "spatial" if category == "complex" else "none",
                    "generation_input": "prompt_only_no_official_eval_outputs",
                },
            })
    tasks_path = tmp_path / "tasks.jsonl"
    write_jsonl(tasks_path, tasks)

    manifest = build_t2icompbenchpp_smoke(
        tasks_path,
        tmp_path / "smoke.jsonl",
        tmp_path / "manifest.json",
        total=16,
        quotas={category: 2 for category in categories},
        seed="unit-test",
    )

    assert manifest["total"] == 16
    assert manifest["uses_official_eval_outputs"] is False
    assert manifest["generated_images_used"] is False
    assert manifest["no_replacement_after_generation"] is True
    assert manifest["selection_inputs"] == ["prompt", "category", "complex_subtype"]
    assert "evaluator_degenerate_in_main" not in manifest
    assert "diagnostic_degenerate_count" not in manifest
    selected = [json.loads(line) for line in (tmp_path / "smoke.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("smoke_profile" in row["metadata"] for row in selected)
    assert "score" not in json.dumps(selected).lower()


def test_t2icompbenchpp_formal_path_has_no_posthoc_selection_fallback_regex_or_taskid_branches():
    leakage_guard_paths = [
        Path("gen_harness/evaluation/t2icompbenchpp.py"),
        Path("gen_harness/experiments/t2icompbenchpp_generate.py"),
        Path("gen_harness/components/policy.py"),
        Path("gen_harness/components/skills.py"),
        Path("gen_harness/components/middleware.py"),
        Path("examples/visual_harness/middleware/tool_router.json"),
        Path("examples/visual_harness/skills/skills.json"),
        Path("configs/backends/t2icompbenchpp_flow_grpo.json"),
        Path("configs/inspection/composite_owlv2_nvila_t2icompbenchpp.json"),
    ]
    route_guard_paths = [
        Path("gen_harness/experiments/t2icompbenchpp_generate.py"),
        Path("gen_harness/components/policy.py"),
        Path("gen_harness/components/skills.py"),
        Path("gen_harness/components/middleware.py"),
        Path("examples/visual_harness/middleware/tool_router.json"),
        Path("examples/visual_harness/skills/skills.json"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in leakage_guard_paths)
    lowered = source.lower()
    source_lines = source.splitlines()

    leakage_forbidden = [
        "uses_official_eval_outputs\": true",
        "official_evaluator_used_for_generation_or_selection\": true",
        "posthoc_candidate_selection_allowed\": true",
        "color_val_",
        "shape_val_",
        "texture_val_",
        "spatial_val_",
        "3d_spatial_val_",
        "numeracy_val_",
        "non_spatial_val_",
        "complex_val_",
    ]
    for token in leakage_forbidden:
        assert token not in lowered
    assert not any(line.strip() == "import re" for line in source_lines)
    assert not any("re.compile" in line or "re.match" in line or "re.search" in line for line in source_lines)

    route_source = "\n".join(path.read_text(encoding="utf-8") for path in route_guard_paths).lower()
    route_forbidden = [
        "scene_preserving_categories",
        "scene_native_categories",
        "evaluator_card_layout_categories",
        "t2icompbenchpp_scene_template_category",
        "routes.get(category",
        "metadata.get(\"category\")",
        "metadata.get('category')",
        "task_metadata.get(\"category\")",
        "task_metadata.get('category')",
        "metadata.get(\"source_index\")",
        "metadata.get('source_index')",
        "task_metadata.get(\"source_index\")",
        "task_metadata.get('source_index')",
        ".get(\"selected_score\")",
        ".get('selected_score')",
        ".get(\"selected_result\")",
        ".get('selected_result')",
        ".get(\"official_score\")",
        ".get('official_score')",
        ".get(\"fallback_image\")",
        ".get('fallback_image')",
        "\"fallback_image\": true",
        "'fallback_image': true",
        "texture_terms",
        "shape_terms",
        "routes.get(\"default\")",
        "routes.get('default')",
    ]
    for token in route_forbidden:
        assert token not in route_source






def test_t2icompbenchpp_run_dirs_have_no_local_executable_code():
    run_root = Path("runs")
    if not run_root.exists():
        return
    stale = [
        str(path)
        for path in run_root.glob("t2icompbenchpp_*/*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    ]
    assert stale == []


def test_t2icompbenchpp_smoke_builder_selection_ignores_source_index(tmp_path):
    from gen_harness.evaluation import build_t2icompbenchpp_smoke
    from gen_harness.io import write_jsonl

    categories = ("color", "shape", "texture", "spatial", "3d_spatial", "numeracy", "non_spatial", "complex")
    task_sets = []
    for offset in (0, 100):
        tasks = []
        for category in categories:
            for index in range(4):
                source_index = offset + index
                tasks.append({
                    "task_id": f"t2icompbenchpp_{category}_val_{source_index:04d}",
                    "prompt": f"three red wooden cups next to two blue bowls marker {category} prompt{index}",
                    "task_family": "compositional",
                    "references": {},
                    "constraints": [],
                    "metadata": {
                        "benchmark": "t2icompbenchpp",
                        "category": category,
                        "source_index": source_index,
                        "complex_subtype": "action" if category == "complex" else "none",
                        "generation_input": "prompt_only_no_official_eval_outputs",
                    },
                })
        tasks_path = tmp_path / f"tasks_{offset}.jsonl"
        write_jsonl(tasks_path, tasks)
        task_sets.append(tasks_path)

    first = build_t2icompbenchpp_smoke(
        task_sets[0],
        tmp_path / "smoke_0.jsonl",
        tmp_path / "manifest_0.json",
        total=16,
        quotas={category: 2 for category in categories},
        seed="unit-test-no-source-index",
    )
    second = build_t2icompbenchpp_smoke(
        task_sets[1],
        tmp_path / "smoke_100.jsonl",
        tmp_path / "manifest_100.json",
        total=16,
        quotas={category: 2 for category in categories},
        seed="unit-test-no-source-index",
    )

    first_prompts = [json.loads(line)["prompt"] for line in (tmp_path / "smoke_0.jsonl").read_text(encoding="utf-8").splitlines()]
    second_prompts = [json.loads(line)["prompt"] for line in (tmp_path / "smoke_100.jsonl").read_text(encoding="utf-8").splitlines()]
    assert first["selection_inputs"] == ["prompt", "category", "complex_subtype"]
    assert second["selection_inputs"] == ["prompt", "category", "complex_subtype"]
    assert first_prompts == second_prompts


def test_partiprompts_evolution_split_filters_benchmark_overlap_and_writes_manifest(tmp_path):
    source = tmp_path / "partiprompts.tsv"
    source.write_text(
        "\n".join([
            "Prompt\tCategory\tChallenge\tNote",
            "benchmark prompt\tObjects\tBasic\tmust be filtered",
            "alpha prompt\tObjects\tBasic\t",
            "beta prompt\tObjects\tBasic\t",
            "alpha prompt\tObjects\tBasic\tduplicate",
            "gamma prompt\tScenes\tComplex\t",
            "delta prompt\tScenes\tComplex\t",
        ]),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl(
        benchmark,
        [{
            "task_id": "bench_0000",
            "prompt": "Benchmark prompt",
            "task_family": "compositional",
            "references": {},
            "constraints": [],
            "metadata": {"benchmark": "unit"},
        }],
    )

    manifest = build_partiprompts_evolution_split(
        str(source),
        tmp_path / "out",
        benchmark_tasks=[benchmark],
        seed=7,
        target_count=2,
        heldout_count=1,
        preservation_count=1,
    )

    target = [json.loads(line) for line in (tmp_path / "out/partiprompts_p2_target500.jsonl").read_text(encoding="utf-8").splitlines()]
    heldout = [json.loads(line) for line in (tmp_path / "out/partiprompts_p2_heldout100.jsonl").read_text(encoding="utf-8").splitlines()]
    preservation = [json.loads(line) for line in (tmp_path / "out/partiprompts_p2_preservation100.jsonl").read_text(encoding="utf-8").splitlines()]
    prompts = [row["prompt"].lower() for row in target + heldout + preservation]
    assert len(target) == 2
    assert len(heldout) == 1
    assert len(preservation) == 1
    assert "benchmark prompt" not in prompts
    assert len(set(prompts)) == 4
    assert manifest["benchmark_overlap_removed"] == 1
    assert manifest["outputs"]["target"].endswith("partiprompts_p2_target500.jsonl")
