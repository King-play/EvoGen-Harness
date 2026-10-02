import importlib.util
import json
from pathlib import Path

from gen_harness.inspection import DetectorObjectRequirementInspection
from gen_harness.schema import VisualConstraint


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_owlv2_selection_applies_classwise_nms_and_uncertainty_band():
    module = load_script("owlv2_open_vocab_detector.py")
    detections, uncertain, best = module.select_detections(
        boxes=[[0, 0, 100, 100], [2, 2, 98, 98], [150, 0, 220, 80]],
        scores=[0.91, 0.72, 0.12],
        labels=[0, 0, 1],
        queries=["cat", "cup"],
        threshold=0.20,
        uncertainty_threshold=0.08,
        nms_threshold=0.55,
    )
    assert [row["class"] for row in detections] == ["cat"]
    assert detections[0]["score"] == 0.91
    assert uncertain == ["cup"]
    assert best == {"cat": 0.91, "cup": 0.12}


def test_owlv2_selection_suppresses_nested_part_boxes_for_counting():
    module = load_script("owlv2_open_vocab_detector.py")
    detections, _, _ = module.select_detections(
        boxes=[[10, 10, 210, 110], [40, 40, 180, 100], [300, 10, 500, 110]],
        scores=[0.9, 0.7, 0.8],
        labels=[0, 0, 0],
        queries=["trumpet"],
        threshold=0.2,
        uncertainty_threshold=0.08,
        nms_threshold=0.55,
    )

    assert len(detections) == 2


def test_detector_uncertainty_becomes_unknown_evidence():
    inspector = DetectorObjectRequirementInspection({"name": "owlv2", "hard_evidence_required": True})
    constraint = VisualConstraint.from_dict({
        "constraint_id": "count_0",
        "constraint_type": "count",
        "text": "one cup",
        "verification_rule": "exact count",
        "metadata": {"required_objects": [{"class": "cup", "count": 1}], "count_mode": "exact"},
    })
    result = {"count_0": {"passed": False, "symptom": "missing_required_object", "confidence": 0.0}}
    inspector._mark_uncertain_constraints_unknown(result, [constraint], {"uncertain_classes": ["cup"]})
    inspector._stamp_detector_evidence_status(result)
    assert result["count_0"]["evidence_status"] == "unknown"
    assert result["count_0"]["symptom"] == "missing_inspection"
    assert result["count_0"]["uncertain_classes"] == ["cup"]


def test_nvila_probability_gate_has_confirmed_mismatch_and_unknown_states():
    module = load_script("nvila_semantic_verifier.py")
    confirmed = module.aggregate_result(
        [{"yes_probability": 0.96, "no_probability": 0.04}], "attribute", 0.9, 0.9
    )
    mismatch = module.aggregate_result(
        [{"yes_probability": 0.03, "no_probability": 0.97}], "attribute", 0.9, 0.9
    )
    unknown = module.aggregate_result(
        [{"yes_probability": 0.62, "no_probability": 0.38}], "attribute", 0.9, 0.9
    )
    assert confirmed["passed"] is True and confirmed["evidence_status"] == "confirmed"
    assert mismatch["passed"] is False and mismatch["evidence_status"] == "mismatch"
    assert unknown["passed"] is False and unknown["evidence_status"] == "unknown"
    assert unknown["decision_state"] == "abstain"
    assert unknown["selection_score"] == 0.62
    assert unknown["hard_evidence_required"] is False


def test_nvila_action_and_depth_abstentions_remain_hard():
    module = load_script("nvila_semantic_verifier.py")
    check = [{"yes_probability": 0.62, "no_probability": 0.38}]
    assert module.aggregate_result(check, "action", 0.9, 0.9)["hard_evidence_required"] is True
    assert module.aggregate_result(check, "spatial", 0.9, 0.9)["hard_evidence_required"] is True


def test_nvila_ask_disables_sampling():
    module = load_script("nvila_semantic_verifier.py")

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    class Torch:
        def inference_mode(self):
            return InferenceMode()

    class Scalar:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return self.value

    class Logits:
        ndim = 1

        def __getitem__(self, index):
            return Scalar({1: 2.0, 2: 0.0}[index])

    class Model:
        def __init__(self):
            self.do_sample = None

        def generate_content(self, payload, do_sample=True):
            self.do_sample = do_sample
            return "yes", [Logits()]

    model = Model()
    result = module.ask(object(), "Is it correct?", {
        "model": model,
        "torch": Torch(),
        "yes_id": 1,
        "no_id": 2,
    })
    assert model.do_sample is False
    assert result["yes_probability"] > result["no_probability"]


def test_nvila_only_owns_depth_spatial_checks():
    module = load_script("nvila_semantic_verifier.py")
    payload = {"task": {"prompt": "a lamp left of a pizza"}}
    planar = {
        "constraint_type": "spatial",
        "metadata": {"spatial_relations": [{"subject": "lamp", "relation": "left of", "object": "pizza"}]},
    }
    depth = {
        "constraint_type": "spatial",
        "metadata": {"spatial_relations": [{"subject": "lamp", "relation": "in front of", "object": "pizza"}]},
    }
    assert module.semantic_checks(payload, planar) == []
    assert module.semantic_checks(payload, depth)[0]["kind"] == "depth"
    assert [row["kind"] for row in module.semantic_checks(payload, depth)] == ["depth", "role_distinctness"]


def test_nvila_attribute_checks_binding_location_not_only_scene_presence():
    module = load_script("nvila_semantic_verifier.py")
    payload = {
        "constraints": [
            {
                "constraint_type": "count",
                "metadata": {
                    "required_objects": [
                        {"class": "croissant", "count": 1},
                        {"class": "plate", "count": 1},
                    ]
                },
            }
        ]
    }
    checks = module.semantic_checks(payload, {
        "constraint_type": "attribute",
        "metadata": {"attribute_bindings": [{"object": "croissant", "attribute": "spotted"}]},
    })
    assert [row["kind"] for row in checks] == ["attribute", "attribute_exclusivity"]
    assert checks[0]["expected_answer"] == "yes"
    assert checks[1]["expected_answer"] == "no"
    assert checks[1]["subject"] == "plate"


def test_strong_verifier_config_has_fixed_capability_owners(monkeypatch):
    config_path = ROOT / "configs" / "inspection" / "composite_owlv2_nvila_strong.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    owners = config["capability_owners"]
    assert config["merge_strategy"] == "capability_partitioned"
    assert owners["count"] == "owlv2_base_open_vocab_geometry"
    assert owners["spatial_2d_detector"] == "owlv2_base_open_vocab_geometry"
    assert owners["attribute"] == "nvila_lite_2b_probability_verifier"
    assert owners["spatial_depth"] == "nvila_lite_2b_probability_verifier"
