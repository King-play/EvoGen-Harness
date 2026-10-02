from PIL import Image

from gen_harness.adapters.scene_assembly_backend import SceneAssemblyBackend
from gen_harness.spatial_prompt import normalize_relation


def test_scene_assembly_layout_propagates_chained_spatial_relations():
    backend = object.__new__(SceneAssemblyBackend)
    backend.canvas_size = 1024
    inventory = [
        {"class": "candle", "count": 7},
        {"class": "toy", "count": 4},
        {"class": "cookie", "count": 5},
    ]
    spatial = [
        {"subject": "candle", "relation": "right of", "object": "toy"},
        {"subject": "toy", "relation": "above", "object": "cookie"},
    ]

    boxes = backend._category_layout(inventory, spatial)
    candle = boxes["candle"]
    toy = boxes["toy"]
    cookie = boxes["cookie"]
    candle_center = ((candle[0] + candle[2]) / 2, (candle[1] + candle[3]) / 2)
    toy_center = ((toy[0] + toy[2]) / 2, (toy[1] + toy[3]) / 2)
    cookie_center = ((cookie[0] + cookie[2]) / 2, (cookie[1] + cookie[3]) / 2)

    assert candle_center[0] > toy_center[0]
    assert toy_center[1] < cookie_center[1]
    assert abs(candle_center[1] - toy_center[1]) <= 1
    assert abs(toy_center[0] - cookie_center[0]) <= 1


def test_spatial_relation_aliases_share_the_executable_ontology():
    assert normalize_relation("on top of") == "above"
    assert normalize_relation("underneath") == "below"


def test_partial_order_keeps_siblings_on_the_same_axis_level():
    backend = object.__new__(SceneAssemblyBackend)
    backend.canvas_size = 1024
    inventory = [
        {"class": "dog", "count": 7},
        {"class": "bird", "count": 7},
        {"class": "flower", "count": 1},
    ]
    spatial = [
        {"subject": "dog", "relation": "right of", "object": "bird"},
        {"subject": "dog", "relation": "below", "object": "flower"},
        {"subject": "bird", "relation": "below", "object": "flower"},
    ]

    boxes = backend._category_layout(inventory, spatial)
    centers = {
        name: ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        for name, box in boxes.items()
    }

    assert centers["dog"][0] > centers["bird"][0]
    assert abs(centers["dog"][1] - centers["bird"][1]) <= 1
    assert centers["dog"][1] > centers["flower"][1]
    assert centers["bird"][1] > centers["flower"][1]


def test_action_asset_prompts_preserve_object_attributes():
    subject_prompt, target_prompt = SceneAssemblyBackend._action_asset_prompts(
        "dog",
        "turtle",
        "chasing",
        subject_attributes=["striped"],
        target_attributes=["brown"],
    )

    assert "stripes" in subject_prompt
    assert "brown" in target_prompt
    assert "no other objects" in subject_prompt
    assert "no other objects" in target_prompt


def test_repair_clause_does_not_cross_contaminate_object_source_prompts():
    program = {
        "fast_loop_repair": {
            "patch": {
                "operation": "append_prompt_clause",
                "value": (
                    "Keep the plastic koalas unmistakable and keep the striped birds "
                    "visually distinct."
                ),
            }
        }
    }

    assert SceneAssemblyBackend._transient_repair_clause(
        program,
        "koala",
        inventory_names=["koala", "bird"],
    ) == ""
    assert SceneAssemblyBackend._transient_repair_clause(
        program,
        "bird",
        inventory_names=["koala", "bird"],
    ) == ""


def test_same_depth_category_lanes_do_not_overlap():
    backend = object.__new__(SceneAssemblyBackend)
    backend.canvas_size = 1024
    boxes = backend._category_layout(
        [
            {"class": "cat", "count": 6},
            {"class": "sheep", "count": 7},
            {"class": "guitar", "count": 2},
        ],
        [
            {"subject": "cat", "relation": "below", "object": "guitar"},
            {"subject": "sheep", "relation": "below", "object": "guitar"},
        ],
    )

    cat, sheep = boxes["cat"], boxes["sheep"]
    assert min(cat[2], sheep[2]) <= max(cat[0], sheep[0])


def test_repeated_instances_have_clear_cell_gaps():
    backend = object.__new__(SceneAssemblyBackend)
    source = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    canvas = Image.new("RGB", (1024, 1024), "white")
    placements = backend._paste_instances(
        canvas,
        source,
        7,
        (50, 50, 500, 500),
        0,
        attributes=["checkered"],
    )
    first_row = placements[:4]
    assert all(left["x"] + left["width"] <= right["x"] for left, right in zip(first_row, first_row[1:]))


def test_stone_is_compiled_into_the_source_asset_prompt():
    prompt = SceneAssemblyBackend._source_asset_prompt("motorcycle", ["stone"])
    assert "stone motorcycle" in prompt
    assert "solid natural stone" in prompt
