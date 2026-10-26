from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from ..backend import VisualBackend
from ..schema import GenerationTask
from ..tool_payloads import public_task_row


class SceneAssemblyBackend(VisualBackend):
    """Deterministic scene assembly around a frozen source image generator.

    The source backend generates fresh object assets from public prompt-derived
    contracts.  This wrapper only performs auditable layout and alpha
    compositing; it never calls an image LLM, benchmark labels, fallback images,
    or official evaluator feedback.
    """

    def __init__(self, config: Dict[str, Any]):
        from ..backends import build_backend_from_data, formal_backend_config_error

        self.config = dict(config)
        self.tool_name = "scene_assembly"
        source_config = dict(self.config.get("source_backend") or {})
        if not source_config:
            source_config = {
                key: value
                for key, value in self.config.items()
                if key not in {"type", "name", "image_cache_dir", "trace_cache_dir", "source_backend"}
            }
            source_config["type"] = "flow_grpo"
        if source_config.get("type") == "scene_assembly":
            raise ValueError("scene_assembly source_backend may not recursively be scene_assembly")
        error = formal_backend_config_error(source_config)
        if error:
            raise ValueError(f"scene_assembly source_backend is not a valid frozen generator: {error}")
        self._reject_image_llm_source(source_config)
        self.source_backend = build_backend_from_data(source_config)
        self.image_cache_dir = Path(self.config.get("image_cache_dir", "runs/scene_assembly_cache"))
        self.trace_cache_dir = Path(self.config.get("trace_cache_dir", "runs/scene_assembly_trace_cache"))
        self.canvas_size = int(self.config.get("canvas_size", 1024))
        if self.canvas_size < 256:
            raise ValueError("scene_assembly canvas_size must be at least 256")
        self.source_steps = int(self.config.get("source_num_inference_steps", 40))
        self.source_guidance = float(self.config.get("source_guidance_scale", 5.0))
        self.default_action_layouts = dict(self.config.get("action_layouts", {}))

    def generate(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
        inventory, attributes, spatial, actions = self._contract_rows(program)
        if not inventory or self._needs_scene_native_generation(inventory, spatial, actions):
            return self._generate_scene_native(task, program, seed=seed)

        assets: Dict[str, Image.Image] = {}
        source_assets: List[Dict[str, Any]] = []
        for index, row in enumerate(inventory):
            name = str(row["class"]).strip().lower()
            source_prompt = self._source_asset_prompt(
                name,
                attributes.get(name, []),
                repair_clause=self._transient_repair_clause(
                    program,
                    name,
                    inventory_names=[str(item["class"]).strip().lower() for item in inventory],
                ),
            )
            source_generation = self._generate_source_asset(
                task,
                program,
                source_prompt,
                seed=self._source_seed(seed, index),
            )
            source_path = source_generation.get("image_path") or source_generation.get("image_uri")
            if not source_path:
                raise ValueError(f"scene_assembly source generation returned no image path for {name}")
            asset = self._extract_foreground(Image.open(source_path).convert("RGBA"))
            assets[name] = asset
            source_assets.append(
                {
                    "role": "object_exemplar",
                    "object": name,
                    "prompt": source_prompt,
                    "generation": source_generation,
                }
            )

        action_assets: Dict[str, Image.Image] = {}
        for action_index, action in enumerate(actions):
            subject = str(action["subject"]).strip().lower()
            target = str(action["object"]).strip().lower()
            verb = str(action["action"]).strip().lower()
            subject_prompt, target_prompt = self._action_asset_prompts(
                subject,
                target,
                verb,
                mode=self._lookup_action_layout(self._action_layouts(program), verb).get("mode", "contact"),
                subject_attributes=attributes.get(subject, []),
                target_attributes=attributes.get(target, []),
                subject_guidance=self._transient_repair_guidance(
                    program,
                    subject,
                    inventory_names=[str(item["class"]).strip().lower() for item in inventory],
                ),
                target_guidance=self._transient_repair_guidance(
                    program,
                    target,
                    inventory_names=[str(item["class"]).strip().lower() for item in inventory],
                ),
            )
            for role, name, prompt in (
                ("action_subject", subject, subject_prompt),
                ("action_target", target, target_prompt),
            ):
                source_generation = self._generate_source_asset(
                    task,
                    program,
                    prompt,
                    seed=self._source_seed(
                        seed,
                        len(inventory) + action_index * 2 + (0 if role == "action_subject" else 1),
                    ),
                )
                source_path = source_generation.get("image_path") or source_generation.get("image_uri")
                if not source_path:
                    raise ValueError(f"scene_assembly action generation returned no image path for {role}:{name}")
                action_asset = self._extract_foreground(Image.open(source_path).convert("RGBA"))
                action_assets[name] = action_asset
                source_assets.append(
                    {
                        "role": role,
                        "subject": subject,
                        "object": target,
                        "action": verb,
                        "object_class": name,
                        "prompt": prompt,
                        "generation": source_generation,
                    }
                )

        image, assembly_plan = self._compose(
            task,
            inventory,
            attributes,
            spatial,
            actions,
            assets,
            action_assets=action_assets,
            action_layouts=self._action_layouts(program),
        )
        image_path = self._save_image(image, task, program, seed)
        result = {
            "backend": self.config.get("name", "scene_assembly"),
            "seed": seed,
            "image_path": str(image_path),
            "image_uri": str(image_path),
            "selected_model": self.config.get("selected_model", "sd3.5-medium-flow-grpo-scene-assembly"),
            "generation_strategy": "fresh_flowgrpo_asset_scene_assembly",
            "source_assets": source_assets,
            "assembly_plan": assembly_plan,
            "official_labels_used": False,
            "fallback_used": False,
            "image_origin": "fresh_generation_for_task",
        }
        trace_path = self._save_trace(image_path, task, program, result, seed)
        result["trace_path"] = str(trace_path)
        return result

    def _generate_scene_native(self, task: GenerationTask, program: Dict[str, Any], *, seed: int) -> Dict[str, Any]:
        generation = self.source_backend.generate(task, program, seed=seed)
        generation = dict(generation)
        generation.setdefault("official_labels_used", False)
        generation.setdefault("fallback_used", False)
        generation.setdefault("image_origin", "fresh_generation_for_task")
        generation.setdefault("generation_strategy", "scene_native_source_generation")
        return generation

    def _generate_source_asset(self, task: GenerationTask, program: Dict[str, Any], prompt: str, *, seed: int) -> Dict[str, Any]:
        source_task = GenerationTask(
            task_id=task.task_id,
            prompt=prompt,
            task_family=task.task_family,
            constraints=[],
            references={},
            metadata={"source_asset_for": task.task_id, "official_labels_used": False},
        )
        source_program = {
            "method": "scene_assembly_source",
            "workflow": {
                "compiled_prompt": prompt,
                "clip_prompt": prompt,
                "t5_prompt": prompt,
                "clip_prompt_segments": [prompt],
                "clip_prompt_segment_types": ["source"],
                "t5_prompt_segments": [prompt],
                "t5_prompt_segment_types": ["source"],
            },
            "tool": {
                "name": "scene_assembly_source_generator",
                "generation_parameters": {
                    "split_prompt_channels": True,
                    "num_inference_steps": self._source_generation_parameter(program, "num_inference_steps", self.source_steps),
                    "guidance_scale": self._source_generation_parameter(program, "guidance_scale", self.source_guidance),
                },
            },
            "component_sources": dict(program.get("component_sources", {})) if isinstance(program, dict) else {},
            "benchmark_generation_input": "prompt_only",
            "official_labels_used": False,
            "fallback_used": False,
        }
        return self.source_backend.generate(source_task, source_program, seed=seed)

    def _action_layouts(self, program: Dict[str, Any]) -> Dict[str, Any]:
        tool = program.get("tool", {}) if isinstance(program, dict) else {}
        parameters = tool.get("generation_parameters", {}) if isinstance(tool, dict) else {}
        configured = parameters.get("action_layouts", {}) if isinstance(parameters, dict) else {}
        if not isinstance(configured, dict):
            configured = {}
        return {**self.default_action_layouts, **configured}

    @staticmethod
    def _reject_image_llm_source(config: Dict[str, Any]) -> None:
        text = json.dumps(config, sort_keys=True).lower()
        forbidden = ("gemini", "gpt-image", "dall-e", "openai_image", "flash-image")
        if any(token in text for token in forbidden):
            raise ValueError("scene_assembly source_backend must be a frozen generator, not an image LLM")

    @staticmethod
    def _contract_rows(program: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, List[str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        inventory: Dict[str, Dict[str, Any]] = {}
        attributes: Dict[str, List[str]] = {}
        spatial: List[Dict[str, Any]] = []
        actions: List[Dict[str, Any]] = []
        contract = program.get("contract", {}) if isinstance(program, dict) else {}
        for constraint in contract.get("constraints", []):
            if not isinstance(constraint, dict):
                continue
            metadata = constraint.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            for row in metadata.get("required_objects", []):
                if isinstance(row, dict) and row.get("class"):
                    name = str(row["class"]).strip().lower()
                    count = max(1, int(row.get("count", 1) or 1))
                    current = inventory.get(name)
                    if current is None or count > int(current.get("count", 1)):
                        inventory[name] = {"class": name, "count": count}
            for row in metadata.get("attribute_bindings", []):
                if isinstance(row, dict) and row.get("object") and row.get("attribute"):
                    name = str(row["object"]).strip().lower()
                    value = str(row["attribute"]).strip().lower()
                    if value and value not in attributes.setdefault(name, []):
                        attributes[name].append(value)
            for row in metadata.get("spatial_relations", []):
                if isinstance(row, dict):
                    relation = {
                        "subject": str(row.get("subject", "")).strip().lower(),
                        "relation": _normalize_relation(row.get("relation")),
                        "object": str(row.get("object", "")).strip().lower(),
                    }
                    if relation["subject"] and relation["relation"] and relation["object"]:
                        spatial.append(relation)
            for row in metadata.get("action_relations", []):
                if isinstance(row, dict):
                    subject = str(row.get("subject", "")).strip().lower()
                    action = " ".join(str(row.get("action", "")).strip().lower().replace("_", " ").split())
                    target = str(row.get("object", "")).strip().lower()
                    if subject and action and target:
                        actions.append({"subject": subject, "action": action, "object": target})
        return list(inventory.values()), attributes, spatial, actions

    @staticmethod
    def _needs_scene_native_generation(
        inventory: List[Dict[str, Any]],
        spatial: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
    ) -> bool:
        names = {str(row.get("class", "")).lower() for row in inventory}
        supported_relations = {"left of", "right of", "above", "below", "in front of", "behind", "near", "next to"}
        return any(
            relation.get("relation") not in supported_relations
            or relation.get("subject") not in names
            or relation.get("object") not in names
            for relation in spatial
        )

    def _compose(
        self,
        task: GenerationTask,
        inventory: List[Dict[str, Any]],
        attributes: Dict[str, List[str]],
        spatial: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        assets: Dict[str, Image.Image],
        *,
        action_assets: Dict[str, Image.Image] | None = None,
        action_layouts: Dict[str, Any] | None = None,
    ) -> tuple[Image.Image, Dict[str, Any]]:
        size = self.canvas_size
        canvas = Image.new(
            "RGB",
            (size, size),
            tuple(self.config.get("background_color", (242, 239, 225))),
        )
        boxes = self._category_layout(inventory, spatial, actions, action_layouts=action_layouts)
        placements = []
        action_assets = action_assets or {}
        for category_index, row in enumerate(inventory):
            name = str(row["class"]).lower()
            count = max(1, int(row.get("count", 1) or 1))
            placement = self._paste_instances(
                canvas,
                action_assets.get(name, assets[name]),
                count,
                boxes[name],
                category_index,
                attributes=attributes.get(name, []),
            )
            placements.append({"object": name, "count": count, "box": boxes[name], "placements": placement})
        return canvas, {
            "schema": "gen_harness.scene_assembly_plan.v1",
            "task_id": task.task_id,
            "canvas_size": size,
            "layout": "deterministic_constraint_indexed",
            "official_labels_used": False,
            "fallback_used": False,
            "actions": actions,
            "categories": placements,
        }

    @staticmethod
    def _action_asset_prompts(
        subject: str,
        target: str,
        action: str,
        *,
        mode: str = "contact",
        subject_attributes: List[str] | None = None,
        target_attributes: List[str] | None = None,
        guidance: str = "",
        subject_guidance: str = "",
        target_guidance: str = "",
    ) -> tuple[str, str]:
        """Create visual action exemplars while keeping class identity explicit."""

        if not subject_guidance:
            subject_guidance = guidance
        if not target_guidance:
            target_guidance = guidance
        subject_descriptor = SceneAssemblyBackend._source_descriptor(
            subject,
            list(subject_attributes or []),
        )
        target_descriptor = SceneAssemblyBackend._source_descriptor(
            target,
            list(target_attributes or []),
        )
        if mode == "directional":
            return (
                f"one complete recognizable {subject_descriptor}, sprinting dynamically toward the right, "
                "legs extended, looking right, visible dust and motion, full-body side-view action photograph, "
                "isolated on a clean solid white studio background, full silhouette visible, "
                f"single subject only, no other objects, no logo, watermark, or caption{subject_guidance}",
                f"one complete recognizable {target_descriptor}, fleeing dynamically toward the right, "
                "legs extended, looking ahead to the right, visible dust and motion, full-body side-view action photograph, "
                "isolated on a clean solid white studio background, full silhouette visible, "
                f"single subject only, no other objects, no logo, watermark, or caption{target_guidance}",
            )
        if mode == "vertical":
            return (
                f"one complete recognizable {subject_descriptor}, airborne with legs extended in a clear jumping pose, "
                "dynamic side-view action photograph, isolated on a solid bright white background, "
                "full silhouette visible, single subject only, no logo, watermark, caption, or additional object"
                f"no other objects, no logo, watermark, or caption{subject_guidance}",
                f"one complete recognizable {target_descriptor}, standing still in a clear ground-level pose, full body visible, "
                "dynamic side-view action photograph, isolated on a solid bright white background, "
                "full silhouette visible, single subject only, no logo, watermark, caption, or additional object"
                f"no other objects, no logo, watermark, or caption{target_guidance}",
            )
        return (
            f"one complete recognizable {subject_descriptor}, playful active pose facing right, "
            "playful contact pose, full-body action photograph, isolated on a solid bright white background, "
            "full silhouette visible, single subject only, no logo, watermark, caption, or additional object"
            f"no other objects, no logo, watermark, or caption{subject_guidance}",
            f"one complete recognizable {target_descriptor}, playful active pose facing left, "
            "playful contact pose, full-body action photograph, isolated on a solid bright white background, "
            "full silhouette visible, single subject only, no logo, watermark, caption, or additional object"
            f"no other objects, no logo, watermark, or caption{target_guidance}",
        )

    def _paste_instances(
        self,
        canvas: Image.Image,
        source: Image.Image,
        count: int,
        box: Tuple[int, int, int, int],
        category_index: int,
        *,
        attributes: List[str],
    ) -> List[Dict[str, Any]]:
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        columns = max(1, math.ceil(math.sqrt(count)))
        if count in {5, 6}:
            columns = 3
        if count >= 7:
            columns = math.ceil(count / 2)
        rows = math.ceil(count / columns)
        cell_w = width / columns
        cell_h = height / rows
        placements = []
        for index in range(count):
            row = index // columns
            col = index % columns
            remaining = count - row * columns
            row_columns = min(columns, remaining)
            row_offset = (columns - row_columns) * cell_w / 2.0
            attribute_scale = 0.88 if attributes else 0.74
            if count >= 6:
                # Never let repeated instances exceed their cell.  The old
                # 1.08 enlargement made adjacent objects overlap and caused
                # VQA count failures even when the source asset was correct.
                scale = min(max(attribute_scale, 1.08), 0.92)
            elif count >= 4:
                scale = max(attribute_scale, 0.84)
            else:
                scale = max(attribute_scale, 0.82)
            target_w = int(cell_w * scale)
            target_h = int(cell_h * scale)
            item = self._fit_asset(source, target_w, target_h, attributes=attributes)
            px = int(x0 + row_offset + col * cell_w + (cell_w - item.width) / 2.0)
            py = int(y0 + row * cell_h + (cell_h - item.height) / 2.0)
            self._paste_rgba(canvas, item, px, py)
            placements.append({"index": index, "x": px, "y": py, "width": item.width, "height": item.height})
        return placements

    @staticmethod
    def _fit_asset(asset: Image.Image, max_w: int, max_h: int, *, attributes: List[str]) -> Image.Image:
        scale = min(max_w / max(1, asset.width), max_h / max(1, asset.height), 1.0)
        new_size = (max(1, int(asset.width * scale)), max(1, int(asset.height * scale)))
        return asset.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def _paste_rgba(canvas: Image.Image, item: Image.Image, x: int, y: int) -> None:
        canvas.paste(item.convert("RGB"), (x, y), item.getchannel("A"))

    def _category_layout(
        self,
        inventory: List[Dict[str, Any]],
        spatial: List[Dict[str, Any]],
        actions: List[Dict[str, Any]] | None = None,
        *,
        action_layouts: Dict[str, Any] | None = None,
    ) -> Dict[str, Tuple[int, int, int, int]]:
        actions = list(actions or [])
        action_layouts = action_layouts or {}
        names = [str(row["class"]).lower() for row in inventory]
        centers = dict(zip(names, self._default_category_centers(len(names))))
        horizontal_edges: List[Tuple[str, str]] = []
        vertical_edges: List[Tuple[str, str]] = []
        for relation in spatial:
            subject = relation.get("subject")
            target = relation.get("object")
            rel = relation.get("relation")
            if subject not in centers or target not in centers:
                continue
            if rel == "left of":
                horizontal_edges.append((subject, target))
            elif rel == "right of":
                horizontal_edges.append((target, subject))
            elif rel == "above":
                vertical_edges.append((subject, target))
            elif rel == "below":
                vertical_edges.append((target, subject))
            elif rel == "in front of":
                vertical_edges.append((target, subject))
            elif rel == "behind":
                vertical_edges.append((subject, target))

        # Solve each constrained axis as an ordered lane.  Unrelated
        # categories keep their default slots instead of being dragged onto a
        # constrained category by coordinate propagation.
        self._apply_axis_constraints(centers, names, horizontal_edges, axis=0)
        horizontal_names = {
            name
            for edge in horizontal_edges
            for name in edge
        }
        self._apply_axis_constraints(
            centers,
            names,
            vertical_edges,
            axis=1,
            perpendicular_anchor_names=horizontal_names,
        )
        # Vertical constraints can move a shared node after a horizontal lane
        # has already been solved.  Re-apply the horizontal equal-depth
        # relation so a chain such as "A right of B above C" keeps A and B in
        # the same row while B remains above C.
        self._apply_axis_constraints(centers, names, horizontal_edges, axis=0)

        # Keep action participants in a shared interaction band.  The source
        # assets remain independently generated by the frozen Flow-GRPO model;
        # only their deterministic placement is changed here.
        for action in actions:
            subject = action.get("subject")
            target = action.get("object")
            verb = str(action.get("action", "")).lower()
            if subject not in centers or target not in centers:
                continue
            tx, ty = centers[target]
            layout = self._lookup_action_layout(action_layouts, verb)
            if layout.get("mode") == "vertical":
                centers[subject] = (tx, ty - 0.20)
            elif layout.get("mode") == "directional":
                centers[subject] = (tx - 0.24, ty)
            else:
                # playing/interacting/holding and unknown contact verbs need
                # visible proximity without collapsing the two categories.
                centers[subject] = (tx - 0.16, ty)
                centers[target] = (tx + 0.08, ty)

        constrained_names = {
            name
            for edge in [*horizontal_edges, *vertical_edges]
            for name in edge
        }
        constrained_names.update(
            name
            for action in actions
            for name in (action.get("subject"), action.get("object"))
            if name in centers
        )
        for row in inventory:
            name = str(row.get("class", "")).lower()
            count = max(1, int(row.get("count", 1) or 1))
            if count >= 4 and name not in constrained_names:
                _, cy = centers[name]
                centers[name] = (0.78 if centers[name][0] >= 0.5 else 0.22, cy)

        centers = {
            name: (
                min(0.86, max(0.14, center[0])),
                min(0.82, max(0.18, center[1])),
            )
            for name, center in centers.items()
        }
        size = self.canvas_size
        if len(names) == 1:
            return {names[0]: (int(0.10 * size), int(0.20 * size), int(0.90 * size), int(0.76 * size))}
        box_w = 0.40 if len(names) <= 2 else 0.30
        box_h = 0.42 if len(names) <= 2 else 0.32
        vertical_names = {
            name
            for relation in spatial
            if relation.get("relation") in {"above", "below", "in front of", "behind"}
            for name in (relation.get("subject"), relation.get("object"))
            if name in centers
        }
        vertical_names.update(
            name
            for action in actions
            if self._lookup_action_layout(action_layouts, str(action.get("action", "")).lower()).get("mode") == "vertical"
            for name in (action.get("subject"), action.get("object"))
            if name in centers
        )
        contact_names = {
            name
            for action in actions
            if self._lookup_action_layout(action_layouts, str(action.get("action", "")).lower()).get("mode") != "vertical"
            for name in (action.get("subject"), action.get("object"))
            if name in centers
        }
        boxes = {}
        for name in names:
            cx, cy = centers[name]
            cx = min(0.86, max(0.14, cx))
            cy = min(0.82, max(0.18, cy))
            count = next(
                max(1, int(row.get("count", 1) or 1))
                for row in inventory
                if str(row.get("class", "")).lower() == name
            )
            # Repeated-object constraints need a readable lane of their own.
            # A narrow three-category box makes seven instances collapse to
            # tiny thumbnails, which defeats both human inspection and VQA.
            if count >= 6:
                category_box_w = 0.44
            elif count >= 4:
                category_box_w = max(box_w, 0.36)
            else:
                category_box_w = 0.40 if name in vertical_names else box_w
            if name in contact_names:
                category_box_w = max(category_box_w, 0.48)
            category_box_h = 0.42 if count >= 6 else box_h
            boxes[name] = (
                int(max(0.03, cx - category_box_w / 2) * size),
                int(max(0.06, cy - category_box_h / 2) * size),
                int(min(0.97, cx + category_box_w / 2) * size),
                int(min(0.94, cy + category_box_h / 2) * size),
            )
        return self._separate_overlapping_lanes(boxes, centers, size)

    @staticmethod
    def _separate_overlapping_lanes(
        boxes: Dict[str, Tuple[int, int, int, int]],
        centers: Dict[str, Tuple[float, float]],
        size: int,
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """Give same-depth repeated categories disjoint horizontal lanes."""

        if len(boxes) < 2:
            return boxes
        updated = dict(boxes)
        names = sorted(
            updated,
            key=lambda name: (
                centers.get(name, (0.5, 0.5))[1],
                centers.get(name, (0.5, 0.5))[0],
            ),
        )
        groups: List[List[str]] = []
        for name in names:
            cy = centers.get(name, (0.5, 0.5))[1]
            for group in groups:
                group_cy = sum(
                    centers.get(item, (0.5, 0.5))[1] for item in group
                ) / len(group)
                if abs(cy - group_cy) <= 0.18:
                    group.append(name)
                    break
            else:
                groups.append([name])

        def overlaps(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> bool:
            return (
                min(left[2], right[2]) > max(left[0], right[0])
                and min(left[3], right[3]) > max(left[1], right[1])
            )

        lane_left, lane_right = int(0.03 * size), int(0.97 * size)
        for group in groups:
            if len(group) < 2:
                continue
            if not any(
                overlaps(updated[left], updated[right])
                for index, left in enumerate(group)
                for right in group[index + 1 :]
            ):
                continue
            lane_width = (lane_right - lane_left) / len(group)
            max_width = int(lane_width * 0.88)
            for index, name in enumerate(group):
                x0, y0, x1, y1 = updated[name]
                width = min(x1 - x0, max_width)
                lane_center = lane_left + lane_width * (index + 0.5)
                next_x0 = int(round(lane_center - width / 2))
                updated[name] = (next_x0, y0, next_x0 + width, y1)
        return updated

    @staticmethod
    def _apply_axis_constraints(
        centers: Dict[str, Tuple[float, float]],
        names: List[str],
        edges: List[Tuple[str, str]],
        *,
        axis: int,
        perpendicular_anchor_names: set[str] | None = None,
    ) -> None:
        if not edges:
            return
        order_index = {name: index for index, name in enumerate(names)}
        adjacency = {name: set() for name in names}
        for before, after in edges:
            adjacency.setdefault(before, set()).add(after)
            adjacency.setdefault(after, set())

        remaining = set(adjacency)
        while remaining:
            root = min(remaining, key=order_index.get)
            component = {root}
            frontier = [root]
            while frontier:
                current = frontier.pop()
                neighbors = {
                    candidate
                    for candidate, successors in adjacency.items()
                    if current in successors
                } | adjacency.get(current, set())
                for neighbor in neighbors:
                    if neighbor not in component:
                        component.add(neighbor)
                        frontier.append(neighbor)
            remaining -= component
            component_edges = [
                (before, after)
                for before, after in edges
                if before in component and after in component
            ]
            if not component_edges:
                continue
            indegree = {name: 0 for name in component}
            predecessors = {name: [] for name in component}
            successors = {name: [] for name in component}
            for before, after in component_edges:
                successors[before].append(after)
                predecessors[after].append(before)
                indegree[after] += 1
            ready = sorted(
                (name for name, degree in indegree.items() if degree == 0),
                key=order_index.get,
            )
            ordered = []
            while ready:
                current = ready.pop(0)
                ordered.append(current)
                for successor in sorted(successors[current], key=order_index.get):
                    indegree[successor] -= 1
                    if indegree[successor] == 0:
                        ready.append(successor)
                        ready.sort(key=order_index.get)
            if len(ordered) != len(component):
                continue
            # A partial order is not a total order.  Objects sharing the same
            # predecessor must occupy the same axis level; topological order
            # alone would incorrectly turn "A and B below C" into C-A-B.
            levels = {name: 0 for name in component}
            for name in ordered:
                if predecessors[name]:
                    levels[name] = max(levels[parent] + 1 for parent in predecessors[name])
            max_level = max(levels.values(), default=0)
            if max_level == 1 and len(component) == 2:
                low, high = ((0.27, 0.64) if axis == 1 else (0.28, 0.72))
            else:
                low, high = 0.18, 0.82
            for name in ordered:
                level = levels[name]
                coordinate = low if max_level == 0 else low + (high - low) * level / max_level
                x, y = centers[name]
                centers[name] = (coordinate, y) if axis == 0 else (x, coordinate)

            level_groups: Dict[int, List[str]] = {}
            for name, level in levels.items():
                level_groups.setdefault(level, []).append(name)
            # Preserve the perpendicular separation of sibling entities.  A
            # singleton parent is centered over its direct children, while a
            # simple chain remains a clean aligned lane.
            if all(len(group) == 1 for group in level_groups.values()):
                anchor = next(
                    (
                        name
                        for name in ordered
                        if name in (perpendicular_anchor_names or set())
                    ),
                    None,
                )
                aligned = (
                    centers[anchor][1 - axis]
                    if anchor is not None
                    else sum(centers[name][1 - axis] for name in ordered) / len(ordered)
                )
                for name in ordered:
                    x, y = centers[name]
                    centers[name] = (x, aligned) if axis == 0 else (aligned, y)
            else:
                for name, group in level_groups.items():
                    if len(group) != 1:
                        continue
                    parent = group[0]
                    children = successors[parent]
                    if not children:
                        continue
                    aligned = sum(centers[child][1 - axis] for child in children) / len(children)
                    x, y = centers[parent]
                    centers[parent] = (x, aligned) if axis == 0 else (aligned, y)

    @staticmethod
    def _lookup_action_layout(action_layouts: Dict[str, Any], action: str) -> Dict[str, Any]:
        normalized = " ".join(str(action).lower().replace("_", " ").split())
        exact = action_layouts.get(normalized)
        if isinstance(exact, dict):
            return exact
        for key, value in action_layouts.items():
            if not isinstance(value, dict):
                continue
            aliases = value.get("aliases", [])
            if isinstance(aliases, list) and normalized in {" ".join(str(alias).lower().replace("_", " ").split()) for alias in aliases}:
                return value
        return {}


    @staticmethod
    def _default_category_centers(count: int) -> List[Tuple[float, float]]:
        if count <= 1:
            return [(0.5, 0.5)]
        if count == 2:
            return [(0.28, 0.52), (0.72, 0.52)]
        if count == 3:
            return [(0.5, 0.25), (0.26, 0.72), (0.74, 0.72)]
        if count == 4:
            return [(0.25, 0.28), (0.75, 0.28), (0.25, 0.72), (0.75, 0.72)]
        return [(0.18, 0.25), (0.50, 0.25), (0.82, 0.25), (0.28, 0.72), (0.62, 0.72)][:count]

    @staticmethod
    def _extract_foreground(image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            pixels = np.asarray(rgb)
            h, w = pixels.shape[:2]
            hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            border_rgb = np.concatenate(
                [pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]],
                axis=0,
            )
            background_rgb = np.median(border_rgb, axis=0)
            background_gray = float(
                np.median(
                    np.concatenate(
                        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]],
                    )
                )
            )
            background_saturation = float(
                np.median(
                    np.concatenate(
                        [hsv[0, :, 1], hsv[-1, :, 1], hsv[:, 0, 1], hsv[:, -1, 1]],
                    )
                )
            )
            color_delta = np.max(
                np.abs(pixels.astype(np.int16) - background_rgb.astype(np.int16)),
                axis=2,
            )
            foreground = (
                (color_delta > 24)
                | (hsv[:, :, 1] > max(35.0, background_saturation + 18.0))
                | (gray < background_gray - 28.0)
                | (gray > background_gray + 32.0)
            ).astype(np.uint8) * 255
            foreground[: max(2, int(foreground.shape[0] * 0.015)), :] = 0
            foreground[-max(2, int(foreground.shape[0] * 0.015)) :, :] = 0
            foreground[:, : max(2, int(foreground.shape[1] * 0.015))] = 0
            foreground[:, -max(2, int(foreground.shape[1] * 0.015)) :] = 0
            foreground = cv2.morphologyEx(
                foreground,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), dtype=np.uint8),
            )
            count, labels, stats, centers = cv2.connectedComponentsWithStats(
                foreground,
                connectivity=8,
            )
            candidates = []
            image_area = float(foreground.shape[0] * foreground.shape[1])
            for index in range(1, count):
                x, y, width, height, area = [int(value) for value in stats[index]]
                if area < max(64, int(image_area * 0.0002)):
                    continue
                if area > int(image_area * 0.70):
                    continue
                cx, cy = [float(value) for value in centers[index]]
                centrality = max(0.0, 1.0 - abs(cx / foreground.shape[1] - 0.5) * 2.0)
                lower_edge_penalty = 0.35 if y + height >= int(foreground.shape[0] * 0.94) else 0.0
                score = area * (0.70 + 0.30 * centrality) * (1.0 - lower_edge_penalty)
                candidates.append((score, index, x, y, width, height))
            if candidates:
                _, primary, x, y, width, height = max(candidates)
                selected = np.zeros_like(foreground)
                selected[labels == primary] = 255
                detector_selected = selected.copy()
                # Refine against the actual RGB image so near-white subject
                # regions remain foreground. A rectangle-initialized GrabCut
                # is deliberately bounded by the detector-derived component;
                # it cannot turn the whole studio canvas into foreground.
                try:
                    detector_spans_canvas = width >= int(w * 0.88) or height >= int(h * 0.88)
                    if detector_spans_canvas:
                        # A smooth studio gradient can connect the object to
                        # the floor and make the detector component span the
                        # whole image.  The source generator places the
                        # exemplar centrally, so a conservative inner rect
                        # gives a downsampled GrabCut a bounded foreground
                        # hypothesis without making every source extraction
                        # pay the full-resolution segmentation cost.
                        max_dim = 512
                        scale = min(1.0, max_dim / max(h, w))
                        small_w = max(64, int(round(w * scale)))
                        small_h = max(64, int(round(h * scale)))
                        small_pixels = cv2.resize(
                            pixels,
                            (small_w, small_h),
                            interpolation=cv2.INTER_AREA,
                        )
                        small_hsv = cv2.cvtColor(small_pixels, cv2.COLOR_RGB2HSV)
                        small_gray = cv2.cvtColor(small_pixels, cv2.COLOR_RGB2GRAY)
                        small_border = np.concatenate(
                            [
                                small_pixels[0, :, :],
                                small_pixels[-1, :, :],
                                small_pixels[:, 0, :],
                                small_pixels[:, -1, :],
                            ],
                            axis=0,
                        )
                        small_background_rgb = np.median(small_border, axis=0)
                        small_background_gray = float(
                            np.median(
                                np.concatenate(
                                    [
                                        small_gray[0, :],
                                        small_gray[-1, :],
                                        small_gray[:, 0],
                                        small_gray[:, -1],
                                    ]
                                )
                            )
                        )
                        small_background_saturation = float(
                            np.median(
                                np.concatenate(
                                    [
                                        small_hsv[0, :, 1],
                                        small_hsv[-1, :, 1],
                                        small_hsv[:, 0, 1],
                                        small_hsv[:, -1, 1],
                                    ]
                                )
                            )
                        )
                        small_delta = np.max(
                            np.abs(
                                small_pixels.astype(np.int16)
                                - small_background_rgb.astype(np.int16)
                            ),
                            axis=2,
                        )
                        seed = (
                            (small_delta > 45)
                            | (
                                small_hsv[:, :, 1]
                                > max(45.0, small_background_saturation + 25.0)
                            )
                            | (small_gray < small_background_gray - 38.0)
                            | (small_gray > small_background_gray + 48.0)
                        ).astype(np.uint8) * 255
                        border_width = max(2, int(min(small_w, small_h) * 0.02))
                        seed[:border_width, :] = 0
                        seed[-border_width:, :] = 0
                        seed[:, :border_width] = 0
                        seed[:, -border_width:] = 0
                        seed = cv2.morphologyEx(
                            seed,
                            cv2.MORPH_CLOSE,
                            np.ones((7, 7), dtype=np.uint8),
                        )
                        small_mask = np.full(
                            (small_h, small_w),
                            cv2.GC_PR_BGD,
                            dtype=np.uint8,
                        )
                        small_mask[seed > 0] = cv2.GC_PR_FGD
                        small_mask[:border_width, :] = cv2.GC_BGD
                        small_mask[-border_width:, :] = cv2.GC_BGD
                        small_mask[:, :border_width] = cv2.GC_BGD
                        small_mask[:, -border_width:] = cv2.GC_BGD
                        bg_model = np.zeros((1, 65), dtype=np.float64)
                        fg_model = np.zeros((1, 65), dtype=np.float64)
                        cv2.grabCut(
                            small_pixels,
                            small_mask,
                            None,
                            bg_model,
                            fg_model,
                            3,
                            cv2.GC_INIT_WITH_MASK,
                        )
                        grab_selected = cv2.resize(
                            np.where(
                                (small_mask == cv2.GC_FGD)
                                | (small_mask == cv2.GC_PR_FGD),
                                255,
                                0,
                            ).astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    else:
                        grab_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
                        rect_x0 = max(1, x - max(8, int(width * 0.04)))
                        rect_y0 = max(1, y - max(8, int(height * 0.04)))
                        rect_x1 = min(w - 2, x + width + max(8, int(width * 0.04)))
                        rect_y1 = min(h - 2, y + height + max(8, int(height * 0.04)))
                        rect = (rect_x0, rect_y0, rect_x1 - rect_x0, rect_y1 - rect_y0)
                        grab_mask[:2, :] = cv2.GC_BGD
                        grab_mask[-2:, :] = cv2.GC_BGD
                        grab_mask[:, :2] = cv2.GC_BGD
                        grab_mask[:, -2:] = cv2.GC_BGD
                        grab_mask[rect_y0:rect_y1, rect_x0:rect_x1] = cv2.GC_PR_FGD
                        bg_model = np.zeros((1, 65), dtype=np.float64)
                        fg_model = np.zeros((1, 65), dtype=np.float64)
                        cv2.grabCut(
                            pixels,
                            grab_mask,
                            rect,
                            bg_model,
                            fg_model,
                            5,
                            cv2.GC_INIT_WITH_RECT,
                        )
                        grab_selected = np.where(
                            (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
                            255,
                            0,
                        ).astype(np.uint8)
                    selected = (
                        grab_selected
                        if detector_spans_canvas
                        else SceneAssemblyBackend._guard_grabcut_collapse(
                            detector_selected,
                            grab_selected,
                            np,
                        )
                    )
                except cv2.error:
                    pass
                ys, xs = np.where(selected > 0)
                if len(xs):
                    pad = max(6, int(max(width, height) * 0.035))
                    crop_x0 = max(0, int(xs.min()) - pad)
                    crop_y0 = max(0, int(ys.min()) - pad)
                    crop_x1 = min(selected.shape[1], int(xs.max()) + pad + 1)
                    crop_y1 = min(selected.shape[0], int(ys.max()) + pad + 1)
                    crop = Image.fromarray(
                        pixels[crop_y0:crop_y1, crop_x0:crop_x1],
                        mode="RGB",
                    ).convert("RGBA")
                    crop.putalpha(Image.fromarray(selected[crop_y0:crop_y1, crop_x0:crop_x1]))
                    return crop

            hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            border = np.concatenate(
                [pixels[0, :, :], pixels[-1, :, :], pixels[:, 0, :], pixels[:, -1, :]],
                axis=0,
            )
            border_gray = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(-1)
            background_gray = float(np.median(border_gray))
            background_saturation = float(np.median(np.concatenate([hsv[0, :, 1], hsv[-1, :, 1], hsv[:, 0, 1], hsv[:, -1, 1]])))
            working = pixels
            working_gray = gray
            working_hsv = hsv
            origin_x = origin_y = 0

            # Flow-GRPO often returns an object on a large inset studio
            # rectangle surrounded by a bright margin.  The outer border is
            # then not the actual background and would select the rectangle.
            dark_inset = (gray < min(180.0, background_gray - 35.0)).astype(np.uint8) * 255
            dark_inset = cv2.morphologyEx(
                dark_inset,
                cv2.MORPH_CLOSE,
                np.ones((21, 21), dtype=np.uint8),
            )
            inset_count, _, inset_stats, _ = cv2.connectedComponentsWithStats(dark_inset, connectivity=8)
            inset_candidate = None
            for index in range(1, inset_count):
                x, y, width, height, area = [int(value) for value in inset_stats[index]]
                if (
                    area >= int(gray.size * 0.25)
                    and width >= int(image.width * 0.55)
                    and height >= int(image.height * 0.55)
                    and x >= int(image.width * 0.02)
                    and y >= int(image.height * 0.02)
                ):
                    score = area + width * height * 0.25
                    if inset_candidate is None or score > inset_candidate[0]:
                        inset_candidate = (score, x, y, width, height)
            if inset_candidate is not None:
                _, x, y, width, height = inset_candidate
                inset_pad = max(2, int(min(width, height) * 0.006))
                origin_x = x + inset_pad
                origin_y = y + inset_pad
                x0 = origin_x
                y0 = origin_y
                x1 = min(image.width, x + width - inset_pad)
                y1 = min(image.height, y + height - inset_pad)
                working = pixels[y0:y1, x0:x1]
                working_gray = gray[y0:y1, x0:x1]
                working_hsv = hsv[y0:y1, x0:x1]
                local_border_gray = np.concatenate(
                    [
                        working_gray[0, :],
                        working_gray[-1, :],
                        working_gray[:, 0],
                        working_gray[:, -1],
                    ]
                )
                local_border_sat = np.concatenate(
                    [
                        working_hsv[0, :, 1],
                        working_hsv[-1, :, 1],
                        working_hsv[:, 0, 1],
                        working_hsv[:, -1, 1],
                    ]
                )
                background_gray = float(np.median(local_border_gray))
                background_saturation = float(np.median(local_border_sat))

            binary = (
                (working_gray > background_gray + 30)
                | (working_hsv[:, :, 1] > background_saturation + 35)
            ).astype(np.uint8) * 255
            if inset_candidate is None and background_gray > 175:
                binary |= (working_gray < background_gray - 40).astype(np.uint8) * 255
            binary[: int(binary.shape[0] * 0.02), :] = 0
            binary[int(binary.shape[0] * 0.98) :, :] = 0
            binary[:, : int(binary.shape[1] * 0.02)] = 0
            binary[:, int(binary.shape[1] * 0.98) :] = 0
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 15), dtype=np.uint8))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
            count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
            candidates = []
            for index in range(1, count):
                x, y, width, height, area = [int(value) for value in stats[index]]
                cx, cy = [float(value) for value in centers[index]]
                if area < max(64, int(binary.size * 0.00025)):
                    continue
                if area > int(binary.size * 0.70):
                    continue
                centrality = max(0.0, 1.0 - abs(cx / binary.shape[1] - 0.5) * 2.0)
                verticality = max(0.0, 1.0 - abs(cy / binary.shape[0] - 0.52) * 1.4)
                candidates.append((area * (0.55 + 0.25 * centrality + 0.20 * verticality), index, x, y, width, height))
            if candidates:
                _, primary, x, y, width, height = max(candidates)
                selected = np.zeros_like(binary)
                selected[labels == primary] = 255
                # Include nearby disconnected highlights or flames, while
                # avoiding detached floor shadows and background texture.
                for _, index, ox, oy, ow, oh in candidates:
                    if index == primary:
                        continue
                    gap_x = max(0, max(x - (ox + ow), ox - (x + width)))
                    gap_y = max(0, max(y - (oy + oh), oy - (y + height)))
                    if gap_x <= max(18, int(width * 0.10)) and gap_y <= max(24, int(height * 0.10)):
                        selected[labels == index] = 255
                detector_selected = selected.copy()
                try:
                    work_h, work_w = working.shape[:2]
                    grab_mask = np.full((work_h, work_w), cv2.GC_PR_BGD, dtype=np.uint8)
                    rect_x0 = max(1, x - max(8, int(width * 0.04)))
                    rect_y0 = max(1, y - max(8, int(height * 0.04)))
                    rect_x1 = min(work_w - 2, x + width + max(8, int(width * 0.04)))
                    rect_y1 = min(work_h - 2, y + height + max(8, int(height * 0.04)))
                    rect = (rect_x0, rect_y0, rect_x1 - rect_x0, rect_y1 - rect_y0)
                    grab_mask[:2, :] = cv2.GC_BGD
                    grab_mask[-2:, :] = cv2.GC_BGD
                    grab_mask[:, :2] = cv2.GC_BGD
                    grab_mask[:, -2:] = cv2.GC_BGD
                    grab_mask[rect_y0:rect_y1, rect_x0:rect_x1] = cv2.GC_PR_FGD
                    bg_model = np.zeros((1, 65), dtype=np.float64)
                    fg_model = np.zeros((1, 65), dtype=np.float64)
                    cv2.grabCut(
                        working,
                        grab_mask,
                        rect,
                        bg_model,
                        fg_model,
                        5,
                        cv2.GC_INIT_WITH_RECT,
                    )
                    grab_selected = np.where(
                        (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
                        255,
                        0,
                    ).astype(np.uint8)
                    selected = SceneAssemblyBackend._guard_grabcut_collapse(
                        detector_selected,
                        grab_selected,
                        np,
                    )
                    floor_rows = np.arange(work_h)[:, None] > int(work_h * 0.62)
                    bright_low_saturation = (
                        (working_gray > 220)
                        & (working_hsv[:, :, 1] < 36)
                        & floor_rows
                    )
                    selected[bright_low_saturation] = 0
                except cv2.error:
                    pass
                ys, xs = np.where(selected > 0)
                if len(xs):
                    pad = max(8, int(max(width, height) * 0.04))
                    crop_x0 = max(0, int(xs.min()) - pad)
                    crop_y0 = max(0, int(ys.min()) - pad)
                    crop_x1 = min(selected.shape[1], int(xs.max()) + pad + 1)
                    crop_y1 = min(selected.shape[0], int(ys.max()) + pad + 1)
                    crop = Image.fromarray(
                        working[crop_y0:crop_y1, crop_x0:crop_x1],
                        mode="RGB",
                    ).convert("RGBA")
                    crop.putalpha(Image.fromarray(selected[crop_y0:crop_y1, crop_x0:crop_x1]))
                    return crop
        except ImportError as exc:
            raise RuntimeError(
                "scene assembly foreground extraction requires OpenCV and NumPy; "
                "the formal path does not provide a fallback"
            ) from exc
        raise RuntimeError(
            "scene assembly foreground extraction produced no verified foreground "
            "component; refusing an implicit full-frame fallback"
        )

    @staticmethod
    def _guard_grabcut_collapse(detector_mask: Any, grabcut_mask: Any, np: Any) -> Any:
        """Keep GrabCut only when it preserves the detected subject extent."""

        detector_ys, detector_xs = np.where(detector_mask > 0)
        grab_ys, grab_xs = np.where(grabcut_mask > 0)
        if len(detector_xs) == 0:
            return grabcut_mask
        if len(grab_xs) == 0:
            return detector_mask
        detector_area = float(len(detector_xs))
        grab_area = float(len(grab_xs))
        detector_width = int(detector_xs.max() - detector_xs.min() + 1)
        detector_height = int(detector_ys.max() - detector_ys.min() + 1)
        grab_width = int(grab_xs.max() - grab_xs.min() + 1)
        grab_height = int(grab_ys.max() - grab_ys.min() + 1)
        if (
            grab_area < detector_area * 0.45
            or grab_width < detector_width * 0.55
            or grab_height < detector_height * 0.55
        ):
            return detector_mask
        return grabcut_mask

    @staticmethod
    def _source_descriptor(name: str, attributes: List[str]) -> str:
        return SceneAssemblyBackend._attribute_noun(name, attributes)

    @staticmethod
    def _source_asset_prompt(name: str, attributes: List[str], *, repair_clause: str = "") -> str:
        clauses = [f"one complete recognizable {name}"]
        if attributes:
            clauses.append("requested attributes: " + "; ".join(dict.fromkeys(attributes)))
        clauses.extend(["centered product photo", "plain background", "full object visible"])
        if repair_clause:
            clauses.append(repair_clause)
        return ", ".join(clauses)

    @staticmethod
    def _transient_repair_clause(
        program: Dict[str, Any],
        object_name: str,
        *,
        inventory_names: List[str] | None = None,
    ) -> str:
        """Expose only an object-relevant LLM clause to its source asset."""

        repair = program.get("fast_loop_repair", {}) if isinstance(program, dict) else {}
        patch = repair.get("patch", {}) if isinstance(repair, dict) else {}
        if not isinstance(patch, dict) or patch.get("operation") != "append_prompt_clause":
            return ""
        value = " ".join(str(patch.get("value") or "").split()).strip().rstrip(".")
        if not value:
            return ""
        names = [
            str(name).strip().lower()
            for name in (inventory_names or [object_name])
            if str(name).strip()
        ]
        mentioned = {
            name
            for name in names
            if SceneAssemblyBackend._mentions_object_name(value, name)
        }
        if mentioned != {object_name.lower()}:
            return ""
        return value

    @classmethod
    def _transient_repair_guidance(
        cls,
        program: Dict[str, Any],
        object_name: str,
        *,
        inventory_names: List[str] | None = None,
    ) -> str:
        clause = cls._transient_repair_clause(
            program,
            object_name,
            inventory_names=inventory_names,
        )
        return f", {clause}" if clause else ""

    @staticmethod
    def _mentions_object_name(text: str, object_name: str) -> bool:
        tokens = [
            token
            for token in str(text).lower().replace("-", " ").replace("_", " ").split()
            if token
        ]
        object_tokens = [
            token
            for token in str(object_name).lower().replace("-", " ").replace("_", " ").split()
            if token
        ]
        if not object_tokens:
            return False
        width = len(object_tokens)
        if any(tokens[index : index + width] == object_tokens for index in range(len(tokens) - width + 1)):
            return True
        return width == 1 and f"{object_tokens[0]}s" in tokens

    @staticmethod
    def _source_generation_parameter(
        program: Dict[str, Any],
        name: str,
        default: int | float,
    ) -> int | float:
        tool = program.get("tool", {}) if isinstance(program, dict) else {}
        parameters = tool.get("generation_parameters", {}) if isinstance(tool, dict) else {}
        value = parameters.get(name) if isinstance(parameters, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return value

    @staticmethod
    def _source_seed(generation_seed: int, asset_index: int) -> int:
        """Derive an independent reproducible stream for each source asset."""
        payload = f"{int(generation_seed)}:{int(asset_index)}".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

    @staticmethod
    def _attribute_noun(name: str, attributes: List[str]) -> str:
        modifiers = list(dict.fromkeys(str(value).strip() for value in attributes if str(value).strip()))
        return " ".join([*modifiers, str(name).strip()])

    def _save_image(self, image: Image.Image, task: GenerationTask, program: Dict[str, Any], seed: int) -> Path:
        workflow = program.get("workflow", {}) if isinstance(program, dict) else {}
        tool = program.get("tool", {}) if isinstance(program, dict) else {}
        digest = hashlib.sha1(
            repr(
                (
                    task.task_id,
                    program.get("method", "gen_harness"),
                    seed,
                    workflow.get("compiled_prompt"),
                    workflow.get("t5_prompt"),
                    tool.get("generation_parameters"),
                    "scene_assembly",
                )
            ).encode()
        ).hexdigest()[:12]
        method = str(program.get("method", "gen_harness"))
        path = self.image_cache_dir / "images" / method / f"{task.task_id}_seed{seed}_{digest}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def _save_trace(self, image_path: Path, task: GenerationTask, program: Dict[str, Any], result: Dict[str, Any], seed: int) -> Path:
        relative = image_path.relative_to(self.image_cache_dir / "images")
        path = self.trace_cache_dir / relative.parent / f"{relative.name}.trace.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": public_task_row(task),
            "seed": seed,
            "program": program,
            "generation": result,
            "image_path": str(image_path),
            "official_labels_used": False,
            "fallback_used": False,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path








def _normalize_relation(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().replace("_", " ").split())
    aliases = {
        "to the left of": "left of",
        "left from": "left of",
        "to the right of": "right of",
        "right from": "right of",
        "on top of": "above",
        "over": "above",
        "underneath": "below",
        "under": "below",
        "in front": "in front of",
        "behind": "behind",
        "next to": "near",
        "beside": "near",
    }
    return aliases.get(text, text)
