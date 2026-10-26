from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ..backend import VisualBackend
from ..executor import HarnessExecutor
from ..fast_loop import require_full_llm_fast_loop
from ..inspection import InspectionRunner
from ..io import write_json, write_jsonl
from ..repository import HarnessRepository
from ..search import SearchBackend
from ..prompt_optimizer import PromptOptimizer
from ..patcher import HarnessLocalizer, PatchProposer, build_harness_localizer, build_patch_proposer
from ..tasks import load_tasks


IMAGE_METRICS = (
    "overall_quality",
    "aesthetic_quality",
    "prompt_alignment",
    "identity_consistency",
    "style_consistency",
    "artifact_free",
    "text_logo_absence",
)


class QualityExperiment:
    """Evaluate Gen-Harness on the same tasks, backend, and inspector."""

    def __init__(
        self,
        repo: HarnessRepository,
        backend: VisualBackend,
        inspector: InspectionRunner,
        search_backend: SearchBackend | None = None,
        prompt_optimizer: PromptOptimizer | None = None,
        fast_loop_attempts: int = 5,
        fast_loop_stochastic_k: int = 4,
        harness_localizer_config: str | Path | None = None,
        patch_proposer_config: str | Path | None = None,
        harness_localizer: HarnessLocalizer | None = None,
        patch_proposer: PatchProposer | None = None,
    ):
        self.repo = repo
        self.backend = backend
        self.inspector = inspector
        self.search_backend = search_backend
        self.prompt_optimizer = prompt_optimizer
        self.fast_loop_attempts = int(fast_loop_attempts)
        self.fast_loop_stochastic_k = int(fast_loop_stochastic_k)
        self.harness_localizer = harness_localizer or build_harness_localizer(
            harness_localizer_config,
            repo=self.repo,
        )
        self.patch_proposer = patch_proposer or build_patch_proposer(
            patch_proposer_config,
            memory=self.repo.build_memory(),
            repo=self.repo,
        )
        require_full_llm_fast_loop(
            inspector=self.inspector,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            max_attempts=self.fast_loop_attempts,
            stochastic_evidence_k=self.fast_loop_stochastic_k,
            context="quality experiment",
        )

    def run(self, tasks_path: str | Path, output_dir: str | Path, seeds: Sequence[int]) -> Dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        tasks = load_tasks(tasks_path)
        seed_list = list(seeds) or [0]
        harness_rows: List[Dict[str, Any]] = []

        executor = HarnessExecutor(
            self.repo,
            self.backend,
            self.inspector,
            self.search_backend,
            prompt_optimizer=self.prompt_optimizer,
            fast_loop_attempts=self.fast_loop_attempts,
            harness_localizer=self.harness_localizer,
            patch_proposer=self.patch_proposer,
            fast_loop_stochastic_k=self.fast_loop_stochastic_k,
        )
        for seed in seed_list:
            for task in tasks:
                for exp in executor.run_task(task, seed=seed, write_memory=False):
                    generation = exp.visual_result.get("raw_generation") or {}
                    image_path = self._save_image(generation, output, "gen_harness", exp.task_id, seed)
                    harness_rows.append(
                        {
                            "method": "gen_harness",
                            "task_id": exp.task_id,
                            "constraint_id": exp.constraint_id,
                            "constraint_type": exp.constraint_type,
                            "seed": seed,
                            "image_uri": image_path or exp.visual_result.get("image_uri"),
                            "verification_result": exp.verification_result,
                            "generation": generation,
                            "program": exp.tool_calls[0].get("arguments", {}).get("program") if exp.tool_calls else None,
                            "metadata": exp.metadata,
                        }
                    )

        write_jsonl(output / "gen_harness_results.jsonl", harness_rows)
        write_jsonl(output / "quality_results.jsonl", harness_rows)
        summary = self._summarize(harness_rows, seed_list)
        write_json(output / "quality_summary.json", summary)
        return summary

    def _summarize(self, harness_rows: List[Dict[str, Any]], seeds: Sequence[int]) -> Dict[str, Any]:
        harness = self._method_summary(harness_rows)
        applicable_metrics = self._applicable_image_metrics(harness_rows)
        harness_metrics = self._image_metric_summary(harness_rows, applicable_metrics)
        return {
            "seeds": list(seeds),
            "num_gen_harness_rows": len(harness_rows),
            "policy_extractor_manifest": self.repo.policy_extractor_manifest(),
            "methods": {
                "gen_harness": harness,
            },
            "image_metrics": {
                "gen_harness": harness_metrics,
                "applicable_metrics": applicable_metrics,
            },
        }

    def _save_image(self, generation: Dict[str, Any], output: Path, method: str, task_id: str, seed: int) -> str | None:
        image_uri = generation.get("image_uri") if isinstance(generation, dict) else None
        if not isinstance(image_uri, str) or not image_uri.startswith("data:") or "," not in image_uri:
            return None
        header, encoded = image_uri.split(",", 1)
        mime = header.split(";", 1)[0].replace("data:", "") or "image/png"
        ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
        }.get(mime, ".img")
        images_dir = output / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        safe_task = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)
        path = images_dir / f"{method}_{safe_task}_seed{seed}{ext}"
        path.write_bytes(base64.b64decode(encoded))
        generation["image_path"] = str(path)
        generation["mime_type"] = mime
        return str(path)

    def _method_summary(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_type.setdefault(row["constraint_type"], []).append(row)
        return {
            "overall": self._rows_summary(rows),
            "by_constraint_type": {ctype: self._rows_summary(ctype_rows) for ctype, ctype_rows in sorted(by_type.items())},
        }

    def _rows_summary(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(rows)
        passed = sum(1 for row in rows if self._passed(row))
        numeric = self._numeric_means(rows)
        return {
            "num_constraints": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "numeric_means": numeric,
        }

    def _numeric_means(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for row in rows:
            result = row.get("verification_result", {})
            for key, value in result.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        return {key: totals[key] / counts[key] for key in sorted(totals)}


    def _image_metric_summary(self, rows: List[Dict[str, Any]], applicable_metrics: List[str]) -> Dict[str, Any]:
        groups = self._image_metric_groups(rows)
        out: Dict[str, Any] = {}
        total_images = len(groups)
        applicable = set(applicable_metrics)
        for metric in IMAGE_METRICS:
            if metric not in applicable:
                out[metric] = {"mean": None, "num_images": 0, "coverage": 0.0, "applicable": False}
                continue
            values = [metrics[metric] for metrics in groups.values() if metric in metrics]
            out[metric] = {
                "mean": sum(values) / len(values) if values else 0.0,
                "num_images": len(values),
                "coverage": len(values) / total_images if total_images else 0.0,
                "applicable": True,
            }
        return out

    def _applicable_image_metrics(self, rows: List[Dict[str, Any]]) -> List[str]:
        constraint_types = {row.get("constraint_type") for row in rows}
        metrics = ["overall_quality", "aesthetic_quality", "prompt_alignment", "artifact_free"]
        if "negative" in constraint_types:
            metrics.append("text_logo_absence")
        if "identity" in constraint_types or "grounded_identity" in constraint_types:
            metrics.append("identity_consistency")
        if "style" in constraint_types:
            metrics.append("style_consistency")
        return [metric for metric in IMAGE_METRICS if metric in set(metrics)]

    def _image_metric_groups(self, rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, float]]:
        groups: Dict[Tuple[str, int], Dict[str, float]] = {}
        for row in rows:
            key = self._image_pair_key(row)
            metrics = self._extract_image_metrics(row)
            if not metrics:
                continue
            groups.setdefault(key, {}).update(metrics)
        return groups

    def _extract_image_metrics(self, row: Dict[str, Any]) -> Dict[str, float]:
        result = row.get("verification_result", {})
        raw_metrics = result.get("image_metrics", {}) if isinstance(result, dict) else {}
        metrics: Dict[str, float] = {}
        if isinstance(raw_metrics, dict):
            for metric in IMAGE_METRICS:
                value = raw_metrics.get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[metric] = min(1.0, max(0.0, float(value)))
        for metric in IMAGE_METRICS:
            value = result.get(metric) if isinstance(result, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics.setdefault(metric, min(1.0, max(0.0, float(value))))
        return metrics


    def _image_pair_key(self, row: Dict[str, Any]) -> Tuple[str, int]:
        return row["task_id"], int(row["seed"])

    def _passed(self, row: Dict[str, Any]) -> bool:
        return bool(row.get("verification_result", {}).get("passed", False))
