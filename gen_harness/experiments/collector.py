from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from ..debugger import FailureDebugger
from ..io import write_json, write_jsonl
from ..schema import VisualExperience


class NaturalFailureCollector:
    def collect(self, experiences: Iterable[VisualExperience], output_dir: str | Path) -> Dict[str, object]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        exps = list(experiences)
        failures = [e for e in exps if not e.verification_result.get("passed")]
        debugger = FailureDebugger()
        diagnoses = debugger.diagnose_many(failures)
        write_jsonl(output / "natural_failures.jsonl", [e.to_compact_dict() for e in failures])
        write_jsonl(output / "failure_diagnoses.jsonl", [d.to_dict() for d in diagnoses])
        by_component: Dict[str, int] = {}
        by_symptom: Dict[str, int] = {}
        by_seed: Dict[str, Dict[str, int]] = {}
        for e in failures:
            by_component[e.suspected_component or "unknown"] = by_component.get(e.suspected_component or "unknown", 0) + 1
            by_symptom[e.failure_symptom or "unknown"] = by_symptom.get(e.failure_symptom or "unknown", 0) + 1
        for e in exps:
            seed = str((e.visual_result or {}).get("seed", "unknown"))
            seed_row = by_seed.setdefault(seed, {"num_experiences": 0, "num_failures": 0})
            seed_row["num_experiences"] += 1
            if not e.verification_result.get("passed"):
                seed_row["num_failures"] += 1
        summary = {
            "num_experiences": len(exps),
            "num_failures": len(failures),
            "failure_rate": len(failures) / len(exps) if exps else 0.0,
            "by_component": by_component,
            "by_symptom": by_symptom,
            "by_seed": {
                seed: {
                    **row,
                    "failure_rate": row["num_failures"] / row["num_experiences"] if row["num_experiences"] else 0.0,
                }
                for seed, row in sorted(by_seed.items())
            },
            "failure_diagnosis": debugger.summary(diagnoses),
        }
        write_json(output / "natural_failure_summary.json", summary)
        return summary
