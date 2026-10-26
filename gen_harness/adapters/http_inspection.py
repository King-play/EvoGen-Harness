from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict

from ..inspection import InspectionRunner
from ..schema import GenerationTask, VisualConstraint
from ..tool_payloads import (
    constraints_for_external_tool,
    public_constraint_ids,
    public_inspection_payload,
    remap_public_constraint_results,
)


class HTTPInspectionRunner(InspectionRunner):
    """Generic HTTP adapter for VLM judges or metric services."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoint = config["endpoint"]
        self.headers = dict(config.get("headers", {}))
        self.timeout = int(config.get("timeout", 300))

    def inspect(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int = 0) -> Dict[str, Dict[str, Any]]:
        constraints = self._constraints_for_inspection(task, program)
        public_constraint_ids = self._public_constraint_ids(constraints)
        payload = {
            "input": public_inspection_payload(
                task,
                program,
                generation,
                constraints=constraints,
                public_ids=public_constraint_ids,
            ),
            "seed": seed,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method=self.config.get("method", "POST"))
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if "constraint_results" not in result:
            raise ValueError("HTTP inspection backend must return constraint_results")
        return self._remap_constraint_results(result["constraint_results"], constraints, public_constraint_ids)

    def _constraints_for_inspection(self, task: GenerationTask, program: Dict[str, Any]) -> list[VisualConstraint]:
        return constraints_for_external_tool(task, program)

    def _public_constraint_ids(self, constraints: list[VisualConstraint]) -> Dict[str, str]:
        return public_constraint_ids(constraints)

    def _remap_constraint_results(
        self,
        rows: Dict[str, Any],
        constraints: list[VisualConstraint],
        public_constraint_ids: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        return remap_public_constraint_results(rows, constraints, public_constraint_ids)
