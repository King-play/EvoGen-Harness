from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import HarnessComponent
from ..backends import VisualBackend
from ..inspection import InspectionRunner
from ..search import SearchBackend
from ..schema import GenerationTask


class VisualToolRegistry(HarnessComponent):
    """Registry for generation/editing/search/inspection tools.

    Tool metadata is used for routing and for component-level attribution when a
    selected tool lacks a required capability.
    """

    component_type = "tools"

    def __init__(self, generation_tools_path: str | Path, inspection_tools_path: str | Path, backend: VisualBackend, inspector: InspectionRunner, search_backend: SearchBackend | None = None):
        super().__init__(generation_tools_path)
        from ..io import read_json
        self.inspection_artifact_path = Path(inspection_tools_path)
        self.inspection_artifact = read_json(self.inspection_artifact_path)
        self.backend = backend
        self.inspector = inspector
        self.search_backend = search_backend

    def get_generation_tool(self, name: str) -> Dict[str, Any]:
        tools = self.artifact.get("tools", {})
        if name not in tools:
            raise KeyError(f"unknown generation tool: {name}")
        return tools[name]

    def get_inspection_tool(self, name: str | None = None) -> Dict[str, Any]:
        tools = self.inspection_artifact.get("tools", {})
        if not tools:
            raise ValueError("no inspection tools registered")
        if name is None:
            name = next(iter(tools))
        if name not in tools:
            raise KeyError(f"unknown inspection tool: {name}")
        return tools[name]

    def supports(self, tool_name: str, capability: str) -> bool:
        tool = self.get_generation_tool(tool_name)
        return bool(tool.get("capabilities", {}).get(capability, False))

    def execute_search(self, task: GenerationTask, contract: Dict[str, Any]) -> Dict[str, Any]:
        if self.search_backend is None:
            raise ValueError("visual contract requires search/evidence, but no search backend was configured")
        return self.search_backend.search(task, contract)

    def execute_generation(self, task: GenerationTask, program: Dict[str, Any], seed: int) -> Dict[str, Any]:
        return self.backend.generate(task, program, seed=seed)

    def execute_inspection(self, task: GenerationTask, program: Dict[str, Any], generation: Dict[str, Any], seed: int) -> Dict[str, Dict[str, Any]]:
        return self.inspector.inspect(task, program, generation, seed=seed)

    def responsible_decisions(self) -> List[str]:
        return ["tool capability declaration", "tool execution", "inspection execution", "tool failure specification"]
