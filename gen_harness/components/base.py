from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..io import read_json, write_json


@dataclass
class ComponentInfo:
    name: str
    component_type: str
    version: str
    artifact_path: str
    validation_scope: str
    rollback_boundary: str


class HarnessComponent:
    component_type = "component"

    def __init__(self, artifact_path: str | Path):
        self.artifact_path = Path(artifact_path)
        self.artifact = read_json(self.artifact_path)
        self.info = ComponentInfo(
            name=self.artifact.get("component_name", self.artifact_path.stem),
            component_type=self.component_type,
            version=str(self.artifact.get("version", "0")),
            artifact_path=str(self.artifact_path),
            validation_scope=self.artifact.get("validation_scope", self.component_type),
            rollback_boundary=self.artifact.get("rollback_boundary", str(self.artifact_path)),
        )

    def save(self) -> None:
        write_json(self.artifact_path, self.artifact)

    def describe(self) -> Dict[str, Any]:
        return {
            "component_name": self.info.name,
            "component_type": self.info.component_type,
            "version": self.info.version,
            "artifact_path": self.info.artifact_path,
            "validation_scope": self.info.validation_scope,
            "rollback_boundary": self.info.rollback_boundary,
        }
