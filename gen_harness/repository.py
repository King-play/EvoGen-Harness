from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from .components import VisualMemory, VisualMiddleware, VisualPolicy, VisualSkillLibrary, VisualToolRegistry
from .backends import VisualBackend
from .inspection import InspectionRunner
from .search import SearchBackend
from .io import read_json, read_jsonl, write_json, write_jsonl
from .schema import PatchManifest


COMPONENTS = ("policy", "tools", "skills", "middleware", "memory")


class HarnessRepository:
    """File-backed visual harness repository with concrete component objects."""

    def __init__(
        self,
        root: str | Path,
        component_roots: Dict[str, Path] | None = None,
        *,
        policy_extractor_config: str | Path | None = None,
        policy_extractor: Any | None = None,
    ):
        self.root = Path(root)
        self.component_roots = dict(component_roots or {})
        self.policy_extractor_config = policy_extractor_config
        self.policy_extractor = policy_extractor

    def component_dir(self, component: str) -> Path:
        if component not in COMPONENTS:
            raise ValueError(f"unknown component: {component}")
        return self.component_roots.get(component, self.root / component)


    def with_component_override(self, component: str, source_repo: "HarnessRepository") -> "HarnessRepository":
        if component not in COMPONENTS:
            raise ValueError(f"unknown component: {component}")
        component_roots = dict(self.component_roots)
        component_roots[component] = source_repo.component_dir(component)
        return HarnessRepository(
            self.root,
            component_roots=component_roots,
            policy_extractor_config=self.policy_extractor_config,
            policy_extractor=self.policy_extractor,
        )

    def artifact_path(self, component: str, artifact: str) -> Path:
        if not isinstance(artifact, str) or artifact in {"", ".", ".."} or Path(artifact).name != artifact:
            raise ValueError("artifact must be a filename within its responsibility directory")
        path = self.component_dir(component) / artifact
        if path.is_symlink():
            raise ValueError("artifact symlinks are not permitted")
        return path

    def load_artifact(self, component: str, artifact: str) -> Dict[str, Any]:
        return read_json(self.artifact_path(component, artifact))

    def save_artifact(self, component: str, artifact: str, data: Dict[str, Any]) -> None:
        write_json(self.artifact_path(component, artifact), data)

    def build_policy(self) -> VisualPolicy:
        return VisualPolicy(
            self.artifact_path("policy", "visual_contract.json"),
            self.artifact_path("policy", "reference_policy.json"),
            self.artifact_path("policy", "preservation_policy.json"),
            policy_extractor_config=self.policy_extractor_config,
            policy_extractor=self.policy_extractor,
        )

    def policy_extractor_manifest(self) -> Dict[str, Any]:
        policy = self.build_policy()
        extractor = policy.extractor.describe()
        return {
            "policy_extractor": extractor,
            "policy_extractor_configured": self.policy_extractor_config is not None,
            "formal_llm_policy_extractor": extractor.get("component_name") == "openai_chat_policy_extractor",
        }

    def build_tools(self, backend: VisualBackend, inspector: InspectionRunner, search_backend: SearchBackend | None = None) -> VisualToolRegistry:
        return VisualToolRegistry(
            self.artifact_path("tools", "generation_tools.json"),
            self.artifact_path("tools", "inspection_tools.json"),
            backend,
            inspector,
            search_backend,
        )

    def build_skills(self) -> VisualSkillLibrary:
        return VisualSkillLibrary(self.artifact_path("skills", "skills.json"))

    def build_middleware(self) -> VisualMiddleware:
        return VisualMiddleware(
            self.artifact_path("middleware", "skill_selector.json"),
            self.artifact_path("middleware", "reference_binder.json"),
            self.artifact_path("middleware", "tool_router.json"),
        )

    def build_memory(self) -> VisualMemory:
        return VisualMemory(self.component_dir("memory"))

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        snap: Dict[str, Dict[str, Any]] = {}
        for component in COMPONENTS:
            cdir = self.component_dir(component)
            snap[component] = {}
            if cdir.exists():
                for path in sorted(cdir.glob("*.json")):
                    snap[component][path.name] = read_json(path)
        return snap

    def snapshot_persistent_state(self) -> Dict[str, Dict[str, Any]]:
        """Capture every editable JSON/JSONL artifact for path-level rollback."""
        state: Dict[str, Dict[str, Any]] = {}
        for component in COMPONENTS:
            component_state: Dict[str, Any] = {}
            cdir = self.component_dir(component)
            if cdir.exists():
                for path in sorted(cdir.glob("*.json")):
                    component_state[path.name] = {"format": "json", "value": read_json(path)}
                for path in sorted(cdir.glob("*.jsonl")):
                    component_state[path.name] = {"format": "jsonl", "value": read_jsonl(path)}
            state[component] = component_state
        return state

    def restore_persistent_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """Restore an exact previously captured editable state.

        Only JSON/JSONL files inside the five component directories are in the
        persistent-state boundary.  Remove files created after the snapshot as
        well as rewriting captured files so a path-level rollback is exact.
        """
        for component in COMPONENTS:
            cdir = self.component_dir(component)
            captured = set(state.get(component, {}))
            if cdir.exists():
                for pattern in ("*.json", "*.jsonl"):
                    for path in cdir.glob(pattern):
                        if path.name not in captured:
                            path.unlink()
            for artifact, record in state.get(component, {}).items():
                path = self.artifact_path(component, artifact)
                if record.get("format") == "jsonl":
                    write_jsonl(path, record.get("value", []))
                elif record.get("format") == "json":
                    write_json(path, record.get("value", {}))

    @classmethod
    def materialize_persistent_state(
        cls,
        root: str | Path,
        state: Dict[str, Dict[str, Any]],
        *,
        policy_extractor_config: str | Path | None = None,
        policy_extractor: Any | None = None,
    ) -> "HarnessRepository":
        repo = cls(
            root,
            policy_extractor_config=policy_extractor_config,
            policy_extractor=policy_extractor,
        )
        for component in COMPONENTS:
            repo.component_dir(component).mkdir(parents=True, exist_ok=True)
        repo.restore_persistent_state(state)
        return repo


    @contextmanager
    def temporary_patch(self, patch: PatchManifest) -> Iterator[Tuple[Any, Any]]:
        path = self.artifact_path(patch.target_component, patch.changed_artifact)
        existed = path.exists()
        before, after = self.apply_patch(patch)
        try:
            yield before, after
        finally:
            if not existed:
                path.unlink(missing_ok=True)
            elif patch.operation == "append_jsonl_unique":
                write_jsonl(path, before)
            else:
                self.save_artifact(patch.target_component, patch.changed_artifact, before)

    def apply_patch(self, patch: PatchManifest) -> Tuple[Any, Any]:
        component, artifact = patch.target_component, patch.changed_artifact
        if patch.operation == "append_jsonl_unique":
            path = self.artifact_path(component, artifact)
            before_rows = read_jsonl(path)
            after_rows = list(before_rows)
            row = dict(patch.payload["row"])
            unique_key = patch.payload.get("unique_key")
            if unique_key:
                exists = any(existing.get(unique_key) == row.get(unique_key) for existing in after_rows)
            else:
                exists = row in after_rows
            if not exists:
                after_rows.append(row)
            write_jsonl(path, after_rows)
            return before_rows, after_rows

        before = self.load_artifact(component, artifact)
        after = copy.deepcopy(before)
        if patch.operation == "set_path":
            path = patch.payload["path"]
            value = patch.payload["value"]
            node = after
            for key in path[:-1]:
                if key not in node or not isinstance(node[key], dict):
                    node[key] = {}
                node = node[key]
            node[path[-1]] = value
        elif patch.operation == "append_unique":
            path = patch.payload["path"]
            value = patch.payload["value"]
            node = after
            for key in path[:-1]:
                node = node.setdefault(key, {})
            arr = node.setdefault(path[-1], [])
            if value not in arr:
                arr.append(value)
        elif patch.operation == "delete_path":
            path = patch.payload["path"]
            node = after
            for key in path[:-1]:
                if key not in node or not isinstance(node[key], dict):
                    raise ValueError(f"delete_path parent does not exist: {'.'.join(path[:-1])}")
                node = node[key]
            if path[-1] not in node:
                raise ValueError(f"delete_path target does not exist: {'.'.join(path)}")
            del node[path[-1]]
        else:
            raise ValueError(f"unsupported patch operation: {patch.operation}")
        self.save_artifact(component, artifact, after)
        return before, after
