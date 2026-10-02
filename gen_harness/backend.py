from __future__ import annotations

from typing import Any, Dict

from .schema import GenerationTask


class VisualBackend:
    """Interface for real image-generation backends.

    Implementations must call an actual generator, workflow engine, or visual
    agent. Implementations are supplied by explicit backend adapters.
    """

    def generate(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
        raise NotImplementedError
