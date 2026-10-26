from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict

from ..schema import GenerationTask
from ..search import SearchBackend
from ..tool_payloads import public_search_input


class HTTPSearchBackend(SearchBackend):
    """Generic HTTP adapter for search/evidence/reference retrieval APIs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoint = config["endpoint"]
        self.headers = dict(config.get("headers", {}))
        self.timeout = int(config.get("timeout", 120))

    def search(self, task: GenerationTask, contract: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"input": public_search_input(task, contract)}
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method=self.config.get("method", "POST"))
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        result.setdefault("evidence", [])
        result.setdefault("references", [])
        return result
