from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict

from ..backend import VisualBackend
from ..leakage_guard import reject_forbidden_payload_context
from ..schema import GenerationTask
from ..tool_payloads import public_generation_input


class HTTPImageGenerationBackend(VisualBackend):
    """Generic HTTP adapter for hosted image-generation APIs.

    The endpoint receives a public semantic `input` payload plus `seed`, not
    the full task or harness program. It must return JSON with at least
    `image_uri` or `images`. This adapter performs no synthetic generation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoint = config["endpoint"]
        self.headers = dict(config.get("headers", {}))
        self.timeout = int(config.get("timeout", 300))
        self.poll_endpoint = config.get("poll_endpoint")
        self.poll_interval = float(config.get("poll_interval", 2.0))

    def generate(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
        request_metadata = self.config.get("request_metadata", {})
        reject_forbidden_payload_context(request_metadata, context="HTTP generation request_metadata")
        payload = {
            "input": public_generation_input(task, program),
            "seed": seed,
            "request_metadata": request_metadata,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method=self.config.get("method", "POST"))
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if self.poll_endpoint and "job_id" in result:
            result = self._poll(result["job_id"])
        if "image_uri" not in result and "images" not in result:
            raise ValueError("HTTP generation backend must return image_uri or images")
        result.setdefault("backend", self.config.get("name", "http_generation"))
        return result

    def _poll(self, job_id: str) -> Dict[str, Any]:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            url = self.poll_endpoint.format(job_id=job_id)
            req = urllib.request.Request(url, headers=self.headers, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            if result.get("status") in {"succeeded", "completed", "done"}:
                return result
            if result.get("status") in {"failed", "error"}:
                raise RuntimeError(f"generation job failed: {result}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"generation job timed out: {job_id}")
