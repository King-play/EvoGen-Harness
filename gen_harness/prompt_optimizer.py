from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

from .io import expand_env_vars, read_json
from .json_extract import parse_json_object_from_text
from .leakage_guard import reject_forbidden_payload_context, sanitize_llm_payload_context
from .schema import GenerationTask


class PromptOptimizer:
    """Interface for real prompt enhancement tools."""

    def optimize(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> str:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}


class OpenAIChatPromptOptimizer(PromptOptimizer):
    """OpenAI-compatible chat adapter that rewrites vague user prompts into image briefs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config["base_url"].rstrip("/")
        self.endpoint = config.get("endpoint") or f"{self.base_url}/chat/completions"
        self.headers = dict(config.get("headers", {}))
        self.timeout = int(config.get("timeout", 120))

    def optimize(self, task: GenerationTask, program: Dict[str, Any], seed: int = 0) -> str:
        workflow = program.get("workflow", {})
        payload = sanitize_llm_payload_context({
            "initial_prompt": task.prompt,
            "draft_harness_prompt": workflow.get("compiled_prompt", task.prompt),
            "contract": program.get("contract", {}),
            "memory_context": program.get("memory_context", {}),
            "selected_skill": workflow.get("skill_name"),
        })
        expected = {"enhanced_prompt": "string"}
        body = {
            "model": self.config["model"],
            "temperature": self.config.get("temperature", 0.2),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": (
                        "Rewrite the initial prompt into a concrete image-generation prompt. "
                        "Return JSON exactly in this shape: "
                        + json.dumps(expected, ensure_ascii=False)
                        + "\n"
                        + json.dumps(payload, ensure_ascii=False, indent=2)
                    ),
                },
            ],
        }
        extra_body = self.config.get("extra_body", {})
        reject_forbidden_payload_context(extra_body, context="prompt optimizer extra_body")
        for key, value in extra_body.items():
            body[key] = value
        result = self._post_json(self.endpoint, body)
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = self._parse_json(content)
        enhanced = parsed.get("enhanced_prompt")
        if not isinstance(enhanced, str) or not enhanced.strip():
            raise ValueError(f"prompt optimizer response did not include enhanced_prompt: {parsed}")
        return enhanced.strip()

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "openai_chat_prompt_optimizer",
            "name": self.config.get("name", "prompt_optimizer"),
            "model": self.config.get("model"),
            "endpoint": self.endpoint,
        }

    def _system_prompt(self) -> str:
        return (
            "You are a prompt planner for a general image-generation harness. "
            "Your job is to convert a vague user prompt into a concrete, faithful image-generation brief. "
            "Preserve the user's requested subject and scene. Do not invent unrelated content. "
            "Make the visual plan explicit: subject, setting, composition, camera/framing, lighting, atmosphere, palette, details, and negative controls. "
            "Do not mention policies, harness internals, evaluation, JSON, or that you improved the prompt in the enhanced_prompt. "
            "Return only valid JSON."
        )

    def _parse_json(self, content: str) -> Dict[str, Any]:
        return parse_json_object_from_text(content)

    def _post_json(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"prompt optimizer request failed: HTTP {exc.code}: {detail}") from exc


def build_prompt_optimizer(config_path: str | Path | None) -> PromptOptimizer | None:
    if not config_path:
        return None
    cfg = expand_env_vars(read_json(config_path))
    typ = cfg.get("type")
    if typ == "openai_chat_prompt_optimizer":
        return OpenAIChatPromptOptimizer(cfg)
    raise ValueError(f"unknown or unsupported prompt optimizer type: {typ}")
