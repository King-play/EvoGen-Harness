from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .components.memory import VisualMemory
from .io import expand_env_vars, read_json
from .json_extract import parse_json_object_from_text
from .leakage_guard import provenance_key_is_forbidden, reject_forbidden_payload_context, reject_forbidden_trace_text
from .schema import PatchManifest, Weakness
from .trace_protocol import (
    TRACE_DEFAULT_K,
    TRACE_EVIDENCE_SCHEMA,
    TRACE_LOCALIZER_OUTPUT_SCHEMA,
    validate_fixed_llm_config,
    validate_localizer_output,
    validate_trace_evidence,
)


COMPONENT_ARTIFACTS: Dict[str, set[str]] = {
    "policy": {"visual_contract.json", "preservation_policy.json", "reference_policy.json"},
    "tools": {"generation_tools.json", "inspection_tools.json"},
    "skills": {"skills.json"},
    "middleware": {"reference_binder.json", "tool_router.json", "skill_selector.json"},
    "memory": {"prompt_guidelines.jsonl"},
}
COMPONENT_STATE_ARTIFACTS: Dict[str, set[str]] = {
    **COMPONENT_ARTIFACTS,
    "memory": {
        "memory_contract.json",
        "visual_experiences.jsonl",
        "failure_clusters.jsonl",
        "weaknesses.jsonl",
        "validated_patches.jsonl",
        "regression_cases.jsonl",
        "prompt_guidelines.jsonl",
    },
}
EDITABLE_RESPONSIBILITIES = tuple(COMPONENT_ARTIFACTS)
NO_PATCH_LABEL = "no_patch"

ALLOWED_OPERATIONS = {"set_path", "append_unique", "append_jsonl_unique", "delete_path"}
LOCALIZER_SYSTEM_PROMPT = """You are the Harness Localizer in TRACE, a fixed LLM role in
an attribution-guided evolution system for visual generation agents.

Your task is to identify where a persistent, model-external harness weakness should be
investigated. You are an attribution component, not a repair component. Visual
verification establishes what failed. You determine which harness decision
responsibilities could explain the failure and should enter the bounded evolution
frontier. You do not propose edits, select candidates, validate patches, change the
generator, or claim that a repair will succeed.

The harness is decomposed into five independent visual decision responsibilities:

1. POLICY
   Policy interprets the request and defines the visual contract, constraint ontology,
   evidence rules, reference requirements, preservation requirements, and negative
   requirements. Score policy high when the harness misses, misclassifies, or
   over-generalizes a visual constraint, or when the failure indicates that the
   requirement itself was represented incorrectly.

2. TOOLS
   Tools describe generator and verifier capabilities, limitations, modes, inputs,
   thresholds, and inspection contracts. Score tools high when repeated evidence
   contradicts what the harness believes a generator or verifier can reliably do, or
   when a capability, limitation, threshold, input mode, or inspection contract is
   wrong or missing.

3. SKILLS
   Skills provide reusable visual generation procedures and prompt-family routines.
   They include decomposition, prompt construction, layout planning, object and
   attribute binding, counting, spatial reasoning, reference use, and verification-
   aware repair procedures. Score skills high when a reusable procedure is inadequate
   across similar requests.

4. MIDDLEWARE
   Middleware performs runtime orchestration: skill selection, routing, reference
   binding, retry decisions, candidate selection, termination, and execution ordering.
   Score middleware high when the available policy, tools, skills, and memory appear
   sufficient but the execution path selected or used them incorrectly.

5. MEMORY
   Memory stores reusable cross-request experience, recurring failure patterns,
   validated repairs, preservation warnings, and lessons that should transfer to
   future requests. Experience, weakness, validation, and regression records are
   controller-owned read-only evidence. Prompt guidelines are the only
   LLM-editable Memory artifact. Score memory high when the evidence reveals a
   recurring pattern or a missing reusable lesson rather than a defect in the
   current policy, tools, skills, or middleware.

6. NO_PATCH
   NO_PATCH means that persistent harness evolution is not justified. Score it high
   when the evidence is better explained by stochastic generation, ambiguous intent,
   verifier uncertainty, insufficient support, a one-off event, or overfitting risk.
   Repeated, constraint-indexed, cross-execution evidence for a reusable weakness
   should reduce the NO_PATCH score.

Perform attribution using the following reasoning procedure:

1. Identify the explicit visual constraint that failed and separate it from the
   observed symptom.
2. Read E_t as constraint-indexed evidence. Each constraint instance contains
   exactly K stochastic executions under the formal protocol, and each execution
   contains the seed, trajectory decisions, tool calls, visual observation,
   verifier evidence, and pass/failure outcome. Passing executions are evidence
   too: they may indicate generation variance rather than a persistent weakness.
3. Determine whether the failure pattern is repeated across the complete K-level
   evidence rather than assuming that every execution must fail.
4. Inspect the execution trajectory and current harness state for the earliest
   responsibility-specific decision that could plausibly explain the failure.
5. Distinguish requirement interpretation, capability knowledge, reusable procedure,
   runtime orchestration, and persistent experience. Do not assign a failure to a
   downstream responsibility merely because it is visible there.
6. Consider multiple responsibilities when the evidence supports interacting causes.
   Do not force a single winner; produce a compact frontier of plausible causes.
7. Prefer NO_PATCH over speculative attribution when the evidence cannot distinguish
   a persistent harness weakness from generation variance or verifier uncertainty.
8. Favor reusable, cross-request explanations and reject task-specific explanations,
   benchmark-specific explanations, and explanations that depend on post-hoc image
   selection.

Use comparable numeric scores from 0 to 1 for every responsibility and NO_PATCH:
0.0 means no evidence, 0.3 means weak or indirect evidence, 0.6 means plausible
responsibility with concrete evidence, and 0.9 means strong repeated evidence that
the responsibility should be explored. Scores are independent evidence strengths,
not probabilities and do not need to sum to one. A positive score does not by itself
authorize a patch. The deterministic controller will retain the Top-r frontier and
will stop when NO_PATCH is preferred.

Use only the trajectory-relative internal-verifier evidence and sanitized current
harness state supplied in the request. Never use official benchmark scores, official
evaluator outputs, hidden labels, benchmark answers, task identifiers, fallback
images, post-hoc image selection, or task-specific fixes. Do not infer a cause from
an external score that is not present in the supplied evidence. Do not invent
observations, executions, or tool calls.

Return one valid JSON object and nothing else. The object must contain exactly
the output schema named gen_harness.trace_localizer_output.v1: a scores object
with exactly the six required keys, a constraint_attributions array containing
one attribution for every failed constraint in E_t, and a non-empty string
rationale. Each constraint attribution must contain constraint_id, scores,
primary_component, confidence, and evidence_summary. Every score and confidence
must be a finite number in the inclusive range [0, 1]. Do not add fields.
The rationale must explain the strongest responsibility, mention meaningful
uncertainty or competing responsibilities when present, and state why NO_PATCH
is or is not preferred. Do not include edits, patch operations, file paths, or
validation claims."""


def _post_chat_json(config: Dict[str, Any], endpoint: str, body: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", **dict(config.get("headers", {}))}
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc


def _chat_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"LLM response did not contain choices: {response}")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"LLM response did not contain message content: {response}")
    return content


def _reject_forbidden(value: Any) -> None:
    reject_forbidden_trace_text(value, context="LLM proposal")
    _reject_forbidden_structured_provenance(value)


def _failed_constraint_ids_from_trace_evidence(weakness: Weakness) -> List[str] | None:
    evidence = weakness.evidence.get("trace_evidence") if isinstance(weakness.evidence, dict) else None
    if not isinstance(evidence, dict):
        return None
    failed: set[str] = set()
    for instance in evidence.get("constraint_instances", []):
        if not isinstance(instance, dict):
            continue
        constraint = instance.get("constraint", {})
        constraint_id = str(constraint.get("constraint_id", "")).strip() if isinstance(constraint, dict) else ""
        if not constraint_id:
            continue
        executions = instance.get("executions", [])
        if any(isinstance(row, dict) and row.get("outcome") == "failure" for row in executions):
            failed.add(constraint_id)
    return sorted(failed)


def _reject_forbidden_structured_provenance(value: Any, *, path: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text != "scores" and provenance_key_is_forbidden(key_text):
                location = f" at {path}.{key_text}" if path else f" at {key_text}"
                raise ValueError(f"LLM proposal contains task, benchmark, score, official, or fallback provenance{location}")
            child_path = f"{path}.{key_text}" if path else key_text
            _reject_forbidden_structured_provenance(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_structured_provenance(child, path=f"{path}[{index}]")


def _reject_task_specific_patch_context(value: Any) -> None:
    """Reject dataset-, task-, or official-evaluator-specific patch content."""

    text = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    forbidden = (
        "geneval",
        "geneval2",
        "t2icompbench",
        "t2i-compbench",
        "wiscore",
        "partiprompts",
        "parti_prompts",
        "soft_tifa",
        "official_score",
        "official evaluator",
        "official_evaluator",
        "source_index",
        "task_id",
        "benchmark",
        "dataset",
        "image_map",
        "vqa_list",
        "question_answers",
    )
    hit = next((token for token in forbidden if token in text), None)
    if hit is not None:
        raise ValueError(f"patch proposal contains task- or benchmark-specific content: {hit}")


@dataclass
class HarnessLocalization:
    weakness_id: str
    scores: Dict[str, float]
    frontier: List[str]
    no_patch_preferred: bool
    rationale: str = ""
    constraint_attributions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationPatch:
    """One transient, schema-checked patch for the next visual attempt."""

    target_component: str
    operation: str
    path: List[str]
    value: Any
    predicted_improvement: str
    possible_regression: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HarnessLocalizer:
    """LLM-backed TRACE harness localizer.

    The localizer decides where evolution should be explored. It returns scores
    over the five editable responsibilities and NO-PATCH; it never proposes an
    edit and never sees official benchmark labels or scores.
    """

    def __init__(self, config: Dict[str, Any], repo: Any | None = None, *, strict_protocol: bool = False):
        if not isinstance(config, dict) or config.get("type") != "openai_chat_harness_localizer":
            raise ValueError("HarnessLocalizer requires an openai_chat_harness_localizer config")
        self.config = config
        self.repo = repo
        self.base_url = str(config["base_url"]).rstrip("/")
        self.endpoint = str(config.get("endpoint") or f"{self.base_url}/chat/completions")
        self.model = str(config["model"])
        self.model_snapshot = str(config.get("model_snapshot", "")).strip()
        self.strict_protocol = bool(strict_protocol)
        if self.strict_protocol:
            validate_fixed_llm_config(config, role="Harness Localizer")
        self.timeout = int(config.get("timeout", 120))
        self.max_localization_attempts = max(1, int(config.get("max_localization_attempts", 2)))
        self.request_count = 0

    def localize(self, weakness: Weakness, *, frontier_size: int = 3) -> HarnessLocalization:
        last_error: Exception | None = None
        for attempt in range(self.max_localization_attempts):
            try:
                row = self._request_localization_row(weakness, retry_error=last_error)
                return self._localization_from_row(weakness, row, frontier_size=frontier_size)
            except (TypeError, ValueError, KeyError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_localization_attempts:
                    raise
        raise RuntimeError("Harness Localizer exhausted without a result")

    def _request_localization_row(
        self,
        weakness: Weakness,
        *,
        retry_error: Exception | None = None,
    ) -> Dict[str, Any]:
        if self.strict_protocol:
            trace_evidence = weakness.evidence.get("trace_evidence") if isinstance(weakness.evidence, dict) else None
            if not isinstance(trace_evidence, dict) or trace_evidence.get("schema") != TRACE_EVIDENCE_SCHEMA:
                raise ValueError("paper TRACE Localizer requires weakness evidence with trace_evidence.v1")
            validate_trace_evidence(trace_evidence, expected_k=TRACE_DEFAULT_K)
        request_payload = {
            "instruction": "Score where persistent harness evolution should be explored. Return strict JSON only.",
            "expected_schema": {
                "scores": {
                    "policy": 0.0,
                    "tools": 0.0,
                    "skills": 0.0,
                    "middleware": 0.0,
                    "memory": 0.0,
                    "no_patch": 0.0,
                },
                "constraint_attributions": [
                    {
                        "constraint_id": "constraint_0",
                        "scores": {
                            "policy": 0.0,
                            "tools": 0.0,
                            "skills": 0.0,
                            "middleware": 0.0,
                            "memory": 0.0,
                            "no_patch": 0.0,
                        },
                        "primary_component": "policy|tools|skills|middleware|memory|no_patch",
                        "confidence": 0.0,
                        "evidence_summary": "brief constraint-specific evidence",
                    }
                ],
                "rationale": "brief evidence-based reason",
            },
            "responsibilities": list(EDITABLE_RESPONSIBILITIES),
            "evidence_schema": TRACE_EVIDENCE_SCHEMA,
            "output_schema": TRACE_LOCALIZER_OUTPUT_SCHEMA,
            "current_harness_state": _current_harness_state_for_llm(self.repo, None),
            "weakness": sanitize_trace_weakness_for_llm(weakness),
        }
        if retry_error is not None:
            request_payload["retry_correction"] = {
                "previous_request_issue": str(retry_error)[:1000],
                "instruction": "Return exactly the required JSON object with finite scores for all six keys and a non-empty rationale.",
            }
        request = {
            "model": self.model_snapshot if self.strict_protocol else self.model,
            "temperature": self.config.get("temperature", 0.0),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(request_payload, ensure_ascii=False, indent=2),
                },
            ],
        }
        extra_body = self.config.get("extra_body", {})
        reject_forbidden_payload_context(extra_body, context="harness localizer extra_body")
        for key, value in extra_body.items():
            request[key] = value
        self.request_count += 1
        response = _post_chat_json(self.config, self.endpoint, request, self.timeout)
        parsed = parse_json_object_from_text(_chat_content(response))
        _reject_forbidden(parsed)
        return parsed

    def _system_prompt(self) -> str:
        return LOCALIZER_SYSTEM_PROMPT

    def _localization_from_row(self, weakness: Weakness, row: Any, *, frontier_size: int) -> HarnessLocalization:
        _reject_forbidden(row)
        validated = validate_localizer_output(
            row,
            expected_constraint_ids=_failed_constraint_ids_from_trace_evidence(weakness),
        )
        scores = validated["scores"]
        normalized: Dict[str, float] = {}
        for key in (*EDITABLE_RESPONSIBILITIES, NO_PATCH_LABEL):
            normalized[key] = float(scores[key])
        ranked = sorted(
            EDITABLE_RESPONSIBILITIES,
            key=lambda name: normalized.get(name, 0.0),
            reverse=True,
        )
        size = max(1, int(frontier_size))
        # r >= |R| is the paper's full-space search ablation: retain every
        # editable responsibility, including a zero-scored one, so the run is
        # genuinely exhaustive over R rather than merely a wider Top-r list.
        frontier = (
            list(ranked)
            if size >= len(EDITABLE_RESPONSIBILITIES)
            else [name for name in ranked[:size] if normalized.get(name, 0.0) > 0.0]
        )
        best_editable = max((normalized[name] for name in EDITABLE_RESPONSIBILITIES), default=0.0)
        no_patch_preferred = normalized[NO_PATCH_LABEL] >= best_editable
        if no_patch_preferred:
            frontier = []
        return HarnessLocalization(
            weakness_id=weakness.weakness_id,
            scores=normalized,
            frontier=frontier,
            no_patch_preferred=no_patch_preferred,
            rationale=validated["rationale"],
            constraint_attributions=validated["constraint_attributions"],
        )


class PatchProposer:
    """LLM-backed responsibility-conditioned patch proposer."""

    def __init__(
        self,
        config: Dict[str, Any],
        memory: VisualMemory | None = None,
        repo: Any | None = None,
        *,
        strict_protocol: bool = False,
    ):
        if not isinstance(config, dict) or config.get("type") != "openai_chat_patch_proposer":
            raise ValueError("PatchProposer requires an openai_chat_patch_proposer config")
        self.config = config
        self.memory = memory
        self.repo = repo
        self.base_url = str(config["base_url"]).rstrip("/")
        self.endpoint = str(config.get("endpoint") or f"{self.base_url}/chat/completions")
        self.model = str(config["model"])
        self.model_snapshot = str(config.get("model_snapshot", "")).strip()
        self.strict_protocol = bool(strict_protocol)
        if self.strict_protocol:
            validate_fixed_llm_config(config, role="Patch Proposer")
        self.timeout = int(config.get("timeout", 120))
        self.max_proposal_attempts = max(1, int(config.get("max_proposal_attempts", 2)))
        self.max_generation_patch_attempts = max(1, int(config.get("max_generation_patch_attempts", 2)))
        self.request_count = 0

    def propose(self, weakness: Weakness, *, responsibilities: List[str] | None = None) -> Optional[PatchManifest]:
        frontier = {weakness.weakness_id: responsibilities} if responsibilities else None
        patches = self.propose_many([weakness], frontier_by_weakness=frontier)
        return patches[0] if patches else None

    def propose_many(
        self,
        weaknesses: List[Weakness],
        *,
        frontier_by_weakness: Dict[str, List[str]] | None = None,
        max_candidates_per_responsibility: int = 3,
    ) -> List[PatchManifest]:
        patches: List[PatchManifest] = []
        seen = set()
        responsibility_counts: Dict[str, int] = {}
        for weakness in weaknesses:
            responsibilities = None
            if frontier_by_weakness is not None:
                responsibilities = frontier_by_weakness.get(weakness.weakness_id, [])
            requests = responsibilities if responsibilities is not None else [None]
            for responsibility in requests:
                if responsibility is not None and responsibility_counts.get(responsibility, 0) >= max_candidates_per_responsibility:
                    continue
                last_error: Exception | None = None
                request_patches: List[PatchManifest] = []
                for attempt in range(self.max_proposal_attempts):
                    try:
                        rows = self._request_patch_rows(weakness, responsibility=responsibility)
                        if responsibility is not None:
                            remaining = max(0, int(max_candidates_per_responsibility)) - responsibility_counts.get(responsibility, 0)
                            rows = rows[:remaining]
                        request_patches = [
                            self._manifest_from_row(weakness, row, index, responsibility=responsibility)
                            for index, row in enumerate(rows)
                        ]
                        last_error = None
                        break
                    except (TypeError, ValueError, KeyError) as exc:
                        last_error = exc
                        if attempt + 1 >= self.max_proposal_attempts:
                            raise
                if last_error is not None:
                    raise last_error
                for patch in request_patches:
                    if self._known_regressive(patch):
                        continue
                    key = (
                        patch.target_component,
                        patch.changed_artifact,
                        patch.operation,
                        json.dumps(patch.payload, sort_keys=True, ensure_ascii=False),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    patches.append(patch)
                    if responsibility is not None:
                        responsibility_counts[responsibility] = responsibility_counts.get(responsibility, 0) + 1
        return patches

    def propose_generation_patch(
        self,
        weakness: Weakness,
        localization: HarnessLocalization,
        program: Dict[str, Any],
    ) -> GenerationPatch:
        """Propose exactly one bounded patch for the next transient attempt.

        This is deliberately separate from persistent artifact patches. The
        fast loop must change the current visual program, not mutate the
        repository or silently invoke a deterministic repair helper.
        """
        if localization.no_patch_preferred or not localization.frontier:
            raise ValueError("generation patch proposal requires a non-empty localized frontier")
        request_payload = {
            "instruction": "Return exactly one transient visual-program patch for the next serial generation attempt.",
            "output_schema": {
                "target_component": "policy|tools|skills|middleware|memory",
                "operation": "append_prompt_clause|set_generation_parameter",
                "path": ["workflow", "t5_prompt"],
                "value": "JSON value",
                "predicted_improvement": "evidence-based reason",
                "possible_regression": "bounded risk",
            },
            "localized_frontier": localization.to_dict(),
            "weakness": sanitize_trace_weakness_for_llm(weakness),
            "current_visual_program": _sanitized_visual_program_for_llm(program),
            "allowed_transient_paths": {
                "prompt": [
                    ["workflow", "t5_prompt"],
                ],
                "generation_parameters": [
                    ["tool", "generation_parameters", "num_inference_steps"],
                    ["tool", "generation_parameters", "guidance_scale"],
                    ["tool", "generation_parameters", "max_sequence_length"],
                ],
            },
            "operation_component_contract": {
                "append_prompt_clause": "target_component must be policy, skills, middleware, or memory; never tools",
                "set_generation_parameter": "target_component must be tools",
            },
            "contract": {
                "one_patch_only": True,
                "repository_mutation": False,
                "image_generation_model_mutation": False,
                "official_evaluator_used": False,
                "posthoc_candidate_selection": False,
            },
        }
        extra_body = self.config.get("extra_body", {})
        reject_forbidden_payload_context(extra_body, context="generation patch proposer extra_body")
        validation_error: str | None = None
        for attempt in range(self.max_generation_patch_attempts):
            payload = dict(request_payload)
            if validation_error is not None:
                payload["schema_correction"] = {
                    "previous_response_rejected": True,
                    "validation_error": validation_error[:1000],
                    "instruction": "Correct the rejected JSON and return one object with exactly the requested six fields.",
                }
            request = {
                "model": self.model_snapshot if self.strict_protocol else self.model,
                "temperature": self.config.get("temperature", 0.1),
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": self._generation_patch_system_prompt(localization)},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, indent=2),
                    },
                ],
            }
            for key, value in extra_body.items():
                request[key] = value
            self.request_count += 1
            response = _post_chat_json(self.config, self.endpoint, request, self.timeout)
            content = _chat_content(response)
            try:
                parsed = parse_json_object_from_text(content)
                _reject_forbidden(parsed)
                return self._generation_patch_from_row(parsed, localization, program=program)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                preview = " ".join(str(content).split())[:1000]
                validation_error = f"{exc}; response_preview={preview!r}"
                if attempt + 1 >= self.max_generation_patch_attempts:
                    raise ValueError(f"generation patch rejected after {attempt + 1} LLM attempts: {validation_error}") from exc
        raise RuntimeError("generation patch proposer exhausted without a result")

    def _generation_patch_system_prompt(self, localization: HarnessLocalization) -> str:
        return (
            "You are the transient visual-program proposer inside TRACE. "
            "The Harness Localizer has already selected the responsibility frontier. "
            "Return exactly one minimal patch for the next serial generation attempt. "
            "The patch changes only the in-memory visual program; it never edits repository files, "
            "never generates an image, never changes the image model, never calls an evaluator, "
            "and never uses official scores or hidden labels. "
            "Use the supplied verifier evidence and current visual program only. "
            "A prompt patch may only append one concrete, concise repair clause to the T5 prompt; "
            "it must preserve every existing contract clause and be directly useful to SD3.5 plus Flow-GRPO. "
            "A generation-parameter patch may change only an explicitly allowed numeric parameter. "
            "A prompt operation must be attributed to policy, skills, middleware, or memory; "
            "a generation-parameter operation must be attributed to tools. "
            "Return one JSON object with exactly the six fields in the requested schema; no markdown."
            f" Localized frontier: {', '.join(localization.frontier)}."
        )

    def _generation_patch_from_row(
        self,
        row: Any,
        localization: HarnessLocalization,
        *,
        program: Dict[str, Any] | None = None,
    ) -> GenerationPatch:
        if not isinstance(row, dict):
            raise ValueError("generation patch must be a JSON object")
        expected = {
            "target_component",
            "operation",
            "path",
            "value",
            "predicted_improvement",
            "possible_regression",
        }
        if set(row) != expected:
            raise ValueError(
                "generation patch schema mismatch; "
                f"expected exactly {sorted(expected)}, found {sorted(row)}"
            )
        component = str(row["target_component"]).strip()
        operation = str(row["operation"]).strip()
        path = row["path"]
        if component not in EDITABLE_RESPONSIBILITIES:
            raise ValueError(f"unsupported generation patch responsibility: {component}")
        if component not in localization.frontier:
            raise ValueError(f"generation patch escaped localized frontier: {component}")
        if operation not in {"append_prompt_clause", "set_generation_parameter"}:
            raise ValueError(f"unsupported transient generation patch operation: {operation}")
        if not isinstance(path, list) or not path or not all(isinstance(part, str) and part.strip() for part in path):
            raise ValueError("generation patch path must be a non-empty string list")
        normalized_path = [str(part).strip() for part in path]
        prompt_paths = {("workflow", "t5_prompt")}
        parameter_paths = {
            ("tool", "generation_parameters", "num_inference_steps"),
            ("tool", "generation_parameters", "guidance_scale"),
            ("tool", "generation_parameters", "max_sequence_length"),
        }
        path_key = tuple(normalized_path)
        if operation == "append_prompt_clause" and path_key not in prompt_paths:
            raise ValueError("prompt patch must target the T5 workflow prompt channel")
        if operation == "set_generation_parameter" and path_key not in parameter_paths:
            raise ValueError("generation parameter patch must target an allowed numeric parameter")
        if operation == "append_prompt_clause":
            if component == "tools":
                raise ValueError("prompt patch cannot be attributed to tools")
            if not isinstance(row["value"], str) or not row["value"].strip():
                raise ValueError("prompt patch value must be a non-empty string")
        else:
            if component != "tools":
                raise ValueError("generation parameter patch must be attributed to tools")
            if isinstance(row["value"], bool) or not isinstance(row["value"], (int, float)) or not math.isfinite(float(row["value"])):
                raise ValueError("generation parameter patch value must be a finite number")
            ranges = {
                ("tool", "generation_parameters", "num_inference_steps"): (1.0, 100.0),
                ("tool", "generation_parameters", "guidance_scale"): (0.0, 30.0),
                ("tool", "generation_parameters", "max_sequence_length"): (1.0, 512.0),
            }
            low, high = ranges[path_key]
            if not low <= float(row["value"]) <= high:
                raise ValueError(f"generation parameter patch value must be within [{low}, {high}]")
        if program is not None:
            workflow = program.get("workflow", {}) if isinstance(program, dict) else {}
            if operation == "append_prompt_clause" and not str(workflow.get("t5_prompt") or "").strip():
                raise ValueError("prompt patch requires an existing workflow.t5_prompt")
        for field in ("predicted_improvement", "possible_regression"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"generation patch field {field} must be a non-empty string")
        return GenerationPatch(
            target_component=component,
            operation=operation,
            path=normalized_path,
            value=row["value"],
            predicted_improvement=row["predicted_improvement"].strip(),
            possible_regression=row["possible_regression"].strip(),
        )

    def _request_patch_rows(self, weakness: Weakness, *, responsibility: str | None = None) -> List[Dict[str, Any]]:
        if responsibility is not None and responsibility not in COMPONENT_ARTIFACTS:
            raise ValueError(f"unsupported responsibility frontier entry: {responsibility}")
        target_component = responsibility or "policy|tools|skills|middleware|memory"
        expected = {
            "patches": [
                {
                    "target_component": target_component,
                    "changed_artifact": "owned artifact filename",
                    "changed_fields": ["dot.path"],
                    "operation": "set_path|append_unique|append_jsonl_unique|delete_path",
                    "payload": {"path": ["json", "path"], "value": "JSON value"},
                    "predicted_improvement": "why this patch should improve the weakness",
                    "possible_regression": "what related behavior could get worse",
                }
            ]
        }
        request = {
            "model": self.model_snapshot if self.strict_protocol else self.model,
            "temperature": self.config.get("temperature", 0.1),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt(responsibility=responsibility)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": "Propose minimal component-scoped harness patches. Return strict JSON only.",
                            "expected_schema": expected,
                            "restricted_target_component": responsibility,
                            "allowed_component_artifacts": {name: sorted(values) for name, values in COMPONENT_ARTIFACTS.items()},
                            "allowed_operations": sorted(ALLOWED_OPERATIONS),
                            "path_contract": (
                                "For set_path, append_unique, and delete_path, payload.path must begin with "
                                "the top-level JSON key owned by the artifact, and changed_fields must use the "
                                "same dotted root. For example, skills/skills.json uses "
                                "payload.path=['skills', 'compositional_generation', ...] and "
                                "changed_fields=['skills.compositional_generation....']; never omit 'skills'."
                            ),
                            "current_harness_state": self._current_harness_state(responsibility),
                            "weakness": sanitize_trace_weakness_for_llm(weakness),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
        }
        extra_body = self.config.get("extra_body", {})
        reject_forbidden_payload_context(extra_body, context="patch proposer extra_body")
        for key, value in extra_body.items():
            request[key] = value
        self.request_count += 1
        response = _post_chat_json(self.config, self.endpoint, request, self.timeout)
        parsed = parse_json_object_from_text(_chat_content(response))
        _reject_forbidden(parsed)
        rows = parsed.get("patches")
        if not isinstance(rows, list):
            raise ValueError("patch proposer JSON must contain a patches list")
        return rows

    def _current_harness_state(self, responsibility: str | None) -> Dict[str, Any]:
        """Return the current editable state visible to a conditioned proposal.

        TRACE proposes an edit relative to its current parent harness. Formal
        runners always construct the proposer with a repository; keeping the
        unavailable marker makes isolated schema tests explicit without
        inventing placeholder state.
        """
        return _current_harness_state_for_llm(self.repo, responsibility)

    def _system_prompt(self, *, responsibility: str | None = None) -> str:
        restriction = (
            f" You are restricted to target_component={responsibility}; do not propose any other component."
            if responsibility
            else ""
        )
        return (
            "You are the candidate proposer inside TRACE. "
            "The Harness Localizer decides where to evolve; you decide how to minimally edit that responsibility. "
            "Use only prompt-derived and internal verifier evidence supplied in the request. "
            "Do not use official benchmark scores, official evaluator outputs, hidden labels, benchmark answers, "
            "post-hoc image selection, fallback images, or task-id-specific fixes. "
            "Do not claim acceptance or promotion; matched validation runs later on target, heldout, and preservation splits. "
            "For JSON path edits, include the artifact's owned top-level key in both payload.path and changed_fields: "
            "for skills/skills.json the path starts with 'skills.compositional_generation', "
            "not merely 'compositional_generation'. "
            "Return only valid JSON."
            + restriction
        )

    def _manifest_from_row(
        self,
        weakness: Weakness,
        row: Any,
        index: int,
        *,
        responsibility: str | None = None,
    ) -> PatchManifest:
        if not isinstance(row, dict):
            raise ValueError("each patch proposal must be a JSON object")
        _reject_forbidden(row)
        _reject_task_specific_patch_context(row)
        component = str(row.get("target_component", "")).strip()
        artifact = str(row.get("changed_artifact", "")).strip()
        operation = str(row.get("operation", "")).strip()
        if responsibility is not None and component != responsibility:
            raise ValueError(f"proposal escaped restricted responsibility {responsibility!r}: {component!r}")
        if component not in COMPONENT_ARTIFACTS:
            raise ValueError(f"unsupported target_component: {component}")
        if artifact not in COMPONENT_ARTIFACTS[component]:
            raise ValueError(f"artifact {artifact!r} is not owned by component {component!r}")
        if operation not in ALLOWED_OPERATIONS:
            raise ValueError(f"unsupported patch operation: {operation}")
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("patch payload must be a JSON object")
        changed_fields = row.get("changed_fields")
        if not isinstance(changed_fields, list) or not changed_fields:
            raise ValueError("changed_fields must be a non-empty list")
        if not all(isinstance(field, str) and field.strip() for field in changed_fields):
            raise ValueError("changed_fields entries must be non-empty strings")
        changed_fields = [str(field).strip() for field in changed_fields]
        self._validate_payload(component, artifact, operation, payload, changed_fields)
        patch = PatchManifest(
            patch_id=str(row.get("patch_id") or f"llm_patch_{weakness.weakness_id}_{component}_{index}"),
            target_component=component,
            changed_artifact=artifact,
            changed_fields=changed_fields,
            operation=operation,
            payload=dict(payload),
            supporting_experience=list(weakness.support_experience_ids),
            predicted_improvement=str(row.get("predicted_improvement") or "LLM-proposed component patch for this weakness."),
            possible_regression=str(row.get("possible_regression") or "May affect related prompts owned by the same component."),
            target_validation_set="target",
            preservation_set="preservation",
            held_out_set="heldout",
            rollback_boundary=f"{component}/{artifact}",
        )
        patch.evidence_summary = self._evidence_summary(weakness, patch, responsibility=responsibility)
        patch.regression_risk = self._regression_risk(weakness, patch)
        self._attach_canonical_edit(patch)
        patch.payload.setdefault("schema_validation", self._schema_validation_hint(patch))
        patch.payload.setdefault("proposal_rationale", self._proposal_rationale(weakness, patch))
        return patch

    def _validate_payload(
        self,
        component: str,
        artifact: str,
        operation: str,
        payload: Dict[str, Any],
        changed_fields: List[str],
    ) -> None:
        allowed_roots = self._allowed_roots(component, artifact)
        changed_roots = {field.split(".", 1)[0] for field in changed_fields}
        if operation in {"set_path", "append_unique", "delete_path"}:
            path = payload.get("path")
            if not isinstance(path, list) or not path:
                raise ValueError(f"{operation} payload requires a non-empty path list")
            if not all(isinstance(part, str) and part.strip() for part in path):
                raise ValueError(f"{operation} payload path entries must be non-empty strings")
            path_root = str(path[0]).strip()
            if allowed_roots and path_root not in allowed_roots:
                raise ValueError(f"{operation} payload path root {path_root!r} is not owned by {component}/{artifact}")
            if changed_roots and path_root not in changed_roots:
                raise ValueError("changed_fields must describe the same artifact path root as payload.path")
            if operation != "delete_path" and "value" not in payload:
                raise ValueError(f"{operation} payload requires value")
        elif operation == "append_jsonl_unique":
            if not isinstance(payload.get("row"), dict):
                raise ValueError("append_jsonl_unique payload requires row object")
            if not isinstance(payload.get("unique_key"), str) or not payload["unique_key"].strip():
                raise ValueError("append_jsonl_unique payload requires unique_key")
            if allowed_roots and allowed_roots != {"*"}:
                row = payload.get("row", {})
                row_roots = {str(key) for key in row}
                unowned = sorted(row_roots - allowed_roots)
                if unowned:
                    raise ValueError(f"append_jsonl_unique row contains fields not owned by {component}/{artifact}: {unowned}")

    def _attach_canonical_edit(self, patch: PatchManifest) -> None:
        path = patch.payload.get("path") if isinstance(patch.payload, dict) else None
        patch.edit_key = ".".join(str(part) for part in path) if isinstance(path, list) else patch.changed_artifact
        patch.new_value = patch.payload.get("value", patch.payload.get("row"))
        if patch.operation == "append_jsonl_unique":
            patch.edit_operator = "ADD"
            patch.old_value = None
            return
        if patch.operation == "append_unique":
            patch.edit_operator = "ADD"
        elif patch.operation == "delete_path":
            patch.edit_operator = "DELETE"
            patch.new_value = None
        else:
            patch.edit_operator = "REPLACE"
        if self.repo is None or not isinstance(path, list):
            return
        node: Any = self.repo.load_artifact(patch.target_component, patch.changed_artifact)
        exists = True
        for part in path:
            if not isinstance(node, dict) or part not in node:
                exists = False
                break
            node = node[part]
        patch.old_value = node if exists else None
        if patch.operation == "set_path" and not exists:
            patch.edit_operator = "ADD"
        if patch.operation == "delete_path" and not exists:
            raise ValueError(f"delete_path target does not exist: {patch.edit_key}")

    def _allowed_roots(self, component: str, artifact: str) -> set[str]:
        roots: Dict[tuple[str, str], set[str]] = {
            ("policy", "visual_contract.json"): {
                "component_name",
                "component_type",
                "version",
                "boundary_contract",
                "constraint_ontology",
                "auto_include_negative_constraints",
                "negative_constraints",
                "verification_rules",
                "evidence_required_for",
                "prompt_governance",
                "validation_scope",
                "rollback_boundary",
            },
            ("policy", "reference_policy.json"): {
                "component_name",
                "component_type",
                "version",
                "boundary_contract",
                "allowed_roles",
                "require_explicit_roles_for_multi_reference",
                "validation_scope",
                "rollback_boundary",
            },
            ("policy", "preservation_policy.json"): {
                "component_name",
                "component_type",
                "version",
                "boundary_contract",
                "default_preservation",
                "repair_preservation",
                "validation_scope",
                "rollback_boundary",
            },
            ("tools", "generation_tools.json"): {
                "component_name",
                "component_type",
                "version",
                "boundary_contract",
                "tools",
                "search_tools",
                "validation_scope",
                "rollback_boundary",
            },
            ("tools", "inspection_tools.json"): {
                "component_name",
                "component_type",
                "version",
                "boundary_contract",
                "tools",
                "validation_scope",
                "rollback_boundary",
            },
            ("skills", "skills.json"): {
                "component_name",
                "component_type",
                "version",
                "boundary_contract",
                "skills",
                "validation_scope",
                "rollback_boundary",
            },
            ("middleware", "skill_selector.json"): {
                "component_name",
                "component_type",
                "version",
                "boundary_contract",
                "default_skill",
                "task_family_to_skill",
                "validation_scope",
                "rollback_boundary",
            },
            ("middleware", "reference_binder.json"): {
                "component_name",
                "component_type",
                "version",
                "boundary_contract",
                "mode",
                "roles",
                "role_overrides",
                "strict_required_references",
                "rules",
                "validation_scope",
                "rollback_boundary",
            },
            ("middleware", "tool_router.json"): {
                "component_name",
                "component_type",
                "version",
                "boundary_contract",
                "routes",
                "default_tool",
                "identity_tool",
                "strict_preferred_route",
                "validation_scope",
                "rollback_boundary",
            },
            ("memory", "prompt_guidelines.jsonl"): {"*"},
        }
        return roots.get((component, artifact), set())

    def _evidence_summary(self, weakness: Weakness, patch: PatchManifest, *, responsibility: str | None) -> Dict[str, Any]:
        return {
            "proposal_source": {
                "type": "openai_chat_patch_proposer",
                "model": self.model_snapshot if self.strict_protocol else self.model,
                "model_snapshot": self.model_snapshot or None,
            },
            "trace_responsibility_condition": responsibility,
            "suspected_component": weakness.suspected_component,
            "constraint_type": weakness.constraint_type,
            "failure_symptom": weakness.failure_symptom,
            "support_count": weakness.support_count,
            "support_experience_ids": weakness.support_experience_ids,
            "hypothesis": weakness.hypothesis,
            "changed_fields": patch.changed_fields,
        }

    def _schema_validation_hint(self, patch: PatchManifest) -> Dict[str, Any]:
        if patch.operation in {"set_path", "append_unique"}:
            required = ["path", "value"]
        elif patch.operation == "delete_path":
            required = ["path"]
        else:
            required = ["row", "unique_key"]
        return {
            "operation": patch.operation,
            "target_component": patch.target_component,
            "changed_artifact": patch.changed_artifact,
            "required_payload_fields": required,
            "rollback_boundary": patch.rollback_boundary,
            "component_boundary_checked": True,
            "dataset_specific_rule_rejected": True,
            "task_specific_rule_rejected": True,
            "official_evaluator_feedback_allowed": False,
        }

    def _proposal_rationale(self, weakness: Weakness, patch: PatchManifest) -> Dict[str, Any]:
        return {
            "why_this_component": weakness.hypothesis,
            "why_this_artifact": f"{patch.changed_artifact} is owned by {patch.target_component}.",
            "minimality": "One component-owned artifact is changed; generator weights, verifier, and benchmark evaluator remain fixed.",
            "validation_required": [patch.target_validation_set, patch.held_out_set, patch.preservation_set],
            "official_feedback_used": False,
        }

    def _regression_risk(self, weakness: Weakness, patch: PatchManifest) -> Dict[str, Any]:
        affected = [weakness.constraint_type]
        if patch.target_component == "skills":
            affected.extend(["prompt_alignment", "quality"])
        elif patch.target_component == "policy":
            affected.extend(["overconstraint", "prompt_alignment"])
        elif patch.target_component == "middleware":
            affected.extend(["routing", "latency"])
        elif patch.target_component == "tools":
            affected.extend(["capability_routing", "cost"])
        return {
            "affected_constraint_families": sorted(set(affected)),
            "requires_preservation_validation": True,
            "requires_heldout_validation": True,
            "known_regression_match": False,
        }

    def _known_regressive(self, patch: PatchManifest) -> bool:
        if self.memory is None:
            return False
        rows = self.memory.retrieve_regression_cases(
            component=patch.target_component,
            constraint_types=[patch.evidence_summary.get("constraint_type")],
            failure_symptoms=[patch.evidence_summary.get("failure_symptom")],
            limit=100,
        )
        for row in rows:
            if (
                row.get("changed_artifact") == patch.changed_artifact
                and row.get("operation") == patch.operation
                and _core_payload(row.get("payload")) == _core_payload(patch.payload)
            ):
                patch.regression_risk["known_regression_match"] = True
                patch.regression_risk["blocked_by_regression_case"] = row.get("patch_id")
                return True
        return False


def sanitize_trace_weakness_for_llm(weakness: Weakness) -> Dict[str, Any]:
    """Return TRACE evidence for LLM localization/proposal without task-level provenance."""

    return {
        "suspected_component": weakness.suspected_component,
        "constraint_type": weakness.constraint_type,
        "failure_symptom": weakness.failure_symptom,
        "support_count": weakness.support_count,
        "hypothesis": _sanitize_trace_value(weakness.hypothesis),
        "evidence": _sanitize_trace_value(weakness.evidence),
        "sanitization": {
            "task_specific_provenance_omitted": True,
            "external_feedback_omitted": True,
            "replacement_image_fields_omitted": True,
        },
    }


def _current_harness_state_for_llm(repo: Any | None, responsibility: str | None) -> Dict[str, Any]:
    """Serialize current editable state without task/benchmark provenance."""
    if repo is None:
        return {"available": False, "reason": "repository_not_attached"}
    snapshot = repo.snapshot_persistent_state()
    components = [responsibility] if responsibility else list(EDITABLE_RESPONSIBILITIES)
    state: Dict[str, Any] = {}
    for component in components:
        if component not in COMPONENT_ARTIFACTS:
            continue
        artifacts: Dict[str, Any] = {}
        for artifact in sorted(COMPONENT_STATE_ARTIFACTS[component]):
            record = snapshot.get(component, {}).get(artifact)
            if not isinstance(record, dict) or "value" not in record:
                continue
            value = record["value"]
            if component == "memory" and isinstance(value, list):
                value = {
                    "row_count": len(value),
                    "visible_rows": value[-32:],
                    "visibility": "sanitized_recent_controller_records",
                }
            artifacts[artifact] = {
                "format": record.get("format"),
                "value": _sanitize_trace_value(value),
            }
        state[component] = artifacts
    return {
        "available": True,
        "scope": "responsibility_conditioned" if responsibility else "all_editable_responsibilities",
        "components": state,
        "official_feedback_omitted": True,
    }


def _sanitized_visual_program_for_llm(program: Dict[str, Any]) -> Dict[str, Any]:
    """Expose only transient generation controls to the fast-loop proposer."""
    if not isinstance(program, dict):
        raise ValueError("visual program must be an object")
    workflow = program.get("workflow", {})
    tool = program.get("tool", {})
    contract = program.get("contract", {})
    if not isinstance(workflow, dict) or not isinstance(tool, dict) or not isinstance(contract, dict):
        raise ValueError("visual program has invalid workflow, tool, or contract state")
    contract_rows = contract.get("constraints", [])
    sanitized_contract = {
        "constraints": _sanitize_trace_value(contract_rows),
    }
    return {
        "contract": sanitized_contract,
        "workflow": {
            key: _sanitize_trace_value(workflow.get(key))
            for key in (
                "compiled_prompt",
                "canonical_prompt",
                "clip_prompt",
                "t5_prompt",
                "selected_prompt_family",
                "clip_prompt_segments",
                "t5_prompt_segments",
            )
            if key in workflow
        },
        "tool": {
            "generation_parameters": _sanitize_trace_value(tool.get("generation_parameters", {})),
        },
        "memory_context": _sanitize_trace_value(program.get("memory_context", {})),
    }


def _sanitize_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            if _trace_key_is_forbidden(str(key)):
                continue
            out[str(key)] = _sanitize_trace_value(child)
        return out
    if isinstance(value, list):
        return [_sanitize_trace_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_trace_text(value)
    return value


def _trace_key_is_forbidden(key: str) -> bool:
    normalized = key.strip().lower()
    forbidden_exact = {
        "benchmark",
        "category",
        "complex_subtype",
        "dataset",
        "eval",
        "evaluation",
        "experience_id",
        "fallback_image",
        "fallback_used",
        "file_path",
        "geneval_index",
        "geneval_metadata",
        "geneval_tag",
        "geneval2_index",
        "hidden_label",
        "hidden_labels",
        "image_index",
        "official_eval",
        "official_evaluator",
        "official_score",
        "original_prompt",
        "question_answers",
        "score",
        "scores",
        "selected_result",
        "selected_score",
        "source_index",
        "split",
        "support_experience_ids",
        "task_id",
        "task_ids",
        "vqa_list",
    }
    forbidden_fragments = (
        "benchmark",
        "fallback",
        "geneval",
        "hidden_label",
        "official",
        "posthoc",
        "post_hoc",
        "selected_score",
        "selected_result",
        "source_index",
        "task_id",
        "vqa",
    )
    return normalized in forbidden_exact or any(fragment in normalized for fragment in forbidden_fragments)


def _sanitize_trace_text(value: str) -> str:
    lowered = value.lower()
    forbidden = (
        "official evaluator",
        "official score",
        "official_score",
        "benchmark label",
        "benchmark answer",
        "hidden label",
        "fallback image",
        "fallback_image",
        "post-hoc",
        "posthoc",
        "selected_score",
        "selected_result",
    )
    if any(token in lowered for token in forbidden):
        return "[omitted_trace_leakage_text]"
    return value


def _core_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    return {
        key: value
        for key, value in payload.items()
        if key not in {"schema_validation", "proposal_rationale"}
    }


def build_harness_localizer(
    config_path: str | Path | None,
    *,
    repo: Any | None = None,
    strict_protocol: bool = False,
) -> HarnessLocalizer:
    if not config_path:
        raise ValueError("LLM harness localizer config is required; no rule-based attribution fallback is available")
    cfg = expand_env_vars(read_json(config_path))
    return HarnessLocalizer(cfg, repo=repo, strict_protocol=strict_protocol)


def build_patch_proposer(
    config_path: str | Path | None,
    *,
    memory: VisualMemory | None = None,
    repo: Any | None = None,
    strict_protocol: bool = False,
) -> PatchProposer:
    if not config_path:
        raise ValueError("LLM patch proposer config is required; no rule-based patch fallback is available")
    cfg = expand_env_vars(read_json(config_path))
    return PatchProposer(cfg, memory=memory, repo=repo, strict_protocol=strict_protocol)
