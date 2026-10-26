from __future__ import annotations

from typing import Any, Dict, Iterable

from .leakage_guard import sanitize_formal_trace_metadata, sanitize_llm_payload_context
from .schema import GenerationTask, VisualConstraint


def constraints_for_external_tool(task: GenerationTask, program: Dict[str, Any]) -> list[VisualConstraint]:
    contract_constraints = program.get("contract", {}).get("constraints", []) if isinstance(program.get("contract"), dict) else []
    if contract_constraints:
        return [VisualConstraint.from_dict(c) for c in contract_constraints]
    return list(task.constraints)


def public_constraint_ids(constraints: Iterable[VisualConstraint]) -> Dict[str, str]:
    return {c.constraint_id: f"constraint_{index}" for index, c in enumerate(constraints)}


def public_constraint_rows(
    constraints: Iterable[VisualConstraint],
    public_ids: Dict[str, str] | None = None,
) -> list[Dict[str, Any]]:
    rows = []
    constraints = list(constraints)
    public_ids = public_ids or public_constraint_ids(constraints)
    for c in constraints:
        rows.append(
            {
                "constraint_id": public_ids[c.constraint_id],
                "constraint_type": c.constraint_type,
                "text": c.text,
                "target": c.target,
                "required_reference_role": c.required_reference_role,
                "metadata": sanitize_llm_payload_context(c.metadata),
            }
        )
    return rows


def compiled_prompt_for_external_tool(task: GenerationTask, program: Dict[str, Any], *, context: str) -> str:
    prompt = str(program.get("workflow", {}).get("compiled_prompt") or task.prompt or "").strip()
    if not prompt:
        raise ValueError(f"{context} requires workflow.compiled_prompt for task {task.task_id}")
    return prompt


def public_task_row(task: GenerationTask) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "task_family": task.task_family,
        "constraints": [],
        "constraint_source": "compiled_visual_contract",
        "references": sanitize_formal_trace_metadata(task.references),
        "metadata": sanitize_formal_trace_metadata(task.metadata),
    }




def public_references_for_llm(task: GenerationTask) -> Dict[str, Any]:
    references = sanitize_llm_payload_context(task.references)
    return references if isinstance(references, dict) else {}


def public_generation_input(task: GenerationTask, program: Dict[str, Any]) -> Dict[str, Any]:
    prompt = compiled_prompt_for_external_tool(task, program, context="generation")
    return sanitize_llm_payload_context(
        {
            "initial_prompt": task.prompt,
            "generation_prompt": prompt,
            "contract": program.get("contract", {}),
            "references": public_references_for_llm(task),
            "tool": program.get("tool", {}),
        }
    )


def public_search_input(task: GenerationTask, contract: Dict[str, Any]) -> Dict[str, Any]:
    return sanitize_llm_payload_context(
        {
            "prompt": task.prompt,
            "references": public_references_for_llm(task),
            "contract": contract,
        }
    )


def public_inspection_payload(
    task: GenerationTask,
    program: Dict[str, Any],
    generation: Dict[str, Any],
    *,
    constraints: list[VisualConstraint] | None = None,
    public_ids: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    constraints = constraints if constraints is not None else constraints_for_external_tool(task, program)
    public_ids = public_ids or public_constraint_ids(constraints)
    prompt = compiled_prompt_for_external_tool(task, program, context="inspection")
    return sanitize_llm_payload_context(
        {
            "initial_prompt": task.prompt,
            "generation_prompt": prompt,
            "prompt": prompt,
            "image_uri": generation.get("image_uri"),
            "image_path": generation.get("image_path") or generation.get("image_uri"),
            "generation": {
                "image_uri": generation.get("image_uri"),
                "image_path": generation.get("image_path") or generation.get("image_uri"),
            },
            "references": public_references_for_llm(task),
            "constraints": public_constraint_rows(constraints, public_ids),
        }
    )


def remap_public_constraint_results(
    rows: Dict[str, Any],
    constraints: list[VisualConstraint],
    public_ids: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rows, dict):
        raise ValueError("constraint_results must be an object")
    out: Dict[str, Dict[str, Any]] = {}
    for c in constraints:
        public_id = public_ids[c.constraint_id]
        raw = rows.get(public_id, {})
        if not raw:
            raw = rows.get(c.constraint_id, {})
        out[c.constraint_id] = raw if isinstance(raw, dict) else {"passed": False, "symptom": "invalid_external_result"}
    return out
