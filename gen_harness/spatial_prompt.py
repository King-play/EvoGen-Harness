"""Shared, evaluator-agnostic language for spatial constraints.

Keeping the canonical clause here makes the prompt that is actually sent to
the backend auditable and prevents any second compiler from replacing a repair.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, List, Sequence, Tuple


Relation = Tuple[str, str, str]


def normalize_relation(value: Any) -> str:
    relation = " ".join(str(value or "").replace("_", " ").strip().lower().split())
    # Map linguistic variants onto the small executable relation ontology used
    # by both prompt compilation and repair.  Without this normalization,
    # valid policy output such as ``under`` fell through to vague prose instead
    # of receiving a concrete top/bottom layout.
    aliases = {
        "under": "below",
        "underneath": "below",
        "beneath": "below",
        "on top": "above",
        "on top of": "above",
        "atop": "above",
        "over": "above",
        "to the left of": "left of",
        "left from": "left of",
        "to the right of": "right of",
        "right from": "right of",
        "in foreground of": "in front of",
        "in the foreground of": "in front of",
        "in background of": "behind",
        "in the background of": "behind",
    }
    return aliases.get(relation, relation)


def canonical_spatial_clause(
    subject: str,
    relation: str,
    target: str,
    *,
    compact: bool = False,
    subject_count: int = 1,
    target_count: int = 1,
) -> str:
    """Return one unambiguous, named-object spatial instruction."""

    subject = str(subject).strip()
    target = str(target).strip()
    relation = normalize_relation(relation)
    subject_count = max(1, int(subject_count))
    target_count = max(1, int(target_count))
    plural = subject_count > 1 or target_count > 1
    subject_group = f"the full group of {subject_count} {subject} instances" if subject_count > 1 else subject
    target_group = f"the full group of {target_count} {target} instances" if target_count > 1 else target
    if compact and plural:
        subject_short = f"all {subject_count} {subject}"
        target_short = f"all {target_count} {target}"
        if relation == "left of":
            layout = f"{subject_short} left; {target_short} right"
        elif relation == "right of":
            layout = f"{subject_short} right; {target_short} left"
        elif relation == "above":
            layout = f"{subject_short} above; {target_short} below"
        elif relation == "below":
            layout = f"{subject_short} below; {target_short} above"
        elif relation == "in front of":
            layout = f"{subject_short} foreground; {target_short} background"
        elif relation == "behind":
            layout = f"{subject_short} background; {target_short} foreground"
        else:
            layout = f"{subject_short} {relation} {target_short}"
        return f"SPATIAL ANCHOR ({relation}): {layout}; preserve counts and categories."
    band = "band" if plural else "edge band"
    if relation == "left of":
        layout = f"{subject_group} in the left {band}; {target_group} in the right {band}"
    elif relation == "right of":
        layout = f"{subject_group} in the right {band}; {target_group} in the left {band}"
    elif relation == "above":
        layout = f"{subject_group} in the top {band}; {target_group} in the bottom {band}"
    elif relation == "below":
        layout = f"{subject_group} in the bottom {band}; {target_group} in the top {band}"
    elif relation == "in front of":
        layout = f"{subject_group} close to camera in the foreground; {target_group} farther away in the background"
    elif relation == "behind":
        layout = f"{target_group} close to camera in the foreground; {subject_group} farther away in the background"
    else:
        layout = f"{subject} clearly {relation} {target}"
    if relation in {"left of", "right of", "above", "below"}:
        gap = "clear empty center lane" if relation in {"left of", "right of"} else "clear empty center strip"
        suffix = "; preserve every counted instance; no swap; no overlap" if plural else "; one complete object of each kind only; no swap; no overlap"
        if compact and not plural:
            suffix = "; one of each only; no swap"
        return f"SPATIAL ANCHOR ({relation}): {layout}; {gap}{suffix}."
    if relation in {"in front of", "behind"}:
        suffix = "preserve every counted instance; no category substitution" if plural else "one object of each kind only; no substitution"
        return f"SPATIAL ANCHOR ({relation}): {layout}; clear depth separation; both categories recognizable; {suffix}."
    suffix = "preserve every counted instance; no swap" if plural else "one complete object of each kind only; no swap"
    return f"SPATIAL ANCHOR: {layout}; {suffix}."


def canonical_spatial_clauses(relations: Iterable[Relation], *, compact: bool = False) -> List[str]:
    out: List[str] = []
    seen = set()
    for subject, relation, target in relations:
        key = (str(subject).strip().lower(), normalize_relation(relation), str(target).strip().lower())
        if not key[0] or not key[1] or not key[2] or key in seen:
            continue
        seen.add(key)
        out.append(canonical_spatial_clause(subject, relation, target, compact=compact))
    return out


def append_canonical_spatial_anchors(prompt: str, relations: Sequence[Relation], *, compact: bool = False) -> str:
    """Append canonical anchors once, preserving the original prompt text."""

    base = " ".join(str(prompt or "").strip().split())
    clauses = canonical_spatial_clauses(relations, compact=compact)
    if not clauses:
        return base
    existing = base.lower()
    additions = [clause for clause in clauses if clause.lower() not in existing]
    if not additions:
        return base
    return " ".join([base, *additions]).strip()


def prompt_provenance(workflow: dict[str, Any]) -> dict[str, Any]:
    """Record both the canonical prompt and the exact backend channels."""

    compiled = str(workflow.get("compiled_prompt") or "").strip()
    clip = str(workflow.get("clip_prompt") or compiled).strip()
    t5 = str(workflow.get("t5_prompt") or compiled).strip()
    workflow["canonical_prompt"] = compiled
    workflow["prompt_provenance"] = {
        "schema": "gen_harness.prompt_provenance.v1",
        "canonical_prompt": compiled,
        "backend_channels": {"clip_prompt": clip, "t5_prompt": t5},
        "sha256": {
            "canonical_prompt": hashlib.sha256(compiled.encode("utf-8")).hexdigest(),
            "clip_prompt": hashlib.sha256(clip.encode("utf-8")).hexdigest(),
            "t5_prompt": hashlib.sha256(t5.encode("utf-8")).hexdigest(),
        },
        "channels_equal": clip == t5 == compiled,
        "canonical_spatial_anchor_present": "SPATIAL ANCHOR" in " ".join((compiled, clip, t5)),
    }
    return workflow["prompt_provenance"]


def spatial_repair_clause(
    subject: str,
    relation: str,
    target: str,
    *,
    actual: str | None = None,
    subject_count: int = 1,
    target_count: int = 1,
) -> str:
    """Return one short executable repair, without repeating the full prompt.

    Repeating both object names through several long constraints caused the
    image model to create extra copies.  The baseline already contains the
    complete contract, so a retry only needs one corrected layout sentence.
    """

    subject = str(subject).strip()
    target = str(target).strip()
    relation = normalize_relation(relation)
    subject_count = max(1, int(subject_count))
    target_count = max(1, int(target_count))
    if subject_count > 1 or target_count > 1:
        return canonical_spatial_clause(
            subject,
            relation,
            target,
            compact=True,
            subject_count=subject_count,
            target_count=target_count,
        ).replace("SPATIAL ANCHOR", "Repair group layout", 1)
    if relation == "left of":
        return f"Repair layout: one {subject} in the left slot and one {target} in the right slot; side-by-side with level centers."
    if relation == "right of":
        return f"Repair layout: one {target} in the left slot and one {subject} in the right slot; side-by-side with level centers."
    if relation == "above":
        return f"Repair layout: one {subject} in the top slot and one {target} in the bottom slot; vertically stacked on one centerline."
    if relation == "below":
        return f"Repair layout: one {target} in the top slot and one {subject} in the bottom slot; vertically stacked on one centerline."
    if relation == "in front of":
        return f"Repair depth: one {subject} close to camera in the foreground and one distinct {target} farther away in the background; no substitution or duplicate {subject}."
    if relation == "behind":
        return f"Repair depth: one {target} close to camera in the foreground and one distinct {subject} farther away in the background; no substitution or duplicate {target}."
    return f"Repair layout: one {subject} clearly {relation} one {target}."
