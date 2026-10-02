#!/usr/bin/env python3
"""Copy an explicitly selected harness and fingerprint its files.

The output is a new local snapshot, never a claim about a historical paper run.
No model weights or external paths are followed. Only the five responsibility
folders are copied; source files and the caller's environment are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gen_harness.component_audit import ComponentBoundaryAuditor
from gen_harness.safety import scan_open_source_safety

COMPONENTS = ("policy", "tools", "skills", "middleware", "memory")


def snapshot(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("output already exists; select a new snapshot directory")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output must not be nested")
    for name in COMPONENTS:
        folder = source / name
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"missing or symlinked responsibility folder: {name}")
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"symlinks are not permitted in snapshots: {path.relative_to(source)}")
            if path.is_file() and path.suffix not in {".json", ".jsonl", ".md", ".txt"}:
                raise ValueError(f"unexpected harness file type: {path.relative_to(source)}")
    audit = ComponentBoundaryAuditor().audit(source)
    if not audit.get("passed"):
        raise ValueError("component boundary audit failed; inspect the selected harness before archiving")
    findings = scan_open_source_safety(source)
    if findings:
        raise ValueError("source safety scan found potentially private content; review before archiving")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".harness-snapshot-", dir=output.parent))
    try:
        for name in COMPONENTS:
            shutil.copytree(source / name, temp / name)
        fingerprints = {p.relative_to(temp).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(temp.rglob("*")) if p.is_file()}
        digest = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
        manifest = {"schema": "evogen.harness_snapshot.v1", "status": "locally_created_snapshot",
                    "files": fingerprints, "tree_sha256": digest,
                    "historical_paper_run_identity": "not_attested_by_this_tool"}
        (temp / "SNAPSHOT.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # rename rather than merge; never overwrite an existing snapshot.
        temp.rename(output)
        return manifest
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def verify(path: Path) -> dict:
    manifest = json.loads((path / "SNAPSHOT.json").read_text(encoding="utf-8"))
    actual = {}
    for name in COMPONENTS:
        for item in sorted((path / name).rglob("*")):
            if item.is_symlink():
                raise ValueError("symlink found in snapshot")
            if item.is_file():
                actual[item.relative_to(path).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    if actual != manifest["files"] or digest != manifest["tree_sha256"]:
        raise ValueError("snapshot contents changed after archiving")
    return {"passed": True, "files": len(actual), "tree_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="explicitly selected harness directory")
    parser.add_argument("--output", type=Path, help="new snapshot directory")
    parser.add_argument("--verify", type=Path, help="verify an existing SNAPSHOT.json inventory")
    args = parser.parse_args()
    try:
        if args.verify:
            if args.source or args.output:
                parser.error("--verify cannot be combined with --source or --output")
            result = verify(args.verify)
        else:
            if not args.source or not args.output:
                parser.error("--source and --output are required when creating a snapshot")
            result = snapshot(args.source, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
