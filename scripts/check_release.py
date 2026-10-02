#!/usr/bin/env python3
"""Offline source-release checks; never execute generation or contact an API."""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import re
import shlex
import sys
import tomllib
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gen_harness.cli import build_parser
from gen_harness.component_audit import ComponentBoundaryAuditor
from gen_harness.safety import scan_open_source_safety
from scripts.prepare_evolution import validate_splits

EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist", "runs", "logs", "docs"}


def source_files(root: Path):
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if p.is_file():
            yield p


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.targets = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"href", "src", "srcset"} and value:
                self.targets.append(value)


def anchors(text: str) -> set[str]:
    result = set(re.findall(r'\bid=["\']([^"\']+)["\']', text))
    seen = {}
    headings = re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M)
    headings += re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", text)
    for heading in headings:
        heading = re.sub(r"<[^>]*>", "", unescape(heading)).lower()
        slug = re.sub(r"[^\w\-\s]", "", heading).replace(" ", "-")
        count = seen.get(slug, 0); seen[slug] = count + 1
        result.add(slug if count == 0 else f"{slug}-{count}")
    return result


def check_docs(root: Path) -> tuple[int, int, list[str]]:
    links = commands = 0; errors = []
    parser = build_parser()
    for p in source_files(root):
        if p.suffix != ".md":
            continue
        text = p.read_text(encoding="utf-8")
        visible = re.sub(r"```.*?```", "", text, flags=re.S)
        targets = re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", visible)
        hp = LinkParser(); hp.feed(visible); targets += hp.targets
        for target in targets:
            u = urlsplit(unescape(target))
            if u.scheme or u.netloc:
                continue
            destination = (p.parent / unquote(u.path)).resolve() if u.path else p
            links += 1
            if not destination.is_relative_to(root.resolve()) or not destination.exists():
                errors.append(f"{p.relative_to(root)}: missing/escaping link {target}")
            elif u.fragment and destination.suffix == ".md":
                if unquote(u.fragment) not in anchors(destination.read_text(encoding="utf-8")):
                    errors.append(f"{p.relative_to(root)}: missing anchor {target}")
        for block in re.findall(r"```(?:bash|sh|shell)\n(.*?)```", text, re.S):
            for line in block.replace("\\\n", " ").splitlines():
                line = line.strip()
                if not line.startswith("python -m gen_harness.cli "):
                    continue
                arguments = shlex.split(line)[3:]
                # Redirects are shell syntax, not CLI arguments.
                if ">" in arguments:
                    arguments = arguments[:arguments.index(">")]
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        ns = parser.parse_args(arguments)
                    for key, value in vars(ns).items():
                        if key.endswith("_config") and value:
                            if not (root / value).is_file():
                                errors.append(f"{p.relative_to(root)}: missing config {value}")
                except SystemExit as exc:
                    if exc.code:
                        errors.append(f"{p.relative_to(root)}: invalid CLI arguments: {line}")
                commands += 1
    return links, commands, errors


def check(root: Path = ROOT) -> dict:
    errors = []
    required = ["README.md", "README_zh-CN.md", "CITATION.bib", "CITATION.cff", "LICENSE", "pyproject.toml",
                "assets/readme/overview.png", "assets/readme/progressive-repair.jpg", "documentation/RELEASE_NOTES.md",
                "documentation/release_validation.json", ".github/workflows/quality.yml"]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"required release file missing: {name}")
    syntax_count = 0
    for p in source_files(root):
        try:
            if p.suffix == ".py":
                ast.parse(p.read_text(encoding="utf-8"), filename=p.name); syntax_count += 1
            elif p.suffix == ".json":
                json.loads(p.read_text(encoding="utf-8"))
            elif p.suffix == ".jsonl":
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        json.loads(line)
        except (ValueError, SyntaxError) as exc:
            errors.append(f"{p.relative_to(root)}: {exc}")
    links, commands, doc_errors = check_docs(root); errors.extend(doc_errors)
    bib = (root / "CITATION.bib").read_text(encoding="utf-8").strip()
    for name in ("README.md", "README_zh-CN.md"):
        if bib not in (root / name).read_text(encoding="utf-8"):
            errors.append(f"{name}: BibTeX differs from CITATION.bib")
    try:
        import yaml
        cff = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
        for name in list((root / ".github").rglob("*.yml")):
            yaml.safe_load(name.read_text(encoding="utf-8"))
        if cff["preferred-citation"]["url"] != "https://arxiv.org/abs/2610.00383":
            errors.append("CITATION.cff: unexpected paper URL")
        if cff["version"] != "0.2.0":
            errors.append("CITATION.cff: version must match the supplied source version")
    except (ImportError, ValueError, KeyError) as exc:
        errors.append(f"citation/workflow validation: {exc}; install .[dev] when dependencies are missing")
    try:
        _, data_report = validate_splits(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"P2 validation: {exc}"); data_report = {"passed": False}
    assets = json.loads((root / "assets/readme/manifest.json").read_text(encoding="utf-8"))
    for name, info in assets["files"].items():
        path = root / "assets/readme" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            errors.append(f"README asset hash mismatch: {name}")
    audit = ComponentBoundaryAuditor().audit(root / "examples/visual_harness")
    if not audit.get("passed"):
        errors.append("component boundary audit failed")
    findings = scan_open_source_safety(root, excludes=["docs"], include_ignored_local=False)
    for finding in findings:
        errors.append(f"safety heuristic: {finding.path}:{finding.line} {finding.kind}")
    for p in source_files(root):
        if p.name.startswith("._") or "__MACOSX" in p.parts or p.name == ".DS_Store":
            errors.append(f"OS metadata in source: {p.relative_to(root)}")
    meta = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    if meta["project"]["name"] != "evogen-harness":
        errors.append("unexpected package name")
    return {"schema": "evogen.release_check.v1", "passed": not errors, "python_files_parsed": syntax_count,
            "local_doc_links_checked": links, "documented_cli_commands_checked": commands,
            "component_audit_passed": bool(audit.get("passed")), "p2_data": data_report,
            "heuristic_safety_findings": len(findings), "errors": errors,
            "scope": "Offline code/data/docs checks; existing docs/ website, network links, GPU inference and historical benchmark replication are not tested here."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
