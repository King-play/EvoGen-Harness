from __future__ import annotations

from fnmatch import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "dist",
    "build",
    "runs",
    "logs",
}

DEFAULT_EXCLUDED_FILE_GLOBS = {
    "[0-9][0-9][0-9].txt",
    "CHAT_RECORD_*.txt",
    "conversation_record_*.txt",
    "CURRENT_*.txt",
    "FULL_GENEVAL_*.txt",
    "GENEVAL_FULL*.txt",
    "PROGRESS_RECORD_*.txt",
    "PROMPT_EFFECTS_SUMMARY_*.md",
    "RESULT_*.txt",
    "TODAY_*.txt",
    "readme",
}

SENSITIVE_FILENAMES = {
    ".env",
    ".genharness.local.env",
    "secrets.json",
    "credentials.json",
}

@dataclass
class SafetyFinding:
    path: str
    line: int
    kind: str
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return {"path": self.path, "line": self.line, "kind": self.kind, "detail": self.detail}


def scan_open_source_safety(
    root: str | Path,
    *,
    excludes: Sequence[str] | None = None,
    include_ignored_local: bool = True,
) -> List[SafetyFinding]:
    root_path = Path(root)
    exclude_names = set(DEFAULT_EXCLUDES)
    if excludes:
        exclude_names.update(excludes)
    findings: List[SafetyFinding] = []
    for path in _iter_files(root_path, exclude_names):
        rel = str(path.relative_to(root_path))
        if path.name in SENSITIVE_FILENAMES:
            if include_ignored_local:
                findings.append(SafetyFinding(rel, 0, "sensitive_file", "local credential file must not be committed or packaged"))
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf", ".zip", ".gz", ".tar"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "${" in line:
                continue
            if _contains_large_inline_image(line):
                findings.append(SafetyFinding(rel, line_no, "inline_image_data", "large base64 image data should stay out of source files"))
            if _contains_local_absolute_path(line):
                findings.append(SafetyFinding(rel, line_no, "local_absolute_path", _redact(line.strip())))
            if _contains_possible_secret(line):
                findings.append(SafetyFinding(rel, line_no, "possible_secret", _redact(line.strip())))
    return findings


def _iter_files(root: Path, exclude_names: set[str]) -> Iterable[Path]:
    if not root.exists():
        return
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for path in children:
            if path.name in exclude_names:
                continue
            if path.is_file() and any(fnmatch(path.name, pattern) for pattern in DEFAULT_EXCLUDED_FILE_GLOBS):
                continue
            if path.is_dir():
                stack.append(path)
            elif path.is_file():
                yield path


def _redact(value: str) -> str:
    if len(value) <= 24:
        return "[redacted]"
    return value[:8] + "...[redacted]..." + value[-4:]


def _contains_large_inline_image(line: str) -> bool:
    marker = "data:image/"
    start = line.find(marker)
    if start < 0:
        return False
    payload_marker = ";base64,"
    payload_start = line.find(payload_marker, start)
    if payload_start < 0:
        return False
    payload = line[payload_start + len(payload_marker) :]
    payload = payload.split()[0].strip("'\"),]}")
    return len(payload) >= 512 and _all_from(payload, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


def _contains_local_absolute_path(line: str) -> bool:
    for marker in ("/data/", "/home/", "/root/"):
        start = line.find(marker)
        if start < 0:
            continue
        before = line[start - 1] if start > 0 else " "
        if before.isalnum() or before in {"_", ".", "-"}:
            continue
        tail = line[start:].split()[0].strip("'\"`),]}")
        if len(tail.split("/")) >= 4:
            return True
    return False


def _contains_possible_secret(line: str) -> bool:
    lowered = line.lower()
    if _contains_openai_key(line) or _contains_google_key(line):
        return True
    if "authorization" in lowered and "bearer" in lowered:
        value = _value_after_casefold_marker(line, "bearer")
        return len(value) >= 12 and "${" not in value
    for label in ("api_key", "api-key", "apikey", "secret", "token"):
        index = lowered.find(label)
        if index < 0:
            continue
        if not _label_has_boundary(lowered, index, label):
            continue
        value = _value_after_separator(line[index + len(label) :])
        if len(value) >= 12 and "${" not in value and _all_from(value, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-./"):
            return True
    return False


def _contains_openai_key(line: str) -> bool:
    for part in _rough_tokens(line):
        if part.startswith("sk-") and len(part) >= 19 and _all_from(part[3:], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"):
            return True
    return False


def _contains_google_key(line: str) -> bool:
    for part in _rough_tokens(line):
        if part.startswith("AIza") and len(part) >= 24 and _all_from(part[4:], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"):
            return True
    return False


def _value_after_separator(text: str) -> str:
    for separator in (":", "="):
        index = text.find(separator)
        if index >= 0:
            value = text[index + 1 :].strip()
            return value.strip("'\" ")
    return text.strip().strip("'\" ")


def _value_after_casefold_marker(text: str, marker: str) -> str:
    index = text.lower().find(marker.lower())
    if index < 0:
        return _value_after_separator(text)
    return _value_after_separator(text[index + len(marker) :])


def _label_has_boundary(text: str, index: int, label: str) -> bool:
    before = text[index - 1] if index > 0 else " "
    after_index = index + len(label)
    after = text[after_index] if after_index < len(text) else " "
    allowed_identifier = "_-"
    return not (before.isalnum() or before in allowed_identifier or after.isalnum() or after in allowed_identifier)


def _rough_tokens(line: str) -> List[str]:
    tokens: List[str] = []
    current: List[str] = []
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-./")
    for char in line:
        if char in allowed:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _all_from(value: str, alphabet: str) -> bool:
    allowed = set(alphabet)
    return bool(value) and all(char in allowed for char in value)
