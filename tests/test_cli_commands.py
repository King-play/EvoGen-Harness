"""Exercise public command entry points without loading models or calling APIs."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pytest

from gen_harness.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = next(
    action.choices for action in build_parser()._actions
    if isinstance(action, argparse._SubParsersAction)
)


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_subcommand_help_runs_in_a_fresh_process(command):
    result = subprocess.run(
        [sys.executable, "-m", "gen_harness.cli", command, "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(("script", "options", "message"), [
    ("owlv2_open_vocab_detector.py", [], "--input and --output are required"),
    ("nvila_semantic_verifier.py", [], "--input and --output are required"),
    ("owlv2_open_vocab_detector.py", ["--threshold", "-1"], "uncertainty-threshold"),
    ("owlv2_open_vocab_detector.py", ["--nms-threshold", "2"], "nms-threshold"),
    ("nvila_semantic_verifier.py", ["--pass-threshold", "2"], "between 0 and 1"),
    ("nvila_semantic_verifier.py", ["--reject-threshold", "nan"], "between 0 and 1"),
])
def test_verifier_invalid_arguments_stop_before_model_loading(script, options, message):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *options],
        cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr
