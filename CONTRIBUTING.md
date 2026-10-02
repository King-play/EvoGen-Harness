# Contributing

Please open a focused issue or pull request describing the problem, expected behavior, and a minimal reproducible command. Read [the architecture](documentation/ARCHITECTURE.md) before changing persistent state or validation boundaries.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q -rs
python scripts/check_release.py
```

Use explicit dependencies/skip conditions for model-dependent tests. CPU tests must not silently invoke paid APIs, download weights, or present synthetic outputs as real model results. Add regression tests for behavior changes. Do not modify paper scores to make a test pass.

Before a pull request, review the diff for credentials, local paths, model weights and unintended changes to `docs/`. The existing `docs/` folder is the project website; development documentation belongs in `documentation/`.

For experimental changes, record input hashes, configs, seeds, model/evaluator revisions, accepted/rejected updates, and real outputs. Keep benchmark labels and official scores out of generation/evolution. Contributors must have permission to submit their changes under the project's existing license and retain third-party notices.
