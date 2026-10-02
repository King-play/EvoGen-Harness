# Source-release preparation notes

This package was rebuilt from the latest uploaded `genharness.zip`, SHA-256:

```text
bc9be5a5a1bcb4fa5bce7b17a14575d18b57a596728ec121c08deea81e90c20a
```

## Presentation and documentation

Replaced the source README with a research/code landing page, local overview and qualitative figures, light/dark logos, citation and runnable entry points. Added a Chinese README, separate installation/evolution/evaluation/architecture guides, contribution/security guidance and safe GitHub merge instructions. The existing project website `docs/` is intentionally not part of this overlay.

Updated the stale P2 data README: this upload already includes the three 500/100/100 files. Retained their exact bytes and original construction manifest. Described only the normalized exact-match overlap scope actually recorded there. Updated the example experiment manifest to reference those existing files; it remains explicitly a template, not an experiment result.

## Packaging and small code corrections

- Added `pyproject.toml` for editable installation and the `evogen-harness` command; preserved the source version `0.2.0`.
- Added a small `.[dev]` environment and CPU CI rather than installing the full CUDA stack for unit tests.
- Kept all original full-environment package pins; added the missing CUDA wheel index in `requirements.txt`.
- Fixed the existing PartiPrompts builder's advertised JSONL input path: multiple JSON objects were incorrectly parsed as one JSON document. TSV/default split behavior and public output filenames are unchanged. Added explicit validation of negative/non-integral/zero-total split sizes.
- Corrected one optional integration test's dependency guard: it checked for PyTorch but then unconditionally asserted Transformers metadata, causing a false failure in environments with PyTorch alone. It now declares both prerequisites; its assertions are unchanged.
- Added offline release/data validation, non-destructive split combination, and explicit harness snapshot/verification helpers, with regression tests.
- Removed macOS resource-fork metadata from the archive, preserved the existing license, and expanded safe ignores without excluding the website or required source/data.

The TRACE search, localizer, proposer, execution, validation, generator adapters, numeric paper results and supplied data were not replaced or retuned. Full source-diff hashes are in [source_integrity.json](source_integrity.json).

## Validation scope

Actual executed checks and skipped integrations are recorded in [release_validation.json](release_validation.json). CPU tests, editable package building, data checks, CLI budget runs, local document links and preview rendering are distinct from full GPU installation, model inference, paid API availability or replication of paper scores. The latter are not claimed by this release preparation.
