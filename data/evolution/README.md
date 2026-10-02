# Included PartiPrompts P2 evolution inputs

This release **already includes** the three prompt-only files:

| File | Rows | `metadata.split` |
| :--- | ---: | :--- |
| [partiprompts_p2_target500.jsonl](partiprompts_p2_target500.jsonl) | 500 | `target` |
| [partiprompts_p2_heldout100.jsonl](partiprompts_p2_heldout100.jsonl) | 100 | `heldout` |
| [partiprompts_p2_preservation100.jsonl](partiprompts_p2_preservation100.jsonl) | 100 | `preservation` |

The JSONL bytes and the supplied [provenance manifest](partiprompts_p2_manifest.json) are unchanged from the latest uploaded code archive. [checksums.json](checksums.json) records their SHA-256 fingerprints for this release.

## What the manifest establishes

It records the public PartiPrompts TSV URL and source hash, seed **2027**, 1,632 source rows, 1,631 deduplicated rows, and filtering by **normalized exact prompt match** against one listed GenEval2 prompt-only file. It reports zero removed benchmark overlaps.

This establishes the recorded construction and the supplied bytes. It does **not** record semantic/near-duplicate filtering or overlap checking against T2I-CompBench++ and WISE; no such broader check is inferred from it. The paper describes a broader filtering protocol. Historical identity with the exact paper run is not established solely by the counts or these filenames.

## Validate and use these exact files

From the repository root:

```bash
python scripts/prepare_evolution.py --output runs/quickstart/evolution_tasks.jsonl
```

This verifies counts, split/source metadata, unique task IDs and source indices, normalized prompt disjointness, checksums, and overlap against the benchmark files named in the supplied manifest. It then combines the existing rows without changing their content or order within each split. The adjacent `.manifest.json` records the working-file hash and validation result. A different pre-existing output is not overwritten without `--force`.

## Build a new split for a new experiment

The implementation also supports rebuilding from the upstream TSV, JSON, or JSONL:

```bash
python -m gen_harness.cli build-evolution-split \
  --output-dir runs/new_p2_split \
  --benchmark-tasks data/geneval2/geneval2_tasks_prompt_only_2026_09_04.jsonl
```

This is a **new dataset construction**, not a way to silently replace this release's inputs. Download access is required unless a local source is supplied with `--source`. Repeat `--benchmark-tasks` for other prepared benchmark prompt files; the builder's filter is still normalized exact matching, not semantic near-duplicate filtering. Preserve the resulting manifest and upstream version. For custom counts, filenames retain the original API's nominal `target500`/`heldout100`/`preservation100` names; actual counts are recorded in the manifest.

Source: [google-research/parti](https://github.com/google-research/parti). Prompt text comes from PartiPrompts; conversion to task JSONL and selection metadata are supplied by this project. See [third-party notices](../../THIRD_PARTY_NOTICES.md).
