# Release contents and reproducibility scope

[Documentation](README.md) · [Paper](https://arxiv.org/abs/2610.00383)

This page inventories the **latest supplied code archive**, not an earlier upload. No missing historical material is replaced by invented data, empty score files, or a renamed example directory.

| Material | What this release actually contains |
| :--- | :--- |
| Method implementation | Five responsibilities, transient execution, persistent evolution, validators and CLI |
| P2 inputs | Real 500/100/100 prompt-only JSONL files and the supplied provenance manifest |
| Dataset checks | Added hashes, count/disjointness checks, and verification of the manifest's named exact-match benchmark scope |
| Harness configuration | The supplied `examples/visual_harness/`, preserved without claiming historical final status |
| Experiment outputs | Tools to create/export real outputs; not the original paper's full generation and score logs |
| Paper presentation | Original overview, progressive repair, additional examples and paper-reported summary scores |
| Run artifact interface | Explicit example manifest, snapshot tool and archival instructions—not a fabricated completed experiment |

## Data provenance is precise, not inferred

The supplied manifest lists normalized exact matching against the included GenEval2 prompt-only file. It does not record near-duplicate removal against all three benchmarks. The paper describes a broader filtering procedure; this release records the distinction in [the data README](../data/evolution/README.md) instead of silently claiming they are identical.

The three supplied JSONL files and their original manifest remain byte-for-byte unchanged. The separate checksum inventory makes later changes visible. New splits should be built into a new working directory, with their own provenance; do not overwrite released inputs while retaining old hashes.

## Three useful levels of use

A reader can inspect the method and run local code checks with no historical experiment bundle. A new experiment can run with the provided data/configuration plus the required models and API configuration. Exact replay or audit of a particular paper result additionally needs an explicitly identified historical state, matching revisions/seeds and real original score/image records.

The provided snapshot helper can archive a real directory you select after a run. It cannot certify that a new snapshot is the original final paper harness. Similarly, passing unit tests or producing a budget estimate is not a reproduction of the reported benchmark scores.

## Validation performed for this package

See [release_validation.json](release_validation.json) and [RELEASE_NOTES.md](RELEASE_NOTES.md). Software checks are recorded with the Python/package environment, skipped integrations and command outcomes. Full GPU dependency resolution, model inference, API availability, and paper metric replication are outside this CPU validation. This is a boundary of the checks, not a claim that the external components are absent from your own lab environment.
