# Reproducibility

[Documentation](README.md) · [Paper](https://arxiv.org/abs/2610.00383)

The repository provides the implementation, example harness state, prompt-only P2 evolution inputs and benchmark adapters. The example harness is a starting configuration, not an authenticated historical final checkpoint. Original paper image archives and raw official score logs are not bundled.

## Evolution data

The supplied P2 subsets contain 500 target, 100 held-out and 100 preservation prompts. The data preparation helper verifies their checksums, metadata and pairwise disjointness. The provenance manifest records normalized exact-match filtering against the supplied GenEval2 prompts. It does not establish near-duplicate filtering against all three benchmarks; see [the data documentation](../data/evolution/README.md).

## Experiment records

For each real run, retain the harness snapshot and its checksum, input hashes, model revisions, selected configs, random seeds, accepted and rejected edits, generated images and independent evaluator outputs. [The artifact manifest](../artifacts/MANIFEST.example.json) describes this interface and is a template rather than an experiment result.

Keep evolution separate from final evaluation. Freeze the evolved harness before generating benchmark images. Internal verification may guide generation; official labels and scores belong only to the separate evaluation stage. Cross-generator adaptation requires a separately evolved harness for each generator.

## Software and model validation

Run `python -m pytest -q -rs` and `python scripts/check_release.py` for local software checks. Unit tests use isolated test doubles and do not measure generation quality. Optional tokenizer tests require the documented local models. GPU inference, external services and official evaluation must be validated separately with the actual models and credentials.

Paper tables contain reported research results. Reproducing them requires real generation and official scoring under matching experimental conditions. Snapshot checks establish file integrity, not model quality or historical provenance.
