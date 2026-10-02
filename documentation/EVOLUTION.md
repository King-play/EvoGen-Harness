# TRACE evolution and harness snapshots

[Documentation](README.md) · [Runtime setup](INSTALL.md)

## 1. Prepare inputs and a writable working harness

The evolution split is separate from benchmark evaluation data. Use all three supplied files through the preparation helper:

```bash
python scripts/prepare_evolution.py --output runs/trace/evolution_tasks.jsonl
python scripts/snapshot_harness.py \
  --source examples/visual_harness \
  --output runs/trace/working_harness
```

The second command copies only the five responsibility directories, after component and heuristic safety checks. It also writes `SNAPSHOT.json` for the initial copy. **That checksum describes the initial state**; it is expected to become outdated once evolution changes the working harness. A new final snapshot is created after the run.

The source directory is not modified. The output must be a new directory. The snapshot tool does not assign historical paper identity or synthesize missing experience records.

## 2. Run persistent evolution after runtime setup

```bash
python -m gen_harness.cli self-evolve-run \
  --harness runs/trace/working_harness \
  --tasks runs/trace/evolution_tasks.jsonl \
  --output-dir runs/trace/evolution \
  --backend-config configs/backends/flux1_dev_diffusers.json \
  --inspection-config configs/inspection/composite_owlv2_nvila_strong.json \
  --policy-extractor-config configs/policy_extractors/openai_chat_policy_extractor.json \
  --harness-localizer-config configs/patch_proposers/openai_chat_harness_localizer.json \
  --patch-proposer-config configs/patch_proposers/openai_chat_patch_proposer.json \
  --paper-trace --seeds 0,1,2,3 --max-rounds 3 \
  --trace-frontier-size 3 --trace-candidate-budget 3 \
  --trace-preservation-weight 0.5 --trace-execution-cost-weight 0.05 \
  --regression-epsilon 0.01 --promote-accepted
```

This command uses real image/verification models and a configured fixed LLM. It can be expensive. Run the budget-only example in the root README before launch. The budget-only example does not set `--paper-trace`, because that full runtime check requires model configuration variables even when estimating cost.

`--promote-accepted` allows validated updates to persist in the **working copy**. The `*_fastloop.json` configs are not substitutes for the persistent localizer/proposer configs shown above. `--max-rounds` is the iterative runner's maximum round count; do not equate a new run with a reported paper run merely because one budget matches.

## 3. Archive the exact post-run state

Once evolution has completed, inspect its summaries and create a new snapshot:

```bash
python scripts/snapshot_harness.py \
  --source runs/trace/working_harness \
  --output artifacts/my_run/final_harness
python scripts/snapshot_harness.py --verify artifacts/my_run/final_harness
```

The new manifest contains per-file SHA-256 hashes and an aggregate tree digest. This verifies identity/integrity, **not quality or historical provenance**. Use the same archived directory for benchmark generation, do not run persistent evolution on benchmark prompts, and re-run verification afterwards.

Archive real run summaries with the existing CLI:

```bash
python -m gen_harness.cli self-evolve-manifest \
  --output-dir runs/trace/evolution \
  --output runs/trace/evolution_manifest.json
```

See [artifacts/README.md](../artifacts/README.md) for matching model revisions, seeds, benchmark inputs, and independent evaluator versions. Do not publish private prompts, credentials, or unreviewed raw traces.

## 4. Evaluate separately

Follow [EVALUATION.md](EVALUATION.md). Generation-time internal verification and official benchmark evaluation are different stages. Official labels and scores must not be supplied to the localizer/proposer or used to choose among generated outputs.
