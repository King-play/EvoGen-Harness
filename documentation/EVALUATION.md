# Benchmark generation and independent evaluation

[Documentation](README.md) · [Runtime setup](INSTALL.md)

First complete model and endpoint setup, then select the harness you actually intend to evaluate. A usage run may use the supplied configuration; a run of an evolved system should use its archived final state:

```bash
export EVOGEN_EVAL_HARNESS="artifacts/my_run/final_harness"
python scripts/snapshot_harness.py --verify "$EVOGEN_EVAL_HARNESS"
```

This path is created by the [evolution workflow](EVOLUTION.md); it is not a pre-bundled paper checkpoint. Select your real existing snapshot. Run commands from the repository root.

## Workflow


Official benchmark labels and scores are never used during generation, localization, proposal, or fast-loop repair. The workflow is always:

1. convert benchmark prompts to prompt-only Gen-Harness tasks;
2. generate the full image set with EVOGEN-HARNESS;
3. freeze the generated images and export the official evaluator input;
4. run the official evaluator separately.

Do not use `--limit` for a full benchmark run. Use it only for smoke tests.

### GenEval2

```bash
python -m gen_harness.cli build-tasks \
  --dataset geneval2 \
  --benchmark-data /path/to/geneval2_data.jsonl \
  --output data/geneval2/tasks.jsonl

python -m gen_harness.cli generate \
  --dataset geneval2 \
  --harness "$EVOGEN_EVAL_HARNESS" \
  --tasks data/geneval2/tasks.jsonl \
  --output-dir runs/geneval2 \
  --backend-config configs/backends/flux1_dev_diffusers.json \
  --inspection-config configs/inspection/composite_owlv2_nvila_strong.json \
  --policy-extractor-config configs/policy_extractors/openai_chat_policy_extractor.json \
  --harness-localizer-config configs/patch_proposers/openai_chat_harness_localizer_fastloop.json \
  --patch-proposer-config configs/patch_proposers/openai_chat_patch_proposer_fastloop.json

python -m gen_harness.cli export-eval \
  --dataset geneval2 \
  --experiences runs/geneval2/gen_harness_generations.jsonl \
  --benchmark-data /path/to/geneval2_data.jsonl \
  --output runs/geneval2/image_map.json
```

Then run the official GenEval2 evaluator from `$GENHARNESS_GENEVAL2_OFFICIAL_REPO` with model weights from `$GENHARNESS_GENEVAL2_EVAL_MODEL` on `runs/geneval2/image_map.json`.

### T2I-CompBench++

```bash
python -m gen_harness.cli build-tasks \
  --dataset t2icompbenchpp \
  --dataset-dir /path/to/T2I-CompBench++ \
  --output data/t2icompbenchpp/tasks.jsonl

python -m gen_harness.cli generate \
  --dataset t2icompbenchpp \
  --harness "$EVOGEN_EVAL_HARNESS" \
  --tasks data/t2icompbenchpp/tasks.jsonl \
  --output-dir runs/t2icompbenchpp \
  --backend-config configs/backends/flux1_dev_diffusers.json \
  --inspection-config configs/inspection/composite_owlv2_nvila_t2icompbenchpp.json \
  --policy-extractor-config configs/policy_extractors/openai_chat_policy_extractor.json \
  --harness-localizer-config configs/patch_proposers/openai_chat_harness_localizer_fastloop.json \
  --patch-proposer-config configs/patch_proposers/openai_chat_patch_proposer_fastloop.json

python -m gen_harness.cli export-eval \
  --dataset t2icompbenchpp \
  --tasks data/t2icompbenchpp/tasks.jsonl \
  --experiences runs/t2icompbenchpp/gen_harness_generations.jsonl \
  --manifest-dir runs/t2icompbenchpp/eval_manifests \
  --sample-root runs/t2icompbenchpp/eval_samples \
  --copy
```

Then run the official T2I-CompBench++ metrics from `$GENHARNESS_T2ICOMPBENCHPP_OFFICIAL_REPO` with assets from `$GENHARNESS_T2ICOMPBENCHPP_ASSET_DIR` on `runs/t2icompbenchpp/eval_manifests` and `runs/t2icompbenchpp/eval_samples`.

### WISE

```bash
python -m gen_harness.cli build-tasks \
  --dataset wise \
  --benchmark-data /path/to/wise.jsonl \
  --output data/wise/tasks.jsonl

python -m gen_harness.cli generate \
  --dataset wise \
  --harness "$EVOGEN_EVAL_HARNESS" \
  --tasks data/wise/tasks.jsonl \
  --output-dir runs/wise \
  --backend-config configs/backends/flux1_dev_diffusers.json \
  --inspection-config configs/inspection/composite_owlv2_nvila_strong.json \
  --policy-extractor-config configs/policy_extractors/openai_chat_policy_extractor.json \
  --harness-localizer-config configs/patch_proposers/openai_chat_harness_localizer_fastloop.json \
  --patch-proposer-config configs/patch_proposers/openai_chat_patch_proposer_fastloop.json

python -m gen_harness.cli export-eval \
  --dataset wise \
  --tasks data/wise/tasks.jsonl \
  --experiences runs/wise/gen_harness_generations.jsonl \
  --output runs/wise/official_input.jsonl
```

Then run the official WISE/WiScore evaluator from `$GENHARNESS_WISE_OFFICIAL_REPO` with model weights from `$GENHARNESS_WISE_EVAL_MODEL` on `runs/wise/official_input.jsonl`.


## Interpretation and completeness

`--limit`, `--allow-missing`, category subsets, and smoke inputs are for functional checks, not full benchmark reporting. Avoid `--continue-on-error` for a formal run unless failures are explicitly accounted for. Do not use official evaluation labels or scores for prompt compilation, repair selection, or persistent evolution.

The `export-eval` commands prepare the official input structures; they do not install or execute the official scoring pipelines. The `GENHARNESS_*_OFFICIAL_REPO` variables document the evaluator checkout locations to invoke separately; exporting the variables does not cause this CLI to run those checkouts automatically. Pin and record the actual upstream revision and follow its versioned instructions.

After generation, verify the selected harness again. Preserve the complete generation JSONL, input manifest, output-image mapping, failure accounting, evaluator revision, scoring settings, and raw official score files. Scores in the root README remain attributed to the paper until you independently reproduce them.

Official project sources: [GenEval2](https://github.com/facebookresearch/GenEval2), [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench), and [WISE](https://github.com/PKU-YuanGroup/WISE). Use the specific release matching the experiment, rather than assuming an arbitrary current default is the original evaluation setup.
