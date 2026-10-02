# Installation and runtime configuration

[Documentation](README.md) · [Quick start](../README.md#quick-start)

## Two installation profiles

**Development / CPU checks.** From the repository root, install `python -m pip install -e ".[dev]"`. This uses `pyproject.toml` and installs the small dependencies needed by the local tools and tests. It does not turn the generation system into a CPU demo. Tests using real tokenizers or optional GPU-stack packages are conditional.

**Inference environment.** The authors' supplied environment targets Python 3.11 with CUDA 12.4 builds of PyTorch 2.6.0 / torchvision 0.21.0. Use a separate environment rather than replacing an existing system-wide installation:

```bash
conda create -n evogen-runtime python=3.11 -y
conda activate evogen-runtime
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

The original package pins in `requirements.txt` are preserved; a CUDA wheel index was added so the `+cu124` packages have a source. This is an **author-supplied environment specification**, not a newly regenerated lockfile or a claim that all future package indexes will provide the same wheels. CUDA build selection follows the [PyTorch 2.6 installation table](https://pytorch.org/get-started/previous-versions/#v260). Git and internet access are required by the VCS/wheel entries. Official benchmark evaluators have their own dependencies and should use separate environments.

The release's local test environment and checks actually performed are recorded in [release validation](release_validation.json). Full installation of the GPU environment and real model inference were not performed in that CPU validation session.

## Credentials and model directories

Copy the template, edit it locally, and export its values. The runtime does not automatically load `.env`:

```bash
cp .env.example .env
# Edit .env: replace the example endpoint/credential and uncomment your model paths.
set -a
source .env
set +a
```

Use shell-compatible assignments and quote paths containing spaces. Never commit `.env`.

| Variable | Required for |
| :--- | :--- |
| `GENHARNESS_LLM_BASE_URL` | OpenAI-compatible endpoint including its `/v1` base |
| `GENHARNESS_LLM_MODEL` | Configured LLM model identifier |
| `GENHARNESS_LLM_MODEL_SNAPSHOT` | Explicit snapshot used by the paper-TRACE contract |
| `GENHARNESS_LLM_API_KEY` | Authentication; supplied privately |
| `GENHARNESS_FLUX_MODEL_ROOT` | Complete local FLUX.1-dev Diffusers model directory |
| `GENHARNESS_OWLV2_MODEL` | Local OWLv2-base-patch16-ensemble model directory |
| `GENHARNESS_NVILA_MODEL` | Local NVILA-Lite-2B-Verifier model directory |

For `--paper-trace`, the source validates the fixed GPT-4.1 snapshot `gpt-4.1-2025-04-14`. Other research configurations must not be represented as this exact setting. There is no rule-based fallback for missing formal LLM configuration.

Model sources: [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble), and [NVILA-Lite-2B-Verifier](https://huggingface.co/Efficient-Large-Model/NVILA-Lite-2B-Verifier). Obtain authorized access and follow each model's setup instructions. The provided adapters expect complete local model directories, not just one weight file.

`nvila_semantic_verifier.py` loads the model with `trust_remote_code=True`. Review and pin the model code/revision you use; do not treat unreviewed remote model code as a passive weight file. The supplied verifier scripts and relative paths require running from the repository root.

## Hardware and first run

The paper reports two NVIDIA A100 80 GB GPUs. The default FLUX config uses 1024 × 1024 images, 28 inference steps, bfloat16, CUDA, and a 32 GB free-memory guard **for the generator**. That guard is not a guarantee that the complete generator-plus-verifier stack fits in 32 GB. Model device placement and available memory must be checked on the actual machine.

First run the small generation command in the [README](../README.md#run-with-real-models). It uses `--limit 1`, but internal evidence collection and repair can still make multiple generator, LLM, and verifier calls. Inspect both the image and the run summary before increasing the workload.

## Optional backends

`configs/backends/` also includes Flow-GRPO/SD3.5, HTTP, external-command, and ComfyUI adapters. Their configuration examples are interfaces, not hosted services. HTTP examples for Qwen-Image and Janus-Pro require your own compatible service. Use `GENHARNESS_FLOW_GRPO_MODEL_ROOT` for the optional Flow-GRPO backend; do not replace the default FLUX experiment silently.

## Common failures

| Symptom | First check |
| :--- | :--- |
| `No module named gen_harness` | Activate the environment and run the editable install from the repository root |
| Missing `+cu124` wheel | Confirm the PyTorch wheel index and Python 3.11/Linux-compatible environment |
| Unset `GENHARNESS_*` variable | The template must be edited and exported; the program does not auto-read `.env` |
| CUDA unavailable or memory guard fails | Verify the driver, PyTorch CUDA build, GPU placement, and free memory |
| Model import or remote-code dependency error | Match the model's expected runtime; do not mask the error with synthetic outputs |
| `paper_p2` split rejection | Prepare all three supplied subsets and preserve their split metadata |

For help, include the command, dependency versions and a redacted traceback in an issue—not API keys, full environment dumps, or private prompts.
