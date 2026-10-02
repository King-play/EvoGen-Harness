# Third-party notices

The [MIT License](LICENSE) applies to the project code as supplied by its authors. It does not replace third-party data/model terms or grant rights to third-party branding.

| Component | Source and scope |
| :--- | :--- |
| PartiPrompts P2 | Prompt text from [google-research/parti](https://github.com/google-research/parti); upstream repository [Apache-2.0 license](https://github.com/google-research/parti/blob/main/LICENSE). The supplied JSONL adds task IDs, selection and split metadata. A standard license copy is included in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). |
| GenEval2 prompt inputs | Derived prompt-only records/subsets from [facebookresearch/GenEval2](https://github.com/facebookresearch/GenEval2). Upstream [CC BY-NC 4.0 notice](https://github.com/facebookresearch/GenEval2/blob/main/LICENSE), copyright Meta Platforms, Inc. and affiliates. Conversion/subsetting are the changes in this bundle; official evaluation labels are not generation inputs. License terms: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). |
| FLUX.1-dev | Model obtained separately under the upstream [FLUX.1-dev non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md). No weights are bundled; do not describe the model license as MIT or as this repository's code license. |
| Other backbones and verifiers | Obtain and use each model under its own model-card/code terms. Optional HTTP adapters do not provide a hosted model or access rights. |
| Official evaluators | Installed separately from their actual pinned upstream releases. Their code, data and model dependencies retain their licenses. |
| Paper figures and EvoGen marks | Author-supplied research figures and project identity assets for presenting this research. No third-party brand artwork is copied. Model/data terms still apply where relevant; the code license is not a trademark license. |

Presentation references (not dependencies or copied source): [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [Kimi K2](https://github.com/MoonshotAI/Kimi-K2), and [SAM 2](https://github.com/facebookresearch/sam2). They informed the separation of resource links, visual results, setup, usage, and citation.
