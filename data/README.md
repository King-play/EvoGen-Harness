# Data inventory

[evolution/](evolution/README.md) contains the supplied PartiPrompts P2 500/100/100 files, source metadata, and release fingerprints.

`geneval2/` contains the supplied 800 prompt-only inputs plus 32-prompt and 16-action-prompt subsets. Subsets are functional checks, not full benchmark results. The source conversion retains prompt/task metadata and excludes official evaluation labels from generation inputs.

`synthetic_compositional/` contains the original development fixtures and their manifests. They are not substitutes for the P2 evolution corpus or the official benchmark releases.

The original files are retained. All third-party data retain their upstream terms; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Benchmark preparation and evaluation are described in [EVALUATION.md](../documentation/EVALUATION.md).
