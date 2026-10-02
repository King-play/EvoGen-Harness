# Archiving a real experiment

The P2 input files are already provided in [data/evolution/](../data/evolution/). Do not create empty substitutes or duplicate them just to satisfy an example directory tree.

A complete historical run archive links an explicitly identified harness state to its input hashes, configs, model revisions, seeds, generated images, run logs and raw independent evaluation scores. [MANIFEST.example.json](MANIFEST.example.json) is a **template**, not a result or a downloadable checkpoint. Its paths are examples to be filled from a real run.

For a new run, use [the evolution guide](../documentation/EVOLUTION.md) to create a working copy and archive the resulting state with `scripts/snapshot_harness.py`. Retain the five responsibility directories and `SNAPSHOT.json`. Copy real generation and scoring outputs only after reviewing privacy, size, and redistribution terms. Large image archives need not be committed to Git history; publish an appropriately reviewed artifact bundle and record its checksum and access link in the real manifest.

Do not label `examples/visual_harness` or an arbitrary newly created snapshot as the final historical paper harness without identifying that correspondence. Do not report benchmark scores from template JSON. The source bundle remains usable for new runs without any invented historical artifacts.
