# Architecture and scope

[Documentation](README.md)

The method is described in Sections 3 and B of the [paper](https://arxiv.org/abs/2610.00383). The following paths refer to the supplied implementation.

| Module | Role |
| :--- | :--- |
| [components/](../gen_harness/components/) | Policy, Tools, Skills, Middleware, and Memory interfaces |
| [repository.py](../gen_harness/repository.py) | Load and manage persistent harness files |
| [executor.py](../gen_harness/executor.py) | Execute compiled requests with frozen tools/models |
| [fast_loop.py](../gen_harness/fast_loop.py) | Per-request stochastic evidence and transient repair |
| [patcher.py](../gen_harness/patcher.py) | Harness localization and responsibility-conditioned edit proposals |
| [validator.py](../gen_harness/validator.py) | Matched target, held-out and preservation evaluation |
| [experiments/](../gen_harness/experiments/) | Persistent evolution and generation runners |
| [trace_protocol.py](../gen_harness/trace_protocol.py) | Seed, LLM-output and split-count contracts |
| [evaluation/](../gen_harness/evaluation/) | Prompt-only converters and official-evaluator exports |
| [leakage_guard.py](../gen_harness/leakage_guard.py) | Guard evaluation-only information from generation inputs |

**Persistent state is not model weights.** Policy contains requirement interpretation, Tools contains capability knowledge, Skills contains reusable procedures, Middleware contains orchestration settings, and Memory contains experience. The model implementation and deterministic base controller are not edit targets.

**Transient repair is not cross-task evolution.** Generation can make bounded internal repairs for a request. The persistent runner separately proposes and validates reusable updates. Evaluation must not promote test-specific changes into the harness used for later benchmark examples.

The supplied [paper protocol config](../configs/experiments/trace_paper_protocol.json) records `r=3`, `K=4`, `M=3`, `L=3`, preservation weight `0.5`, cost weight `0.05`, and regression tolerance `0.01`. Treat this as a configuration contract, not proof that an arbitrary directory contains a previously evaluated final state.

The paper's concise localizer prompt and the implementation's stricter JSON schema operate at different documentation levels. Use the actual config/interface of the checked-out code rather than pasting a paper snippet over a runtime schema.

## Required state and optional adapters

The example harness needs three Policy records (`visual_contract.json`, `reference_policy.json`, and `preservation_policy.json`) and one Skills record (`skills.json`). They are consumed by `HarnessRepository` and define editable state; they do not each configure a separate model. Tool and Middleware records are similarly read by their corresponding components. Memory JSONL stores may start empty and are created as runs produce records; the bundled prompt guidelines are example reusable knowledge, not historical experiment results.

Accepted and rejected patch records use `validated_patches.jsonl` and `regression_cases.jsonl`.

A default FLUX run uses one backend config, one composite inspection config, one policy-extractor config, and a localizer/proposer pair. Persistent evolution and request-local repair select their respective documented pairs. Other backend, search, fault-injection, and synthetic-dataset configs are optional examples; a default run does not load every JSON file in `configs/`.

Model integrations load their heavy dependencies only when selected. CPU tests use explicit test doubles and cannot establish generation quality or paper-score reproduction. Request constraints are rendered without object-specific recipe tables or fixed-count layouts. Failure diagnostics retain observed evidence and explicit LLM localization results; an unlocalized cause remains unknown.
