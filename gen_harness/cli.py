from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .backends import build_backend, validate_formal_backend_config
from .inspection import build_inspector, validate_formal_inspection_config
from .search import build_search
from .prompt_optimizer import build_prompt_optimizer
from .executor import HarnessExecutor
from .io import read_jsonl, write_json, write_jsonl
from .miner import WeaknessMiner
from .patcher import build_harness_localizer, build_patch_proposer
from .repository import HarnessRepository
from .schema import PatchManifest, VisualExperience, Weakness
from .tasks import load_tasks, split_tasks
from .validator import PatchValidator
from .datasets import (
    DEFAULT_PARTIPROMPTS_URL,
    DatasetBuilder,
    build_compositional_suite,
    build_partiprompts_evolution_split,
)
from .faults import FaultInjector
from .experiments import (
    QualityExperiment,
    GenEvalImageGenerationExperiment,
    T2ICompBenchPPImageGenerationExperiment,
    SuiteEvaluationExperiment,
    SlowLoopEvolutionRunner,
    SelfEvolveRunner,
)
from .experiments.slow_loop import estimate_slow_loop_budget
from .component_audit import ComponentBoundaryAuditor, GoalReadinessAuditor, RunArtifactAuditor
from .evaluation import (
    build_geneval2_tasks,
    build_wise_tasks,
    build_t2icompbenchpp_tasks,
    export_geneval2_image_map,
    export_wise_manifest,
    prepare_t2icompbenchpp_eval_manifest,
)
from .safety import scan_open_source_safety
from .self_evolve import build_self_evolve_manifest


def _parse_seeds(value: str | None, default_seed: int) -> List[int]:
    if not value:
        return [default_seed]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_trace_seeds(value: str | None, default_seed: int) -> List[int]:
    """Return TRACE's paper-default K=4 matched stochastic conditions."""
    if value:
        return _parse_seeds(value, default_seed)
    return [default_seed + offset for offset in range(4)]


def _formal_policy_extractor_config(args: argparse.Namespace) -> str:
    config = getattr(args, "policy_extractor_config", None)
    if not config:
        raise ValueError("--policy-extractor-config is required for every generation and evolution run")
    return config


def _build_executor(args: argparse.Namespace) -> HarnessExecutor:
    policy_extractor_config = _formal_policy_extractor_config(args)
    repo = HarnessRepository(
        args.harness,
        policy_extractor_config=policy_extractor_config,
    )
    backend = build_backend(args.backend_config)
    inspector = build_inspector(args.inspection_config)
    search = build_search(args.search_config, required=False) if args.search_config else None
    prompt_optimizer = build_prompt_optimizer(getattr(args, "prompt_optimizer_config", None))
    harness_localizer = build_harness_localizer(args.harness_localizer_config, repo=repo)
    patch_proposer = build_patch_proposer(
        args.patch_proposer_config,
        memory=repo.build_memory(),
        repo=repo,
    )
    return HarnessExecutor(
        repo,
        backend,
        inspector,
        search,
        prompt_optimizer=prompt_optimizer,
        fast_loop_attempts=getattr(args, "fast_loop_attempts", 5),
        harness_localizer=harness_localizer,
        patch_proposer=patch_proposer,
        fast_loop_stochastic_k=getattr(args, "fast_loop_stochastic_k", 4),
    )


def cmd_run(args: argparse.Namespace) -> None:
    executor = _build_executor(args)
    tasks = load_tasks(args.tasks)
    experiences = []
    seed_list = _parse_seeds(getattr(args, "seeds", None), args.seed)
    for run_seed in seed_list:
        for task in tasks:
            experiences.extend([e.to_dict() for e in executor.run_task(task, seed=run_seed, write_memory=args.write_memory)])
    output_path = Path(args.output)
    write_jsonl(output_path, experiences)
    write_json(
        output_path.parent / "run_summary.json",
        {
            "schema": "gen_harness.run_summary.v1",
            "num_tasks": len(tasks),
            "num_experiences": len(experiences),
            "seeds": seed_list,
            "output": str(output_path),
            "policy_extractor_manifest": executor.repo.policy_extractor_manifest(),
            "leakage_and_fallback_guards": {
                "official_evaluator_used_for_generation_or_selection": False,
                "official_labels_used_for_prompt_compilation": False,
                "fallback_image_substitution_allowed": False,
            },
        },
    )
    print(f"wrote {len(experiences)} experiences to {args.output}")


def cmd_mine(args: argparse.Namespace) -> None:
    repo = HarnessRepository(args.harness) if args.harness else None
    exps = [VisualExperience.from_dict(row) for row in read_jsonl(args.experiences)]
    if args.split:
        exps = [e for e in exps if e.metadata.get("split") == args.split]
    weaknesses = WeaknessMiner(min_support=args.min_support).mine(exps)
    write_jsonl(args.output, [w.to_dict() for w in weaknesses])
    if repo is not None and args.write_memory:
        repo.build_memory().write_weaknesses(weaknesses)
    print(f"wrote {len(weaknesses)} weaknesses to {args.output}")


def cmd_patch(args: argparse.Namespace) -> None:
    weaknesses = [Weakness(**row) for row in read_jsonl(args.weaknesses)]
    repo = HarnessRepository(args.harness) if getattr(args, "harness", None) else None
    memory = repo.build_memory() if repo is not None else None
    patches = build_patch_proposer(args.patch_proposer_config, memory=memory, repo=repo).propose_many(weaknesses)
    write_jsonl(args.output, [p.to_dict() for p in patches])
    print(f"wrote {len(patches)} patch manifests to {args.output}")


def cmd_validate(args: argparse.Namespace) -> None:
    policy_extractor_config = _formal_policy_extractor_config(args)
    repo = HarnessRepository(
        args.harness,
        policy_extractor_config=policy_extractor_config,
    )
    backend = build_backend(args.backend_config)
    inspector = build_inspector(args.inspection_config)
    tasks = split_tasks(load_tasks(args.tasks))
    patches = [PatchManifest.from_dict(row) for row in read_jsonl(args.patches)]
    search = build_search(args.search_config, required=False) if args.search_config else None
    prompt_optimizer = build_prompt_optimizer(getattr(args, "prompt_optimizer_config", None))
    validator = PatchValidator(
        repo,
        backend,
        inspector,
        search,
        regression_epsilon=args.regression_epsilon,
        prompt_optimizer=prompt_optimizer,
        metric_mode=getattr(args, "validation_metric_mode", "pass_rate"),
        fast_loop_attempts=getattr(args, "fast_loop_attempts", 5),
    )
    memory = repo.build_memory()
    reports = []
    for patch in patches:
        report = validator.validate(
            patch,
            target_tasks=tasks.get("target", []),
            heldout_tasks=tasks.get("heldout", []),
            preservation_tasks=tasks.get("preservation", []),
            seed=args.seed,
            seeds=_parse_seeds(getattr(args, "seeds", None), args.seed),
        )
        if args.promote and report.accepted:
            validator.promote_if_accepted(patch, report)
        memory.write_patch_result(patch, report.to_dict())
        reports.append(report.to_dict())
    write_jsonl(args.output, reports)
    print(f"wrote {len(reports)} validation reports to {args.output}")


def cmd_build_dataset(args: argparse.Namespace) -> None:
    DatasetBuilder(seed=args.seed).build_splits(
        args.input_manifest,
        args.output,
        target_ratio=args.target_ratio,
        heldout_ratio=args.heldout_ratio,
        preservation_ratio=args.preservation_ratio,
    )
    print(f"wrote task dataset to {args.output}")


def cmd_build_compositional_suite(args: argparse.Namespace) -> None:
    manifest = build_compositional_suite(
        args.output,
        seed=args.seed,
        target_per_family=args.target_per_family,
        heldout_per_family=args.heldout_per_family,
        preservation_per_family=args.preservation_per_family,
        include_action=args.include_action,
    )
    print(manifest)


def cmd_build_evolution_split(args: argparse.Namespace) -> None:
    manifest = build_partiprompts_evolution_split(
        args.source,
        args.output_dir,
        benchmark_tasks=args.benchmark_tasks,
        seed=args.seed,
        target_count=args.target_count,
        heldout_count=args.heldout_count,
        preservation_count=args.preservation_count,
    )
    print(manifest)


def cmd_inject_faults(args: argparse.Namespace) -> None:
    rows = FaultInjector(args.harness, args.output_dir).generate(include_compound=args.include_compound)
    print(f"wrote {len(rows)} controlled fault harnesses to {args.output_dir}")


def cmd_slow_loop_evolve(args: argparse.Namespace) -> None:
    seeds = _parse_trace_seeds(getattr(args, "seeds", None), args.seed)
    paper_trace = bool(getattr(args, "paper_trace", False))
    strict_trace_protocol = paper_trace
    dataset_protocol = "paper_p2" if paper_trace else getattr(args, "trace_dataset_protocol", "development")
    allow_trace_seed_ablation = bool(getattr(args, "allow_trace_seed_ablation", False))
    if strict_trace_protocol and allow_trace_seed_ablation:
        raise ValueError("--paper-trace cannot be combined with --allow-trace-seed-ablation")
    if strict_trace_protocol and bool(getattr(args, "allow_benchmark_tasks", False)):
        raise ValueError("--paper-trace cannot be combined with --allow-benchmark-tasks")
    if strict_trace_protocol:
        validate_formal_backend_config(args.backend_config)
        validate_formal_inspection_config(args.inspection_config)
    policy_extractor_config = _formal_policy_extractor_config(args)
    if getattr(args, "dry_run_budget", False):
        from .experiments.slow_loop import estimate_slow_loop_budget

        print(
            estimate_slow_loop_budget(
                args.tasks,
                seeds=seeds,
                fast_loop_attempts=getattr(args, "fast_loop_attempts", 5),
                allow_benchmark_tasks=args.allow_benchmark_tasks,
                trace_frontier_size=getattr(args, "trace_frontier_size", 3),
                trace_candidate_budget=getattr(args, "trace_candidate_budget", 3),
                strict_trace_protocol=strict_trace_protocol,
                dataset_protocol=dataset_protocol,
                allow_trace_seed_ablation=allow_trace_seed_ablation,
            )
        )
        return

    repo = HarnessRepository(
        args.harness,
        policy_extractor_config=policy_extractor_config,
    )
    backend = build_backend(args.backend_config)
    inspector = build_inspector(args.inspection_config)
    search = build_search(args.search_config, required=False) if args.search_config else None
    prompt_optimizer = build_prompt_optimizer(getattr(args, "prompt_optimizer_config", None))
    runner = SlowLoopEvolutionRunner(
        repo,
        backend,
        inspector,
        search,
        prompt_optimizer,
        validation_metric_mode=getattr(args, "validation_metric_mode", "pass_rate"),
        fast_loop_attempts=getattr(args, "fast_loop_attempts", 5),
        fast_loop_stochastic_k=getattr(args, "fast_loop_stochastic_k", 4),
        regression_epsilon=getattr(args, "regression_epsilon", 0.01),
        harness_localizer_config=getattr(args, "harness_localizer_config", None),
        trace_frontier_size=getattr(args, "trace_frontier_size", 3),
        trace_candidate_budget=getattr(args, "trace_candidate_budget", 3),
        trace_preservation_weight=getattr(args, "trace_preservation_weight", 0.5),
        trace_execution_cost_weight=getattr(args, "trace_execution_cost_weight", 0.05),
        patch_proposer_config=getattr(args, "patch_proposer_config", None),
        strict_trace_protocol=strict_trace_protocol,
        dataset_protocol=dataset_protocol,
        allow_trace_seed_ablation=allow_trace_seed_ablation,
    )
    summary = runner.run(
        args.tasks,
        args.output_dir,
        seeds=seeds,
        min_support=args.min_support,
        promote=args.promote,
        write_memory=not args.no_write_memory,
        allow_benchmark_tasks=args.allow_benchmark_tasks,
    )
    print(summary)


def cmd_self_evolve_manifest(args: argparse.Namespace) -> None:
    manifest = build_self_evolve_manifest(args.output_dir, write_path=args.output)
    print({"output": args.output, "schema": manifest["schema"], "num_changes": len(manifest["changes"])})


def cmd_self_evolve_run(args: argparse.Namespace) -> None:
    seeds = _parse_trace_seeds(getattr(args, "seeds", None), args.seed)
    paper_trace = bool(getattr(args, "paper_trace", False))
    strict_trace_protocol = paper_trace
    dataset_protocol = "paper_p2" if paper_trace else getattr(args, "trace_dataset_protocol", "development")
    allow_trace_seed_ablation = bool(getattr(args, "allow_trace_seed_ablation", False))
    if strict_trace_protocol and allow_trace_seed_ablation:
        raise ValueError("--paper-trace cannot be combined with --allow-trace-seed-ablation")
    if strict_trace_protocol:
        validate_formal_backend_config(args.backend_config)
        validate_formal_inspection_config(args.inspection_config)
    policy_extractor_config = _formal_policy_extractor_config(args)
    if getattr(args, "dry_run_budget", False):
        audit = ComponentBoundaryAuditor().audit(args.harness, tasks_path=args.tasks)
        if not audit.get("passed"):
            print({
                "mode": "self_evolve_run_budget",
                "component_audit_passed": False,
                "num_issues": len(audit.get("issues", [])),
                "issues": audit.get("issues", [])[:5],
            })
            return
        per_round = estimate_slow_loop_budget(
            args.tasks,
            seeds=seeds,
            fast_loop_attempts=getattr(args, "fast_loop_attempts", 2),
            allow_benchmark_tasks=False,
            trace_frontier_size=getattr(args, "trace_frontier_size", 3),
            trace_candidate_budget=getattr(args, "trace_candidate_budget", 3),
            strict_trace_protocol=strict_trace_protocol,
            dataset_protocol=dataset_protocol,
            allow_trace_seed_ablation=allow_trace_seed_ablation,
        )
        max_rounds = max(1, int(args.max_rounds))
        report = {
            "mode": "self_evolve_run_budget",
            "component_audit_passed": True,
            "allow_benchmark_tasks": False,
            "max_rounds": max_rounds,
            "per_round": per_round,
            "upper_bound": {
                "exploration_generation_calls": per_round["exploration_generation_calls"] * max_rounds,
                "exploration_internal_inspection_calls": per_round["exploration_internal_inspection_calls"] * max_rounds,
                "validation_generation_calls_per_patch": per_round["validation_generation_calls_per_patch"],
                "validation_internal_inspection_calls_per_patch": per_round["validation_internal_inspection_calls_per_patch"],
                "candidate_edits": per_round["max_candidate_edits_per_step"] * max_rounds,
                "validation_generation_calls": per_round["max_validation_generation_calls_per_step"] * max_rounds,
                "validation_internal_inspection_calls": per_round["max_validation_internal_inspection_calls_per_step"] * max_rounds,
                "note": "Validation calls are per patch per round; total depends on mined weaknesses and patch proposals.",
            },
            "leakage_and_fallback_guards": {
                "official_evaluator_used_for_generation_or_selection": False,
                "official_labels_used_for_prompt_compilation": False,
                "fallback_image_substitution_allowed": False,
            },
        }
        print(report)
        return
    repo = HarnessRepository(
        args.harness,
        policy_extractor_config=policy_extractor_config,
    )
    backend = build_backend(args.backend_config)
    inspector = build_inspector(args.inspection_config)
    search = build_search(args.search_config, required=False) if args.search_config else None
    prompt_optimizer = build_prompt_optimizer(getattr(args, "prompt_optimizer_config", None))
    runner = SelfEvolveRunner(
        repo,
        backend,
        inspector,
        search,
        prompt_optimizer,
        validation_metric_mode=getattr(args, "validation_metric_mode", "pass_rate"),
        fast_loop_attempts=getattr(args, "fast_loop_attempts", 2),
        regression_epsilon=getattr(args, "regression_epsilon", 0.01),
        harness_localizer_config=getattr(args, "harness_localizer_config", None),
        trace_frontier_size=getattr(args, "trace_frontier_size", 3),
        trace_candidate_budget=getattr(args, "trace_candidate_budget", 3),
        trace_preservation_weight=getattr(args, "trace_preservation_weight", 0.5),
        trace_execution_cost_weight=getattr(args, "trace_execution_cost_weight", 0.05),
        patch_proposer_config=getattr(args, "patch_proposer_config", None),
        strict_trace_protocol=strict_trace_protocol,
        dataset_protocol=dataset_protocol,
        allow_trace_seed_ablation=allow_trace_seed_ablation,
    )
    summary = runner.run(
        args.tasks,
        args.output_dir,
        seeds=seeds,
        max_rounds=args.max_rounds,
        min_support=args.min_support,
        promote_accepted=args.promote_accepted,
        write_memory=not args.no_write_memory,
    )
    print({
        "output": str(Path(args.output_dir) / "self_evolve_run_summary.json"),
        "schema": summary["schema"],
        "stop_reason": summary["stop_reason"],
        "rounds": len(summary["rounds"]),
    })


def cmd_audit_components(args: argparse.Namespace) -> None:
    report = ComponentBoundaryAuditor().audit(args.harness, tasks_path=args.tasks, write_path=args.output)
    if args.output is None:
        print(report)
    else:
        print({"output": args.output, "schema": report["schema"], "passed": report["passed"], "num_issues": len(report["issues"])})


def cmd_audit_run_artifacts(args: argparse.Namespace) -> None:
    report = RunArtifactAuditor().audit(
        tasks_path=args.tasks,
        experiences_path=args.experiences,
        inspection_config_path=args.inspection_config,
        backend_config_path=args.backend_config,
        write_path=args.output,
    )
    if args.output is None:
        print(report)
    else:
        print({"output": args.output, "schema": report["schema"], "passed": report["passed"], "num_issues": len(report["issues"])})


def cmd_audit_goal_readiness(args: argparse.Namespace) -> None:
    report = GoalReadinessAuditor().audit(
        args.run_root,
        smoke_min_am=args.smoke_min_am,
        smoke_min_gm=args.smoke_min_gm,
        full_target_gm=args.full_target_gm,
        write_path=args.output,
    )
    if args.output is None:
        print(report)
    else:
        print({"output": args.output, "schema": report["schema"], "passed": report["passed"], "num_issues": len(report["issues"])})


def cmd_evaluate_suite(args: argparse.Namespace) -> None:
    seeds = _parse_seeds(getattr(args, "seeds", None), args.seed)
    policy_extractor_config = _formal_policy_extractor_config(args)
    if getattr(args, "dry_run_budget", False):
        tasks = load_tasks(args.tasks)
        if args.limit is not None:
            tasks = tasks[:args.limit]
        max_attempts = args.fast_loop_attempts
        print({
            "mode": "generic_suite_internal_inspection",
            "num_tasks": len(tasks),
            "num_seeds": len(seeds),
            "max_generation_calls": len(tasks) * len(seeds) * max_attempts,
            "max_internal_inspection_calls": len(tasks) * len(seeds) * max_attempts,
            "fast_loop_attempts": max_attempts,
            "note": "Actual calls may be lower because repair stops after the first internally passing attempt.",
        })
        return
    summary = SuiteEvaluationExperiment(
        args.harness,
        args.backend_config,
        args.inspection_config,
        args.search_config,
        prompt_optimizer_config=getattr(args, "prompt_optimizer_config", None),
        fast_loop_attempts=getattr(args, "fast_loop_attempts", 5),
        fast_loop_stochastic_k=getattr(args, "fast_loop_stochastic_k", 4),
        harness_localizer_config=args.harness_localizer_config,
        patch_proposer_config=args.patch_proposer_config,
        policy_extractor_config=policy_extractor_config,
    ).run(
        args.tasks,
        args.output_dir,
        seeds=seeds,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
    )
    print(summary)


def cmd_quality_exp(args: argparse.Namespace) -> None:
    policy_extractor_config = _formal_policy_extractor_config(args)
    repo = HarnessRepository(
        args.harness,
        policy_extractor_config=policy_extractor_config,
    )
    backend = build_backend(args.backend_config)
    inspector = build_inspector(args.inspection_config)
    search = build_search(args.search_config, required=False) if args.search_config else None
    prompt_optimizer = build_prompt_optimizer(getattr(args, "prompt_optimizer_config", None))
    summary = QualityExperiment(
        repo,
        backend,
        inspector,
        search,
        prompt_optimizer=prompt_optimizer,
        fast_loop_attempts=getattr(args, "fast_loop_attempts", 5),
        fast_loop_stochastic_k=getattr(args, "fast_loop_stochastic_k", 4),
        harness_localizer_config=args.harness_localizer_config,
        patch_proposer_config=args.patch_proposer_config,
    ).run(
        args.tasks,
        args.output_dir,
        seeds=_parse_seeds(getattr(args, "seeds", None), args.seed),
    )
    print(summary)


def cmd_geneval_generate(args: argparse.Namespace) -> None:
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    seeds = _parse_seeds(getattr(args, "seeds", None), args.seed)
    policy_extractor_config = _formal_policy_extractor_config(args)
    if getattr(args, "dry_run_budget", False):
        tasks = load_tasks(args.tasks)
        if args.limit is not None:
            tasks = tasks[:args.limit]
        gen_harness_multiplier = args.fast_loop_attempts if "gen_harness" in methods else 0
        non_harness_methods = [m for m in methods if m != "gen_harness"]
        print({
            "mode": f"{getattr(args, 'benchmark', 'geneval')}_formal_fast_loop_generation",
            "num_tasks": len(tasks),
            "num_seeds": len(seeds),
            "methods": methods,
            "total_generation_calls": len(tasks) * len(seeds) * (len(non_harness_methods) + gen_harness_multiplier),
            "total_internal_inspection_calls": len(tasks) * len(seeds) * args.fast_loop_attempts if "gen_harness" in methods else 0,
            "note": f"Scores must come from the official {getattr(args, 'benchmark', 'geneval')} evaluator, not this command.",
        })
        return

    backend = build_backend(args.backend_config)
    search = build_search(args.search_config, required=False) if args.search_config else None
    prompt_optimizer = build_prompt_optimizer(getattr(args, "prompt_optimizer_config", None))
    inspector = build_inspector(args.inspection_config)
    summary = GenEvalImageGenerationExperiment(
        args.harness,
        backend,
        search,
        prompt_optimizer,
        inspector=inspector,
        fast_loop_attempts=args.fast_loop_attempts,
        fast_loop_stochastic_k=getattr(args, "fast_loop_stochastic_k", 4),
        policy_extractor_config=policy_extractor_config,
        harness_localizer_config=getattr(args, "harness_localizer_config", None),
        patch_proposer_config=getattr(args, "patch_proposer_config", None),
        benchmark_name=getattr(args, "benchmark", "geneval"),
        export_geneval_layout=getattr(args, "benchmark", "geneval") == "geneval",
    ).run(
        args.tasks,
        args.output_dir,
        methods=methods,
        seeds=seeds,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
    )
    print(summary)


def _add_prompt_benchmark_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--harness", required=True)
    parser.add_argument("--tasks", required=True, help="prompt-only benchmark task JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend-config", required=True, help="real image backend adapter config")
    parser.add_argument("--inspection-config", required=True, help="real internal inspection adapter config")
    parser.add_argument("--search-config", help="real search/evidence adapter config, required only for contracts that require evidence")
    parser.add_argument("--prompt-optimizer-config", help="optional prompt enhancer used by Gen-Harness prompt-enhancement skills")
    parser.add_argument("--policy-extractor-config", required=True, help="OpenAI-compatible LLM policy extractor config")
    parser.add_argument("--harness-localizer-config", required=True, help="OpenAI-compatible LLM Harness Localizer config")
    parser.add_argument("--patch-proposer-config", required=True, help="OpenAI-compatible LLM transient patch proposer config")
    parser.add_argument("--methods", default="gen_harness")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", help="comma-separated generation seeds, e.g. 0,1,2")
    parser.add_argument("--limit", type=int, help="optional first-N task limit for smoke runs")
    parser.add_argument("--fast-loop-attempts", type=int, default=5, help="serial LLM Fast Loop attempts; must be greater than K")
    parser.add_argument("--fast-loop-stochastic-k", type=int, default=4, help="K distinct internal stochastic executions required before LLM localization")
    parser.add_argument("--dry-run-budget", action="store_true", help="estimate generation calls without invoking backends")
    parser.add_argument("--continue-on-error", action="store_true", help="record failed generations and continue")


def cmd_open_source_check(args: argparse.Namespace) -> None:
    findings = scan_open_source_safety(args.root, include_ignored_local=not args.ignore_local_files)
    for finding in findings:
        location = finding.path if finding.line == 0 else f"{finding.path}:{finding.line}"
        print(f"{location}: {finding.kind}: {finding.detail}")
    if findings and args.fail_on_findings:
        raise SystemExit(1)
    print({"num_findings": len(findings), "root": args.root})


def _require_option(args: argparse.Namespace, name: str, *, dataset: str) -> str:
    value = getattr(args, name, None)
    if value is None or value == "":
        raise ValueError(f"--{name.replace('_', '-')} is required for --dataset {dataset}")
    return value


def cmd_build_tasks(args: argparse.Namespace) -> None:
    dataset = args.dataset
    split = args.split or ("val" if dataset == "t2icompbenchpp" else "heldout")
    if dataset == "geneval2":
        tasks = build_geneval2_tasks(
            _require_option(args, "benchmark_data", dataset=dataset),
            args.output,
            split=split,
            limit=args.limit,
        )
    elif dataset == "wise":
        tasks = build_wise_tasks(
            _require_option(args, "benchmark_data", dataset=dataset),
            args.output,
            split=split,
            limit=args.limit,
        )
    elif dataset == "t2icompbenchpp":
        categories = [part.strip() for part in args.categories.split(",") if part.strip()] if args.categories else None
        tasks = build_t2icompbenchpp_tasks(
            _require_option(args, "dataset_dir", dataset=dataset),
            args.output,
            split=split,
            categories=categories,
            limit_per_category=args.limit_per_category,
        )
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    print({"dataset": dataset, "output": args.output, "num_tasks": len(tasks)})


def cmd_generate(args: argparse.Namespace) -> None:
    if args.dataset == "t2icompbenchpp":
        cmd_t2icompbenchpp_generate(args)
        return
    if args.dataset in {"geneval2", "wise"}:
        args.benchmark = args.dataset
        cmd_geneval_generate(args)
        return
    raise ValueError(f"unsupported dataset: {args.dataset}")


def cmd_export_eval(args: argparse.Namespace) -> None:
    dataset = args.dataset
    if dataset == "geneval2":
        image_map = export_geneval2_image_map(
            args.experiences,
            _require_option(args, "benchmark_data", dataset=dataset),
            _require_option(args, "output", dataset=dataset),
            seed=args.seed,
            require_all=not args.allow_missing,
            prompts_file=args.prompts_file,
        )
        print({"dataset": dataset, "output": args.output, "num_mapped_prompts": len(image_map)})
    elif dataset == "wise":
        rows = export_wise_manifest(
            _require_option(args, "tasks", dataset=dataset),
            args.experiences,
            _require_option(args, "output", dataset=dataset),
            seed=args.seed,
            require_all=not args.allow_missing,
        )
        print({"dataset": dataset, "output": args.output, "num_mapped_prompts": len(rows)})
    elif dataset == "t2icompbenchpp":
        summary = prepare_t2icompbenchpp_eval_manifest(
            _require_option(args, "tasks", dataset=dataset),
            args.experiences,
            _require_option(args, "manifest_dir", dataset=dataset),
            _require_option(args, "sample_root", dataset=dataset),
            strict_complete=not args.allow_missing,
            symlink=not args.copy,
        )
        print(summary)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")


def cmd_t2icompbenchpp_generate(args: argparse.Namespace) -> None:
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    seeds = _parse_seeds(getattr(args, "seeds", None), args.seed)
    policy_extractor_config = _formal_policy_extractor_config(args)
    if getattr(args, "dry_run_budget", False):
        tasks = load_tasks(args.tasks)
        if args.limit is not None:
            tasks = tasks[: args.limit]
        gen_harness_multiplier = args.fast_loop_attempts if "gen_harness" in methods else 0
        non_harness_methods = [method for method in methods if method != "gen_harness"]
        print({
            "mode": "t2icompbenchpp_official_compatible_generation_only",
            "num_tasks": len(tasks),
            "num_seeds": len(seeds),
            "methods": methods,
            "total_generation_calls": len(tasks) * len(seeds) * (len(non_harness_methods) + gen_harness_multiplier),
            "total_internal_inspection_calls": (
                len(tasks) * len(seeds) * args.fast_loop_attempts
                if "gen_harness" in methods
                else 0
            ),
            "official_evaluator_used_for_generation_or_selection": False,
            "official_labels_used_for_prompt_compilation": False,
            "no_replacement_after_generation": True,
            "posthoc_candidate_selection_allowed": False,
            "internal_fast_loop_attempt_policy": "bounded_internal_inspection_only",
        })
        return

    backend = build_backend(args.backend_config)
    search = build_search(args.search_config, required=False) if args.search_config else None
    prompt_optimizer = build_prompt_optimizer(getattr(args, "prompt_optimizer_config", None))
    inspector = build_inspector(args.inspection_config)
    summary = T2ICompBenchPPImageGenerationExperiment(
        args.harness,
        backend,
        search,
        prompt_optimizer,
        inspector=inspector,
        fast_loop_attempts=args.fast_loop_attempts,
        fast_loop_stochastic_k=getattr(args, "fast_loop_stochastic_k", 4),
        policy_extractor_config=policy_extractor_config,
        harness_localizer_config=args.harness_localizer_config,
        patch_proposer_config=args.patch_proposer_config,
    ).run(
        args.tasks,
        args.output_dir,
        methods=methods,
        seeds=seeds,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
    )
    print(summary)


def _add_real_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend-config", required=True, help="real backend adapter config")
    parser.add_argument("--inspection-config", required=True, help="real inspection adapter config")
    parser.add_argument("--search-config", help="real search/evidence adapter config, required only for contracts that require evidence")
    parser.add_argument("--prompt-optimizer-config", help="optional real prompt enhancement adapter config used by Gen-Harness, never by direct-prompt baseline")
    parser.add_argument("--policy-extractor-config", required=True, help="OpenAI-compatible LLM policy extractor config")
    parser.add_argument("--validation-metric-mode", choices=["pass_rate", "image_metrics"], default="pass_rate", help="metric used to accept/reject evolver patches")
    parser.add_argument("--fast-loop-attempts", type=int, default=5, help="serial LLM Fast Loop attempts; must be greater than K")
    parser.add_argument("--fast-loop-stochastic-k", type=int, default=4, help="K distinct internal stochastic executions required before LLM localization")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gen-Harness CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="execute harness and write Visual Experience JSONL")
    run.add_argument("--harness", required=True)
    run.add_argument("--tasks", required=True)
    run.add_argument("--output", required=True)
    _add_real_runtime_args(run)
    run.add_argument("--harness-localizer-config", required=True, help="OpenAI-compatible LLM Harness Localizer config")
    run.add_argument("--patch-proposer-config", required=True, help="OpenAI-compatible LLM patch proposer config")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--seeds", help="comma-separated generation seeds, e.g. 0,1,2")
    run.add_argument("--write-memory", action="store_true")
    run.set_defaults(func=cmd_run)

    mine = sub.add_parser("mine", help="mine repeated component weaknesses")
    mine.add_argument("--experiences", required=True)
    mine.add_argument("--output", required=True)
    mine.add_argument("--harness")
    mine.add_argument("--min-support", type=int, default=2)
    mine.add_argument("--split", default="target")
    mine.add_argument("--write-memory", action="store_true")
    mine.set_defaults(func=cmd_mine)

    patch = sub.add_parser("patch", help="propose component-scoped patch manifests")
    patch.add_argument("--weaknesses", required=True)
    patch.add_argument("--output", required=True)
    patch.add_argument("--harness", help="optional harness root so patch proposal can use memory")
    patch.add_argument("--patch-proposer-config", help="OpenAI-compatible LLM patch proposer config; no rule-based fallback is available")
    patch.set_defaults(func=cmd_patch)

    val = sub.add_parser("validate", help="validate patch manifests")
    val.add_argument("--harness", required=True)
    val.add_argument("--tasks", required=True)
    val.add_argument("--patches", required=True)
    val.add_argument("--output", required=True)
    _add_real_runtime_args(val)
    val.add_argument("--seed", type=int, default=0)
    val.add_argument("--seeds", help="comma-separated validation seeds, e.g. 0,1,2")
    val.add_argument("--regression-epsilon", type=float, default=0.01)
    val.add_argument("--promote", action="store_true")
    val.set_defaults(func=cmd_validate)

    build_ds = sub.add_parser("build-dataset", help="validate and split a real task manifest")
    build_ds.add_argument("--input-manifest", required=True)
    build_ds.add_argument("--output", required=True)
    build_ds.add_argument("--seed", type=int, default=0)
    build_ds.add_argument("--target-ratio", type=float, default=0.5)
    build_ds.add_argument("--heldout-ratio", type=float, default=0.25)
    build_ds.add_argument("--preservation-ratio", type=float, default=0.25)
    build_ds.set_defaults(func=cmd_build_dataset)

    compositional_suite = sub.add_parser("build-compositional-suite", help="build a non-GenEval compositional dev/heldout/preservation suite")
    compositional_suite.add_argument("--output", required=True)
    compositional_suite.add_argument("--seed", type=int, default=0)
    compositional_suite.add_argument("--target-per-family", type=int, default=8)
    compositional_suite.add_argument("--heldout-per-family", type=int, default=8)
    compositional_suite.add_argument("--preservation-per-family", type=int, default=4)
    compositional_suite.add_argument("--include-action", action="store_true", help="include non-benchmark action/verb tasks")
    compositional_suite.set_defaults(func=cmd_build_compositional_suite)

    evolution_split = sub.add_parser("build-evolution-split", help="build the PartiPrompts P2 500/100/100 self-evolution split")
    evolution_split.add_argument("--source", default=DEFAULT_PARTIPROMPTS_URL, help="PartiPrompts TSV/JSON/JSONL path or URL")
    evolution_split.add_argument("--output-dir", required=True)
    evolution_split.add_argument("--benchmark-tasks", action="append", default=[], help="prompt-only benchmark task JSONL to exclude by normalized prompt overlap")
    evolution_split.add_argument("--seed", type=int, default=2027)
    evolution_split.add_argument("--target-count", type=int, default=500)
    evolution_split.add_argument("--heldout-count", type=int, default=100)
    evolution_split.add_argument("--preservation-count", type=int, default=100)
    evolution_split.set_defaults(func=cmd_build_evolution_split)

    faults = sub.add_parser("inject-faults", help="generate controlled component fault harnesses")
    faults.add_argument("--harness", required=True)
    faults.add_argument("--output-dir", required=True)
    faults.add_argument("--include-compound", action="store_true", help="also build all ten unordered pairs of the five responsibility faults")
    faults.set_defaults(func=cmd_inject_faults)

    slow = sub.add_parser("slow-loop-evolve", help="run clean slow-loop harness evolution: explore, mine, patch, validate, and optionally promote")
    slow.add_argument("--harness", required=True)
    slow.add_argument("--tasks", required=True, help="non-benchmark task JSONL with target/heldout/preservation splits")
    slow.add_argument("--output-dir", required=True)
    _add_real_runtime_args(slow)
    slow.add_argument("--seed", type=int, default=0)
    slow.add_argument("--seeds", help="comma-separated exploration/validation seeds, e.g. 0,1,2")
    slow.add_argument("--min-support", type=int, default=2)
    slow.add_argument("--regression-epsilon", type=float, default=0.01)
    slow.add_argument("--harness-localizer-config", required=True, help="OpenAI-compatible LLM harness localizer config")
    slow.add_argument("--patch-proposer-config", required=True, help="OpenAI-compatible LLM patch proposer config")
    slow.add_argument("--trace-frontier-size", type=int, choices=range(1, 6), default=3, help="TRACE Top-r responsibility frontier size; 1=Hard Top-1, 5=full-space")
    slow.add_argument("--trace-candidate-budget", type=int, default=3, help="maximum LLM edit proposals per localized responsibility")
    slow.add_argument("--trace-preservation-weight", type=float, default=0.5, help="TRACE lambda preservation-regression penalty")
    slow.add_argument("--trace-execution-cost-weight", type=float, default=0.05, help="TRACE mu additional-execution-cost penalty")
    slow.add_argument("--trace-dataset-protocol", choices=["development", "paper_p2"], default="development", help="TRACE task split contract; paper_p2 enforces the published 500/100/100 split")
    slow.add_argument("--paper-trace", action="store_true", help="enforce the full paper protocol, including K=4 and pinned GPT-4.1 configs")
    slow.add_argument("--allow-trace-seed-ablation", action="store_true", help="explicitly allow non-K=4 seeds for sensitivity experiments")
    slow.add_argument("--promote", action="store_true", help="apply accepted patches to the harness after target/heldout/preservation validation")
    slow.add_argument("--no-write-memory", action="store_true", help="write run artifacts only; do not append experiences, weaknesses, or patch results to harness memory")
    slow.add_argument("--allow-benchmark-tasks", action="store_true", help="diagnostic-only escape hatch; also requires GENHARNESS_DIAGNOSTIC_ALLOW_BENCHMARK_EVOLVE=1")
    slow.add_argument("--dry-run-budget", action="store_true", help="estimate real backend and inspection calls without running evolution")
    slow.set_defaults(func=cmd_slow_loop_evolve)

    self_manifest = sub.add_parser("self-evolve-manifest", help="build an auditable self-evolve manifest from slow-loop or iterative self-evolve artifacts")
    self_manifest.add_argument("--output-dir", required=True, help="slow-loop output directory or self-evolve-run output directory")
    self_manifest.add_argument("--output", required=True, help="self-evolve manifest JSON path")
    self_manifest.set_defaults(func=cmd_self_evolve_manifest)

    self_run = sub.add_parser("self-evolve-run", help="run iterative leakage-safe Gen-Harness self-evolution on non-benchmark dev tasks")
    self_run.add_argument("--harness", required=True)
    self_run.add_argument("--tasks", required=True, help="non-benchmark task JSONL with target/heldout/preservation splits")
    self_run.add_argument("--output-dir", required=True)
    _add_real_runtime_args(self_run)
    self_run.add_argument("--seed", type=int, default=0)
    self_run.add_argument("--seeds", help="comma-separated exploration/validation seeds, e.g. 0,1,2")
    self_run.add_argument("--max-rounds", type=int, default=3)
    self_run.add_argument("--min-support", type=int, default=2)
    self_run.add_argument("--regression-epsilon", type=float, default=0.01)
    self_run.add_argument("--harness-localizer-config", required=True, help="OpenAI-compatible LLM harness localizer config")
    self_run.add_argument("--patch-proposer-config", required=True, help="OpenAI-compatible LLM patch proposer config")
    self_run.add_argument("--trace-frontier-size", type=int, choices=range(1, 6), default=3, help="TRACE Top-r responsibility frontier size; 1=Hard Top-1, 5=full-space")
    self_run.add_argument("--trace-candidate-budget", type=int, default=3, help="maximum LLM edit proposals per localized responsibility")
    self_run.add_argument("--trace-preservation-weight", type=float, default=0.5, help="TRACE lambda preservation-regression penalty")
    self_run.add_argument("--trace-execution-cost-weight", type=float, default=0.05, help="TRACE mu additional-execution-cost penalty")
    self_run.add_argument("--trace-dataset-protocol", choices=["development", "paper_p2"], default="development", help="TRACE task split contract; paper_p2 enforces the published 500/100/100 split")
    self_run.add_argument("--paper-trace", action="store_true", help="enforce the full paper protocol, including K=4 and pinned GPT-4.1 configs")
    self_run.add_argument("--allow-trace-seed-ablation", action="store_true", help="explicitly allow non-K=4 seeds for sensitivity experiments")
    self_run.add_argument("--promote-accepted", dest="promote_accepted", action="store_true", help="apply accepted patches after each target/heldout/preservation gate")
    self_run.add_argument("--no-promote-accepted", dest="promote_accepted", action="store_false", help="validate accepted patches without writing them back to the harness")
    self_run.add_argument("--no-write-memory", action="store_true", help="write run artifacts only; do not append experiences, weaknesses, or patch results to harness memory")
    self_run.add_argument("--dry-run-budget", action="store_true", help="estimate self-evolve calls and audits without loading generation or verifier backends")
    # One baseline attempt followed by one inspection-driven repair.
    # The slow loop owns patch proposals and promotion.
    self_run.set_defaults(promote_accepted=True, fast_loop_attempts=5)
    self_run.set_defaults(func=cmd_self_evolve_run)

    audit = sub.add_parser("audit-components", help="audit Gen-Harness component boundaries, leakage guards, and fallback absence")
    audit.add_argument("--harness", required=True)
    audit.add_argument("--tasks", help="optional prompt-only task JSONL to audit for benchmark label leakage")
    audit.add_argument("--output", help="optional JSON report path")
    audit.set_defaults(func=cmd_audit_components)

    run_audit = sub.add_parser("audit-run-artifacts", help="audit run traces/configs for benchmark label leakage, official evaluator usage, and fallback image substitution")
    run_audit.add_argument("--tasks", help="optional prompt-only task JSONL to audit")
    run_audit.add_argument("--experiences", help="optional Gen-Harness experiences JSONL to audit")
    run_audit.add_argument("--inspection-config", help="optional internal inspection config to audit")
    run_audit.add_argument("--backend-config", help="optional backend config to audit")
    run_audit.add_argument("--output", help="optional JSON report path")
    run_audit.set_defaults(func=cmd_audit_run_artifacts)

    readiness = sub.add_parser("audit-goal-readiness", help="audit whether a self-evolve -> smoke -> full GenEval2 run satisfies the target goal")
    readiness.add_argument("--run-root", required=True, help="run root containing self_evolve_dev, geneval2_smoke32, and geneval2_full800")
    readiness.add_argument("--smoke-min-am", type=float, default=88.0)
    readiness.add_argument("--smoke-min-gm", type=float, default=67.0)
    readiness.add_argument("--full-target-gm", type=float, default=72.0)
    readiness.add_argument("--output", help="optional JSON report path")
    readiness.set_defaults(func=cmd_audit_goal_readiness)

    suite_eval = sub.add_parser("evaluate-suite", help="run generic task suite with internal Gen-Harness inspection")
    suite_eval.add_argument("--harness", required=True)
    suite_eval.add_argument("--tasks", required=True)
    suite_eval.add_argument("--output-dir", required=True)
    _add_real_runtime_args(suite_eval)
    suite_eval.add_argument("--harness-localizer-config", required=True, help="OpenAI-compatible LLM Harness Localizer config")
    suite_eval.add_argument("--patch-proposer-config", required=True, help="OpenAI-compatible LLM patch proposer config")
    suite_eval.add_argument("--seed", type=int, default=0)
    suite_eval.add_argument("--seeds", help="comma-separated generation seeds, e.g. 0,1,2")
    suite_eval.add_argument("--limit", type=int, help="optional first-N task limit for smoke runs")
    suite_eval.add_argument("--dry-run-budget", action="store_true", help="estimate generation and internal inspection calls without invoking backends")
    suite_eval.add_argument("--continue-on-error", action="store_true")
    suite_eval.set_defaults(func=cmd_evaluate_suite)


    quality = sub.add_parser("quality-exp", help="evaluate Gen-Harness with internal image quality metrics")
    quality.add_argument("--harness", required=True)
    quality.add_argument("--tasks", required=True)
    quality.add_argument("--output-dir", required=True)
    quality.add_argument("--backend-config", required=True, help="real backend adapter config")
    quality.add_argument("--inspection-config", required=True, help="real inspection adapter config")
    quality.add_argument("--search-config", help="real search/evidence adapter config, required only for contracts that require evidence")
    quality.add_argument("--prompt-optimizer-config", help="optional real prompt enhancement adapter config used by Gen-Harness, never by direct-prompt baseline")
    quality.add_argument("--policy-extractor-config", required=True, help="OpenAI-compatible LLM policy extractor config")
    quality.add_argument("--harness-localizer-config", required=True, help="OpenAI-compatible LLM Harness Localizer config")
    quality.add_argument("--patch-proposer-config", required=True, help="OpenAI-compatible LLM patch proposer config")
    quality.add_argument("--seed", type=int, default=0)
    quality.add_argument("--seeds", help="comma-separated generation/evaluation seeds, e.g. 0,1,2")
    quality.add_argument("--fast-loop-attempts", type=int, default=5, help="serial LLM Fast Loop attempts; must be greater than K")
    quality.add_argument("--fast-loop-stochastic-k", type=int, default=4)
    quality.set_defaults(func=cmd_quality_exp)

    build_tasks = sub.add_parser("build-tasks", help="convert a benchmark split into prompt-only Gen-Harness tasks")
    build_tasks.add_argument("--dataset", choices=("geneval2", "t2icompbenchpp", "wise"), required=True)
    build_tasks.add_argument("--benchmark-data", help="input JSON/JSONL for GenEval2 or WISE")
    build_tasks.add_argument("--dataset-dir", help="input dataset directory for T2I-CompBench++")
    build_tasks.add_argument("--output", required=True, help="output prompt-only task JSONL")
    build_tasks.add_argument("--split", help="benchmark split; defaults to heldout for GenEval2/WISE and val for T2I-CompBench++")
    build_tasks.add_argument("--limit", type=int, help="optional smoke-test prompt limit")
    build_tasks.add_argument("--categories", help="T2I-CompBench++ comma-separated category subset")
    build_tasks.add_argument("--limit-per-category", type=int, help="T2I-CompBench++ optional first-N limit per category")
    build_tasks.set_defaults(func=cmd_build_tasks)

    generate = sub.add_parser("generate", help="generate benchmark images through full EVOGEN-HARNESS")
    generate.add_argument("--dataset", choices=("geneval2", "t2icompbenchpp", "wise"), required=True)
    _add_prompt_benchmark_generation_args(generate)
    generate.set_defaults(func=cmd_generate)

    export_eval = sub.add_parser("export-eval", help="export frozen generations for the selected official evaluator")
    export_eval.add_argument("--dataset", choices=("geneval2", "t2icompbenchpp", "wise"), required=True)
    export_eval.add_argument("--tasks", help="prompt-only task JSONL, required for T2I-CompBench++ and WISE")
    export_eval.add_argument("--experiences", required=True, help="Gen-Harness generation JSONL")
    export_eval.add_argument("--benchmark-data", help="official benchmark JSON/JSONL, required for GenEval2")
    export_eval.add_argument("--output", help="output official evaluator input for GenEval2 or WISE")
    export_eval.add_argument("--manifest-dir", help="T2I-CompBench++ output evaluator manifest directory")
    export_eval.add_argument("--sample-root", help="T2I-CompBench++ output sample root")
    export_eval.add_argument("--seed", type=int, help="optional seed to export when multiple seeds are present")
    export_eval.add_argument("--prompts-file", help="optional GenEval2 prompt-only subset JSONL")
    export_eval.add_argument("--allow-missing", action="store_true", help="write a partial export for smoke tests")
    export_eval.add_argument("--copy", action="store_true", help="copy samples instead of symlinking when possible")
    export_eval.set_defaults(func=cmd_export_eval)

    safety = sub.add_parser("open-source-check", help="scan source tree for local credentials, hard-coded secrets, and oversized inline artifacts")
    safety.add_argument("--root", default=".")
    safety.add_argument("--ignore-local-files", action="store_true", help="do not report ignored local files such as .genharness.local.env")
    safety.add_argument("--fail-on-findings", action="store_true")
    safety.set_defaults(func=cmd_open_source_check)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
