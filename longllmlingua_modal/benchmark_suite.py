from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console

from latent_rollback_modal.benchmark_datasets import load_benchmark
from latent_rollback_modal.benchmark_repobench import load_repobench

from .benchmark_longbench import (
    QA_RATIO_PRESETS,
    print_summary as print_longbench_summary,
    run_model as run_longbench_model,
    save_results as save_longbench_results,
)
from .benchmark_repobench import (
    CODE_RATIO_PRESETS,
    print_summary as print_repobench_summary,
    run_model as run_repobench_model,
    save_results as save_repobench_results,
)
from .config import (
    DEFAULT_COMPRESSOR_DEVICE,
    DEFAULT_COMPRESSOR_MODEL,
    MODEL_MATRIX,
    results_path,
)

console = Console()
SUITE_RESULTS_DIR = results_path("longllmlingua_suite_results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LongLLMLingua LongBench and/or RepoBench sequentially across models"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["longbench", "repobench"],
        choices=["longbench", "repobench"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_MATRIX.keys()),
        choices=list(MODEL_MATRIX.keys()),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compressor-model", default=DEFAULT_COMPRESSOR_MODEL)
    parser.add_argument("--compressor-device", default=DEFAULT_COMPRESSOR_DEVICE)
    parser.add_argument("--run-prefix", default=None)

    parser.add_argument(
        "--longbench-tasks",
        nargs="+",
        default=["hotpotqa", "2wikimqa"],
        choices=["hotpotqa", "2wikimqa"],
    )
    parser.add_argument(
        "--longbench-ratios",
        nargs="+",
        default=list(QA_RATIO_PRESETS.keys()),
        choices=list(QA_RATIO_PRESETS.keys()),
    )
    parser.add_argument(
        "--longbench-total",
        type=int,
        default=200,
        help="Total LongBench examples across all selected tasks",
    )
    parser.add_argument(
        "--longbench-max-tokens",
        type=int,
        default=80,
        dest="longbench_max_tokens",
    )

    parser.add_argument("--repobench-n", type=int, default=200)
    parser.add_argument("--repobench-split", default="cross_file_first")
    parser.add_argument(
        "--repobench-level",
        default="all",
        help='RepoBench level bucket (e.g. "2k") or "all" for no filter',
    )
    parser.add_argument(
        "--repobench-ratios",
        nargs="+",
        default=list(CODE_RATIO_PRESETS.keys()),
        choices=list(CODE_RATIO_PRESETS.keys()),
    )
    parser.add_argument(
        "--repobench-max-tokens",
        type=int,
        default=60,
        dest="repobench_max_tokens",
    )
    return parser


def _longbench_n_per_task(total: int, tasks: list[str]) -> int:
    if total <= 0:
        raise ValueError("longbench total must be positive")
    if not tasks:
        raise ValueError("at least one LongBench task is required")
    return max(1, total // len(tasks))


def _write_suite_manifest(run_prefix: str, payload: dict, suffix: str) -> Path:
    SUITE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SUITE_RESULTS_DIR / f"{run_prefix}_{suffix}.manifest.json"
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def main() -> int:
    args = build_parser().parse_args()
    run_prefix = args.run_prefix or f"longllmlingua_suite_{int(time.time())}"

    console.rule("[bold magenta]LongLLMLingua Suite[/bold magenta]")
    console.print(f"  Benchmarks : {args.benchmarks}")
    console.print(f"  Models     : {args.models}")
    console.print(f"  Seed       : {args.seed}")
    console.print(f"  Run Prefix : {run_prefix}")

    if "longbench" in args.benchmarks:
        n_per_task = _longbench_n_per_task(args.longbench_total, args.longbench_tasks)
        console.rule("[bold]Preparing LongBench example set[/bold]")
        console.print(f"  Tasks      : {args.longbench_tasks}")
        console.print(f"  Total goal : {args.longbench_total}")
        console.print(f"  N/task     : {n_per_task}")
        longbench_examples = load_benchmark(
            tasks=tuple(args.longbench_tasks),
            n_per_task=n_per_task,
            seed=args.seed,
        )
        longbench_records = []
        longbench_runs = []
        for model_key in args.models:
            try:
                model_records = run_longbench_model(
                    model_key=model_key,
                    model_cfg=MODEL_MATRIX[model_key],
                    examples=longbench_examples,
                    ratio_names=args.longbench_ratios,
                    compressor_model=args.compressor_model,
                    compressor_device=args.compressor_device,
                    max_new_tokens=args.longbench_max_tokens,
                )
                longbench_records.extend(model_records)
                longbench_runs.append(
                    {
                        "model_key": model_key,
                        "status": "ok",
                        "record_count": len(model_records),
                    }
                )
            except Exception as exc:
                console.print(f"[red]LongBench run failed for {model_key}: {exc}[/red]")
                longbench_runs.append(
                    {
                        "model_key": model_key,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            save_longbench_results(longbench_records, f"{run_prefix}_longbench_partial")
        print_longbench_summary(longbench_records)
        save_longbench_results(longbench_records, f"{run_prefix}_longbench")
        _write_suite_manifest(
            run_prefix,
            {
                "benchmark": "longbench",
                "tasks": args.longbench_tasks,
                "total_goal": args.longbench_total,
                "n_per_task": n_per_task,
                "models": args.models,
                "runs": longbench_runs,
                "example_ids": [example.id for example in longbench_examples],
            },
            "longbench_suite",
        )

    if "repobench" in args.benchmarks:
        console.rule("[bold]Preparing RepoBench example set[/bold]")
        console.print(f"  Split : {args.repobench_split}")
        console.print(f"  Level : {args.repobench_level}")
        console.print(f"  N     : {args.repobench_n}")
        repobench_examples = load_repobench(
            split=args.repobench_split,
            level=None if args.repobench_level == "all" else args.repobench_level,
            n=args.repobench_n,
            seed=args.seed,
        )
        repobench_records = []
        repobench_runs = []
        for model_key in args.models:
            try:
                model_records = run_repobench_model(
                    model_key=model_key,
                    model_cfg=MODEL_MATRIX[model_key],
                    examples=repobench_examples,
                    ratio_names=args.repobench_ratios,
                    compressor_model=args.compressor_model,
                    compressor_device=args.compressor_device,
                    max_new_tokens=args.repobench_max_tokens,
                )
                repobench_records.extend(model_records)
                repobench_runs.append(
                    {
                        "model_key": model_key,
                        "status": "ok",
                        "record_count": len(model_records),
                    }
                )
            except Exception as exc:
                console.print(f"[red]RepoBench run failed for {model_key}: {exc}[/red]")
                repobench_runs.append(
                    {
                        "model_key": model_key,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            save_repobench_results(repobench_records, f"{run_prefix}_repobench_partial")
        print_repobench_summary(repobench_records)
        save_repobench_results(repobench_records, f"{run_prefix}_repobench")
        _write_suite_manifest(
            run_prefix,
            {
                "benchmark": "repobench",
                "split": args.repobench_split,
                "level": args.repobench_level,
                "n": args.repobench_n,
                "models": args.models,
                "runs": repobench_runs,
                "example_ids": [example.id for example in repobench_examples],
            },
            "repobench_suite",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
