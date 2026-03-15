"""
Cross-Model Latent Rollback Benchmark Runner

Tests whether residual state injection (Latent Rollback) reduces input token
consumption while preserving answer quality across multiple model architectures.

Model matrix (all 4-bit quantized, sequential loading to fit 24GB M4 Pro):
  llama3-8b       mlx-community/Meta-Llama-3-8B-Instruct-4bit       ~4.5 GB
  qwen25-7b       mlx-community/Qwen2.5-7B-Instruct-4bit            ~4.5 GB
  mistral-24b     mlx-community/Mistral-Small-24B-Instruct-2501-4bit ~13  GB
  deepseek-14b    mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit   ~8   GB

  Optional (uncomment deepseek-32b in MODEL_MATRIX):
    mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit  ~16 GB
    WARNING: may OOM on 24GB — close all other apps first.

  NOTE: mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit does NOT
  exist on HuggingFace. Replaced with Qwen2.5-7B-Instruct-4bit.

Tasks (from LongBench — THUDM/LongBench):
  hotpotqa   — Multi-document QA, answer requires combining 2 documents
  2wikimqa   — Two-hop Wikipedia reasoning

Metrics per (model, task, example):
  input_token_reduction : fraction of input tokens saved (higher = better)
  total_cost_ratio      : (injection in+out) / (baseline in+out)
  baseline_f1 / injection_f1 : QA accuracy

Usage:
  source .venv/bin/activate
  python benchmark_runner.py
  python benchmark_runner.py --models llama3-8b qwen25-7b
  python benchmark_runner.py --tasks hotpotqa --n 5
  python benchmark_runner.py --sweep-layer          # calibrate f(M) per model
  python benchmark_runner.py --sweep-scale          # find optimal injection scale
  python benchmark_runner.py --layer 14             # override injection layer directly
  python benchmark_runner.py --condition injection  # skip baseline pass
  python benchmark_runner.py --resume benchmark_1773255147  # skip already-done examples
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from rich.console import Console
from rich import box
from rich.table import Table

from .backend_torch import load_model, MLXModelWrapper, clear_backend_cache
from .benchmark_datasets import load_benchmark, BenchmarkExample, grade_qa
from .config import MODEL_MATRIX, results_path
from .context_injector import (
    extract_context_state,
    generate_baseline_qa,
    generate_with_context_injection,
    compute_token_metrics,
    sweep_injection_scale,
)
from .layer_selector import (
    select_layer_heuristic,
    select_layer_sweep,
    print_sweep_table,
)

console = Console()
RESULTS_DIR = results_path("benchmark_results")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRecord:
    model_key: str
    model_hf_id: str
    task: str
    example_id: str
    injection_layer: int
    injection_scale: float
    pool_strategy: str
    n_context_words: int

    baseline_input_tokens: int
    injection_input_tokens: int
    baseline_output_tokens: int
    injection_output_tokens: int
    input_token_reduction: float
    total_cost_ratio: float

    baseline_exact_match: bool
    injection_exact_match: bool
    baseline_f1: float
    injection_f1: float

    baseline_answer: str
    injection_answer: str
    gold_answers: str

    elapsed_baseline_s: float
    elapsed_injection_s: float


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

def run_example(
    wrapper: MLXModelWrapper,
    example: BenchmarkExample,
    model_key: str,
    model_hf_id: str,
    injection_layer: int,
    injection_scale: float = 1.0,
    pool: str = "mean",
    max_new_tokens: int = 80,
    condition: str = "both",
    verbose: bool = False,
) -> BenchmarkRecord:
    full_prompt = example.full_prompt()
    question_prompt = example.question_prompt()

    # Baseline: context + question in prompt
    if condition in ("both", "baseline"):
        t0 = time.time()
        baseline_text, n_baseline_input = generate_baseline_qa(
            wrapper, full_prompt, max_new_tokens
        )
        baseline_elapsed = time.time() - t0
        n_baseline_output = len(wrapper.encode(baseline_text))
    else:
        baseline_text, n_baseline_input, n_baseline_output, baseline_elapsed = "", 0, 0, 0.0

    # Injection: extract context state, run question-only with injection
    if condition in ("both", "injection"):
        t0 = time.time()
        context_vector, _ = extract_context_state(
            wrapper, example.context, injection_layer, pool=pool
        )
        injection_text, n_injection_input = generate_with_context_injection(
            wrapper, question_prompt, context_vector,
            layer=injection_layer, scale=injection_scale,
            max_new_tokens=max_new_tokens,
        )
        injection_elapsed = time.time() - t0
        n_injection_output = len(wrapper.encode(injection_text))
    else:
        injection_text, n_injection_input, n_injection_output, injection_elapsed = "", 0, 0, 0.0

    baseline_grades = grade_qa(baseline_text, example.gold_answers)
    injection_grades = grade_qa(injection_text, example.gold_answers)
    token_metrics = compute_token_metrics(
        n_baseline_input, n_injection_input,
        n_baseline_output, n_injection_output,
    )

    if verbose:
        console.print(f"    baseline  ({n_baseline_input} in): {baseline_text.strip()[:80]}")
        console.print(f"    injection ({n_injection_input} in): {injection_text.strip()[:80]}")
        console.print(f"    gold: {example.gold_answers}")
        console.print(
            f"    BL F1={baseline_grades['f1']:.3f} EM={baseline_grades['exact_match']}  "
            f"INJ F1={injection_grades['f1']:.3f} EM={injection_grades['exact_match']}  "
            f"reduction={token_metrics['input_token_reduction']:.1%}"
        )

    return BenchmarkRecord(
        model_key=model_key,
        model_hf_id=model_hf_id,
        task=example.task,
        example_id=example.id,
        injection_layer=injection_layer,
        injection_scale=injection_scale,
        pool_strategy=pool,
        n_context_words=example.context_word_len,
        baseline_input_tokens=n_baseline_input,
        injection_input_tokens=n_injection_input,
        baseline_output_tokens=n_baseline_output,
        injection_output_tokens=n_injection_output,
        input_token_reduction=token_metrics["input_token_reduction"],
        total_cost_ratio=token_metrics["total_cost_ratio"],
        baseline_exact_match=baseline_grades["exact_match"],
        injection_exact_match=injection_grades["exact_match"],
        baseline_f1=baseline_grades["f1"],
        injection_f1=injection_grades["f1"],
        baseline_answer=baseline_text.strip()[:120],
        injection_answer=injection_text.strip()[:120],
        gold_answers=str(example.gold_answers),
        elapsed_baseline_s=round(baseline_elapsed, 2),
        elapsed_injection_s=round(injection_elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    model_cfg: dict,
    examples: list[BenchmarkExample],
    injection_scale: float = 1.0,
    pool: str = "mean",
    max_new_tokens: int = 80,
    sweep_layer: bool = False,
    sweep_scale: bool = False,
    layer_override: int | None = None,
    condition: str = "both",
    done_ids: set[str] | None = None,
    verbose: bool = False,
) -> list[BenchmarkRecord]:
    hf_id = model_cfg["hf_id"]
    console.rule(f"[bold cyan]Model: {model_key}[/bold cyan]")
    console.print(f"  HF ID : {hf_id}")
    console.print(f"  Size  : ~{model_cfg['size_gb']} GB")
    console.print(f"  Note  : {model_cfg['note']}")

    try:
        wrapper = load_model(hf_id)
    except Exception as exc:
        console.print(f"  [red]Failed to load {hf_id}: {exc}[/red]")
        return []

    # Select injection layer via f(M)
    if layer_override is not None:
        injection_layer = layer_override
        console.print(f"  Injection layer: {injection_layer} (override)")
    elif sweep_layer and examples:
        ex = examples[0]
        best_layer, sweep_results = select_layer_sweep(
            wrapper, ex.context[:2000], ex.question
        )
        print_sweep_table(sweep_results, best_layer)
        injection_layer = best_layer
    else:
        injection_layer = select_layer_heuristic(wrapper, hf_id)

    console.print(f"  Injection layer: {injection_layer} / {wrapper.n_layers}")

    # Optional scale sweep on first example
    if sweep_scale and examples:
        ex = examples[0]
        ctx_v, _ = extract_context_state(wrapper, ex.context, injection_layer)
        scale_results = sweep_injection_scale(
            wrapper, ex.question_prompt(), ctx_v, injection_layer,
            ex.gold_answers, max_new_tokens=50
        )
        console.print("  Scale sweep:")
        for r in scale_results:
            mark = "[green]EM[/green]" if r["exact_match"] else f"F1={r['f1']:.2f}"
            console.print(f"    scale={r['scale']:.2f} {mark} '{r['text'][:60]}'")

    # Run all examples
    records: list[BenchmarkRecord] = []
    skipped = 0
    for i, ex in enumerate(examples):
        run_key = f"{model_key}:{ex.id}"
        if done_ids and run_key in done_ids:
            skipped += 1
            continue
        console.print(
            f"\n  [{i+1}/{len(examples)}] {ex.task}/{ex.id[:24]}  "
            f"({ex.context_word_len}w ctx)"
        )
        try:
            rec = run_example(
                wrapper, ex,
                model_key=model_key,
                model_hf_id=hf_id,
                injection_layer=injection_layer,
                injection_scale=injection_scale,
                pool=pool,
                max_new_tokens=max_new_tokens,
                condition=condition,
                verbose=verbose,
            )
            records.append(rec)
        except Exception as exc:
            console.print(f"  [red]ERROR: {exc}[/red]")
    if skipped:
        console.print(f"  [dim]Skipped {skipped} already-done examples[/dim]")

    _print_model_summary(model_key, records)

    del wrapper
    gc.collect()
    clear_backend_cache()

    return records


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _print_model_summary(model_key: str, records: list[BenchmarkRecord]) -> None:
    if not records:
        return
    n = len(records)
    avg_red = sum(r.input_token_reduction for r in records) / n
    avg_cost = sum(r.total_cost_ratio for r in records) / n
    b_em = sum(1 for r in records if r.baseline_exact_match) / n
    i_em = sum(1 for r in records if r.injection_exact_match) / n
    b_f1 = sum(r.baseline_f1 for r in records) / n
    i_f1 = sum(r.injection_f1 for r in records) / n
    f1_delta = i_f1 - b_f1

    console.rule(f"[bold]{model_key} summary[/bold]")
    console.print(f"  Examples          : {n}")
    console.print(f"  Avg input reduction: {avg_red:.1%}")
    console.print(f"  Avg total cost ratio: {avg_cost:.3f}")
    color = "green" if f1_delta >= -0.05 else "yellow" if f1_delta >= -0.15 else "red"
    console.print(f"  Baseline EM/F1    : {b_em:.1%} / {b_f1:.3f}")
    console.print(f"  Injection EM/F1   : {i_em:.1%} / {i_f1:.3f}")
    console.print(f"  F1 delta          : [{color}]{f1_delta:+.3f}[/{color}]")


def print_benchmark_summary(all_records: list[BenchmarkRecord]) -> None:
    from collections import defaultdict
    console.rule("[bold magenta]CROSS-MODEL BENCHMARK SUMMARY[/bold magenta]")

    by_model: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for r in all_records:
        by_model[r.model_key].append(r)

    table = Table(
        title="Latent Rollback: Token Efficiency vs QA Accuracy",
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    table.add_column("Model", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Layer", justify="right")
    table.add_column("Input Reduction", justify="right")
    table.add_column("Cost Ratio", justify="right")
    table.add_column("BL-EM", justify="right")
    table.add_column("INJ-EM", justify="right")
    table.add_column("BL-F1", justify="right")
    table.add_column("INJ-F1", justify="right")
    table.add_column("F1 Delta", justify="right")

    for model_key, recs in by_model.items():
        n = len(recs)
        avg_red = sum(r.input_token_reduction for r in recs) / n
        avg_cost = sum(r.total_cost_ratio for r in recs) / n
        b_em = sum(1 for r in recs if r.baseline_exact_match) / n
        i_em = sum(1 for r in recs if r.injection_exact_match) / n
        b_f1 = sum(r.baseline_f1 for r in recs) / n
        i_f1 = sum(r.injection_f1 for r in recs) / n
        f1_delta = i_f1 - b_f1
        layer = recs[0].injection_layer if recs else "-"
        color = "green" if f1_delta >= -0.05 else "yellow" if f1_delta >= -0.15 else "red"
        table.add_row(
            model_key, str(n), str(layer),
            f"{avg_red:.1%}", f"{avg_cost:.3f}",
            f"{b_em:.1%}", f"{i_em:.1%}",
            f"{b_f1:.3f}", f"{i_f1:.3f}",
            f"[{color}]{f1_delta:+.3f}[/{color}]",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(records: list[BenchmarkRecord], run_id: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"{run_id}.csv"
    json_path = RESULTS_DIR / f"{run_id}.json"

    if records:
        fields = list(asdict(records[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(r) for r in records)

    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    console.print(f"\n[dim]Saved:[/dim]\n  {csv_path}\n  {json_path}")
    return csv_path, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-Model Latent Rollback Token Efficiency Benchmark"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_MATRIX.keys()),
        choices=list(MODEL_MATRIX.keys()),
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["hotpotqa", "2wikimqa"],
        choices=["hotpotqa", "2wikimqa"],
    )
    parser.add_argument("--n", type=int, default=10, help="Examples per task")
    parser.add_argument("--scale", type=float, default=1.0, help="Injection scale")
    parser.add_argument(
        "--pool", choices=["mean", "last", "cls"], default="mean"
    )
    parser.add_argument("--max-tokens", type=int, default=80, dest="max_tokens")
    parser.add_argument("--sweep-layer", action="store_true", dest="sweep_layer")
    parser.add_argument("--sweep-scale", action="store_true", dest="sweep_scale")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--run-id", default=None, dest="run_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=None,
                        help="Override injection layer for all models (skips heuristic/sweep)")
    parser.add_argument(
        "--condition", choices=["both", "baseline", "injection"], default="both",
        help="Run only baseline pass, only injection pass, or both (default: both)",
    )
    parser.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help="Resume a previous run: load its CSV and skip already-completed examples",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"benchmark_{int(time.time())}"

    # Load already-done example keys from a prior run
    done_ids: set[str] = set()
    if args.resume:
        resume_path = RESULTS_DIR / f"{args.resume}.csv"
        if not resume_path.exists():
            resume_path = RESULTS_DIR / f"{args.resume}_partial.csv"
        if resume_path.exists():
            with open(resume_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_ids.add(f"{row['model_key']}:{row['example_id']}")
            console.print(f"  [dim]Resuming from {resume_path.name}: {len(done_ids)} examples already done[/dim]")
            run_id = args.resume  # append to same run file
        else:
            console.print(f"  [yellow]Resume file not found: {resume_path}[/yellow]")

    console.rule("[bold magenta]Latent Rollback Cross-Model Benchmark[/bold magenta]")
    console.print(f"  Models : {args.models}")
    console.print(f"  Tasks  : {args.tasks}")
    console.print(f"  N/task : {args.n}")
    console.print(f"  Scale  : {args.scale}")
    console.print(f"  Run ID : {run_id}")

    console.rule("[bold]Loading benchmark dataset[/bold]")
    examples = load_benchmark(
        tasks=tuple(args.tasks),
        n_per_task=args.n,
        seed=args.seed,
    )
    console.print(f"  Total examples: {len(examples)}")

    all_records: list[BenchmarkRecord] = []
    for model_key in args.models:
        model_cfg = MODEL_MATRIX[model_key]
        records = run_model(
            model_key=model_key,
            model_cfg=model_cfg,
            examples=examples,
            injection_scale=args.scale,
            pool=args.pool,
            max_new_tokens=args.max_tokens,
            sweep_layer=args.sweep_layer,
            sweep_scale=args.sweep_scale,
            layer_override=args.layer,
            condition=args.condition,
            done_ids=done_ids,
            verbose=args.verbose,
        )
        all_records.extend(records)
        # Checkpoint after each model
        save_results(all_records, f"{run_id}_partial")

    print_benchmark_summary(all_records)
    save_results(all_records, run_id)

    # Success criterion: any model shows >=30% token reduction
    # with F1 degradation < 0.1
    n_success = sum(
        1 for r in all_records
        if r.input_token_reduction >= 0.3
        and (r.injection_f1 >= r.baseline_f1 - 0.1)
    )
    total = len(all_records)
    console.print(
        f"\n[bold]{n_success}/{total} examples: "
        f">=30% token reduction with F1 within 0.1 of baseline[/bold]"
    )
    return 0 if total > 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
