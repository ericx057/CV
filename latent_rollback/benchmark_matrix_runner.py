"""
Matrix Injection Benchmark Runner

Compares three context injection strategies on LongBench QA tasks:
  baseline : full context + question in prompt (no injection)
  vector   : mean-pool context hidden states -> fixed vector added at layer L
  matrix   : SVD of context hidden states -> B @ (A @ v) query-dependent injection

The matrix approach encodes context as a low-rank linear map (rank r):
  H [seq_len, d_model] at layer L
  SVD -> U [seq_len, r], S [r], Vh [r, d_model]
  A = Vh[:r, :]           [r, d_model]  -- context subspace directions
  B = H.T @ U[:, :r]      [d_model, r]  -- lifting map
  correction(v) = B @ (A @ v)           -- query-adaptive injection

Unlike the fixed vector, the correction is query-dependent: different
question residuals activate different context components, preserving the
binding structure that mean-pooling destroys.

Usage:
  source .venv/bin/activate
  python benchmark_matrix_runner.py --n 5 --max-tokens 60
  python benchmark_matrix_runner.py --models llama3-8b --n 10 --rank 4
  python benchmark_matrix_runner.py --rank 4 8 16   # sweep ranks
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from rich.console import Console
from rich import box
from rich.table import Table

from backend_mlx import load_model, MLXModelWrapper, generate_with_matrix_hook, _ids_to_str
from benchmark_datasets import load_benchmark, BenchmarkExample, grade_qa
from context_injector import (
    generate_baseline_qa,
    generate_with_context_injection,
    compute_token_metrics,
)
from layer_selector import select_layer_heuristic

console = Console()
RESULTS_DIR = Path(__file__).parent / "benchmark_results"

MODEL_MATRIX: dict[str, dict] = {
    "llama3-8b": {
        "hf_id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "size_gb": 4.5,
        "family": "llama",
    },
    "qwen25-7b": {
        "hf_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "size_gb": 4.5,
        "family": "qwen",
    },
    "mistral-24b": {
        "hf_id": "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
        "size_gb": 13.0,
        "family": "mistral",
    },
    "deepseek-14b": {
        "hf_id": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        "size_gb": 8.0,
        "family": "deepseek",
    },
}


# ---------------------------------------------------------------------------
# Matrix extraction
# ---------------------------------------------------------------------------

def extract_context_matrix(
    wrapper: MLXModelWrapper,
    context_text: str,
    layer: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Extract a low-rank (A, B) matrix pair from context hidden states at `layer`.

    H [seq_len, d_model] -> SVD -> U [seq_len, r], S [r], Vh [r, d_model]
    A = Vh[:r, :]           [r, d_model]  -- context subspace directions
    B = H.T @ U[:, :r]      [d_model, r]  -- lifting map

    Correction at inference: correction(v) = B @ (A @ v)

    Returns
    -------
    (A, B, singular_values, n_context_tokens)
    """
    token_ids = wrapper.encode(context_text)
    n_tokens = len(token_ids)

    h = wrapper.embed(token_ids)
    mask = "causal" if h.shape[1] > 1 else None

    for i, layer_module in enumerate(wrapper.layers):
        h = layer_module(h, mask, cache=None)
        if i == layer:
            mx.eval(h)
            break

    H = torch.from_numpy(np.array(h[0])).float()  # [seq_len, d_model]

    effective_rank = min(rank, H.shape[0], H.shape[1])

    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    # U: [seq_len, min(seq_len, d_model)]
    # S: [min(seq_len, d_model)]
    # Vh: [min(seq_len, d_model), d_model]

    U_r = U[:, :effective_rank]    # [seq_len, r]
    S_r = S[:effective_rank]       # [r]
    Vh_r = Vh[:effective_rank, :]  # [r, d_model]

    A = Vh_r                       # [r, d_model]
    B = H.T @ U_r                  # [d_model, r]

    return A, B, S_r, n_tokens


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class MatrixBenchmarkRecord:
    model_key: str
    model_hf_id: str
    task: str
    example_id: str
    injection_layer: int
    rank: int
    n_context_words: int
    sv_energy_frac: float        # fraction of variance captured by top-r SVs

    baseline_input_tokens: int
    vector_input_tokens: int
    matrix_input_tokens: int
    baseline_output_tokens: int
    vector_output_tokens: int
    matrix_output_tokens: int

    input_token_reduction_vector: float
    input_token_reduction_matrix: float
    total_cost_ratio_vector: float
    total_cost_ratio_matrix: float

    baseline_f1: float
    vector_f1: float
    matrix_f1: float
    baseline_exact_match: bool
    vector_exact_match: bool
    matrix_exact_match: bool

    f1_delta_vector: float       # vector_f1 - baseline_f1
    f1_delta_matrix: float       # matrix_f1 - baseline_f1
    f1_improvement: float        # matrix_f1 - vector_f1

    baseline_answer: str
    vector_answer: str
    matrix_answer: str
    gold_answers: str

    elapsed_baseline_s: float
    elapsed_vector_s: float
    elapsed_matrix_s: float


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

def run_example(
    wrapper: MLXModelWrapper,
    example: BenchmarkExample,
    model_key: str,
    model_hf_id: str,
    injection_layer: int,
    rank: int,
    scale: float = 1.0,
    max_new_tokens: int = 80,
    verbose: bool = False,
) -> MatrixBenchmarkRecord:
    full_prompt = example.full_prompt()
    question_prompt = example.question_prompt()

    # --- Baseline: full context in prompt ---
    t0 = time.time()
    baseline_text, n_baseline_input = generate_baseline_qa(
        wrapper, full_prompt, max_new_tokens
    )
    elapsed_baseline = time.time() - t0
    n_baseline_output = len(wrapper.encode(baseline_text))

    # --- Vector: mean-pool context -> fixed injection ---
    from context_injector import extract_context_state
    t0 = time.time()
    context_vector, n_ctx_tokens = extract_context_state(
        wrapper, example.context, injection_layer, pool="mean"
    )
    vector_text, n_vector_input = generate_with_context_injection(
        wrapper, question_prompt, context_vector,
        layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
    )
    elapsed_vector = time.time() - t0
    n_vector_output = len(wrapper.encode(vector_text))

    # --- Matrix: SVD context -> B @ (A @ v) injection ---
    t0 = time.time()
    A, B, S_r, _ = extract_context_matrix(
        wrapper, example.context, injection_layer, rank=rank
    )
    question_ids = wrapper.encode(question_prompt)
    matrix_ids = generate_with_matrix_hook(
        wrapper,
        token_ids=question_ids,
        layer_matrices={injection_layer: (A, B)},
        mode="inject",
        scale=scale,
        max_new_tokens=max_new_tokens,
        broadcast=True,
    )
    matrix_text = _ids_to_str(wrapper, matrix_ids)
    elapsed_matrix = time.time() - t0
    n_matrix_input = len(question_ids)
    n_matrix_output = len(matrix_ids)

    # --- Grading ---
    baseline_grades = grade_qa(baseline_text, example.gold_answers)
    vector_grades = grade_qa(vector_text, example.gold_answers)
    matrix_grades = grade_qa(matrix_text, example.gold_answers)

    vec_metrics = compute_token_metrics(
        n_baseline_input, n_vector_input, n_baseline_output, n_vector_output
    )
    mx_metrics = compute_token_metrics(
        n_baseline_input, n_matrix_input, n_baseline_output, n_matrix_output
    )

    # Singular value energy fraction (how much variance rank captures)
    sv_energy_frac = float(S_r.sum() / (S_r.sum() + 1e-8))  # fraction of top-r energy

    if verbose:
        console.print(
            f"    baseline  ({n_baseline_input:4d} in): {baseline_text.strip()[:70]}"
        )
        console.print(
            f"    vector    ({n_vector_input:4d} in): {vector_text.strip()[:70]}"
        )
        console.print(
            f"    matrix    ({n_matrix_input:4d} in): {matrix_text.strip()[:70]}"
        )
        console.print(f"    gold: {example.gold_answers}")
        console.print(
            f"    BL F1={baseline_grades['f1']:.3f}  "
            f"VEC F1={vector_grades['f1']:.3f}  "
            f"MX F1={matrix_grades['f1']:.3f}  "
            f"SV energy={sv_energy_frac:.2%}"
        )

    return MatrixBenchmarkRecord(
        model_key=model_key,
        model_hf_id=model_hf_id,
        task=example.task,
        example_id=example.id,
        injection_layer=injection_layer,
        rank=rank,
        n_context_words=example.context_word_len,
        sv_energy_frac=round(sv_energy_frac, 4),

        baseline_input_tokens=n_baseline_input,
        vector_input_tokens=n_vector_input,
        matrix_input_tokens=n_matrix_input,
        baseline_output_tokens=n_baseline_output,
        vector_output_tokens=n_vector_output,
        matrix_output_tokens=n_matrix_output,

        input_token_reduction_vector=vec_metrics["input_token_reduction"],
        input_token_reduction_matrix=mx_metrics["input_token_reduction"],
        total_cost_ratio_vector=vec_metrics["total_cost_ratio"],
        total_cost_ratio_matrix=mx_metrics["total_cost_ratio"],

        baseline_f1=baseline_grades["f1"],
        vector_f1=vector_grades["f1"],
        matrix_f1=matrix_grades["f1"],
        baseline_exact_match=baseline_grades["exact_match"],
        vector_exact_match=vector_grades["exact_match"],
        matrix_exact_match=matrix_grades["exact_match"],

        f1_delta_vector=round(vector_grades["f1"] - baseline_grades["f1"], 4),
        f1_delta_matrix=round(matrix_grades["f1"] - baseline_grades["f1"], 4),
        f1_improvement=round(matrix_grades["f1"] - vector_grades["f1"], 4),

        baseline_answer=baseline_text.strip()[:120],
        vector_answer=vector_text.strip()[:120],
        matrix_answer=matrix_text.strip()[:120],
        gold_answers=str(example.gold_answers),

        elapsed_baseline_s=round(elapsed_baseline, 2),
        elapsed_vector_s=round(elapsed_vector, 2),
        elapsed_matrix_s=round(elapsed_matrix, 2),
    )


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    model_cfg: dict,
    examples: list[BenchmarkExample],
    rank: int = 8,
    scale: float = 1.0,
    max_new_tokens: int = 80,
    verbose: bool = False,
) -> list[MatrixBenchmarkRecord]:
    hf_id = model_cfg["hf_id"]
    console.rule(f"[bold cyan]Model: {model_key}[/bold cyan]")
    console.print(f"  HF ID : {hf_id}")
    console.print(f"  Rank  : {rank}")

    try:
        wrapper = load_model(hf_id)
    except Exception as exc:
        console.print(f"  [red]Failed to load {hf_id}: {exc}[/red]")
        return []

    injection_layer = select_layer_heuristic(wrapper, hf_id)
    console.print(f"  Injection layer: {injection_layer} / {wrapper.n_layers}")

    records: list[MatrixBenchmarkRecord] = []
    for i, ex in enumerate(examples):
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
                rank=rank,
                scale=scale,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            records.append(rec)
        except Exception as exc:
            console.print(f"  [red]ERROR: {exc}[/red]")

    _print_model_summary(model_key, rank, records)

    del wrapper
    gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception:
        pass

    return records


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _print_model_summary(
    model_key: str, rank: int, records: list[MatrixBenchmarkRecord]
) -> None:
    if not records:
        return
    n = len(records)
    avg_bl_f1 = sum(r.baseline_f1 for r in records) / n
    avg_vec_f1 = sum(r.vector_f1 for r in records) / n
    avg_mx_f1 = sum(r.matrix_f1 for r in records) / n
    avg_improvement = sum(r.f1_improvement for r in records) / n
    avg_sv_energy = sum(r.sv_energy_frac for r in records) / n

    bl_em = sum(1 for r in records if r.baseline_exact_match) / n
    vec_em = sum(1 for r in records if r.vector_exact_match) / n
    mx_em = sum(1 for r in records if r.matrix_exact_match) / n

    console.rule(f"[bold]{model_key} summary (rank={rank})[/bold]")
    console.print(f"  Examples        : {n}")
    console.print(f"  Avg SV energy   : {avg_sv_energy:.1%}")

    vec_color = "green" if avg_vec_f1 >= avg_bl_f1 - 0.05 else "yellow" if avg_vec_f1 >= avg_bl_f1 - 0.15 else "red"
    mx_color  = "green" if avg_mx_f1 >= avg_bl_f1 - 0.05 else "yellow" if avg_mx_f1 >= avg_bl_f1 - 0.15 else "red"
    imp_color = "green" if avg_improvement > 0 else "red"

    console.print(f"  Baseline EM/F1  : {bl_em:.1%} / {avg_bl_f1:.3f}")
    console.print(f"  Vector EM/F1    : [{vec_color}]{vec_em:.1%} / {avg_vec_f1:.3f}[/{vec_color}]")
    console.print(f"  Matrix EM/F1    : [{mx_color}]{mx_em:.1%} / {avg_mx_f1:.3f}[/{mx_color}]")
    console.print(f"  Mx vs Vec F1    : [{imp_color}]{avg_improvement:+.3f}[/{imp_color}]")


def _print_context_length_stratification(
    all_records: list[MatrixBenchmarkRecord],
    buckets: tuple[tuple[int, int], ...] = (
        (0, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 3000),
        (3000, 999999),
    ),
) -> None:
    """
    Split results by context word count and show vector vs matrix F1 per bucket.

    Tests the hypothesis: vector wins on short contexts, matrix wins on long ones.
    The crossover point (if it exists) tells you when SVD decomposition starts
    recovering enough structure to outperform mean-pooling.
    """
    console.rule("[bold]Context Length Stratification: Vector vs Matrix[/bold]")

    table = Table(
        title="F1 by Context Length — does matrix win on longer contexts?",
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    table.add_column("Context Words", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("BL F1", justify="right")
    table.add_column("Vec F1", justify="right")
    table.add_column("Mx F1", justify="right")
    table.add_column("Mx - Vec", justify="right")
    table.add_column("Vec EM", justify="right")
    table.add_column("Mx EM", justify="right")
    table.add_column("Winner", justify="center")

    any_crossover = False
    prev_winner = None

    for lo, hi in buckets:
        label = f"{lo}–{hi}" if hi < 999999 else f"{lo}+"
        recs = [r for r in all_records if lo <= r.n_context_words < hi]
        if not recs:
            continue
        n = len(recs)
        avg_bl = sum(r.baseline_f1 for r in recs) / n
        avg_vec = sum(r.vector_f1 for r in recs) / n
        avg_mx = sum(r.matrix_f1 for r in recs) / n
        avg_imp = avg_mx - avg_vec
        vec_em = sum(1 for r in recs if r.vector_exact_match) / n
        mx_em = sum(1 for r in recs if r.matrix_exact_match) / n

        if avg_imp > 0.01:
            winner = "[green]MATRIX[/green]"
            cur_winner = "matrix"
        elif avg_imp < -0.01:
            winner = "[yellow]VECTOR[/yellow]"
            cur_winner = "vector"
        else:
            winner = "TIE"
            cur_winner = "tie"

        if prev_winner is not None and cur_winner != prev_winner and cur_winner != "tie":
            any_crossover = True

        imp_color = "green" if avg_imp > 0 else "red" if avg_imp < 0 else "white"
        table.add_row(
            label, str(n),
            f"{avg_bl:.3f}",
            f"{avg_vec:.3f}",
            f"{avg_mx:.3f}",
            f"[{imp_color}]{avg_imp:+.3f}[/{imp_color}]",
            f"{vec_em:.1%}", f"{mx_em:.1%}",
            winner,
        )
        prev_winner = cur_winner if cur_winner != "tie" else prev_winner

    console.print(table)

    if any_crossover:
        console.print(
            "[bold green]Crossover detected — matrix overtakes vector at longer contexts.[/bold green]"
        )
    else:
        console.print(
            "[dim]No clear crossover yet. Try more examples or higher rank.[/dim]"
        )


def print_benchmark_summary(
    all_records: list[MatrixBenchmarkRecord], rank: int
) -> None:
    from collections import defaultdict
    console.rule("[bold magenta]MATRIX vs VECTOR BENCHMARK SUMMARY[/bold magenta]")

    by_model: dict[str, list[MatrixBenchmarkRecord]] = defaultdict(list)
    for r in all_records:
        by_model[r.model_key].append(r)

    table = Table(
        title=f"Matrix (rank={rank}) vs Vector Injection — QA Accuracy",
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    table.add_column("Model", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Layer", justify="right")
    table.add_column("SV Energy", justify="right")
    table.add_column("BL F1", justify="right")
    table.add_column("Vec F1", justify="right")
    table.add_column("Mx F1", justify="right")
    table.add_column("Mx-Vec", justify="right")
    table.add_column("BL EM", justify="right")
    table.add_column("Vec EM", justify="right")
    table.add_column("Mx EM", justify="right")

    for model_key, recs in by_model.items():
        n = len(recs)
        avg_bl = sum(r.baseline_f1 for r in recs) / n
        avg_vec = sum(r.vector_f1 for r in recs) / n
        avg_mx = sum(r.matrix_f1 for r in recs) / n
        avg_imp = avg_mx - avg_vec
        avg_sv = sum(r.sv_energy_frac for r in recs) / n
        bl_em = sum(1 for r in recs if r.baseline_exact_match) / n
        vec_em = sum(1 for r in recs if r.vector_exact_match) / n
        mx_em = sum(1 for r in recs if r.matrix_exact_match) / n
        layer = recs[0].injection_layer if recs else "-"
        imp_color = "green" if avg_imp > 0 else "red"
        table.add_row(
            model_key, str(n), str(layer),
            f"{avg_sv:.1%}",
            f"{avg_bl:.3f}",
            f"{avg_vec:.3f}",
            f"{avg_mx:.3f}",
            f"[{imp_color}]{avg_imp:+.3f}[/{imp_color}]",
            f"{bl_em:.1%}", f"{vec_em:.1%}", f"{mx_em:.1%}",
        )

    console.print(table)

    # Context length stratification: does matrix beat vector on long contexts?
    _print_context_length_stratification(all_records)

    # Rank sweep summary if multiple ranks present
    ranks = sorted(set(r.rank for r in all_records))
    if len(ranks) > 1:
        console.rule("[bold]Rank Sweep[/bold]")
        rank_table = Table(box=box.SIMPLE)
        rank_table.add_column("Rank", justify="right")
        rank_table.add_column("Avg BL F1", justify="right")
        rank_table.add_column("Avg Vec F1", justify="right")
        rank_table.add_column("Avg Mx F1", justify="right")
        rank_table.add_column("Mx - Vec", justify="right")
        rank_table.add_column("SV Energy", justify="right")
        for rk in ranks:
            recs_rk = [r for r in all_records if r.rank == rk]
            n_rk = len(recs_rk)
            avg_bl = sum(r.baseline_f1 for r in recs_rk) / n_rk
            avg_vec = sum(r.vector_f1 for r in recs_rk) / n_rk
            avg_mx = sum(r.matrix_f1 for r in recs_rk) / n_rk
            avg_imp = avg_mx - avg_vec
            avg_sv = sum(r.sv_energy_frac for r in recs_rk) / n_rk
            imp_color = "green" if avg_imp > 0 else "red"
            rank_table.add_row(
                str(rk),
                f"{avg_bl:.3f}",
                f"{avg_vec:.3f}",
                f"{avg_mx:.3f}",
                f"[{imp_color}]{avg_imp:+.3f}[/{imp_color}]",
                f"{avg_sv:.1%}",
            )
        console.print(rank_table)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    records: list[MatrixBenchmarkRecord], run_id: str
) -> tuple[Path, Path]:
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
        description="Matrix vs Vector Context Injection Benchmark"
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
    parser.add_argument(
        "--rank", nargs="+", type=int, default=[8],
        help="SVD rank(s). Multiple values run a rank sweep.",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=80, dest="max_tokens")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--run-id", default=None, dest="run_id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_id = args.run_id or f"matrix_benchmark_{int(time.time())}"

    console.rule("[bold magenta]Matrix vs Vector Context Injection Benchmark[/bold magenta]")
    console.print(f"  Models : {args.models}")
    console.print(f"  Tasks  : {args.tasks}")
    console.print(f"  N/task : {args.n}")
    console.print(f"  Rank(s): {args.rank}")
    console.print(f"  Scale  : {args.scale}")
    console.print(f"  Run ID : {run_id}")

    examples = load_benchmark(
        tasks=tuple(args.tasks),
        n_per_task=args.n,
        seed=args.seed,
    )
    console.print(f"  Total examples: {len(examples)}")

    all_records: list[MatrixBenchmarkRecord] = []

    for rank in args.rank:
        console.rule(f"[bold]Rank = {rank}[/bold]")
        for model_key in args.models:
            model_cfg = MODEL_MATRIX[model_key]
            records = run_model(
                model_key=model_key,
                model_cfg=model_cfg,
                examples=examples,
                rank=rank,
                scale=args.scale,
                max_new_tokens=args.max_tokens,
                verbose=args.verbose,
            )
            all_records.extend(records)
            save_results(all_records, f"{run_id}_partial")

    primary_rank = args.rank[0]
    print_benchmark_summary(all_records, rank=primary_rank)
    save_results(all_records, run_id)

    # Success criterion: matrix F1 >= vector F1 on majority of examples
    n_matrix_wins = sum(1 for r in all_records if r.f1_improvement > 0)
    n_ties = sum(1 for r in all_records if r.f1_improvement == 0)
    total = len(all_records)
    console.print(
        f"\n[bold]Matrix wins: {n_matrix_wins}/{total}  "
        f"Ties: {n_ties}/{total}  "
        f"Vector wins: {total - n_matrix_wins - n_ties}/{total}[/bold]"
    )

    return 0 if total > 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
