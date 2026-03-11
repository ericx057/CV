"""
Matrix Injection Benchmark Runner

Compares five context injection strategies on LongBench QA tasks:
  baseline   : full context + question in prompt (no injection)
  vector     : C injection only          [C + P]
  vector+F   : C injection + fact block  [C + F + P]
  matrix     : matrix injection only     [M + P]
  matrix+F   : matrix injection + facts  [M + F + P]

Context injection:
  C (vector): mean-pool context hidden states -> fixed vector added at layer L
  M (matrix): SVD of context hidden states -> B @ (A @ v) query-dependent injection
    H [seq_len, d_model] at layer L
    SVD -> U [seq_len, r], S [r], Vh [r, d_model]
    A = Vh[:r, :]           [r, d_model]  -- context subspace directions
    B = H.T @ U[:, :r]      [d_model, r]  -- lifting map
    correction(v) = B @ (A @ v)

F block strategies (h(D)):
  ner    : regex NER extraction (practical, no extra model)
  oracle : gold-answer-guided sentence extraction (upper bound)

Usage:
  source .venv/bin/activate
  python benchmark_matrix_runner.py --n 5 --max-tokens 60
  python benchmark_matrix_runner.py --models llama3-8b --n 10 --rank 4
  python benchmark_matrix_runner.py --rank 4 8 16        # rank sweep
  python benchmark_matrix_runner.py --fact-mode oracle   # upper bound F
  python benchmark_matrix_runner.py --fact-mode both     # NER + oracle
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
from benchmark_ablation import extract_facts_ner, extract_facts_oracle

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
    fact_mode: str               # "ner" | "oracle" | "none"
    n_context_words: int
    sv_energy_frac: float        # fraction of variance captured by top-r SVs

    # Token counts
    baseline_input_tokens: int
    vector_input_tokens: int
    vector_f_input_tokens: int
    matrix_input_tokens: int
    matrix_f_input_tokens: int
    baseline_output_tokens: int
    vector_output_tokens: int
    vector_f_output_tokens: int
    matrix_output_tokens: int
    matrix_f_output_tokens: int

    # Token efficiency
    input_token_reduction_vector: float
    input_token_reduction_vector_f: float
    input_token_reduction_matrix: float
    input_token_reduction_matrix_f: float
    total_cost_ratio_vector: float
    total_cost_ratio_vector_f: float
    total_cost_ratio_matrix: float
    total_cost_ratio_matrix_f: float

    # Accuracy
    baseline_f1: float
    vector_f1: float
    vector_f_f1: float
    matrix_f1: float
    matrix_f_f1: float
    baseline_exact_match: bool
    vector_exact_match: bool
    vector_f_exact_match: bool
    matrix_exact_match: bool
    matrix_f_exact_match: bool

    # Deltas (relative to baseline)
    f1_delta_vector: float
    f1_delta_vector_f: float
    f1_delta_matrix: float
    f1_delta_matrix_f: float

    # F contribution (how much F adds on top of injection alone)
    f_contrib_vector: float      # vector_f_f1 - vector_f1
    f_contrib_matrix: float      # matrix_f_f1 - matrix_f1

    # Matrix vs vector (without and with F)
    mx_vs_vec: float             # matrix_f1 - vector_f1
    mx_vs_vec_f: float           # matrix_f_f1 - vector_f_f1

    # Answers
    baseline_answer: str
    vector_answer: str
    vector_f_answer: str
    matrix_answer: str
    matrix_f_answer: str
    fact_block: str
    gold_answers: str

    elapsed_baseline_s: float
    elapsed_vector_s: float
    elapsed_vector_f_s: float
    elapsed_matrix_s: float
    elapsed_matrix_f_s: float


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
    fact_mode: str = "ner",
    scale: float = 1.0,
    max_new_tokens: int = 80,
    verbose: bool = False,
) -> MatrixBenchmarkRecord:
    full_prompt = example.full_prompt()
    question_prompt = example.question_prompt()

    # --- F block: h(D) ---
    if fact_mode == "oracle":
        fact_block = extract_facts_oracle(example.context, example.gold_answers)
    else:
        fact_block = extract_facts_ner(example.context)
    f_prompt = f"{fact_block}\n{question_prompt}" if fact_block else question_prompt

    # --- Baseline: full context in prompt ---
    t0 = time.time()
    baseline_text, n_baseline_input = generate_baseline_qa(
        wrapper, full_prompt, max_new_tokens
    )
    elapsed_baseline = time.time() - t0
    n_baseline_output = len(wrapper.encode(baseline_text))

    # --- Vector [C + P]: mean-pool context -> fixed injection, no F ---
    from context_injector import extract_context_state
    t0 = time.time()
    context_vector, _ = extract_context_state(
        wrapper, example.context, injection_layer, pool="mean"
    )
    vector_text, n_vector_input = generate_with_context_injection(
        wrapper, question_prompt, context_vector,
        layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
    )
    elapsed_vector = time.time() - t0
    n_vector_output = len(wrapper.encode(vector_text))

    # --- Vector+F [C + F + P]: same vector injection, F prepended to prompt ---
    t0 = time.time()
    vector_f_text, n_vector_f_input = generate_with_context_injection(
        wrapper, f_prompt, context_vector,
        layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
    )
    elapsed_vector_f = time.time() - t0
    n_vector_f_output = len(wrapper.encode(vector_f_text))

    # --- Matrix [M + P]: SVD context -> B @ (A @ v), no F ---
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

    # --- Matrix+F [M + F + P]: same matrix injection, F prepended to prompt ---
    t0 = time.time()
    f_ids = wrapper.encode(f_prompt)
    matrix_f_ids = generate_with_matrix_hook(
        wrapper,
        token_ids=f_ids,
        layer_matrices={injection_layer: (A, B)},
        mode="inject",
        scale=scale,
        max_new_tokens=max_new_tokens,
        broadcast=True,
    )
    matrix_f_text = _ids_to_str(wrapper, matrix_f_ids)
    elapsed_matrix_f = time.time() - t0
    n_matrix_f_input = len(f_ids)
    n_matrix_f_output = len(matrix_f_ids)

    # --- Grading ---
    baseline_grades  = grade_qa(baseline_text,  example.gold_answers)
    vector_grades    = grade_qa(vector_text,    example.gold_answers)
    vector_f_grades  = grade_qa(vector_f_text,  example.gold_answers)
    matrix_grades    = grade_qa(matrix_text,    example.gold_answers)
    matrix_f_grades  = grade_qa(matrix_f_text,  example.gold_answers)

    vec_m    = compute_token_metrics(n_baseline_input, n_vector_input,   n_baseline_output, n_vector_output)
    vec_f_m  = compute_token_metrics(n_baseline_input, n_vector_f_input, n_baseline_output, n_vector_f_output)
    mx_m     = compute_token_metrics(n_baseline_input, n_matrix_input,   n_baseline_output, n_matrix_output)
    mx_f_m   = compute_token_metrics(n_baseline_input, n_matrix_f_input, n_baseline_output, n_matrix_f_output)

    sv_energy_frac = float(S_r.sum() / (S_r.sum() + 1e-8))

    if verbose:
        console.print(f"    fact block: {fact_block[:80]}")
        console.print(f"    baseline  ({n_baseline_input:4d} in): {baseline_text.strip()[:70]}")
        console.print(f"    vector    ({n_vector_input:4d} in): {vector_text.strip()[:70]}")
        console.print(f"    vector+F  ({n_vector_f_input:4d} in): {vector_f_text.strip()[:70]}")
        console.print(f"    matrix    ({n_matrix_input:4d} in): {matrix_text.strip()[:70]}")
        console.print(f"    matrix+F  ({n_matrix_f_input:4d} in): {matrix_f_text.strip()[:70]}")
        console.print(f"    gold: {example.gold_answers}")
        console.print(
            f"    BL={baseline_grades['f1']:.3f}  "
            f"VEC={vector_grades['f1']:.3f}  VEC+F={vector_f_grades['f1']:.3f}  "
            f"MX={matrix_grades['f1']:.3f}  MX+F={matrix_f_grades['f1']:.3f}"
        )

    bl_f1  = baseline_grades["f1"]
    vec_f1 = vector_grades["f1"]
    vf_f1  = vector_f_grades["f1"]
    mx_f1  = matrix_grades["f1"]
    mf_f1  = matrix_f_grades["f1"]

    return MatrixBenchmarkRecord(
        model_key=model_key,
        model_hf_id=model_hf_id,
        task=example.task,
        example_id=example.id,
        injection_layer=injection_layer,
        rank=rank,
        fact_mode=fact_mode,
        n_context_words=example.context_word_len,
        sv_energy_frac=round(sv_energy_frac, 4),

        baseline_input_tokens=n_baseline_input,
        vector_input_tokens=n_vector_input,
        vector_f_input_tokens=n_vector_f_input,
        matrix_input_tokens=n_matrix_input,
        matrix_f_input_tokens=n_matrix_f_input,
        baseline_output_tokens=n_baseline_output,
        vector_output_tokens=n_vector_output,
        vector_f_output_tokens=n_vector_f_output,
        matrix_output_tokens=n_matrix_output,
        matrix_f_output_tokens=n_matrix_f_output,

        input_token_reduction_vector=vec_m["input_token_reduction"],
        input_token_reduction_vector_f=vec_f_m["input_token_reduction"],
        input_token_reduction_matrix=mx_m["input_token_reduction"],
        input_token_reduction_matrix_f=mx_f_m["input_token_reduction"],
        total_cost_ratio_vector=vec_m["total_cost_ratio"],
        total_cost_ratio_vector_f=vec_f_m["total_cost_ratio"],
        total_cost_ratio_matrix=mx_m["total_cost_ratio"],
        total_cost_ratio_matrix_f=mx_f_m["total_cost_ratio"],

        baseline_f1=bl_f1,
        vector_f1=vec_f1,
        vector_f_f1=vf_f1,
        matrix_f1=mx_f1,
        matrix_f_f1=mf_f1,
        baseline_exact_match=baseline_grades["exact_match"],
        vector_exact_match=vector_grades["exact_match"],
        vector_f_exact_match=vector_f_grades["exact_match"],
        matrix_exact_match=matrix_grades["exact_match"],
        matrix_f_exact_match=matrix_f_grades["exact_match"],

        f1_delta_vector=round(vec_f1 - bl_f1, 4),
        f1_delta_vector_f=round(vf_f1 - bl_f1, 4),
        f1_delta_matrix=round(mx_f1 - bl_f1, 4),
        f1_delta_matrix_f=round(mf_f1 - bl_f1, 4),

        f_contrib_vector=round(vf_f1 - vec_f1, 4),
        f_contrib_matrix=round(mf_f1 - mx_f1, 4),

        mx_vs_vec=round(mx_f1 - vec_f1, 4),
        mx_vs_vec_f=round(mf_f1 - vf_f1, 4),

        baseline_answer=baseline_text.strip()[:120],
        vector_answer=vector_text.strip()[:120],
        vector_f_answer=vector_f_text.strip()[:120],
        matrix_answer=matrix_text.strip()[:120],
        matrix_f_answer=matrix_f_text.strip()[:120],
        fact_block=fact_block[:200],
        gold_answers=str(example.gold_answers),

        elapsed_baseline_s=round(elapsed_baseline, 2),
        elapsed_vector_s=round(elapsed_vector, 2),
        elapsed_vector_f_s=round(elapsed_vector_f, 2),
        elapsed_matrix_s=round(elapsed_matrix, 2),
        elapsed_matrix_f_s=round(elapsed_matrix_f, 2),
    )


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    model_cfg: dict,
    examples: list[BenchmarkExample],
    rank: int = 8,
    fact_mode: str = "ner",
    scale: float = 1.0,
    max_new_tokens: int = 80,
    verbose: bool = False,
) -> list[MatrixBenchmarkRecord]:
    hf_id = model_cfg["hf_id"]
    console.rule(f"[bold cyan]Model: {model_key}[/bold cyan]")
    console.print(f"  HF ID     : {hf_id}")
    console.print(f"  Rank      : {rank}")
    console.print(f"  Fact mode : {fact_mode}")

    try:
        wrapper = load_model(hf_id)
    except Exception as exc:
        console.print(f"  [red]Failed to load {hf_id}: {exc}[/red]")
        return []

    injection_layer = select_layer_heuristic(wrapper, hf_id)
    console.print(f"  Injection layer: {injection_layer} / {wrapper.n_layers}")

    records: list[MatrixBenchmarkRecord] = []

    modes = ["ner", "oracle"] if fact_mode == "both" else [fact_mode]
    for mode in modes:
        for i, ex in enumerate(examples):
            console.print(
                f"\n  [{i+1}/{len(examples)}] {ex.task}/{ex.id[:24]}  "
                f"({ex.context_word_len}w ctx)  [dim]fact={mode}[/dim]"
            )
            try:
                rec = run_example(
                    wrapper, ex,
                    model_key=model_key,
                    model_hf_id=hf_id,
                    injection_layer=injection_layer,
                    rank=rank,
                    fact_mode=mode,
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

def _avg(records: list[MatrixBenchmarkRecord], attr: str) -> float:
    vals = [getattr(r, attr) for r in records]
    return sum(vals) / len(vals) if vals else 0.0


def _em_rate(records: list[MatrixBenchmarkRecord], attr: str) -> float:
    return sum(1 for r in records if getattr(r, attr)) / len(records) if records else 0.0


def _print_model_summary(
    model_key: str, rank: int, records: list[MatrixBenchmarkRecord]
) -> None:
    if not records:
        return

    from collections import defaultdict
    by_mode: dict[str, list[MatrixBenchmarkRecord]] = defaultdict(list)
    for r in records:
        by_mode[r.fact_mode].append(r)

    for mode, recs in by_mode.items():
        n = len(recs)
        bl  = _avg(recs, "baseline_f1")
        vec = _avg(recs, "vector_f1")
        vf  = _avg(recs, "vector_f_f1")
        mx  = _avg(recs, "matrix_f1")
        mf  = _avg(recs, "matrix_f_f1")

        console.rule(f"[bold]{model_key} | rank={rank} | fact={mode}[/bold]")
        console.print(f"  N={n}  SV energy={_avg(recs, 'sv_energy_frac'):.1%}")

        def _row(label, f1, em, delta=None):
            col = "green" if f1 >= bl - 0.05 else "yellow" if f1 >= bl - 0.15 else "red"
            d = f"  Δ=[{col}]{delta:+.3f}[/{col}]" if delta is not None else ""
            console.print(f"  {label:<14}: EM={em:.1%}  F1=[{col}]{f1:.3f}[/{col}]{d}")

        _row("Baseline",  bl,  _em_rate(recs, "baseline_exact_match"))
        _row("Vector",    vec, _em_rate(recs, "vector_exact_match"),   vec - bl)
        _row("Vector+F",  vf,  _em_rate(recs, "vector_f_exact_match"), vf  - bl)
        _row("Matrix",    mx,  _em_rate(recs, "matrix_exact_match"),   mx  - bl)
        _row("Matrix+F",  mf,  _em_rate(recs, "matrix_f_exact_match"), mf  - bl)
        console.print(
            f"  F contrib → vec: [{('green' if vf>vec else 'red')}]{vf-vec:+.3f}[/{'green' if vf>vec else 'red'}]"
            f"   matrix: [{('green' if mf>mx else 'red')}]{mf-mx:+.3f}[/{'green' if mf>mx else 'red'}]"
        )


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
    Split results by context word count and show all five conditions per bucket.

    Key question: at what context length does matrix overtake vector,
    and does F narrow or widen that gap?
    """
    # Run separately per fact_mode
    modes = sorted(set(r.fact_mode for r in all_records))
    for mode in modes:
        recs_mode = [r for r in all_records if r.fact_mode == mode]
        console.rule(f"[bold]Context Length Stratification (fact={mode})[/bold]")

        table = Table(
            title="F1 by Context Length — Vec vs Vec+F vs Mx vs Mx+F",
            box=box.MINIMAL_DOUBLE_HEAD,
        )
        table.add_column("Words",    style="cyan")
        table.add_column("N",        justify="right")
        table.add_column("BL",       justify="right")
        table.add_column("Vec",      justify="right")
        table.add_column("Vec+F",    justify="right")
        table.add_column("Mx",       justify="right")
        table.add_column("Mx+F",     justify="right")
        table.add_column("Mx-Vec",   justify="right")
        table.add_column("MxF-VF",   justify="right")
        table.add_column("Winner",   justify="center")

        any_crossover = False
        prev_winner = None

        for lo, hi in buckets:
            label = f"{lo}–{hi}" if hi < 999999 else f"{lo}+"
            recs = [r for r in recs_mode if lo <= r.n_context_words < hi]
            if not recs:
                continue
            bl  = _avg(recs, "baseline_f1")
            vec = _avg(recs, "vector_f1")
            vf  = _avg(recs, "vector_f_f1")
            mx  = _avg(recs, "matrix_f1")
            mf  = _avg(recs, "matrix_f_f1")
            mx_v  = mx - vec
            mf_vf = mf - vf

            # Winner = best injection (excluding baseline)
            best = max(vec, vf, mx, mf)
            if best == mf:      winner, cur = "[green]Mx+F[/green]",   "matrix_f"
            elif best == mx:    winner, cur = "[green]Mx[/green]",      "matrix"
            elif best == vf:    winner, cur = "[yellow]Vec+F[/yellow]", "vector_f"
            else:               winner, cur = "[yellow]Vec[/yellow]",   "vector"

            if prev_winner and cur != prev_winner:
                any_crossover = True

            def _c(v): return "green" if v > 0.005 else "red" if v < -0.005 else "white"

            table.add_row(
                label, str(len(recs)),
                f"{bl:.3f}", f"{vec:.3f}", f"{vf:.3f}", f"{mx:.3f}", f"{mf:.3f}",
                f"[{_c(mx_v)}]{mx_v:+.3f}[/{_c(mx_v)}]",
                f"[{_c(mf_vf)}]{mf_vf:+.3f}[/{_c(mf_vf)}]",
                winner,
            )
            prev_winner = cur

        console.print(table)
        if any_crossover:
            console.print("[bold green]Crossover detected across context length buckets.[/bold green]")
        else:
            console.print("[dim]No crossover yet — try more examples or higher rank.[/dim]")


def print_benchmark_summary(
    all_records: list[MatrixBenchmarkRecord], rank: int
) -> None:
    from collections import defaultdict
    console.rule("[bold magenta]FULL BENCHMARK SUMMARY[/bold magenta]")

    # One table per fact_mode
    modes = sorted(set(r.fact_mode for r in all_records))
    for mode in modes:
        recs_mode = [r for r in all_records if r.fact_mode == mode]
        by_model: dict[str, list[MatrixBenchmarkRecord]] = defaultdict(list)
        for r in recs_mode:
            by_model[r.model_key].append(r)

        table = Table(
            title=f"rank={rank}  fact={mode} — Baseline / Vec / Vec+F / Mx / Mx+F",
            box=box.MINIMAL_DOUBLE_HEAD,
        )
        table.add_column("Model", style="cyan")
        table.add_column("N",        justify="right")
        table.add_column("BL F1",    justify="right")
        table.add_column("Vec F1",   justify="right")
        table.add_column("Vec+F F1", justify="right")
        table.add_column("Mx F1",    justify="right")
        table.add_column("Mx+F F1",  justify="right")
        table.add_column("F→Vec",    justify="right")
        table.add_column("F→Mx",     justify="right")
        table.add_column("Mx-Vec",   justify="right")
        table.add_column("MxF-VF",   justify="right")

        for model_key, recs in by_model.items():
            bl  = _avg(recs, "baseline_f1")
            vec = _avg(recs, "vector_f1")
            vf  = _avg(recs, "vector_f_f1")
            mx  = _avg(recs, "matrix_f1")
            mf  = _avg(recs, "matrix_f_f1")

            def _c(val): return "green" if val > 0.005 else "red" if val < -0.005 else "white"

            table.add_row(
                model_key, str(len(recs)),
                f"{bl:.3f}",
                f"{vec:.3f}",
                f"{vf:.3f}",
                f"{mx:.3f}",
                f"{mf:.3f}",
                f"[{_c(vf-vec)}]{vf-vec:+.3f}[/{_c(vf-vec)}]",
                f"[{_c(mf-mx)}]{mf-mx:+.3f}[/{_c(mf-mx)}]",
                f"[{_c(mx-vec)}]{mx-vec:+.3f}[/{_c(mx-vec)}]",
                f"[{_c(mf-vf)}]{mf-vf:+.3f}[/{_c(mf-vf)}]",
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
    parser.add_argument(
        "--fact-mode", choices=["ner", "oracle", "both"], default="ner",
        dest="fact_mode",
        help="ner: regex extraction (practical); oracle: gold-guided (upper bound); both: run both",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--run-id", default=None, dest="run_id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_id = args.run_id or f"matrix_benchmark_{int(time.time())}"

    console.rule("[bold magenta]Matrix vs Vector Context Injection Benchmark[/bold magenta]")
    console.print(f"  Models    : {args.models}")
    console.print(f"  Tasks     : {args.tasks}")
    console.print(f"  N/task    : {args.n}")
    console.print(f"  Rank(s)   : {args.rank}")
    console.print(f"  Fact mode : {args.fact_mode}")
    console.print(f"  Scale     : {args.scale}")
    console.print(f"  Run ID    : {run_id}")
    console.print()
    console.print("  Conditions:")
    console.print("    1. Baseline  — [ctx + P]")
    console.print("    2. Vector    — [C + P]")
    console.print("    3. Vector+F  — [C + F + P]")
    console.print("    4. Matrix    — [M + P]")
    console.print("    5. Matrix+F  — [M + F + P]")

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
                fact_mode=args.fact_mode,
                scale=args.scale,
                max_new_tokens=args.max_tokens,
                verbose=args.verbose,
            )
            all_records.extend(records)
            save_results(all_records, f"{run_id}_partial")

    primary_rank = args.rank[0]
    print_benchmark_summary(all_records, rank=primary_rank)
    save_results(all_records, run_id)

    # Per-example winner across all five conditions
    total = len(all_records)
    best_mf  = sum(1 for r in all_records if max(r.vector_f1, r.vector_f_f1, r.matrix_f1, r.matrix_f_f1) == r.matrix_f_f1)
    best_mx  = sum(1 for r in all_records if max(r.vector_f1, r.vector_f_f1, r.matrix_f1, r.matrix_f_f1) == r.matrix_f1
                   and r.matrix_f1 > r.matrix_f_f1)
    best_vf  = sum(1 for r in all_records if max(r.vector_f1, r.vector_f_f1, r.matrix_f1, r.matrix_f_f1) == r.vector_f_f1
                   and r.vector_f_f1 > r.matrix_f_f1 and r.vector_f_f1 > r.matrix_f1)
    best_vec = total - best_mf - best_mx - best_vf
    console.print(
        f"\n[bold]Per-example best injection:[/bold]  "
        f"Mx+F={best_mf}/{total}  Mx={best_mx}/{total}  "
        f"Vec+F={best_vf}/{total}  Vec={best_vec}/{total}"
    )

    return 0 if total > 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
