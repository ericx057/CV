"""
RepoBench-C integration for latent context compression evaluation.

RepoBench-C (cross_file_first split) measures repository-level next-line code
completion where the model must predict the next line given:
  - Cross-file context: relevant snippets from other files in the repo
  - In-file context: import block + preceding lines of the current file

This maps directly onto the latent injection setup:
  - context = cross-file snippets  →  gets compressed into the injection vector
  - query   = in-file context      →  stays in the token stream
  - gold    = next_line

For a coding assistant (Claude Code use case), this is the primary scenario:
compress knowledge of the broader codebase so each call only needs the
immediate in-file context, saving O(N * repo_tokens) per session.

Conditions:
  baseline     — full cross-file + in-file context in prompt (standard)
  vector       — cross-file context compressed to vector; in-file in prompt
  vector_f     — vector + code-fact F block (extract_facts_code)
  vector_f_sum — vector + model-written summary of cross-file context
  matrix       — SVD low-rank projection of cross-file context; in-file in prompt
  matrix_f     — matrix + code-fact F block

Metrics:
  exact_match    — predicted.rstrip() == next_line.rstrip()
  edit_sim       — difflib SequenceMatcher ratio (0.0–1.0)
  prefix_match   — generated starts with stripped gold (partial credit)

Usage:
  source .venv/bin/activate
  python benchmark_repobench.py
  python benchmark_repobench.py --models llama3-8b --n 50 --level 2k
  python benchmark_repobench.py --split cross_file_random --n 100
  python benchmark_repobench.py --conditions baseline vector vector_f
  python benchmark_repobench.py --layer 14 --rank 8
  python benchmark_repobench.py --resume repobench_run_123
"""

from __future__ import annotations

import argparse
import csv
import difflib
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich import box
from rich.table import Table

from .backend_torch import (
    load_model,
    MLXModelWrapper,
    clear_backend_cache,
    generate_with_matrix_hook,
    _ids_to_str,
)
from .context_injector import (
    extract_context_state,
    generate_baseline_qa,
    generate_with_context_injection,
    truncate_at_stop,
)
from .benchmark_matrix_runner import extract_context_matrix
from .benchmark_ablation import extract_facts_code
from .benchmark_code_refactor import build_summary_prompt
from .layer_selector import select_layer_heuristic
from .benchmark_runner import MODEL_MATRIX
from .config import results_path

console = Console()
RESULTS_DIR = results_path("repobench_results")

# Stop at newline — we predict exactly one line
COMPLETION_STOP = ("\n",)

ALL_CONDITIONS = (
    "baseline",
    "vector",
    "vector_f",
    "vector_f_sum",
    "matrix",
    "matrix_f",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class RepoBenchExample:
    id: str                      # repo_name + file_path + index
    repo_name: str
    file_path: str
    cross_file_context: str      # formatted cross-file snippets (the "context" to compress)
    in_file_context: str         # import block + cropped code (the query prompt)
    next_line: str               # gold label
    gold_snippet_index: int      # which snippet is the oracle answer
    token_num: int               # original token count for the prompt
    level: str                   # "2k", "4k", "8k", etc.
    split: str                   # "cross_file_first" | "cross_file_random" | "in_file"


def _format_cross_file(example: dict) -> str:
    """Assemble cross-file snippets into a single context string."""
    parts = [f"# Repo: {example['repo_name']}"]
    for snippet in example["context"]:
        parts.append(f"# Path: {snippet['path']}\n{snippet['snippet']}")
    return "\n\n".join(parts)


def _format_in_file(example: dict) -> str:
    """Assemble in-file context (imports + body) as the query prompt."""
    return (
        f"# Path: {example['file_path']}\n"
        f"{example['import_statement']}\n"
        f"{example['cropped_code']}"
    )


def load_repobench(
    split: str = "cross_file_first",
    level: str | None = "2k",
    n: int = 50,
    seed: int = 42,
    language: str = "python",
) -> list[RepoBenchExample]:
    """
    Load RepoBench-C examples from HuggingFace.

    Parameters
    ----------
    split : "cross_file_first" | "cross_file_random" | "in_file"
        cross_file_first is hardest (no prior in-file hints for the imported symbol).
    level : "2k" | "4k" | "8k" | "all" | None
        Token budget bucket. Use "all" or None to disable level filtering.
    n : int
        Number of examples to load.
    language : "python" | "java"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace datasets not installed. "
            "Run: pip install datasets"
        )

    dataset_id = (
        "tianyang/repobench_python_v1.1"
        if language == "python"
        else "tianyang/repobench_java_v1.1"
    )

    console.print(f"  [dim]Loading RepoBench-C ({language}, {split}, level={level}) from {dataset_id}[/dim]")

    try:
        ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load RepoBench: {exc}") from exc

    # Filter to requested level unless explicitly disabled
    if level in (None, "", "all"):
        ds_level = ds
    else:
        ds_level = ds.filter(lambda x: x["level"] == level)
        if len(ds_level) == 0:
            console.print(f"  [yellow]Warning: no examples found at level={level}, using all levels[/yellow]")
            ds_level = ds

    # Deterministic shuffle and subsample
    ds_level = ds_level.shuffle(seed=seed)
    n_take = min(n, len(ds_level))
    subset = ds_level.select(range(n_take))

    examples = []
    for i, row in enumerate(subset):
        cross_file = _format_cross_file(row)
        in_file = _format_in_file(row)
        examples.append(RepoBenchExample(
            id=f"{row['repo_name']}/{row['file_path']}#{i}",
            repo_name=row["repo_name"],
            file_path=row["file_path"],
            cross_file_context=cross_file,
            in_file_context=in_file,
            next_line=row["next_line"],
            gold_snippet_index=row.get("gold_snippet_index", -1),
            token_num=row.get("token_num", 0),
            level=row.get("level", level),
            split=split,
        ))

    console.print(f"  [dim]Loaded {len(examples)} examples[/dim]")
    return examples


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_completion(generated: str, gold: str) -> dict:
    """
    Grade a next-line completion against the gold next_line.

    exact_match  : bool  — stripped strings match exactly
    edit_sim     : float — difflib SequenceMatcher ratio (0.0–1.0)
    prefix_match : bool  — generated starts with gold (partial credit for long lines)
    """
    gen = generated.rstrip()
    ref = gold.rstrip()

    exact = gen == ref
    edit_sim = difflib.SequenceMatcher(None, gen, ref).ratio()
    prefix = ref.startswith(gen.lstrip()) if gen.strip() else False

    return {
        "exact_match": exact,
        "edit_sim": round(edit_sim, 4),
        "prefix_match": prefix,
    }


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class RepoBenchRecord:
    model_key: str
    repo_name: str
    file_path: str
    split: str
    level: str
    gold_snippet_index: int
    injection_layer: int
    rank: int

    # Token counts
    baseline_input_tokens: int
    vector_input_tokens: int
    vector_f_input_tokens: int
    vector_f_sum_input_tokens: int
    matrix_input_tokens: int
    matrix_f_input_tokens: int

    # Input token reduction vs baseline
    input_reduction_vector: float
    input_reduction_vector_f: float
    input_reduction_vector_f_sum: float
    input_reduction_matrix: float
    input_reduction_matrix_f: float

    # Edit similarity scores
    baseline_edit_sim: float
    vector_edit_sim: float
    vector_f_edit_sim: float
    vector_f_sum_edit_sim: float
    matrix_edit_sim: float
    matrix_f_edit_sim: float

    # Exact match
    baseline_exact_match: bool
    vector_exact_match: bool
    vector_f_exact_match: bool
    vector_f_sum_exact_match: bool
    matrix_exact_match: bool
    matrix_f_exact_match: bool

    # Delta edit_sim vs baseline
    delta_vector: float
    delta_vector_f: float
    delta_vector_f_sum: float
    delta_matrix: float
    delta_matrix_f: float

    # F contribution
    f_contrib_vector: float   # vector_f_edit_sim - vector_edit_sim
    f_contrib_matrix: float   # matrix_f_edit_sim - matrix_edit_sim
    sum_vs_code_f: float      # vector_f_sum_edit_sim - vector_f_edit_sim

    # Generated outputs
    baseline_output: str
    vector_output: str
    vector_f_output: str
    vector_f_sum_output: str
    matrix_output: str
    matrix_f_output: str
    gold_next_line: str
    fact_block: str
    summary_block: str

    elapsed_baseline_s: float
    elapsed_vector_s: float
    elapsed_vector_f_s: float
    elapsed_vector_f_sum_s: float
    elapsed_matrix_s: float
    elapsed_matrix_f_s: float


# ---------------------------------------------------------------------------
# F-block helpers
# ---------------------------------------------------------------------------

def _build_code_fblock(cross_file: str, in_file_tail: str) -> str:
    """
    Build a code F block using extract_facts_code.
    Uses the tail of in_file_context as the BM25 query (the prediction context).
    """
    # Last 300 chars of in-file context = the immediate prediction context
    query = in_file_tail[-300:] if len(in_file_tail) > 300 else in_file_tail
    return extract_facts_code(cross_file, query)


def _build_summary_fblock(
    wrapper: MLXModelWrapper,
    cross_file: str,
    max_new_tokens: int = 200,
) -> tuple[str, int]:
    """
    Generate a model-written structured summary of the cross-file context.
    Returns (summary_text, n_tokens_generated).
    """
    prompt = build_summary_prompt(cross_file)
    summary, n_input = generate_baseline_qa(
        wrapper, prompt, max_new_tokens, stop_strings=("\n\n",)
    )
    return f"Summary:\n{summary.strip()}", n_input


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

def run_example(
    wrapper: MLXModelWrapper,
    example: RepoBenchExample,
    model_key: str,
    injection_layer: int,
    rank: int = 8,
    scale: float = 1.0,
    max_new_tokens: int = 60,
    conditions: tuple[str, ...] = ALL_CONDITIONS,
    verbose: bool = False,
) -> RepoBenchRecord:
    """
    Run all conditions for one RepoBench-C example.

    The cross-file context is the "context" that gets compressed.
    The in-file context is the "query" that stays in the token stream.
    The model predicts the next line.
    """
    cross_file = example.cross_file_context
    in_file = example.in_file_context
    gold = example.next_line

    run_baseline      = "baseline"      in conditions
    run_vector        = "vector"        in conditions
    run_vector_f      = "vector_f"      in conditions
    run_vector_f_sum  = "vector_f_sum"  in conditions
    run_matrix        = "matrix"        in conditions
    run_matrix_f      = "matrix_f"      in conditions

    # F blocks (built once, reused across conditions)
    fact_block = ""
    summary_block = ""
    if run_vector_f or run_matrix_f:
        fact_block = _build_code_fblock(cross_file, in_file)

    n_summary_setup = 0
    if run_vector_f_sum:
        summary_block, n_summary_setup = _build_summary_fblock(
            wrapper, cross_file, max_new_tokens=200
        )

    # Baseline prompt: cross-file + in-file, then model continues
    full_prompt = f"{cross_file}\n\n{in_file}"

    # Query prompts (what the model sees when injection carries the context)
    q_prompt_plain = in_file
    q_prompt_f = f"{fact_block}\n{in_file}" if fact_block else in_file
    q_prompt_sum = f"{summary_block}\n{in_file}" if summary_block else in_file

    # --- Baseline ---
    if run_baseline:
        t0 = time.time()
        baseline_text, n_baseline_input = generate_baseline_qa(
            wrapper, full_prompt, max_new_tokens, stop_strings=COMPLETION_STOP
        )
        elapsed_baseline = time.time() - t0
        n_baseline_output = len(wrapper.encode(baseline_text))
    else:
        baseline_text, n_baseline_input, n_baseline_output, elapsed_baseline = "", 0, 0, 0.0

    # --- Extract context vector (shared by vector, vector_f, vector_f_sum) ---
    if run_vector or run_vector_f or run_vector_f_sum or run_matrix or run_matrix_f:
        ctx_vector, _ = extract_context_state(
            wrapper, cross_file, injection_layer, pool="mean"
        )
    else:
        ctx_vector = None

    # --- Vector (no F block) ---
    if run_vector:
        t0 = time.time()
        vector_text, n_vector_input = generate_with_context_injection(
            wrapper, q_prompt_plain, ctx_vector,
            layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
            stop_strings=COMPLETION_STOP,
        )
        elapsed_vector = time.time() - t0
    else:
        vector_text, n_vector_input, elapsed_vector = "", 0, 0.0

    # --- Vector + code F block ---
    if run_vector_f:
        t0 = time.time()
        vector_f_text, n_vector_f_input = generate_with_context_injection(
            wrapper, q_prompt_f, ctx_vector,
            layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
            stop_strings=COMPLETION_STOP,
        )
        elapsed_vector_f = time.time() - t0
    else:
        vector_f_text, n_vector_f_input, elapsed_vector_f = "", 0, 0.0

    # --- Vector + model summary F block ---
    if run_vector_f_sum:
        t0 = time.time()
        vector_f_sum_text, n_vector_f_sum_input = generate_with_context_injection(
            wrapper, q_prompt_sum, ctx_vector,
            layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
            stop_strings=COMPLETION_STOP,
        )
        elapsed_vector_f_sum = time.time() - t0
    else:
        vector_f_sum_text, n_vector_f_sum_input, elapsed_vector_f_sum = "", 0, 0.0

    # --- Matrix: SVD low-rank projection ---
    if run_matrix or run_matrix_f:
        A, B, S_r, S_full, _ = extract_context_matrix(
            wrapper, cross_file, injection_layer, rank=rank
        )
    else:
        A, B = None, None

    if run_matrix:
        t0 = time.time()
        q_ids = wrapper.encode(q_prompt_plain)
        matrix_ids = generate_with_matrix_hook(
            wrapper,
            token_ids=q_ids,
            layer_matrices={injection_layer: (A, B)},
            mode="inject",
            scale=scale,
            max_new_tokens=max_new_tokens,
            broadcast=True,
        )
        matrix_text = truncate_at_stop(_ids_to_str(wrapper, matrix_ids), COMPLETION_STOP)
        elapsed_matrix = time.time() - t0
        n_matrix_input = len(q_ids)
    else:
        matrix_text, n_matrix_input, elapsed_matrix = "", 0, 0.0

    if run_matrix_f:
        t0 = time.time()
        qf_ids = wrapper.encode(q_prompt_f)
        matrix_f_ids = generate_with_matrix_hook(
            wrapper,
            token_ids=qf_ids,
            layer_matrices={injection_layer: (A, B)},
            mode="inject",
            scale=scale,
            max_new_tokens=max_new_tokens,
            broadcast=True,
        )
        matrix_f_text = truncate_at_stop(_ids_to_str(wrapper, matrix_f_ids), COMPLETION_STOP)
        elapsed_matrix_f = time.time() - t0
        n_matrix_f_input = len(qf_ids)
    else:
        matrix_f_text, n_matrix_f_input, elapsed_matrix_f = "", 0, 0.0

    # --- Grade ---
    def _g(text): return grade_completion(text, gold)

    g_bl  = _g(baseline_text)
    g_vec = _g(vector_text)
    g_vf  = _g(vector_f_text)
    g_vfs = _g(vector_f_sum_text)
    g_mx  = _g(matrix_text)
    g_mxf = _g(matrix_f_text)

    # --- Token reduction ---
    def _red(n_inj):
        return round(1.0 - n_inj / max(n_baseline_input, 1), 4)

    if verbose:
        console.print(f"    gold:         {gold[:80]!r}")
        console.print(f"    baseline:     {baseline_text[:80]!r}  [{g_bl['edit_sim']:.2f}]")
        console.print(f"    vector:       {vector_text[:80]!r}  [{g_vec['edit_sim']:.2f}]")
        console.print(f"    vector_f:     {vector_f_text[:80]!r}  [{g_vf['edit_sim']:.2f}]")
        console.print(f"    vector_f_sum: {vector_f_sum_text[:80]!r}  [{g_vfs['edit_sim']:.2f}]")
        console.print(f"    matrix:       {matrix_text[:80]!r}  [{g_mx['edit_sim']:.2f}]")
        console.print(f"    matrix_f:     {matrix_f_text[:80]!r}  [{g_mxf['edit_sim']:.2f}]")
        console.print(f"    token_reduction (vector): {_red(n_vector_input):.1%}")

    bl_es  = g_bl["edit_sim"]
    vec_es = g_vec["edit_sim"]
    vf_es  = g_vf["edit_sim"]
    vfs_es = g_vfs["edit_sim"]
    mx_es  = g_mx["edit_sim"]
    mxf_es = g_mxf["edit_sim"]

    return RepoBenchRecord(
        model_key=model_key,
        repo_name=example.repo_name,
        file_path=example.file_path,
        split=example.split,
        level=example.level,
        gold_snippet_index=example.gold_snippet_index,
        injection_layer=injection_layer,
        rank=rank,

        baseline_input_tokens=n_baseline_input,
        vector_input_tokens=n_vector_input,
        vector_f_input_tokens=n_vector_f_input,
        vector_f_sum_input_tokens=n_vector_f_sum_input,
        matrix_input_tokens=n_matrix_input,
        matrix_f_input_tokens=n_matrix_f_input,

        input_reduction_vector=_red(n_vector_input),
        input_reduction_vector_f=_red(n_vector_f_input),
        input_reduction_vector_f_sum=_red(n_vector_f_sum_input),
        input_reduction_matrix=_red(n_matrix_input),
        input_reduction_matrix_f=_red(n_matrix_f_input),

        baseline_edit_sim=bl_es,
        vector_edit_sim=vec_es,
        vector_f_edit_sim=vf_es,
        vector_f_sum_edit_sim=vfs_es,
        matrix_edit_sim=mx_es,
        matrix_f_edit_sim=mxf_es,

        baseline_exact_match=g_bl["exact_match"],
        vector_exact_match=g_vec["exact_match"],
        vector_f_exact_match=g_vf["exact_match"],
        vector_f_sum_exact_match=g_vfs["exact_match"],
        matrix_exact_match=g_mx["exact_match"],
        matrix_f_exact_match=g_mxf["exact_match"],

        delta_vector=round(vec_es - bl_es, 4),
        delta_vector_f=round(vf_es - bl_es, 4),
        delta_vector_f_sum=round(vfs_es - bl_es, 4),
        delta_matrix=round(mx_es - bl_es, 4),
        delta_matrix_f=round(mxf_es - bl_es, 4),

        f_contrib_vector=round(vf_es - vec_es, 4),
        f_contrib_matrix=round(mxf_es - mx_es, 4),
        sum_vs_code_f=round(vfs_es - vf_es, 4),

        baseline_output=baseline_text.rstrip()[:200],
        vector_output=vector_text.rstrip()[:200],
        vector_f_output=vector_f_text.rstrip()[:200],
        vector_f_sum_output=vector_f_sum_text.rstrip()[:200],
        matrix_output=matrix_text.rstrip()[:200],
        matrix_f_output=matrix_f_text.rstrip()[:200],
        gold_next_line=gold.rstrip()[:200],
        fact_block=fact_block[:300],
        summary_block=summary_block[:300],

        elapsed_baseline_s=round(elapsed_baseline, 2),
        elapsed_vector_s=round(elapsed_vector, 2),
        elapsed_vector_f_s=round(elapsed_vector_f, 2),
        elapsed_vector_f_sum_s=round(elapsed_vector_f_sum, 2),
        elapsed_matrix_s=round(elapsed_matrix, 2),
        elapsed_matrix_f_s=round(elapsed_matrix_f, 2),
    )


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    model_cfg: dict,
    examples: list[RepoBenchExample],
    rank: int = 8,
    scale: float = 1.0,
    max_new_tokens: int = 60,
    layer_override: int | None = None,
    conditions: tuple[str, ...] = ALL_CONDITIONS,
    done_ids: set[str] | None = None,
    verbose: bool = False,
) -> list[RepoBenchRecord]:
    hf_id = model_cfg["hf_id"]
    console.rule(f"[bold cyan]{model_key}[/bold cyan]")
    console.print(f"  HF ID: {hf_id}")
    console.print(f"  Conditions: {list(conditions)}")

    try:
        wrapper = load_model(hf_id)
    except Exception as exc:
        console.print(f"  [red]Failed to load {hf_id}: {exc}[/red]")
        return []

    if layer_override is not None:
        injection_layer = layer_override
    else:
        injection_layer = select_layer_heuristic(wrapper, hf_id)
    console.print(f"  Injection layer: {injection_layer} / {wrapper.n_layers}")

    records: list[RepoBenchRecord] = []
    skipped = 0

    for i, ex in enumerate(examples):
        run_key = f"{model_key}:{ex.id}"
        if done_ids and run_key in done_ids:
            skipped += 1
            continue
        console.print(
            f"\n  [{i+1}/{len(examples)}] {ex.repo_name[:30]}  "
            f"(level={ex.level}  tok={ex.token_num})"
        )
        try:
            rec = run_example(
                wrapper, ex,
                model_key=model_key,
                injection_layer=injection_layer,
                rank=rank,
                scale=scale,
                max_new_tokens=max_new_tokens,
                conditions=conditions,
                verbose=verbose,
            )
            records.append(rec)
        except Exception as exc:
            console.print(f"  [red]ERROR: {exc}[/red]")

    if skipped:
        console.print(f"  [dim]Skipped {skipped} already-done examples[/dim]")

    _print_summary(model_key, records)

    del wrapper
    gc.collect()
    clear_backend_cache()

    return records


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _avg(records: list[RepoBenchRecord], attr: str) -> float:
    vals = [getattr(r, attr) for r in records]
    return sum(vals) / len(vals) if vals else 0.0


def _em_rate(records: list[RepoBenchRecord], attr: str) -> float:
    return sum(1 for r in records if getattr(r, attr)) / len(records) if records else 0.0


def _print_summary(model_key: str, records: list[RepoBenchRecord]) -> None:
    if not records:
        return

    n = len(records)
    bl  = _avg(records, "baseline_edit_sim")
    vec = _avg(records, "vector_edit_sim")
    vf  = _avg(records, "vector_f_edit_sim")
    vfs = _avg(records, "vector_f_sum_edit_sim")
    mx  = _avg(records, "matrix_edit_sim")
    mxf = _avg(records, "matrix_f_edit_sim")

    def _c(v, ref): return "green" if v >= ref - 0.02 else "yellow" if v >= ref - 0.10 else "red"

    console.rule(f"[bold]{model_key}  N={n}[/bold]")

    table = Table(
        title="RepoBench-C — Edit Similarity vs Baseline",
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    table.add_column("Condition",   style="cyan")
    table.add_column("EditSim",     justify="right")
    table.add_column("EM%",         justify="right")
    table.add_column("Δ",           justify="right")
    table.add_column("Tok Reduction", justify="right")

    def _row(label, es, em_attr, delta, red_attr):
        col = _c(es, bl)
        d_col = "green" if delta > 0.005 else "red" if delta < -0.005 else "white"
        table.add_row(
            label,
            f"[{col}]{es:.3f}[/{col}]",
            f"{_em_rate(records, em_attr):.1%}",
            f"[{d_col}]{delta:+.3f}[/{d_col}]",
            f"{_avg(records, red_attr):.1%}" if red_attr else "—",
        )

    table.add_row("Baseline",     f"{bl:.3f}", f"{_em_rate(records, 'baseline_exact_match'):.1%}", "—", "—")
    _row("Vector",        vec, "vector_exact_match",        vec - bl, "input_reduction_vector")
    _row("Vector+F",      vf,  "vector_f_exact_match",      vf  - bl, "input_reduction_vector_f")
    _row("Vector+F_sum",  vfs, "vector_f_sum_exact_match",  vfs - bl, "input_reduction_vector_f_sum")
    _row("Matrix",        mx,  "matrix_exact_match",        mx  - bl, "input_reduction_matrix")
    _row("Matrix+F",      mxf, "matrix_f_exact_match",      mxf - bl, "input_reduction_matrix_f")

    console.print(table)
    console.print(
        f"  F contribution → vec: [{_c(vf, vec)}]{vf-vec:+.3f}[/{_c(vf, vec)}]"
        f"  summary vs code_f: [{_c(vfs, vf)}]{vfs-vf:+.3f}[/{_c(vfs, vf)}]"
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    records: list[RepoBenchRecord], run_id: str
) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path  = RESULTS_DIR / f"{run_id}.csv"
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
        description="RepoBench-C Latent Context Compression Benchmark"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_MATRIX.keys()),
        choices=list(MODEL_MATRIX.keys()),
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Examples per model"
    )
    parser.add_argument(
        "--split",
        choices=["cross_file_first", "cross_file_random", "in_file"],
        default="cross_file_first",
        help="cross_file_first (hardest) tests cross-file knowledge only available in the compressed context",
    )
    parser.add_argument(
        "--level",
        default="2k",
        choices=["2k", "4k", "8k", "12k", "16k"],
        help="Token budget bucket — use 2k for 4k-context models",
    )
    parser.add_argument(
        "--language",
        choices=["python", "java"],
        default="python",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=60, dest="max_tokens")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--conditions", nargs="+",
        default=list(ALL_CONDITIONS),
        choices=list(ALL_CONDITIONS),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--run-id", default=None, dest="run_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help="Resume a previous run — skip already-completed examples",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"repobench_{int(time.time())}"

    done_ids: set[str] = set()
    if args.resume:
        resume_path = RESULTS_DIR / f"{args.resume}.csv"
        if resume_path.exists():
            with open(resume_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_ids.add(f"{row['model_key']}:{row['repo_name']}/{row['file_path']}#?")
            console.print(f"  [dim]Resuming: {len(done_ids)} examples already done[/dim]")
            run_id = args.resume

    console.rule("[bold magenta]RepoBench-C  Latent Context Compression[/bold magenta]")
    console.print(f"  Models     : {args.models}")
    console.print(f"  Split      : {args.split}")
    console.print(f"  Level      : {args.level}")
    console.print(f"  Language   : {args.language}")
    console.print(f"  N          : {args.n}")
    console.print(f"  Rank       : {args.rank}")
    console.print(f"  Conditions : {args.conditions}")
    console.print(f"  Run ID     : {run_id}")
    console.print()
    console.print("  Key insight: cross-file context is COMPRESSED via injection.")
    console.print("  In-file context stays in the token stream as the query.")
    console.print("  This models the Claude Code / coding assistant use case.")

    examples = load_repobench(
        split=args.split,
        level=args.level,
        n=args.n,
        seed=args.seed,
        language=args.language,
    )
    console.print(f"  Total examples: {len(examples)}")

    all_records: list[RepoBenchRecord] = []

    for model_key in args.models:
        model_cfg = MODEL_MATRIX[model_key]
        records = run_model(
            model_key=model_key,
            model_cfg=model_cfg,
            examples=examples,
            rank=args.rank,
            scale=args.scale,
            max_new_tokens=args.max_tokens,
            layer_override=args.layer,
            conditions=tuple(args.conditions),
            done_ids=done_ids or None,
            verbose=args.verbose,
        )
        all_records.extend(records)
        save_results(all_records, f"{run_id}_partial")

    if all_records:
        save_results(all_records, run_id)

    total = len(all_records)
    if total:
        console.rule("[bold]Final Summary[/bold]")
        bl_avg  = _avg(all_records, "baseline_edit_sim")
        vfs_avg = _avg(all_records, "vector_f_sum_edit_sim")
        red_avg = _avg(all_records, "input_reduction_vector_f_sum")
        console.print(
            f"  Baseline EditSim:        {bl_avg:.3f}\n"
            f"  Vector+F_sum EditSim:    {vfs_avg:.3f}  (Δ{vfs_avg-bl_avg:+.3f})\n"
            f"  Avg token reduction:     {red_avg:.1%}"
        )

    return 0 if total > 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
