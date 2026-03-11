"""
RSCE Ablation Benchmark — Part 2

3-way ablation to measure the contribution of the fact block F:

  Condition 1 — Baseline  : [ctx + P]       full context in token stream
  Condition 2 — RSCE only : [C + P]         context vector injection, no F
  Condition 3 — RSCE + F  : [C + F + P]     context vector + fact block

Two fact extraction strategies for h(D):
  - NER  : regex-based named entity + number extraction (practical, no extra model)
  - Oracle: sentences from ctx containing a gold answer string (upper bound)

The F1 delta between conditions 2 and 3 measures how much F contributes.
The F1 delta between condition 3 and baseline measures the residual gap.

Usage:
  source .venv/bin/activate
  python benchmark_ablation.py
  python benchmark_ablation.py --models llama3-8b --n 10 --fact-mode ner
  python benchmark_ablation.py --fact-mode oracle        # upper bound experiment
  python benchmark_ablation.py --fact-mode both          # run both (default)
  python benchmark_ablation.py --layer 14                # override injection layer
  python benchmark_ablation.py --condition rsce_f        # only run RSCE+F pass
  python benchmark_ablation.py --resume ablation_123     # skip already-done examples
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import mlx.core as mx
from rich.console import Console
from rich import box
from rich.table import Table

from backend_mlx import load_model, MLXModelWrapper
from benchmark_datasets import load_benchmark, BenchmarkExample, grade_qa
from context_injector import (
    extract_context_state,
    generate_baseline_qa,
    generate_with_context_injection,
    compute_token_metrics,
)
from layer_selector import select_layer_heuristic
from benchmark_runner import MODEL_MATRIX

console = Console()
RESULTS_DIR = Path(__file__).parent / "ablation_results"


# ---------------------------------------------------------------------------
# h(D): Fact extraction for QA contexts
# ---------------------------------------------------------------------------

def extract_facts_ner(ctx: str, max_facts: int = 15) -> str:
    """
    NER-based h(D): extract named entities and numbers from ctx using regex.

    Targets:
      - Capitalized multi-word sequences (proper nouns / named entities)
      - 4-digit years
      - Standalone numbers
      - Quoted strings

    Returns a compact fact block string, e.g.:
      "Facts: Marie Curie; Nobel Prize; 1903; Warsaw; Paris"
    """
    facts: list[str] = []
    seen: set[str] = set()

    def add(f: str) -> None:
        f = f.strip()
        if f and f.lower() not in seen and len(f) > 1:
            seen.add(f.lower())
            facts.append(f)

    # Capitalized multi-word proper nouns (2-5 words, each capitalized)
    for m in re.finditer(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b', ctx
    ):
        add(m.group(1))

    # Single capitalized words that aren't sentence starters
    # (appear mid-sentence after lowercase word)
    for m in re.finditer(r'(?<=[a-z,;]\s)([A-Z][a-z]{2,})\b', ctx):
        add(m.group(1))

    # 4-digit years
    for m in re.finditer(r'\b(1[0-9]{3}|20[0-2][0-9])\b', ctx):
        add(m.group(1))

    # Numbers with units or standalone numbers
    for m in re.finditer(r'\b(\d+(?:\.\d+)?(?:\s*(?:km|m|kg|mph|°|percent|%))?)\b', ctx):
        val = m.group(1).strip()
        if val.isdigit() and len(val) == 1:
            continue  # skip single digits
        add(val)

    # Quoted strings
    for m in re.finditer(r'"([^"]{3,40})"', ctx):
        add(m.group(1))

    # Deduplicate while preserving order, trim to max_facts
    final = facts[:max_facts]
    if not final:
        return ""
    return "Facts: " + "; ".join(final)


def extract_facts_oracle(ctx: str, gold_answers: list[str]) -> str:
    """
    Oracle h(D): extract sentences from ctx that contain a gold answer string.

    This is an upper bound — it uses knowledge of the correct answer to
    select the most relevant sentences. Shows the ceiling of what F can achieve.
    """
    sentences = re.split(r'(?<=[.!?])\s+', ctx)
    relevant: list[str] = []

    for sent in sentences:
        for gold in gold_answers:
            if gold.lower() in sent.lower():
                relevant.append(sent.strip())
                break

    if not relevant:
        # Fallback: first 2 sentences (at least some context)
        relevant = sentences[:2]

    # Keep it short — max 3 sentences
    block = " ".join(relevant[:3])
    # Trim to ~200 chars
    if len(block) > 200:
        block = block[:200].rsplit(" ", 1)[0] + "..."
    return f"Context: {block}"


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class AblationRecord:
    model_key: str
    task: str
    example_id: str
    injection_layer: int
    n_context_words: int
    fact_mode: str          # "ner" | "oracle"
    fact_block: str         # the actual F string used

    # Token counts
    baseline_input_tokens: int
    rsce_only_input_tokens: int
    rsce_f_input_tokens: int

    # Quality
    baseline_exact_match: bool
    rsce_only_exact_match: bool
    rsce_f_exact_match: bool

    baseline_f1: float
    rsce_only_f1: float
    rsce_f_f1: float

    # Derived
    f_contribution_f1: float      # rsce_f_f1 - rsce_only_f1
    residual_gap_f1: float        # baseline_f1 - rsce_f_f1
    input_token_reduction: float  # 1 - rsce_only_input / baseline_input

    baseline_answer: str
    rsce_only_answer: str
    rsce_f_answer: str
    gold_answers: str


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

ALL_CONDITIONS = ("baseline", "rsce_only", "rsce_f")


def run_example_ablation(
    wrapper: MLXModelWrapper,
    example: BenchmarkExample,
    model_key: str,
    injection_layer: int,
    fact_mode: str,
    scale: float = 1.0,
    max_new_tokens: int = 80,
    conditions: tuple[str, ...] = ALL_CONDITIONS,
    verbose: bool = False,
) -> list[AblationRecord]:
    """
    Run 3-way ablation for one example.

    If fact_mode == "both", returns two records (one NER, one oracle).
    Otherwise returns one record.
    """
    modes = ["ner", "oracle"] if fact_mode == "both" else [fact_mode]
    records = []

    run_baseline  = "baseline"  in conditions
    run_rsce_only = "rsce_only" in conditions
    run_rsce_f    = "rsce_f"    in conditions

    # --- Condition 1: Baseline ---
    if run_baseline:
        baseline_text, n_baseline = generate_baseline_qa(
            wrapper, example.full_prompt(), max_new_tokens
        )
        baseline_grades = grade_qa(baseline_text, example.gold_answers)
    else:
        baseline_text, n_baseline = "", 0
        baseline_grades = {"exact_match": False, "f1": 0.0}

    # --- Condition 2: RSCE only ---
    if run_rsce_only or run_rsce_f:
        ctx_v, _ = extract_context_state(wrapper, example.context, injection_layer)
    else:
        ctx_v = None

    if run_rsce_only:
        rsce_only_text, n_rsce_only = generate_with_context_injection(
            wrapper, example.question_prompt(), ctx_v,
            layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
        )
        rsce_only_grades = grade_qa(rsce_only_text, example.gold_answers)
        token_metrics = compute_token_metrics(
            n_baseline, n_rsce_only,
            len(wrapper.encode(baseline_text)) if baseline_text else 0,
            len(wrapper.encode(rsce_only_text)),
        )
    else:
        rsce_only_text, n_rsce_only = "", 0
        rsce_only_grades = {"exact_match": False, "f1": 0.0}
        token_metrics = {"input_token_reduction": 0.0}

    if verbose:
        if run_baseline:
            console.print(f"    [dim]baseline:   {baseline_text.strip()[:80]}[/dim]")
        if run_rsce_only:
            console.print(f"    [dim]rsce-only:  {rsce_only_text.strip()[:80]}[/dim]")
        console.print(f"    [dim]gold:       {example.gold_answers}[/dim]")

    # --- Condition 3: RSCE + F (once per mode) ---
    for mode in modes:
        if run_rsce_f:
            if mode == "ner":
                fact_block = extract_facts_ner(example.context)
            else:
                fact_block = extract_facts_oracle(example.context, example.gold_answers)

            rsce_f_prompt = f"{fact_block}\n{example.question_prompt()}" if fact_block else example.question_prompt()
            rsce_f_text, n_rsce_f = generate_with_context_injection(
                wrapper, rsce_f_prompt, ctx_v,
                layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
            )
            rsce_f_grades = grade_qa(rsce_f_text, example.gold_answers)

            if verbose:
                console.print(f"    [dim]rsce+F ({mode}): {rsce_f_text.strip()[:80]}[/dim]")
                console.print(f"    [dim]fact block: {fact_block[:80]}[/dim]")
        else:
            fact_block = ""
            rsce_f_text, n_rsce_f = "", 0
            rsce_f_grades = {"exact_match": False, "f1": 0.0}

        records.append(AblationRecord(
            model_key=model_key,
            task=example.task,
            example_id=example.id,
            injection_layer=injection_layer,
            n_context_words=example.context_word_len,
            fact_mode=mode,
            fact_block=fact_block[:200],
            baseline_input_tokens=n_baseline,
            rsce_only_input_tokens=n_rsce_only,
            rsce_f_input_tokens=n_rsce_f,
            baseline_exact_match=baseline_grades["exact_match"],
            rsce_only_exact_match=rsce_only_grades["exact_match"],
            rsce_f_exact_match=rsce_f_grades["exact_match"],
            baseline_f1=baseline_grades["f1"],
            rsce_only_f1=rsce_only_grades["f1"],
            rsce_f_f1=rsce_f_grades["f1"],
            f_contribution_f1=round(rsce_f_grades["f1"] - rsce_only_grades["f1"], 4),
            residual_gap_f1=round(baseline_grades["f1"] - rsce_f_grades["f1"], 4),
            input_token_reduction=token_metrics["input_token_reduction"],
            baseline_answer=baseline_text.strip()[:120],
            rsce_only_answer=rsce_only_text.strip()[:120],
            rsce_f_answer=rsce_f_text.strip()[:120],
            gold_answers=str(example.gold_answers),
        ))

    return records


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model_ablation(
    model_key: str,
    model_cfg: dict,
    examples: list[BenchmarkExample],
    fact_mode: str = "both",
    scale: float = 1.0,
    max_new_tokens: int = 80,
    layer_override: int | None = None,
    conditions: tuple[str, ...] = ALL_CONDITIONS,
    done_ids: set[str] | None = None,
    verbose: bool = False,
) -> list[AblationRecord]:
    hf_id = model_cfg["hf_id"]
    console.rule(f"[bold cyan]Ablation: {model_key}[/bold cyan]")
    console.print(f"  HF ID      : {hf_id}")
    console.print(f"  Fact mode  : {fact_mode}")
    console.print(f"  Conditions : {list(conditions)}")

    try:
        wrapper = load_model(hf_id)
    except Exception as exc:
        console.print(f"  [red]Load failed: {exc}[/red]")
        return []

    if layer_override is not None:
        injection_layer = layer_override
        console.print(f"  Layer f(M): {injection_layer} (override)")
    else:
        injection_layer = select_layer_heuristic(wrapper, hf_id)
        console.print(f"  Layer f(M): {injection_layer} / {wrapper.n_layers}")

    all_records: list[AblationRecord] = []
    skipped = 0

    modes = ["ner", "oracle"] if fact_mode == "both" else [fact_mode]
    for i, ex in enumerate(examples):
        # Check if all modes for this example are already done
        all_done = done_ids and all(
            f"{model_key}:{ex.id}:{mode}" in done_ids for mode in modes
        )
        if all_done:
            skipped += 1
            continue
        console.print(
            f"\n  [{i+1}/{len(examples)}] {ex.task}/{ex.id[:24]} "
            f"({ex.context_word_len}w)"
        )
        try:
            recs = run_example_ablation(
                wrapper, ex,
                model_key=model_key,
                injection_layer=injection_layer,
                fact_mode=fact_mode,
                scale=scale,
                max_new_tokens=max_new_tokens,
                conditions=conditions,
                verbose=verbose,
            )
            all_records.extend(recs)
        except Exception as exc:
            console.print(f"  [red]ERROR: {exc}[/red]")
    if skipped:
        console.print(f"  [dim]Skipped {skipped} already-done examples[/dim]")

    _print_model_ablation_summary(model_key, all_records)

    del wrapper
    gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception:
        pass

    return all_records


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _print_model_ablation_summary(
    model_key: str, records: list[AblationRecord]
) -> None:
    if not records:
        return

    for mode in ["ner", "oracle"]:
        mode_recs = [r for r in records if r.fact_mode == mode]
        if not mode_recs:
            continue

        n = len(mode_recs)
        b_f1  = sum(r.baseline_f1 for r in mode_recs) / n
        r_f1  = sum(r.rsce_only_f1 for r in mode_recs) / n
        rf_f1 = sum(r.rsce_f_f1 for r in mode_recs) / n
        f_contrib = sum(r.f_contribution_f1 for r in mode_recs) / n
        gap = sum(r.residual_gap_f1 for r in mode_recs) / n
        red = sum(r.input_token_reduction for r in mode_recs) / n

        b_em  = sum(1 for r in mode_recs if r.baseline_exact_match) / n
        r_em  = sum(1 for r in mode_recs if r.rsce_only_exact_match) / n
        rf_em = sum(1 for r in mode_recs if r.rsce_f_exact_match) / n

        contrib_color = "green" if f_contrib > 0 else "yellow" if f_contrib == 0 else "red"
        gap_color = "green" if gap < 0.05 else "yellow" if gap < 0.15 else "red"

        console.rule(f"[bold]{model_key} | fact_mode={mode}[/bold]")
        console.print(f"  N examples         : {n}")
        console.print(f"  Input token reduction: {red:.1%}")
        console.print(f"  Baseline     EM/F1 : {b_em:.1%} / {b_f1:.3f}")
        console.print(f"  RSCE only    EM/F1 : {r_em:.1%} / {r_f1:.3f}")
        console.print(f"  RSCE + F     EM/F1 : {rf_em:.1%} / {rf_f1:.3f}")
        console.print(
            f"  F contribution F1  : [{contrib_color}]{f_contrib:+.3f}[/{contrib_color}]"
            f"  (RSCE+F minus RSCE-only)"
        )
        console.print(
            f"  Residual gap   F1  : [{gap_color}]{gap:+.3f}[/{gap_color}]"
            f"  (baseline minus RSCE+F)"
        )


def print_ablation_summary(all_records: list[AblationRecord]) -> None:
    from collections import defaultdict
    console.rule("[bold magenta]ABLATION SUMMARY — F Contribution[/bold magenta]")

    # Group by (model, fact_mode)
    groups: dict[tuple[str, str], list[AblationRecord]] = defaultdict(list)
    for r in all_records:
        groups[(r.model_key, r.fact_mode)].append(r)

    table = Table(
        title="RSCE 3-Way Ablation: Baseline vs RSCE-only vs RSCE+F",
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    table.add_column("Model", style="cyan")
    table.add_column("F mode")
    table.add_column("N", justify="right")
    table.add_column("BL-F1", justify="right")
    table.add_column("C-only F1", justify="right")
    table.add_column("C+F F1", justify="right")
    table.add_column("F contrib", justify="right")
    table.add_column("Gap vs BL", justify="right")
    table.add_column("Tok saved", justify="right")

    for (model_key, mode), recs in sorted(groups.items()):
        n = len(recs)
        b_f1  = sum(r.baseline_f1 for r in recs) / n
        r_f1  = sum(r.rsce_only_f1 for r in recs) / n
        rf_f1 = sum(r.rsce_f_f1 for r in recs) / n
        f_contrib = rf_f1 - r_f1
        gap = b_f1 - rf_f1
        red = sum(r.input_token_reduction for r in recs) / n

        contrib_color = "green" if f_contrib > 0.01 else "yellow" if f_contrib >= 0 else "red"
        gap_color = "green" if gap < 0.05 else "yellow" if gap < 0.15 else "red"

        table.add_row(
            model_key, mode, str(n),
            f"{b_f1:.3f}",
            f"{r_f1:.3f}",
            f"{rf_f1:.3f}",
            f"[{contrib_color}]{f_contrib:+.3f}[/{contrib_color}]",
            f"[{gap_color}]{gap:+.3f}[/{gap_color}]",
            f"{red:.1%}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_ablation(records: list[AblationRecord], run_id: str) -> None:
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

    console.print(f"\n[dim]Saved:\n  {csv_path}\n  {json_path}[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RSCE Ablation: measure F contribution to quality"
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
    parser.add_argument("--n", type=int, default=10, help="Examples per task (default: 10)")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=80, dest="max_tokens")
    parser.add_argument(
        "--fact-mode", choices=["ner", "oracle", "both"], default="both",
        dest="fact_mode",
        help=(
            "ner: regex NER extraction (practical); "
            "oracle: gold-answer-guided extraction (upper bound); "
            "both: run both (default)"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--run-id", default=None, dest="run_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=None,
                        help="Override injection layer for all models (skips heuristic)")
    parser.add_argument(
        "--condition", nargs="+", default=list(ALL_CONDITIONS),
        choices=list(ALL_CONDITIONS), dest="conditions",
        help="Which conditions to run (default: all three)",
    )
    parser.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help="Resume a previous run: load its CSV and skip already-completed examples",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"ablation_{int(time.time())}"

    # Load already-done keys from a prior run
    done_ids: set[str] = set()
    if args.resume:
        resume_path = RESULTS_DIR / f"{args.resume}.csv"
        if not resume_path.exists():
            resume_path = RESULTS_DIR / f"{args.resume}_partial.csv"
        if resume_path.exists():
            with open(resume_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_ids.add(f"{row['model_key']}:{row['example_id']}:{row['fact_mode']}")
            console.print(f"  [dim]Resuming from {resume_path.name}: {len(done_ids)} examples already done[/dim]")
            run_id = args.resume
        else:
            console.print(f"  [yellow]Resume file not found: {resume_path}[/yellow]")

    console.rule("[bold magenta]RSCE Ablation Benchmark — Part 2[/bold magenta]")
    console.print(f"  Models    : {args.models}")
    console.print(f"  Tasks     : {args.tasks}")
    console.print(f"  N/task    : {args.n}")
    console.print(f"  Fact mode : {args.fact_mode}")
    console.print(f"  Run ID    : {run_id}")
    console.print()
    console.print("  Conditions:")
    console.print("    1. Baseline  — [ctx + P]     full context in token stream")
    console.print("    2. RSCE only — [C + P]       vector injection, no fact block")
    console.print("    3. RSCE + F  — [C + F + P]   vector injection + fact block")

    examples = load_benchmark(
        tasks=tuple(args.tasks),
        n_per_task=args.n,
        seed=args.seed,
    )
    console.print(f"\n  Total examples: {len(examples)}")

    all_records: list[AblationRecord] = []

    for model_key in args.models:
        recs = run_model_ablation(
            model_key=model_key,
            model_cfg=MODEL_MATRIX[model_key],
            examples=examples,
            fact_mode=args.fact_mode,
            scale=args.scale,
            max_new_tokens=args.max_tokens,
            layer_override=args.layer,
            conditions=tuple(args.conditions),
            done_ids=done_ids,
            verbose=args.verbose,
        )
        all_records.extend(recs)
        # Checkpoint after each model
        save_ablation(all_records, f"{run_id}_partial")

    print_ablation_summary(all_records)
    save_ablation(all_records, run_id)

    return 0 if all_records else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
