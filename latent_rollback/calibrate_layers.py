"""
Layer calibration: run accuracy-based sweep to find f(M) for each model.

For each model in the matrix, loads it, runs select_layer_by_accuracy on
a small calibration set, prints the sweep table, and saves the results to
calibration_results.json.

Usage:
  source .venv/bin/activate
  python calibrate_layers.py
  python calibrate_layers.py --models llama3-8b qwen25-7b
  python calibrate_layers.py --n 5 --step 2 --scale 1.0
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from rich.console import Console
from rich import box
from rich.table import Table

from backend_mlx import load_model
from benchmark_datasets import load_benchmark, _hardcoded_fallback
from layer_selector import (
    select_layer_by_accuracy,
    select_layer_heuristic,
    print_accuracy_sweep_table,
)
from benchmark_runner import MODEL_MATRIX

console = Console()
RESULTS_PATH = Path(__file__).parent / "calibration_results.json"


def calibrate_model(
    model_key: str,
    model_cfg: dict,
    examples: list,
    step: int = 2,
    scale: float = 1.0,
    max_new_tokens: int = 40,
) -> dict:
    hf_id = model_cfg["hf_id"]
    console.rule(f"[bold cyan]Calibrating: {model_key}[/bold cyan]")
    console.print(f"  HF ID : {hf_id}")
    console.print(f"  Size  : ~{model_cfg['size_gb']} GB")

    try:
        wrapper = load_model(hf_id)
    except Exception as exc:
        console.print(f"  [red]Load failed: {exc}[/red]")
        return {"model_key": model_key, "error": str(exc)}

    heuristic_layer = select_layer_heuristic(wrapper, hf_id)
    console.print(f"  Heuristic f(M): layer {heuristic_layer}")

    t0 = time.time()
    best_layer, sweep_results = select_layer_by_accuracy(
        wrapper,
        examples,
        step=step,
        scale=scale,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - t0

    print_accuracy_sweep_table(sweep_results, best_layer)

    depth_frac = best_layer / wrapper.n_layers
    console.print(
        f"\n  f(M) = layer {best_layer} / {wrapper.n_layers} "
        f"({depth_frac:.0%} depth)  [heuristic was {heuristic_layer}]"
    )

    record = {
        "model_key": model_key,
        "hf_id": hf_id,
        "n_layers": wrapper.n_layers,
        "d_model": wrapper.d_model,
        "heuristic_layer": heuristic_layer,
        "optimal_layer": best_layer,
        "optimal_depth_fraction": round(depth_frac, 3),
        "n_calibration_examples": len(examples),
        "step": step,
        "scale": scale,
        "elapsed_s": round(elapsed, 1),
        "sweep": sweep_results,
    }

    del wrapper
    gc.collect()
    try:
        mx.metal.clear_cache()
    except Exception:
        pass

    return record


def print_final_table(records: list[dict]) -> None:
    console.rule("[bold magenta]CALIBRATION SUMMARY — f(M) per model[/bold magenta]")
    table = Table(box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Model", style="cyan")
    table.add_column("Layers", justify="right")
    table.add_column("Heuristic", justify="right")
    table.add_column("f(M) Optimal", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("Best F1", justify="right")
    table.add_column("Match", justify="center")

    for r in records:
        if "error" in r:
            table.add_row(r["model_key"], "-", "-", "-", "-", "-", "[red]LOAD ERROR[/red]")
            continue

        heur = r["heuristic_layer"]
        opt = r["optimal_layer"]
        match = "[green]YES[/green]" if heur == opt else f"[yellow]off by {abs(heur-opt)}[/yellow]"

        best_sweep = max(r["sweep"], key=lambda x: x["avg_f1"]) if r["sweep"] else {}

        table.add_row(
            r["model_key"],
            str(r["n_layers"]),
            str(heur),
            str(opt),
            f"{r['optimal_depth_fraction']:.0%}",
            f"{best_sweep.get('avg_f1', 0.0):.4f}",
            match,
        )

    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description="Accuracy-based layer calibration: f(M)")
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_MATRIX.keys()),
        choices=list(MODEL_MATRIX.keys()),
    )
    parser.add_argument("--n", type=int, default=5, help="Calibration examples (default: 5)")
    parser.add_argument("--step", type=int, default=2, help="Layer step size (default: 2)")
    parser.add_argument("--scale", type=float, default=1.0, help="Injection scale")
    parser.add_argument("--max-tokens", type=int, default=40, dest="max_tokens")
    parser.add_argument(
        "--use-longbench", action="store_true", dest="use_longbench",
        help="Download LongBench examples (requires internet). Default: hardcoded fallback.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    console.rule("[bold magenta]Layer Calibration: f(M) Accuracy Sweep[/bold magenta]")

    # Load calibration examples
    if args.use_longbench:
        examples = load_benchmark(
            tasks=("hotpotqa",),
            n_per_task=args.n,
            seed=args.seed,
        )
    else:
        examples = _hardcoded_fallback("hotpotqa") + _hardcoded_fallback("2wikimqa")
        examples = examples[:args.n]

    console.print(f"  Calibration examples: {len(examples)}")
    for ex in examples:
        console.print(f"  [{ex.task}] {ex.question[:60]}  gold={ex.gold_answers}")

    # Run calibration per model
    all_records = []
    for model_key in args.models:
        record = calibrate_model(
            model_key=model_key,
            model_cfg=MODEL_MATRIX[model_key],
            examples=examples,
            step=args.step,
            scale=args.scale,
            max_new_tokens=args.max_tokens,
        )
        all_records.append(record)

        # Save after each model
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_records, f, indent=2)
        console.print(f"  [dim]Saved to {RESULTS_PATH}[/dim]")

    print_final_table(all_records)

    # Print update instructions for benchmark_runner
    console.rule("[bold]UPDATE INSTRUCTIONS[/bold]")
    console.print("To use these optimal layers in benchmark_runner, update _LAYER_FRACTION_TABLE:")
    for r in all_records:
        if "error" not in r:
            frac = r["optimal_depth_fraction"]
            console.print(
                f"  ({r['model_key']!r}, {frac})  "
                f"# {r['optimal_layer']}/{r['n_layers']} layers"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
