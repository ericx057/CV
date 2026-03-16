"""
Q-only baseline: question + no context, no vector, no fact block.
Loads the same 42 examples from vec_f_n50.csv and runs llama3-8b.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import mlx.core as mx

from backend_mlx import load_model
from benchmark_datasets import load_benchmark, grade_qa
from context_injector import generate_baseline_qa, QA_STOP_STRINGS
from benchmark_runner import MODEL_MATRIX

RESULTS_CSV = Path(__file__).parent / "benchmark_results" / "vec_f_n50.csv"
MODEL_KEY = "llama3-8b"
SEED = 42


def main() -> None:
    # Load the example IDs from the prior run
    prior_ids: set[str] = set()
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            prior_ids.add(row["example_id"])
    print(f"Prior example IDs loaded: {len(prior_ids)}")

    # Load benchmark examples matching those IDs
    examples = load_benchmark(
        tasks=("hotpotqa", "2wikimqa"),
        n_per_task=50,
        seed=SEED,
    )
    examples = [e for e in examples if e.id in prior_ids]
    print(f"Matched examples: {len(examples)}")

    # Load model
    cfg = MODEL_MATRIX[MODEL_KEY]
    print(f"Loading {MODEL_KEY} …")
    wrapper = load_model(cfg["hf_id"])


    # Run Q-only inference
    results = []
    for i, ex in enumerate(examples):
        prompt = ex.question_prompt()          # just the question, no context
        t0 = time.time()
        text, n_toks = generate_baseline_qa(wrapper, prompt, max_new_tokens=80,
                                             stop_strings=QA_STOP_STRINGS)
        elapsed = time.time() - t0
        grades = grade_qa(text, ex.gold_answers)
        results.append({
            "task": ex.task,
            "example_id": ex.id,
            "f1": grades["f1"],
            "em": grades["exact_match"],
            "output": text,
            "gold": ex.gold_answers,
            "elapsed": elapsed,
            "n_input_tokens": n_toks,
        })
        print(
            f"[{i+1:02d}/{len(examples)}] {ex.task} "
            f"F1={grades['f1']:.3f} EM={int(grades['exact_match'])} "
            f"| q: {ex.question[:50]!r} → {text[:40]!r}"
        )

    # Summary
    f1_avg   = sum(r["f1"] for r in results) / len(results)
    em_avg   = sum(r["em"] for r in results) / len(results)
    f1_hot   = [r["f1"] for r in results if r["task"] == "hotpotqa"]
    f1_wiki  = [r["f1"] for r in results if r["task"] == "2wikimqa"]
    em_hot   = [r["em"] for r in results if r["task"] == "hotpotqa"]
    em_wiki  = [r["em"] for r in results if r["task"] == "2wikimqa"]

    print()
    print("=== Q-only baseline results (llama3-8b) ===")
    print(f"  Aggregate   F1={f1_avg:.3f}  EM={em_avg:.3f}")
    print(f"  HotpotQA    F1={sum(f1_hot)/len(f1_hot):.3f}  EM={sum(em_hot)/len(em_hot):.3f}  (n={len(f1_hot)})")
    print(f"  2WikiMQA    F1={sum(f1_wiki)/len(f1_wiki):.3f}  EM={sum(em_wiki)/len(em_wiki):.3f}  (n={len(f1_wiki)})")

    print()
    print("=== Comparison table ===")
    print(f"{'Condition':<20} {'F1':>6} {'EM':>6}")
    print("-" * 35)
    print(f"{'Q-only (new)':<20} {f1_avg:>6.3f} {em_avg:>6.3f}")
    print(f"{'Vec (prior)':<20} {'0.000':>6} {'0.000':>6}")
    print(f"{'Vec+F (prior)':<20} {'0.218':>6} {'0.333':>6}")
    print(f"{'Baseline (prior)':<20} {'0.165':>6} {'0.714':>6}")


if __name__ == "__main__":
    main()
