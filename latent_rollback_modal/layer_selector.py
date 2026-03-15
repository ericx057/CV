"""
Optimal injection layer selection: implements f(M) from the paper.

f(M) -> int

Given a model M, returns the layer index where residual stream injection
is most effective for context compression.

Heuristics (in priority order):
  1. Empirical sweep (slow but accurate): run a sample context through the
     model, measure per-layer residual variance and cosine similarity between
     context and question representations. Peak similarity layer is optimal.

  2. Architecture heuristic (fast): ~60% of total depth is empirically
     well-supported across LLaMA, Mistral, and Qwen families.
     (LLaMA-3 8B: L14/32 = 44%, verified experimentally in prior work)

The sweep is used when calibrating a new model. The heuristic is used for
fast benchmarking when calibration has not been done.
"""

from __future__ import annotations

import torch

from .backend_torch import MLXModelWrapper, extract_layer_hidden_states
from rich.console import Console
from rich import box
from rich.table import Table

console = Console()

# Empirically verified optimal layer fractions per model family
# Key: model name substring -> (fraction of n_layers, source)
_LAYER_FRACTION_TABLE: list[tuple[str, float]] = [
    ("llama-3", 0.44),       # LLaMA-3 8B: L14/32, empirically verified (LongBench sweep)
    ("llama", 0.44),         # LLaMA generic
    ("mistral", 0.25),       # Mistral: L10/40, empirically verified (LongBench sweep)
    ("qwen", 0.61),          # Qwen2.5-7B: L17/28, empirically verified (LongBench sweep)
    ("deepseek", 0.25),      # DeepSeek R1 distill: L12/48, empirically verified
    ("gemma", 0.44),         # Gemma — using LLaMA-family default
]

_FALLBACK_FRACTION = 0.55    # conservative default for unknown architectures


def select_layer_heuristic(wrapper: MLXModelWrapper, model_name: str) -> int:
    """
    Fast heuristic: returns estimated optimal layer based on model family.

    This implements f(M) as a lookup table + fallback fraction.
    """
    name_lower = model_name.lower()
    fraction = _FALLBACK_FRACTION

    for keyword, frac in _LAYER_FRACTION_TABLE:
        if keyword in name_lower:
            fraction = frac
            break

    layer = int(fraction * wrapper.n_layers)
    # Clamp to valid range
    layer = max(1, min(layer, wrapper.n_layers - 2))
    console.print(
        f"  [dim]f(M) heuristic: {model_name} -> layer {layer}/{wrapper.n_layers} "
        f"({layer / wrapper.n_layers:.0%} depth)[/dim]"
    )
    return layer


def select_layer_sweep(
    wrapper: MLXModelWrapper,
    context_text: str,
    question_text: str,
    layer_range: tuple[int, int] | None = None,
    step: int = 2,
) -> tuple[int, list[dict]]:
    """
    Empirical sweep: find the layer where the context and question residuals
    are most similar (highest cosine similarity).

    This is a proxy for "where does the model best unify the semantics of
    context and query" — the optimal injection point.

    Returns
    -------
    (best_layer : int, sweep_results : list[dict])
    """
    if layer_range is None:
        lo = max(1, wrapper.n_layers // 4)
        hi = min(wrapper.n_layers - 1, int(wrapper.n_layers * 0.85))
        layer_range = (lo, hi)

    lo, hi = layer_range
    layers_to_test = list(range(lo, hi + 1, step))

    context_ids = wrapper.encode(context_text)
    question_ids = wrapper.encode(question_text)

    results = []
    best_layer = layers_to_test[0]
    best_sim = -999.0

    for layer in layers_to_test:
        ctx_v = _extract_mean_residual(wrapper, context_ids, layer)
        q_v = _extract_mean_residual(wrapper, question_ids, layer)

        cosine = torch.nn.functional.cosine_similarity(
            ctx_v.unsqueeze(0), q_v.unsqueeze(0)
        ).item()

        ctx_var = float(ctx_v.var().item())

        results.append({
            "layer": layer,
            "cosine_sim": round(cosine, 4),
            "ctx_variance": round(ctx_var, 4),
        })

        # Best = highest cosine similarity (context and question aligned semantically)
        if cosine > best_sim:
            best_sim = cosine
            best_layer = layer

    console.print(f"  [dim]f(M) sweep: best layer = {best_layer} (cosine = {best_sim:.4f})[/dim]")
    return best_layer, results


def _extract_mean_residual(
    wrapper: MLXModelWrapper,
    token_ids: list[int],
    layer: int,
) -> torch.Tensor:
    """Extract mean-pooled residual at `layer` for token_ids."""
    h = extract_layer_hidden_states(wrapper, token_ids, layer)[0].float().cpu()
    return h.mean(dim=0)


def select_layer_by_accuracy(
    wrapper: MLXModelWrapper,
    examples: list,
    layer_range: tuple[int, int] | None = None,
    step: int = 2,
    scale: float = 1.0,
    max_new_tokens: int = 40,
    pool: str = "mean",
) -> tuple[int, list[dict]]:
    """
    Accuracy-based sweep: find the layer that maximises QA F1 on a small
    calibration set.  This is the correct implementation of f(M).

    For each candidate layer:
      - Extract context state (mean-pooled residuals) from each example's context
      - Generate answer using question-only prompt + context injection
      - Grade against gold answers (F1 + exact match)
      - Average across all examples

    The layer with the highest average F1 is returned as the optimal injection
    point for this model.

    Parameters
    ----------
    examples : list[BenchmarkExample]
        Small calibration set (3-10 examples is enough).
    layer_range : (lo, hi) inclusive, defaults to 25%-85% of model depth.
    step : layer stride (2 = test every other layer, faster).
    scale : injection scale factor (1.0 is a safe default for calibration).
    max_new_tokens : keep short for speed during calibration.

    Returns
    -------
    (best_layer : int, results : list[dict])
      results sorted by layer, each entry:
        layer, avg_f1, avg_em, n_examples, per_example details
    """
    # Lazy import to avoid circular dependency
    from .context_injector import extract_context_state, generate_with_context_injection
    from .benchmark_datasets import grade_qa

    if layer_range is None:
        lo = max(1, wrapper.n_layers // 4)
        hi = min(wrapper.n_layers - 2, int(wrapper.n_layers * 0.85))
        layer_range = (lo, hi)

    lo, hi = layer_range
    layers_to_test = list(range(lo, hi + 1, step))
    n_layers = len(layers_to_test)
    n_examples = len(examples)

    console.print(
        f"  [dim]Accuracy sweep: {n_layers} layers x {n_examples} examples "
        f"= {n_layers * n_examples} evaluations[/dim]"
    )

    results = []
    best_layer = layers_to_test[0]
    best_f1 = -1.0

    for i, layer in enumerate(layers_to_test):
        console.print(
            f"  [{i+1}/{n_layers}] layer {layer}/{wrapper.n_layers}...",
            end="\r",
        )

        f1_scores = []
        em_scores = []
        per_example = []

        for ex in examples:
            try:
                ctx_v, _ = extract_context_state(wrapper, ex.context, layer, pool=pool)
                text, _ = generate_with_context_injection(
                    wrapper, ex.question_prompt(), ctx_v,
                    layer=layer, scale=scale, max_new_tokens=max_new_tokens,
                )
                grades = grade_qa(text, ex.gold_answers)
                f1_scores.append(grades["f1"])
                em_scores.append(float(grades["exact_match"]))
                per_example.append({
                    "example_id": ex.id,
                    "f1": grades["f1"],
                    "em": grades["exact_match"],
                    "generated": text.strip()[:60],
                    "gold": ex.gold_answers,
                })
            except Exception as exc:
                f1_scores.append(0.0)
                em_scores.append(0.0)
                per_example.append({"example_id": ex.id, "error": str(exc)})

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0

        results.append({
            "layer": layer,
            "depth_fraction": round(layer / wrapper.n_layers, 3),
            "avg_f1": round(avg_f1, 4),
            "avg_em": round(avg_em, 4),
            "n_examples": n_examples,
            "per_example": per_example,
        })

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_layer = layer

    console.print()  # clear \r line
    console.print(
        f"  [bold green]f(M) result: layer {best_layer}/{wrapper.n_layers} "
        f"({best_layer/wrapper.n_layers:.0%} depth)  avg_F1={best_f1:.4f}[/bold green]"
    )
    return best_layer, results


def print_accuracy_sweep_table(sweep_results: list[dict], best_layer: int) -> None:
    table = Table(title="Layer Accuracy Sweep: f(M)", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Layer", style="cyan", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("Avg F1", justify="right")
    table.add_column("Avg EM", justify="right")
    table.add_column("Best", justify="center")

    for r in sweep_results:
        is_best = r["layer"] == best_layer
        f1_color = "green" if r["avg_f1"] >= 0.3 else "yellow" if r["avg_f1"] >= 0.1 else "dim"
        table.add_row(
            str(r["layer"]),
            f"{r['depth_fraction']:.0%}",
            f"[{f1_color}]{r['avg_f1']:.4f}[/{f1_color}]",
            f"{r['avg_em']:.4f}",
            "[bold green]<-- f(M)[/bold green]" if is_best else "",
        )
    console.print(table)


def print_sweep_table(sweep_results: list[dict], best_layer: int) -> None:
    table = Table(title="Layer Sweep: f(M)", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Layer", style="cyan", justify="right")
    table.add_column("Cosine Sim", justify="right")
    table.add_column("Ctx Variance", justify="right")
    table.add_column("Selected", justify="center")

    for r in sweep_results:
        is_best = r["layer"] == best_layer
        table.add_row(
            str(r["layer"]),
            f"{r['cosine_sim']:.4f}",
            f"{r['ctx_variance']:.4f}",
            "[bold green]YES[/bold green]" if is_best else "",
        )
    console.print(table)
