"""
Context state extraction and injection for the Latent Rollback benchmark.

Core mechanism:
  1. Extract: run context_text through the model up to `layer`, then
     mean-pool residuals across ALL context token positions -> context_vector [d_model]

  2. Inject: when running question-only prompt, add context_vector to
     the residual stream at `layer` for every position.

This implements I(m) + F from the paper where:
  m  = context_vector (compressed context representation)
  F  = fact_vector (optional, from FactStore)
  I  = injection at the optimal layer

Token efficiency:
  - Baseline input = encode(context + question) -> len() tokens
  - Injection input = encode(question) -> len() tokens
  - Reduction ratio = 1 - injection_input / baseline_input

The injection_scale is the key hyperparameter.  Too low: context ignored.
Too high: context overwrites the question. Empirically, 0.5-1.5 works well.
"""

from __future__ import annotations

import numpy as np
import torch
import mlx.core as mx

from backend_mlx import MLXModelWrapper, _run_layers, _ids_to_str
from rich.console import Console

console = Console()

_EPSILON = 1e-8

# Stop strings for QA generation: truncate at the first paragraph break or
# when the model starts generating a new question/fact-block prefix.
QA_STOP_STRINGS: tuple[str, ...] = ("\n", "\nQuestion:", "\nFacts:", "\nContext:")


def truncate_at_stop(text: str, stop_strings: tuple[str, ...] = QA_STOP_STRINGS) -> str:
    """
    Truncate `text` at the first occurrence of any stop string.

    Used to prevent the model from generating beyond the answer boundary
    (e.g. regenerating the fact block or the next question in a QA prompt).
    Applied consistently to ALL conditions (baseline, vector, matrix) so
    scores are comparable.
    """
    best = len(text)
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1 and idx < best:
            best = idx
    return text[:best]


# ---------------------------------------------------------------------------
# Context state extraction
# ---------------------------------------------------------------------------

def extract_context_state(
    wrapper: MLXModelWrapper,
    context_text: str,
    layer: int,
    pool: str = "mean",
) -> tuple[torch.Tensor, int]:
    """
    Extract a compressed representation of context_text from `layer`.

    Parameters
    ----------
    pool : "mean" | "last" | "cls"
        Pooling strategy across the context token positions.
        "mean"  — average across all positions (default, robust)
        "last"  — last token only (fast but positionally biased)
        "cls"   — first token position (good for instruction-tuned models)

    Returns
    -------
    (context_vector : torch.Tensor [d_model], n_context_tokens : int)
    """
    if layer < 0 or layer >= wrapper.n_layers:
        raise ValueError(
            f"layer {layer} out of range for {wrapper.n_layers}-layer model"
        )

    token_ids = wrapper.encode(context_text)
    n_tokens = len(token_ids)

    h = wrapper.embed(token_ids)
    mask = "causal" if h.shape[1] > 1 else None

    # Run layers up to (and including) target layer, stop early
    for i, layer_module in enumerate(wrapper.layers):
        h = layer_module(h, mask, cache=None)
        if i == layer:
            mx.eval(h)
            break

    # h shape: [1, seq_len, d_model]
    h_np = np.array(h[0])  # [seq_len, d_model]

    if pool == "mean":
        v = h_np.mean(axis=0)
    elif pool == "last":
        v = h_np[-1]
    elif pool == "cls":
        v = h_np[0]
    else:
        raise ValueError(f"Unknown pool strategy: {pool!r}")

    return torch.from_numpy(v).float(), n_tokens


# ---------------------------------------------------------------------------
# Context-injected generation
# ---------------------------------------------------------------------------

def generate_with_context_injection(
    wrapper: MLXModelWrapper,
    question_text: str,
    context_vector: torch.Tensor,
    layer: int,
    scale: float = 1.0,
    max_new_tokens: int = 100,
    normalize: bool = True,
    stop_strings: tuple[str, ...] = QA_STOP_STRINGS,
) -> tuple[str, int]:
    """
    Generate a response to question_text with context_vector injected at `layer`.

    The context_vector is added to the residual stream at `layer` for all
    token positions (both prompt and generated). This implements persistent
    context injection: the context information is encoded at every step
    without occupying input token positions.

    Parameters
    ----------
    normalize : bool
        If True, normalize context_vector to unit norm before applying scale.
        This separates the direction from the magnitude, making scale
        interpretable as an absolute intervention strength.

    Returns
    -------
    (generated_text : str, n_question_tokens : int)
    """
    if normalize:
        norm = context_vector.norm().item()
        if norm < _EPSILON:
            raise ValueError("context_vector has near-zero norm")
        ctx_v = context_vector / norm
    else:
        ctx_v = context_vector

    ctx_mx = mx.array(ctx_v.numpy())

    question_ids = wrapper.encode(question_text)
    n_question_tokens = len(question_ids)

    def injection_hook(h: mx.array) -> mx.array:
        c = ctx_mx.astype(h.dtype)
        s = mx.array(scale, dtype=h.dtype)
        return h + s * c[None, None, :]

    tokens = list(question_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits, _ = _run_layers(wrapper, tokens, hook_layer=layer, hook_fn=injection_hook)
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        tokens.append(next_id)
        if wrapper.eos_token_id is not None and next_id == wrapper.eos_token_id:
            break

    text = _ids_to_str(wrapper, generated)
    if stop_strings:
        text = truncate_at_stop(text, stop_strings)
    return text, n_question_tokens


# ---------------------------------------------------------------------------
# Baseline generation (context + question in prompt)
# ---------------------------------------------------------------------------

def generate_baseline_qa(
    wrapper: MLXModelWrapper,
    full_prompt: str,
    max_new_tokens: int = 100,
    stop_strings: tuple[str, ...] = QA_STOP_STRINGS,
) -> tuple[str, int]:
    """
    Standard generation with full context in the prompt.

    Returns
    -------
    (generated_text : str, n_input_tokens : int)
    """
    from backend_mlx import generate_baseline

    token_ids = wrapper.encode(full_prompt)
    n_input_tokens = len(token_ids)
    generated_ids = generate_baseline(wrapper, token_ids, max_new_tokens)
    text = _ids_to_str(wrapper, generated_ids)
    if stop_strings:
        text = truncate_at_stop(text, stop_strings)
    return text, n_input_tokens


# ---------------------------------------------------------------------------
# Scale sweep helper
# ---------------------------------------------------------------------------

def sweep_injection_scale(
    wrapper: MLXModelWrapper,
    question_text: str,
    context_vector: torch.Tensor,
    layer: int,
    gold_answers: list[str],
    scales: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
    max_new_tokens: int = 50,
) -> list[dict]:
    """
    Sweep injection scales and return graded results.
    Used to find the optimal scale for a given model + layer.
    """
    from benchmark_datasets import grade_qa

    results = []
    for scale in scales:
        try:
            text, _ = generate_with_context_injection(
                wrapper, question_text, context_vector, layer,
                scale=scale, max_new_tokens=max_new_tokens,
            )
            grades = grade_qa(text, gold_answers)
            results.append({
                "scale": scale,
                "text": text.strip()[:80],
                "exact_match": grades["exact_match"],
                "f1": grades["f1"],
            })
        except Exception as exc:
            results.append({
                "scale": scale,
                "text": f"ERROR: {exc}",
                "exact_match": False,
                "f1": 0.0,
            })
    return results


# ---------------------------------------------------------------------------
# Token efficiency metrics
# ---------------------------------------------------------------------------

def compute_token_metrics(
    n_baseline_input: int,
    n_injection_input: int,
    n_baseline_output: int,
    n_injection_output: int,
) -> dict:
    """
    Compute token efficiency metrics.

    token_reduction_ratio:
        Fraction of input tokens saved. 1.0 = perfect (question-only).
        0.0 = no savings (same input length).

    total_cost_ratio:
        Total tokens (in + out) for injection / baseline.
        Values < 1.0 indicate computational savings.
    """
    baseline_total = n_baseline_input + n_baseline_output
    injection_total = n_injection_input + n_injection_output

    input_reduction = 1.0 - (n_injection_input / max(n_baseline_input, 1))
    total_ratio = injection_total / max(baseline_total, 1)

    return {
        "baseline_input_tokens": n_baseline_input,
        "injection_input_tokens": n_injection_input,
        "baseline_output_tokens": n_baseline_output,
        "injection_output_tokens": n_injection_output,
        "input_token_reduction": round(input_reduction, 4),
        "total_cost_ratio": round(total_ratio, 4),
        "baseline_total_tokens": baseline_total,
        "injection_total_tokens": injection_total,
    }
