"""
MLX backend for the Latent Rollback experiment.

Uses mlx-community/Meta-Llama-3-8B-Instruct-4bit — publicly available on
HuggingFace with no token required (~4.5 GB download).

Key differences from the TransformerLens backend:
  - mlx.core arrays instead of torch.Tensor inside the model
  - Layer-by-layer forward pass for extraction and rollback injection
  - MLX arrays are converted to torch tensors only for shared vector arithmetic
    (compute_delta, vector_stats) which live in extraction.py
  - Attention mask is the string "causal" per mlx-lm convention (the attention
    layer resolves this internally to a real matrix)

Layer convention: `hook_resid_post` equivalent is the output of
`model.model.layers[i](h, mask, cache)`, which is the post-MLP residual stream.
This is the same quantity TransformerLens calls `hook_resid_post`.
"""

from __future__ import annotations

import numpy as np
import torch
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask
from rich.console import Console

from config import ExperimentConfig, EXPECTED_A, EXPECTED_B
from vector_math import compute_delta, vector_stats
from evaluation import (
    ExperimentResult,
    grade_output,
    print_step,
    print_vector_stats,
    print_rollback_verdict,
    print_ablation_table,
    save_result,
)

console = Console()

_EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Wrapper — provides a uniform interface over (mlx_model, tokenizer)
# ---------------------------------------------------------------------------

class MLXModelWrapper:
    """Thin wrapper around an mlx-lm model + HF tokenizer."""

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self.n_layers: int = len(model.model.layers)
        # d_model from the embedding weight shape
        self.d_model: int = int(model.model.embed_tokens.weight.shape[1])

    # ---- Token / string helpers -------------------------------------------

    def encode(self, prompt: str) -> list[int]:
        """Encode `prompt` to a list of token IDs (no special tokens added)."""
        return self._tokenizer.encode(prompt)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self._tokenizer, "eos_token_id", None)

    # ---- Layer access ------------------------------------------------------

    @property
    def layers(self):
        return self._model.model.layers

    def embed(self, token_ids: list[int]) -> mx.array:
        """Embed a token list and return [1, seq_len, d_model]."""
        return self._model.model.embed_tokens(mx.array([token_ids]))

    def norm_and_lm_head(self, h: mx.array) -> mx.array:
        """Apply final norm + language-model head -> [1, seq_len, vocab].

        Handles different mlx-lm architecture attribute names:
          - Standard (LLaMA, Mistral, Qwen, DeepSeek): model.model.norm
          - Fallback: model.model.final_layernorm
        """
        model_inner = self._model.model
        if hasattr(model_inner, "norm"):
            normed = model_inner.norm(h)
        elif hasattr(model_inner, "final_layernorm"):
            normed = model_inner.final_layernorm(h)
        else:
            # Last resort: skip norm (rare, architecture-specific)
            normed = h
        return self._model.lm_head(normed)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> MLXModelWrapper:
    """
    Download (if needed) and load an MLX model from HuggingFace.

    model_name examples (all keyless):
      mlx-community/Meta-Llama-3-8B-Instruct-4bit
      mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
    """
    console.print(f"[bold cyan]Loading MLX model:[/bold cyan] {model_name}")
    console.print("  (downloads ~4.5 GB on first run, cached afterwards)")

    model, tokenizer = mlx_load(model_name)
    wrapper = MLXModelWrapper(model, tokenizer)

    console.print(
        f"[green]Model loaded.[/green] "
        f"Layers: {wrapper.n_layers}  d_model: {wrapper.d_model}"
    )
    return wrapper


# ---------------------------------------------------------------------------
# Core layer-by-layer runner (extraction + rollback share this)
# ---------------------------------------------------------------------------

def _run_layers(
    wrapper: MLXModelWrapper,
    token_ids: list[int],
    hook_layer: int | None = None,
    hook_fn=None,
) -> tuple[mx.array, mx.array | None]:
    """
    Run a full forward pass layer-by-layer.

    If `hook_layer` and `hook_fn` are provided, `hook_fn` is called after
    layer `hook_layer` and its return value replaces the hidden state h.

    Returns
    -------
    (logits [1, seq_len, vocab], h_at_hook [1, seq_len, d_model] | None)
    """
    h = wrapper.embed(token_ids)
    # mlx-lm uses the string "causal" as a sentinel that attention layers
    # resolve to a real causal mask internally. Pass None for single tokens.
    mask = "causal" if h.shape[1] > 1 else None

    h_at_hook = None

    for i, layer in enumerate(wrapper.layers):
        h = layer(h, mask, cache=None)
        if hook_layer is not None and i == hook_layer:
            h_at_hook = h
            if hook_fn is not None:
                h = hook_fn(h)

    logits = wrapper.norm_and_lm_head(h)
    mx.eval(logits)
    return logits, h_at_hook


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_residual_vector(
    wrapper: MLXModelWrapper,
    prompt: str,
    layer: int,
    token_position: int = -1,
) -> tuple[torch.Tensor, str]:
    """
    Run `prompt` through the model, extract the residual stream at `layer`
    at `token_position` (default -1 = last token).

    Returns
    -------
    (vector : torch.Tensor [d_model] float32 on CPU,
     next_token_str : str  — sanity check greedy prediction)
    """
    token_ids = wrapper.encode(prompt)
    actual_pos = token_position % len(token_ids)

    logits, h_at_hook = _run_layers(wrapper, token_ids, hook_layer=layer)

    # h_at_hook shape: [1, seq_len, d_model]
    v_mx = h_at_hook[0, actual_pos, :]
    mx.eval(v_mx)

    # Convert to CPU torch tensor for shared vector arithmetic
    v_torch = torch.from_numpy(np.array(v_mx)).float()

    next_token_id = int(mx.argmax(logits[0, -1, :]).item())
    next_token_str = wrapper.decode([next_token_id])

    return v_torch, next_token_str


# ---------------------------------------------------------------------------
# All-layer extraction
# ---------------------------------------------------------------------------

def extract_all_layer_vectors(
    wrapper: MLXModelWrapper,
    prompt: str,
    token_position: int = -1,
) -> list[torch.Tensor]:
    """
    Single forward pass that captures the residual stream at every layer.

    Returns
    -------
    list of length n_layers, each a torch.Tensor [d_model] float32 on CPU.
    Index 0 = output of layer 0 (post-MLP), index n-1 = output of last layer.
    """
    token_ids = wrapper.encode(prompt)
    actual_pos = token_position % len(token_ids)

    h = wrapper.embed(token_ids)
    mask = "causal" if h.shape[1] > 1 else None

    vectors: list[torch.Tensor] = []
    for layer in wrapper.layers:
        h = layer(h, mask, cache=None)
        mx.eval(h)
        v_mx = h[0, actual_pos, :]
        vectors.append(torch.from_numpy(np.array(v_mx)).float())

    return vectors


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_baseline(
    wrapper: MLXModelWrapper,
    token_ids: list[int],
    max_new_tokens: int = 80,
) -> list[int]:
    """Greedy generation with no intervention."""
    tokens = list(token_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits, _ = _run_layers(wrapper, tokens)
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        tokens.append(next_id)
        if wrapper.eos_token_id is not None and next_id == wrapper.eos_token_id:
            break

    return generated


def generate_with_rollback(
    wrapper: MLXModelWrapper,
    token_ids: list[int],
    delta: torch.Tensor,
    layer: int,
    scale: float = 1.0,
    max_new_tokens: int = 80,
    broadcast: bool = False,
) -> list[int]:
    """
    Greedy generation with the rollback hook applied at `layer`.

    Parameters
    ----------
    broadcast : bool
        If False (default), subtract delta only at position `prompt_len - 1`.
        If True, subtract delta from ALL prompt token positions. This is
        needed for cross-prompt generalization (Test 1) and multi-hop (Test 2)
        where the semantic state is distributed across the full sequence,
        not concentrated at the final query token.
    """
    prompt_len = len(token_ids)
    target_pos = prompt_len - 1

    delta_mx = mx.array(delta.numpy())

    def rollback_hook(h: mx.array) -> mx.array:
        # h: [1, seq_len, d_model]
        if h.shape[1] <= target_pos:
            return h

        d = delta_mx.astype(h.dtype)
        scale_mx = mx.array(scale, dtype=h.dtype)
        modification = scale_mx * d  # [d_model]

        if broadcast:
            # Subtract delta from ALL prompt positions (0..prompt_len-1).
            # Leave generated-token positions (prompt_len..) untouched.
            prompt_part = h[:, :prompt_len, :] - modification[None, None, :]
            rest = h[:, prompt_len:, :]
            return mx.concatenate([prompt_part, rest], axis=1)
        else:
            # Single-position subtraction (original Phase 2 behaviour)
            original_row = h[0, target_pos, :]
            new_row = original_row - modification
            before = h[:, :target_pos, :]
            after = h[:, target_pos + 1:, :]
            return mx.concatenate(
                [before, new_row[None, None, :], after], axis=1
            )

    tokens = list(token_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits, _ = _run_layers(wrapper, tokens, hook_layer=layer, hook_fn=rollback_hook)
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        tokens.append(next_id)
        if wrapper.eos_token_id is not None and next_id == wrapper.eos_token_id:
            break

    return generated


# ---------------------------------------------------------------------------
# Probe-then-steer generation
# ---------------------------------------------------------------------------

def generate_with_probe_steer(
    wrapper: MLXModelWrapper,
    token_ids: list[int],
    delta: torch.Tensor,
    layer: int,
    min_projection: float = 0.5,
    max_new_tokens: int = 80,
    broadcast: bool = True,
) -> tuple[list[int], dict]:
    """
    Generation with probe-then-steer: only intervene if the concept is encoded.

    At the hook layer, projects the current residual onto the delta unit vector.
    If the projection magnitude is below `min_projection`, the context doesn't
    meaningfully encode the concept — skip the intervention to avoid corrupting
    the representation. If above threshold, scale the intervention proportionally
    to the projection magnitude (stronger encoding → stronger push).

    Returns
    -------
    (generated_ids : list[int], probe_info : dict)
      probe_info contains: projection, threshold_triggered, scale_applied
    """
    prompt_len = len(token_ids)
    target_pos = prompt_len - 1

    delta_np  = delta.numpy()
    delta_mx  = mx.array(delta_np)
    delta_norm = float(delta.norm().item())
    if delta_norm < _EPSILON:
        raise ValueError("delta has near-zero norm — cannot compute unit vector")
    delta_unit_mx = delta_mx / delta_norm

    probe_info: dict = {
        "layer": layer,
        "min_projection": min_projection,
        "projection": None,
        "triggered": False,
        "scale_applied": 0.0,
    }

    def probe_hook(h: mx.array) -> mx.array:
        if h.shape[1] <= target_pos:
            return h

        # Project last prompt token residual onto delta unit vector
        v_curr = h[0, target_pos, :]           # [d_model]
        mx.eval(v_curr)
        v_torch = torch.from_numpy(np.array(v_curr)).float()

        delta_unit_torch = delta / delta_norm
        projection = torch.dot(v_torch, delta_unit_torch).item()
        probe_info["projection"] = projection

        if abs(projection) < min_projection:
            # Concept not meaningfully encoded — skip, return h unchanged
            probe_info["triggered"] = False
            probe_info["scale_applied"] = 0.0
            return h

        # Scale proportional to how strongly the concept is encoded.
        # Projection > 0 means residual points toward state B — subtract.
        # Projection < 0 means residual already toward state A — small push.
        adaptive_scale = abs(projection) / delta_norm
        probe_info["triggered"] = True
        probe_info["scale_applied"] = adaptive_scale

        d = delta_mx.astype(h.dtype)
        scale_mx = mx.array(adaptive_scale, dtype=h.dtype)
        modification = scale_mx * d

        if broadcast:
            prompt_part = h[:, :prompt_len, :] - modification[None, None, :]
            rest = h[:, prompt_len:, :]
            return mx.concatenate([prompt_part, rest], axis=1)
        else:
            original_row = h[0, target_pos, :]
            new_row = original_row - modification
            before = h[:, :target_pos, :]
            after  = h[:, target_pos + 1:, :]
            return mx.concatenate([before, new_row[None, None, :], after], axis=1)

    tokens = list(token_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits, _ = _run_layers(wrapper, tokens, hook_layer=layer, hook_fn=probe_hook)
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        tokens.append(next_id)
        if wrapper.eos_token_id is not None and next_id == wrapper.eos_token_id:
            break

    return generated, probe_info


# ---------------------------------------------------------------------------
# Multi-layer rollback (stack rollback support)
# ---------------------------------------------------------------------------

def generate_with_multi_rollback(
    wrapper: MLXModelWrapper,
    token_ids: list[int],
    layer_deltas: dict[int, torch.Tensor],
    scale: float = 1.0,
    max_new_tokens: int = 80,
    broadcast: bool = True,
) -> list[int]:
    """
    Greedy generation with rollback hooks applied at multiple distinct layers.

    Each entry in `layer_deltas` is {layer_index: delta_tensor}. At each
    specified layer the corresponding delta is subtracted from the residual
    stream independently. This is the correct approach for sequential
    stack rollback where different commits select different optimal layers.

    Parameters
    ----------
    layer_deltas : dict[int, torch.Tensor]
        Map of layer index → combined delta tensor for that layer.
        Build this by grouping per-commit deltas by their optimal probe layer.
    """
    prompt_len = len(token_ids)

    # Pre-convert deltas to MLX arrays once, outside the token loop.
    mx_deltas: dict[int, mx.array] = {
        L: mx.array(d.numpy()) for L, d in layer_deltas.items()
    }

    def run_forward(toks: list[int]) -> mx.array:
        h = wrapper.embed(toks)
        mask = "causal" if h.shape[1] > 1 else None

        for i, layer in enumerate(wrapper.layers):
            h = layer(h, mask, cache=None)
            if i in mx_deltas:
                d = mx_deltas[i].astype(h.dtype)
                scale_mx = mx.array(scale, dtype=h.dtype)
                modification = scale_mx * d  # [d_model]

                if broadcast:
                    cur_prompt_len = min(prompt_len, h.shape[1])
                    prompt_part = h[:, :cur_prompt_len, :] - modification[None, None, :]
                    rest = h[:, cur_prompt_len:, :]
                    h = mx.concatenate([prompt_part, rest], axis=1)
                else:
                    target_pos = prompt_len - 1
                    if h.shape[1] > target_pos:
                        original_row = h[0, target_pos, :]
                        new_row = original_row - modification
                        before = h[:, :target_pos, :]
                        after = h[:, target_pos + 1:, :]
                        h = mx.concatenate([before, new_row[None, None, :], after], axis=1)

        logits = wrapper.norm_and_lm_head(h)
        mx.eval(logits)
        return logits

    tokens = list(token_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits = run_forward(tokens)
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        tokens.append(next_id)
        if wrapper.eos_token_id is not None and next_id == wrapper.eos_token_id:
            break

    return generated


# ---------------------------------------------------------------------------
# Matrix hook generation (LoRA-style context injection/rollback)
# ---------------------------------------------------------------------------

def generate_with_matrix_hook(
    wrapper: MLXModelWrapper,
    token_ids: list[int],
    layer_matrices: dict[int, tuple[torch.Tensor, torch.Tensor]],
    mode: str = "rollback",
    scale: float = 1.0,
    max_new_tokens: int = 80,
    broadcast: bool = True,
) -> list[int]:
    """
    Greedy generation with LoRA-style matrix hooks at multiple layers.

    Instead of subtracting a fixed vector, applies a low-rank linear map
    to the residual: correction = B @ (A @ v). The correction is query-
    dependent — different residuals produce different corrections.

    Parameters
    ----------
    layer_matrices : dict[int, (A, B)]
        A : torch.Tensor [r, d_model]  — projects residual into context subspace
        B : torch.Tensor [d_model, r]  — lifts correction back to full space
    mode : "rollback" (subtract correction) or "inject" (add correction)
    """
    prompt_len = len(token_ids)
    sign = -1.0 if mode == "rollback" else 1.0

    # Pre-convert to MLX
    mx_matrices: dict[int, tuple[mx.array, mx.array]] = {}
    for L, (A, B) in layer_matrices.items():
        mx_matrices[L] = (mx.array(A.numpy()), mx.array(B.numpy()))

    def run_forward(toks: list[int]) -> mx.array:
        h = wrapper.embed(toks)
        mask = "causal" if h.shape[1] > 1 else None

        for i, layer in enumerate(wrapper.layers):
            h = layer(h, mask, cache=None)
            if i in mx_matrices:
                A_mx, B_mx = mx_matrices[i]
                A_mx = A_mx.astype(h.dtype)
                B_mx = B_mx.astype(h.dtype)
                sign_mx = mx.array(sign * scale, dtype=h.dtype)

                cur_prompt_len = min(prompt_len, h.shape[1])
                if broadcast:
                    # h_prompt: [1, prompt_len, n]
                    h_prompt = h[:, :cur_prompt_len, :]
                    # proj: [1, prompt_len, r] = h_prompt @ A^T
                    proj = mx.matmul(h_prompt, mx.transpose(A_mx))
                    # correction: [1, prompt_len, n] = proj @ B^T
                    correction = mx.matmul(proj, mx.transpose(B_mx))
                    prompt_part = h[:, :cur_prompt_len, :] + sign_mx * correction
                    rest = h[:, cur_prompt_len:, :]
                    h = mx.concatenate([prompt_part, rest], axis=1)
                else:
                    target_pos = prompt_len - 1
                    if h.shape[1] > target_pos:
                        v = h[:, target_pos:target_pos + 1, :]  # [1, 1, n]
                        proj = mx.matmul(v, mx.transpose(A_mx))  # [1, 1, r]
                        correction = mx.matmul(proj, mx.transpose(B_mx))  # [1, 1, n]
                        new_row = v + sign_mx * correction
                        before = h[:, :target_pos, :]
                        after = h[:, target_pos + 1:, :]
                        h = mx.concatenate([before, new_row, after], axis=1)

        logits = wrapper.norm_and_lm_head(h)
        mx.eval(logits)
        return logits

    tokens = list(token_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits = run_forward(tokens)
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        generated.append(next_id)
        tokens.append(next_id)
        if wrapper.eos_token_id is not None and next_id == wrapper.eos_token_id:
            break

    return generated


# ---------------------------------------------------------------------------
# Five-step experiment (mirrors experiment.py but uses MLX primitives)
# ---------------------------------------------------------------------------

def run_experiment_mlx(
    wrapper: MLXModelWrapper,
    cfg: ExperimentConfig,
    run_ablation: bool = False,
    output_path: str = "results_mlx.json",
) -> ExperimentResult:
    """Full five-step Latent Rollback experiment using the MLX backend."""

    _validate_layer(wrapper, cfg.extraction_layer)

    result = ExperimentResult(
        model_name=cfg.model_name,
        device="mps-mlx",
        dtype="mlx-native",
        extraction_layer=cfg.extraction_layer,
        rollback_scale=cfg.rollback_scale,
    )

    # Step 1 — State A
    console.rule("[bold cyan]STEP 1 — State A (Base Truth)[/bold cyan]")
    try:
        v_a, sanity_a = extract_residual_vector(
            wrapper, cfg.prompt_a, cfg.extraction_layer, cfg.extraction_position
        )
    except Exception as exc:
        _fatal("Step 1 (extract v_A)", exc)

    console.print(f"  Sanity next-token: '{sanity_a}'  (expect near '5432')")

    ids_a = _safe_gen_baseline(wrapper, wrapper.encode(cfg.prompt_a), cfg.max_new_tokens, "Step 1")
    text_a = _ids_to_str(wrapper, ids_a)
    result.state_a_text = text_a
    _, _, outcome_a = grade_output(text_a, EXPECTED_A, EXPECTED_B)
    result.state_a_outcome = outcome_a

    print_step("State A", cfg.prompt_a, text_a, outcome_a,
               extra_lines=[f"  v_A norm: {v_a.norm().item():.4f}"])
    _warn_baseline(outcome_a, EXPECTED_A, text_a, "State A")

    # Step 2 — State B
    console.rule("[bold cyan]STEP 2 — State B (The Change)[/bold cyan]")
    try:
        v_b, sanity_b = extract_residual_vector(
            wrapper, cfg.prompt_b, cfg.extraction_layer, cfg.extraction_position
        )
    except Exception as exc:
        _fatal("Step 2 (extract v_B)", exc)

    console.print(f"  Sanity next-token: '{sanity_b}'  (expect near '8080')")

    token_ids_b = wrapper.encode(cfg.prompt_b)
    ids_b = _safe_gen_baseline(wrapper, token_ids_b, cfg.max_new_tokens, "Step 2")
    text_b = _ids_to_str(wrapper, ids_b)
    result.state_b_text = text_b
    _, _, outcome_b = grade_output(text_b, EXPECTED_A, EXPECTED_B)
    result.state_b_outcome = outcome_b

    print_step("State B", cfg.prompt_b, text_b, outcome_b,
               extra_lines=[f"  v_B norm: {v_b.norm().item():.4f}"])
    _warn_baseline(outcome_b, EXPECTED_B, text_b, "State B")

    # Step 3 — Delta
    console.rule("[bold cyan]STEP 3 — Latent Delta (The Commit)[/bold cyan]")
    delta = compute_delta(v_a, v_b)

    stats_a = vector_stats("v_A", v_a)
    stats_b = vector_stats("v_B", v_b)
    stats_d = vector_stats("delta", delta)

    result.v_a_norm = stats_a["norm"]
    result.v_b_norm = stats_b["norm"]
    result.delta_norm = stats_d["norm"]

    print_vector_stats(stats_a["norm"], stats_b["norm"], stats_d["norm"])

    cosine = torch.nn.functional.cosine_similarity(
        v_a.unsqueeze(0), v_b.unsqueeze(0)
    ).item()
    console.print(f"  Cosine similarity(v_A, v_B): {cosine:.4f}")
    console.print(
        f"  Delta / v_A norm ratio: {stats_d['norm'] / max(stats_a['norm'], _EPSILON):.4f}"
    )

    # Step 4 — Rollback
    console.rule("[bold cyan]STEP 4 — Rollback (The Checkout)[/bold cyan]")
    console.print(f"  Layer {cfg.extraction_layer}, scale={cfg.rollback_scale}")

    try:
        ids_rollback = generate_with_rollback(
            wrapper, token_ids_b, delta,
            layer=cfg.extraction_layer,
            scale=cfg.rollback_scale,
            max_new_tokens=cfg.max_new_tokens,
        )
    except Exception as exc:
        _fatal("Step 4 (rollback generation)", exc)

    result.rollback_text = _ids_to_str(wrapper, ids_rollback)

    # Step 5 — Evaluate
    console.rule("[bold cyan]STEP 5 — Evaluation[/bold cyan]")
    contains_a, contains_b, outcome = grade_output(result.rollback_text, EXPECTED_A, EXPECTED_B)
    result.rollback_outcome = outcome

    print_step(
        "ROLLBACK", "Prompt B  [hook applied]", result.rollback_text, outcome,
        extra_lines=[f"  contains '5432': {contains_a}", f"  contains '8080': {contains_b}"],
    )
    print_rollback_verdict(outcome)

    # Ablation
    if run_ablation:
        console.rule("[bold cyan]ABLATION SWEEP[/bold cyan]")
        result.ablation_rows = _run_ablation(wrapper, token_ids_b, delta, cfg)
        print_ablation_table(result.ablation_rows)

    save_result(result, output_path)
    return result


# ---------------------------------------------------------------------------
# Ablation sweep
# ---------------------------------------------------------------------------

def _run_ablation(
    wrapper: MLXModelWrapper,
    token_ids_b: list[int],
    delta: torch.Tensor,
    cfg: ExperimentConfig,
) -> list[dict]:
    rows = []
    n_total = len(cfg.layer_sweep) * len(cfg.scale_sweep)
    done = 0

    for layer in cfg.layer_sweep:
        if layer >= wrapper.n_layers:
            console.print(f"  [yellow]Skip layer {layer} — model has {wrapper.n_layers} layers.[/yellow]")
            continue
        for scale in cfg.scale_sweep:
            done += 1
            console.print(f"  [{done}/{n_total}] layer={layer}  scale={scale}", end="\r")
            try:
                ids = generate_with_rollback(
                    wrapper, token_ids_b, delta, layer=layer, scale=scale, max_new_tokens=30
                )
                text = _ids_to_str(wrapper, ids)
                _, _, outcome = grade_output(text, EXPECTED_A, EXPECTED_B)
                snippet = text.strip()[:60]
            except Exception as exc:
                outcome = "ERROR"
                snippet = str(exc)[:60]

            rows.append({"layer": layer, "scale": scale, "outcome": outcome, "snippet": snippet})

    console.print()
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_layer(wrapper: MLXModelWrapper, layer: int) -> None:
    if layer < 0 or layer >= wrapper.n_layers:
        raise ValueError(
            f"extraction_layer {layer} out of range for {wrapper.n_layers}-layer model "
            f"(valid: 0 to {wrapper.n_layers - 1})"
        )


def _safe_gen_baseline(wrapper, token_ids, max_new_tokens, label):
    try:
        return generate_baseline(wrapper, token_ids, max_new_tokens)
    except Exception as exc:
        _fatal(f"{label} (baseline generation)", exc)


def _ids_to_str(wrapper: MLXModelWrapper, ids: list[int]) -> str:
    if not ids:
        console.print("[yellow]Warning: generation produced zero tokens.[/yellow]")
        return ""
    return wrapper.decode(ids)


def _fatal(label: str, exc: Exception) -> None:
    console.print(f"[red]{label} failed: {exc}[/red]")
    raise SystemExit(1)


def _warn_baseline(outcome: str, expected: str, text: str, label: str) -> None:
    norm = text.lower().replace(",", "").replace(".", "")
    if expected not in norm:
        console.print(
            f"[yellow]WARNING: {label} baseline didn't output '{expected}'. "
            f"Got: {text.strip()[:80]}[/yellow]"
        )
