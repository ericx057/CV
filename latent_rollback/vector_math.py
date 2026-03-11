"""
Pure torch vector arithmetic — no transformer_lens dependency.
Shared by both the TransformerLens and MLX backends.
"""

import torch


def compute_delta(v_a: torch.Tensor, v_b: torch.Tensor) -> torch.Tensor:
    """delta = v_B - v_A  (the latent 'state change' direction)."""
    return v_b - v_a


def vector_stats(name: str, v: torch.Tensor) -> dict:
    return {
        "name": name,
        "shape": tuple(v.shape),
        "norm": v.norm().item(),
        "mean": v.mean().item(),
        "std": v.std().item(),
    }
