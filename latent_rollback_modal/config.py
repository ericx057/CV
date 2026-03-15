"""
Experiment configuration for the Latent Rollback (Phase 2 State Reversibility) test.

Model hierarchy by size and M4 RAM budget:
  - gpt2-xl        (~1.5B, ~6GB)  : quick smoke test, no auth required
  - mistralai/Mistral-7B-v0.1     : primary experiment target (~14GB fp16)
  - meta-llama/Meta-Llama-3-8B    : alternative, requires HF auth token
"""

from dataclasses import dataclass, field
import os
from pathlib import Path


# --- Prompt Pair -----------------------------------------------------------

# Prompts end mid-sentence so the model's very next token is the port number.
# This avoids multiple-choice formatting from instruct fine-tuning and makes
# the rollback signal unambiguous — either the first token is "5432" or "8080".
PROMPT_A = (
    "Production database config:\n"
    "  host: db.prod.internal\n"
    "  port: 5432\n\n"
    "The application should connect to port"
)

PROMPT_B = (
    "Production database config:\n"
    "  host: db.prod.internal\n"
    "  port: 5432\n\n"
    "MIGRATION COMPLETE: port updated to 8080\n\n"
    "The application should connect to port"
)

# Expected outputs (used for evaluation grading)
EXPECTED_A = "5432"
EXPECTED_B = "8080"
EXPECTED_ROLLBACK = "5432"  # goal: model ignores the UPDATE, reverts to State A


# --- Experiment Config -----------------------------------------------------

@dataclass
class ExperimentConfig:
    # Model
    model_name: str = "mistralai/Mistral-7B-v0.1"

    # Device: "mps" for M4 Apple Silicon, "cuda" for NVIDIA, "cpu" for fallback
    device: str = "mps"

    # dtype: float32 is safest for MPS; bfloat16 works on recent PyTorch+M4
    dtype: str = "float32"

    # Layer to hook (0-indexed). Mistral-7B has 32 layers; Layer 16 is mid-to-late.
    extraction_layer: int = 16

    # Token position to extract: -1 = last token in the prompt (the query token)
    extraction_position: int = -1

    # Scale factor for delta subtraction (1.0 = full rollback, <1 = partial)
    rollback_scale: float = 1.0

    # Generation
    max_new_tokens: int = 80
    temperature: float = 0.0  # greedy decoding for reproducibility

    # Ablation sweep ranges
    layer_sweep: list = field(default_factory=lambda: list(range(8, 28, 2)))
    scale_sweep: list = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5])

    # Prompts (bind to module-level constants)
    prompt_a: str = PROMPT_A
    prompt_b: str = PROMPT_B


# Shorthand config presets

def mistral_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="mistralai/Mistral-7B-v0.1",
        device="mps",
        dtype="float32",
        extraction_layer=16,
    )


def llama3_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="meta-llama/Meta-Llama-3-8B",
        device="mps",
        dtype="float32",
        extraction_layer=16,
    )


def gpt2xl_config() -> ExperimentConfig:
    """Quick smoke test — no auth required, 48 layers total."""
    return ExperimentConfig(
        model_name="gpt2-xl",
        device="mps",
        dtype="float32",
        extraction_layer=24,   # mid-to-late for 48-layer GPT-2 XL
        layer_sweep=list(range(12, 44, 4)),
    )


def llama3_mlx_config() -> ExperimentConfig:
    """
    Llama 3 8B Instruct via MLX — 4-bit quantized, keyless (~4.5 GB).
    Kept for reference; prefer llama3_mlx_base for the experiment.
    """
    return ExperimentConfig(
        model_name="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        device="mps",
        dtype="float32",
        extraction_layer=16,
        layer_sweep=list(range(8, 28, 2)),
    )


def qwen25_7b_config() -> ExperimentConfig:
    """Qwen 2.5 7B Instruct via MLX — 4-bit, ~4.5 GB."""
    return ExperimentConfig(
        model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
        device="mps",
        dtype="bfloat16",
        extraction_layer=15,   # ~55% of 28 layers
        layer_sweep=list(range(8, 24, 2)),
    )


def mistral_24b_config() -> ExperimentConfig:
    """Mistral Small 24B via MLX — 4-bit, ~13 GB."""
    return ExperimentConfig(
        model_name="mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
        device="mps",
        dtype="bfloat16",
        extraction_layer=26,   # ~55% of 48 layers
        layer_sweep=list(range(16, 40, 2)),
    )


def deepseek_14b_config() -> ExperimentConfig:
    """DeepSeek R1 Distill Qwen 14B via MLX — 4-bit, ~8 GB."""
    return ExperimentConfig(
        model_name="mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        device="mps",
        dtype="bfloat16",
        extraction_layer=22,   # ~55% of 40 layers
        layer_sweep=list(range(12, 36, 2)),
    )


def llama3_mlx_base_config() -> ExperimentConfig:
    """
    Llama 3 8B BASE model via MLX in BF16 — keyless, ~16 GB, fits in 24 GB.
    Base (non-instruct) models are far more susceptible to residual stream
    interventions because RLHF has not baked in strong answer-format priors.
    BF16 preserves residual stream geometry vs 4-bit quantization.
    """
    return ExperimentConfig(
        model_name="mlx-community/Meta-Llama-3-8B",
        device="mps",
        dtype="bfloat16",
        extraction_layer=16,
        layer_sweep=list(range(8, 28, 2)),
        scale_sweep=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    )


MODEL_MATRIX: dict[str, dict] = {
    "llama3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "size_gb": 16.0,
        "family": "llama",
        "note": "Modal Torch backend",
    },
    "qwen25-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "size_gb": 15.0,
        "family": "qwen",
        "note": "Modal Torch backend",
    },
    "mistral-24b": {
        "hf_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "size_gb": 48.0,
        "family": "mistral",
        "note": "Modal Torch backend",
    },
    "deepseek-14b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "size_gb": 30.0,
        "family": "deepseek",
        "note": "Modal Torch backend",
    },
}


@dataclass(frozen=True)
class RuntimePaths:
    hf_home: Path
    datasets_cache: Path
    results_root: Path
    hf_volume_name: str
    datasets_volume_name: str
    results_volume_name: str


def runtime_paths() -> RuntimePaths:
    results_root = Path(os.environ.get("LATENT_ROLLBACK_RESULTS_ROOT", "/vol/results"))
    return RuntimePaths(
        hf_home=Path("/vol/hf-cache"),
        datasets_cache=Path("/vol/datasets-cache"),
        results_root=results_root,
        hf_volume_name="latent-rollback-hf-cache",
        datasets_volume_name="latent-rollback-datasets-cache",
        results_volume_name="latent-rollback-results",
    )


def results_path(name: str) -> Path:
    return runtime_paths().results_root / name
