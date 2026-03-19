from __future__ import annotations

import os
from pathlib import Path

from latent_rollback_modal.config import MODEL_MATRIX, runtime_paths

DEFAULT_GPU = "A100-80GB"
DEFAULT_FUNCTION_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60 * 60
DEFAULT_PILOT_PROBES = 50
DEFAULT_TOP_K_HEADS = 8
DEFAULT_POOL_KERNEL = 3
DEFAULT_PILOT_TARGET_LENGTH = 2000
DEFAULT_MAX_ATTENTION_GB = 8.0

MODEL_ALIASES = {
    "llama": "llama3-8b",
    "qwen": "qwen25-7b",
    "deepseek": "deepseek-14b",
    "mistral": "mistral-24b",
}


def results_path(name: str) -> Path:
    return runtime_paths().results_root / name


def build_env() -> dict[str, str]:
    paths = runtime_paths()
    return {
        "HF_HOME": str(paths.hf_home),
        "HF_DATASETS_CACHE": str(paths.datasets_cache),
        "TRANSFORMERS_CACHE": str(paths.hf_home / "transformers"),
        "LATENT_ROLLBACK_RESULTS_ROOT": str(paths.results_root),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
    }


def resolve_model_key(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def packaged_head_config_path(model_key: str) -> Path:
    return Path(__file__).resolve().parent / "head_configs" / f"{resolve_model_key(model_key)}.json"


def head_config_path(model_key: str) -> Path:
    return results_path("ehpc_head_configs") / f"{resolve_model_key(model_key)}.json"
