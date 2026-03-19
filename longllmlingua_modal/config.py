from __future__ import annotations

import os
from pathlib import Path

from latent_rollback_modal.config import MODEL_MATRIX, runtime_paths

DEFAULT_GPU = "A100-80GB"
DEFAULT_FUNCTION_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60 * 60
DEFAULT_COMPRESSOR_MODEL = "NousResearch/Llama-2-7b-hf"
DEFAULT_COMPRESSOR_DEVICE = "cuda"


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
