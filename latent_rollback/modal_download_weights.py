"""
Download model weights to the rsce-hf-weights volume using a CPU-only container.
Run once per account to pre-cache weights before benchmark runs.

Usage:
  modal run modal_download_weights.py --model-key deepseek-67b
  modal run modal_download_weights.py --model-key qwen25-32b
  modal run modal_download_weights.py --model-key llama31-70b
"""

from __future__ import annotations
import modal

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime",
        add_python="3.11",
    )
    .pip_install(
        "transformers>=4.43.0,<5.0.0",
        "accelerate>=0.30.0",
        "huggingface_hub>=0.23.0",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App("rsce-download-weights", image=image)
vol = modal.Volume.from_name("rsce-hf-weights", create_if_missing=True)

MODEL_REGISTRY = {
    "qwen25-32b":  "Qwen/Qwen2.5-32B-Instruct",
    "deepseek-67b": "deepseek-ai/deepseek-llm-67b-chat",
    "llama31-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    # smaller models
    "llama3-8b":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen25-7b":   "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "mistral-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
}

@app.function(
    cpu=4,
    memory=16384,
    timeout=86400,  # 24h — large models take a while
    volumes={"/weights": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download(model_key: str) -> None:
    import os
    import time
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/weights"

    hf_id = MODEL_REGISTRY[model_key]
    print(f"Downloading {hf_id} -> /weights ...")
    t0 = time.time()

    snapshot_download(
        repo_id=hf_id,
        cache_dir="/weights",
        resume_download=True,
        ignore_patterns=["*.pt", "original/*"],  # skip redundant formats
    )

    vol.commit()
    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min")


@app.local_entrypoint()
def main(model_key: str):
    print(f"Spawning CPU download for {model_key} ...")
    download.remote(model_key)
    print("Done.")
