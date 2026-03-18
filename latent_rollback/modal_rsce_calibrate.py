"""
Modal job: calibrate f(M) for each model.

Sweeps candidate injection layers over a small held-out RepoBench-C sample
(n=20, seed=99 — distinct from the benchmark seed=42) and returns the layer
that maximises mean EditSim under the Vec-only condition.

Usage:
  # calibrate a single model
  modal run modal_rsce_calibrate.py --model-key qwen25-32b
  modal run modal_rsce_calibrate.py --model-key deepseek-67b
  modal run modal_rsce_calibrate.py --model-key llama31-70b

Output printed to stdout and saved to rsce-repobench-results volume as
  calibration_{model_key}.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image / app — same as benchmark
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime",
        add_python="3.11",
    )
    .pip_install(
        "transformers>=4.43.0,<5.0.0",
        "accelerate>=0.30.0",
        "datasets>=2.20.0,<3.0.0",
        "huggingface_hub>=0.23.0",
        "rank_bm25>=0.2.2",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App("rsce-calibrate", image=image)
vol         = modal.Volume.from_name("rsce-hf-weights",        create_if_missing=True)
results_vol = modal.Volume.from_name("rsce-repobench-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Model registry — must include n_layers for sweep bounds
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "qwen25-32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "n_layers": 64,
    },
    "deepseek-67b": {
        "hf_id": "deepseek-ai/deepseek-llm-67b-chat",
        "n_layers": 95,
    },
    "llama31-70b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "n_layers": 80,
    },
}

CALIB_N      = 20    # examples per calibration run
CALIB_SEED   = 99    # different from benchmark seed=42
MAX_NEW_TOKENS = 50
LAYER_STRIDE   = 2   # step size for layer sweep


# ---------------------------------------------------------------------------
# Core calibration logic (runs on GPU)
# ---------------------------------------------------------------------------

def _calibrate(model_key: str) -> dict:
    import gc
    import difflib
    import random
    import re
    import os
    import torch
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/weights"
    os.environ["TRANSFORMERS_CACHE"] = "/weights"

    cfg     = MODEL_REGISTRY[model_key]
    hf_id   = cfg["hf_id"]
    n_layers = cfg["n_layers"]

    # Sweep range: [25%, 85%] of total layers, stride 2
    lo  = max(1, n_layers // 4)
    hi  = int(0.85 * n_layers)
    candidates = list(range(lo, hi + 1, LAYER_STRIDE))
    print(f"Sweeping layers {lo}..{hi} (stride {LAYER_STRIDE}) — {len(candidates)} candidates")

    CODE_STOP = ("\n\n", "\ndef ", "\nclass ", "```")

    def truncate_stop(text):
        best = len(text)
        for s in CODE_STOP:
            idx = text.find(s)
            if 0 <= idx < best:
                best = idx
        return text[:best].rstrip()

    def edit_sim(a, b):
        return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()

    def clear():
        gc.collect()
        torch.cuda.empty_cache()

    # ---- load model -------------------------------------------------------
    for attempt in range(10):
        try:
            print(f"Loading {hf_id} (attempt {attempt+1}) ...")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id, cache_dir="/weights", resume_download=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="/weights",
                resume_download=True,
            )
            break
        except Exception as e:
            print(f"  Load failed: {e}")
            if attempt == 9:
                raise
            time.sleep(min(30, 5 * (attempt + 1)))

    model.eval()
    print(f"Loaded. Layers: {model.config.num_hidden_layers}")
    clear()

    input_device = next(model.parameters()).device

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    # ---- load calibration dataset ----------------------------------------
    print("Loading RepoBench-C calibration split ...")
    ds   = load_dataset("tianyang/repobench_python_v1.1",
                        split="cross_file_first", trust_remote_code=True)
    rows = [r for r in ds if r.get("next_line", "").strip()]
    rng  = random.Random(CALIB_SEED)
    selected = rng.sample(rows, min(CALIB_N, len(rows)))
    print(f"  {len(selected)} calibration examples")

    def format_cross_file(row):
        parts = [f"# Repo: {row.get('repo_name', '')}"]
        for snippet in row.get("context", []):
            if isinstance(snippet, dict):
                parts.append(f"# Path: {snippet.get('path', '')}\n{snippet.get('snippet', '')}")
            else:
                parts.append(str(snippet))
        return "\n\n".join(p for p in parts if p)

    def format_in_file(row):
        return (f"# Path: {row.get('file_path', '')}\n"
                f"{row.get('import_statement', '')}\n"
                f"{row.get('cropped_code', '')}")

    # Pre-compute cross-file texts (shared across all layer trials)
    samples = [
        {
            "cross_file": format_cross_file(r),
            "in_file":    format_in_file(r),
            "next_line":  r.get("next_line", "").strip(),
        }
        for r in selected
    ]

    # ---- sweep layers -----------------------------------------------------
    results: dict[int, float] = {}

    for layer in candidates:
        layer_device = next(model.model.layers[layer].parameters()).device
        scores = []

        for s in samples:
            cross_file = s["cross_file"]
            query_prompt = f"# Complete the next line:\n{s['in_file']}"
            next_line   = s["next_line"]

            # Extract vector at this candidate layer
            try:
                ids = encode(cross_file)
                inp = torch.tensor([ids], dtype=torch.long).to(input_device)
                with torch.no_grad():
                    out = model(inp, output_hidden_states=True)
                h = out.hidden_states[layer + 1][0].mean(dim=0).float().cpu()
                del inp, out
                clear()

                norm = h.norm().item()
                if norm < 1e-8:
                    scores.append(0.0)
                    continue

                v = (h / norm).to(layer_device, dtype=torch.bfloat16)

                def hook(module, inp_, output):
                    if isinstance(output, tuple):
                        return (output[0] + v[None, None, :],) + output[1:]
                    return output + v[None, None, :]

                handle = model.model.layers[layer].register_forward_hook(hook)
                q_ids = encode(query_prompt)
                q_inp = torch.tensor([q_ids], dtype=torch.long).to(input_device)
                try:
                    with torch.no_grad():
                        out2 = model.generate(
                            q_inp,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    new_ids = out2[0][q_inp.shape[1]:].tolist()
                    del out2
                finally:
                    handle.remove()
                    del q_inp, v
                    clear()

                pred = truncate_stop(decode(new_ids))
                scores.append(edit_sim(pred, next_line))

            except Exception as e:
                print(f"  [WARN] layer={layer} example failed: {e}")
                clear()
                scores.append(0.0)

        mean_es = sum(scores) / len(scores) if scores else 0.0
        results[layer] = round(mean_es, 4)
        print(f"  layer={layer:3d}  mean_EditSim={mean_es:.4f}")

    best_layer = max(results, key=results.__getitem__)
    best_score = results[best_layer]

    print(f"\n=== BEST LAYER for {model_key}: {best_layer} "
          f"({best_layer/n_layers:.0%} depth)  EditSim={best_score:.4f} ===")

    return {
        "model_key":   model_key,
        "hf_id":       hf_id,
        "n_layers":    n_layers,
        "best_layer":  best_layer,
        "best_depth":  round(best_layer / n_layers, 3),
        "best_editsim": best_score,
        "sweep":       results,
    }


# ---------------------------------------------------------------------------
# GPU functions — one per tier
# ---------------------------------------------------------------------------

@app.function(gpu="H100:2", memory=163840, timeout=7200,
              volumes={"/weights": vol, "/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def calibrate_large(model_key: str) -> dict:
    """2x H100 — Qwen2.5-32B."""
    result = _calibrate(model_key)
    path = f"/results/calibration_{model_key}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(f"Saved {path}")
    return result


@app.function(gpu="H100:4", memory=327680, timeout=10800,
              volumes={"/weights": vol, "/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def calibrate_xl(model_key: str) -> dict:
    """4x H100 — DeepSeek-67B, LLaMA-3.1-70B."""
    result = _calibrate(model_key)
    path = f"/results/calibration_{model_key}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(f"Saved {path}")
    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

XL_MODELS = {"deepseek-67b", "llama31-70b"}

@app.local_entrypoint()
def main(model_key: str = ""):
    keys = [model_key] if model_key else list(MODEL_REGISTRY.keys())
    print(f"Calibrating: {keys}")

    handles = {}
    for k in keys:
        fn = calibrate_xl if k in XL_MODELS else calibrate_large
        handles[k] = fn.spawn(k)
        print(f"  Spawned {k}")

    print("\nWaiting for results...")
    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)

    for k, handle in handles.items():
        try:
            result = handle.get()
            print(f"\n{k}: best_layer={result['best_layer']} "
                  f"({result['best_depth']:.0%} depth) "
                  f"EditSim={result['best_editsim']:.4f}")
            (out_dir / f"calibration_{k}.json").write_text(
                json.dumps(result, indent=2))
        except Exception as e:
            print(f"  {k}: FAILED — {e}")
