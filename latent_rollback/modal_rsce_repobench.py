"""
Modal job: RSCE RepoBench-C benchmark (full scale, 4 models).

Each model runs in its own isolated function with its own GPU allocation:
  llama3-8b    1x H100  (8B  bfloat16 ~16GB)
  qwen25-7b    1x H100  (7B  bfloat16 ~14GB)
  deepseek-14b 2x H100  (14B bfloat16 ~28GB + activations)
  mistral-24b  2x H100  (24B bfloat16 ~48GB + activations)

Cache eviction (torch.cuda.empty_cache + explicit del) after every inference.
"""

from __future__ import annotations

import csv
import difflib
import io
import re
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image and app
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

app = modal.App("rsce-repobench-benchmark", image=image)
vol = modal.Volume.from_name("rsce-hf-weights", create_if_missing=True)

COMMON = dict(
    timeout=7200,
    volumes={"/weights": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)

MODEL_REGISTRY: dict[str, dict] = {
    "llama3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "injection_layer": 14,   # 44% of 32 layers
    },
    "qwen25-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "injection_layer": 17,   # 61% of 28 layers
    },
    "deepseek-14b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "injection_layer": 29,   # 60% of 48 layers
    },
    "mistral-24b": {
        "hf_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "injection_layer": 10,   # 25% of 40 layers
    },
    # Larger models for scaling study
    "qwen25-32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "injection_layer": 32,   # 50% of 64 layers
    },
    "deepseek-67b": {
        "hf_id": "deepseek-ai/deepseek-llm-67b-chat",
        "injection_layer": 57,   # 60% of 95 layers
    },
    "llama31-70b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "injection_layer": 35,   # 44% of 80 layers (same depth % as 8B)
    },
}

N_PER_MODEL = 200
SEED = 42
MAX_NEW_TOKENS = 50


# ---------------------------------------------------------------------------
# Shared benchmark logic (inlined per function to avoid import issues)
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY = 10  # flush partial CSV to volume every N examples


def _run_benchmark(model_key: str, checkpoint_path: str | None = None) -> str:
    import gc
    import random
    import torch
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_REGISTRY[model_key]
    hf_id = cfg["hf_id"]
    layer = cfg["injection_layer"]
    _EPS = 1e-8

    CODE_STOP = ("\n\n", "\ndef ", "\nclass ", "```")

    def truncate_stop(text, stops=CODE_STOP):
        best = len(text)
        for s in stops:
            idx = text.find(s)
            if 0 <= idx < best:
                best = idx
        return text[:best].rstrip()

    def edit_sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()

    def clear():
        gc.collect()
        torch.cuda.empty_cache()

    def extract_facts_code(ctx: str, query: str, top_k: int = 5) -> str:
        candidates: list[str] = []
        for m in re.finditer(
            r'^\s*(?:async\s+)?def\s+(\w+\s*\([^)]*\)(?:\s*->\s*[^\n:]+)?)',
            ctx, re.MULTILINE
        ):
            candidates.append(m.group(1).strip())
        for m in re.finditer(r'^\s*class\s+(\w+(?:\s*\([^)]*\))?)', ctx, re.MULTILINE):
            candidates.append(m.group(1).strip())
        for m in re.finditer(r'^\s*(?:from\s+\S+\s+)?import\s+.+', ctx, re.MULTILINE):
            candidates.append(m.group().strip())
        for m in re.finditer(r'^\s*([A-Z_]{2,}\s*=\s*[^\n]+)', ctx, re.MULTILINE):
            candidates.append(m.group(1).strip())
        if not candidates:
            return ""
        tok = [c.lower().split() for c in candidates]
        bm25 = BM25Okapi(tok)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(scores, candidates), reverse=True)
        return "Facts: " + "; ".join(c for _, c in ranked[:top_k])

    def format_cross_file(row: dict) -> str:
        parts = [f"# Repo: {row.get('repo_name', '')}"]
        for snippet in row.get("context", []):
            if isinstance(snippet, dict):
                parts.append(f"# Path: {snippet.get('path', '')}\n{snippet.get('snippet', '')}")
            else:
                parts.append(str(snippet))
        return "\n\n".join(p for p in parts if p)

    def format_in_file(row: dict) -> str:
        return (
            f"# Path: {row.get('file_path', '')}\n"
            f"{row.get('import_statement', '')}\n"
            f"{row.get('cropped_code', '')}"
        )

    # ---- load model (retry up to 3x for transient HF CDN errors) ----------
    import os
    os.environ["HF_HOME"] = "/weights"
    os.environ["TRANSFORMERS_CACHE"] = "/weights"

    for attempt in range(10):
        try:
            print(f"Loading {hf_id} (attempt {attempt+1}) ...")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id, cache_dir="/weights",
                resume_download=True,
            )
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
            print(f"  Load failed (attempt {attempt+1}): {e}")
            if attempt == 9:
                raise
            wait = min(30, 5 * (attempt + 1))
            print(f"  Retrying in {wait}s ...")
            time.sleep(wait)
    model.eval()
    print(f"Loaded. Layers: {model.config.num_hidden_layers}")
    clear()

    # With device_map="auto" the model may span multiple GPUs.
    # Use the first parameter's device for input tensors (embedding is always first).
    input_device = next(model.parameters()).device
    # The injection layer's actual device (may differ from input_device on multi-GPU).
    layer_device = next(model.model.layers[layer].parameters()).device
    print(f"  input_device={input_device}  layer_device={layer_device}")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    def gen(input_ids):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0][input_ids.shape[1]:].tolist()
        del out
        return new_ids

    def extract_vector(text: str) -> tuple[torch.Tensor, int]:
        ids = encode(text)
        inp = torch.tensor([ids], dtype=torch.long).to(input_device)
        with torch.no_grad():
            out = model(inp, output_hidden_states=True)
        h = out.hidden_states[layer + 1][0].mean(dim=0).float().cpu()
        del inp, out
        clear()
        return h, len(ids)

    def gen_with_injection(prompt: str, ctx_vec: torch.Tensor, scale: float = 1.0) -> str:
        norm = ctx_vec.norm().item()
        if norm < _EPS:
            return ""
        # Pin v to the injection layer's device — avoids cross-GPU mismatch.
        v = (ctx_vec / norm).to(layer_device, dtype=torch.bfloat16)

        def hook(module, inp, output):
            if isinstance(output, tuple):
                return (output[0] + scale * v[None, None, :],) + output[1:]
            return output + scale * v[None, None, :]

        handle = model.model.layers[layer].register_forward_hook(hook)
        ids = encode(prompt)
        inp = torch.tensor([ids], dtype=torch.long).to(input_device)
        try:
            new_ids = gen(inp)
        finally:
            handle.remove()
            del inp, v
            clear()
        return truncate_stop(decode(new_ids))

    # ---- load dataset -----------------------------------------------------
    print("Loading RepoBench-C ...")
    ds = load_dataset("tianyang/repobench_python_v1.1", split="cross_file_first", trust_remote_code=True)
    rows = [r for r in ds if r.get("next_line", "").strip()]
    rng = random.Random(SEED)
    selected = rng.sample(rows, min(N_PER_MODEL, len(rows)))
    print(f"  Selected {len(selected)} examples")

    # ---- resume from checkpoint if available ------------------------------
    results = []
    resume_from = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as _f:
                _reader = csv.DictReader(_f)
                results = list(_reader)
            resume_from = len(results)
            print(f"  Resuming from checkpoint: {resume_from} examples already done")
        except Exception as _e:
            print(f"  Checkpoint load failed ({_e}), starting fresh")
            results = []
            resume_from = 0

    def _flush_checkpoint():
        if checkpoint_path and results:
            try:
                with open(checkpoint_path, "w") as _f:
                    _w = csv.DictWriter(_f, fieldnames=list(results[0].keys()))
                    _w.writeheader()
                    _w.writerows(results)
                print(f"  Checkpoint saved: {len(results)} rows")
            except Exception as _e:
                print(f"  Checkpoint save failed: {_e}")

    # ---- run conditions ---------------------------------------------------
    for i, row in enumerate(selected):
        if i < resume_from:
            continue
        cross_file = format_cross_file(row)
        in_file    = format_in_file(row)
        next_line  = row.get("next_line", "").strip()
        ex_id      = f"{row.get('repo_name', '')}#{i}"
        query      = in_file[-200:] if len(in_file) > 200 else in_file

        baseline_prompt = f"# Cross-file context:\n{cross_file}\n\n# Complete the next line:\n{in_file}"
        query_prompt    = f"# Complete the next line:\n{in_file}"

        n_baseline = len(encode(baseline_prompt))
        n_query    = len(encode(query_prompt))
        tok_red    = round(1.0 - n_query / n_baseline, 4) if n_baseline > 0 else 0.0

        t0 = time.time()

        try:
            # baseline
            b_inp  = torch.tensor([encode(baseline_prompt)], dtype=torch.long).to(input_device)
            b_ids  = gen(b_inp)
            del b_inp
            clear()
            b_text = truncate_stop(decode(b_ids))
            b_es   = edit_sim(b_text, next_line)

            # vec
            ctx_vec, n_ctx = extract_vector(cross_file)
            v_text = gen_with_injection(query_prompt, ctx_vec)
            v_es   = edit_sim(v_text, next_line)

            # vec+f
            facts     = extract_facts_code(cross_file, query)
            vf_prompt = f"{facts}\n\n# Complete the next line:\n{in_file}" if facts else query_prompt
            vf_text   = gen_with_injection(vf_prompt, ctx_vec)
            vf_es     = edit_sim(vf_text, next_line)
            del ctx_vec
            clear()
        except Exception as _ex:
            print(f"  [WARN] example {i} failed: {_ex} — skipping")
            clear()
            continue

        elapsed = time.time() - t0

        results.append({
            "model": model_key,
            "example_id": ex_id,
            "n_context_tokens": n_ctx,
            "n_baseline_tokens": n_baseline,
            "token_reduction": tok_red,
            "baseline_editsim": round(b_es, 4),
            "vec_editsim":      round(v_es, 4),
            "vecf_editsim":     round(vf_es, 4),
            "baseline_output":  b_text[:150],
            "vec_output":       v_text[:150],
            "vecf_output":      vf_text[:150],
            "gold":             next_line[:150],
            "elapsed":          round(elapsed, 2),
        })

        print(
            f"[{model_key}] {i+1}/{len(selected)} "
            f"base={b_es:.3f} vec={v_es:.3f} vecf={vf_es:.3f} "
            f"tokred={tok_red:.1%}"
        )

        if len(results) % CHECKPOINT_EVERY == 0:
            _flush_checkpoint()

    def avg(key):
        return sum(r[key] for r in results) / len(results) if results else 0.0

    print(f"\n=== {model_key} RESULTS ===")
    print(f"  baseline  EditSim={avg('baseline_editsim'):.3f}")
    print(f"  vec       EditSim={avg('vec_editsim'):.3f}  delta={avg('vec_editsim')-avg('baseline_editsim'):+.3f}")
    print(f"  vec+f     EditSim={avg('vecf_editsim'):.3f}  delta={avg('vecf_editsim')-avg('baseline_editsim'):+.3f}")
    print(f"  avg token reduction: {avg('token_reduction'):.1%}")

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# One function per GPU tier — each runs in a fully isolated container
# ---------------------------------------------------------------------------

results_vol = modal.Volume.from_name("rsce-repobench-results", create_if_missing=True)

SMALL_MODELS = {"llama3-8b", "qwen25-7b"}        # 1x H100 (~7-8B bfloat16)
LARGE_MODELS  = {"deepseek-14b", "mistral-24b", "qwen25-32b"}  # 2x H100 (~14-32B)
XL_MODELS     = {"deepseek-67b", "llama33-70b"}  # 4x H100 (~67-70B bfloat16)


@app.function(gpu="H100", memory=81920, timeout=7200,
              volumes={"/weights": vol, "/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_model_small(model_key: str) -> None:
    """1x H100 — for 7-8B models."""
    import os
    checkpoint = f"/results/repobench_{model_key}_partial.csv"
    csv_data = _run_benchmark(model_key, checkpoint_path=checkpoint)
    out_path = f"/results/repobench_{model_key}.csv"
    with open(out_path, "w") as f:
        f.write(csv_data)
    results_vol.commit()
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print(f"Saved {out_path}")


@app.function(gpu="H100:2", memory=163840, timeout=7200,
              volumes={"/weights": vol, "/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_model(model_key: str) -> None:
    """2x H100 — for 14-32B models."""
    import os
    checkpoint = f"/results/repobench_{model_key}_partial.csv"
    csv_data = _run_benchmark(model_key, checkpoint_path=checkpoint)
    out_path = f"/results/repobench_{model_key}.csv"
    with open(out_path, "w") as f:
        f.write(csv_data)
    results_vol.commit()
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print(f"Saved {out_path}")


@app.function(gpu="H100:4", memory=327680, timeout=10800,
              volumes={"/weights": vol, "/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_model_xl(model_key: str) -> None:
    """4x H100 — for 67-70B models."""
    import os
    checkpoint = f"/results/repobench_{model_key}_partial.csv"
    csv_data = _run_benchmark(model_key, checkpoint_path=checkpoint)
    out_path = f"/results/repobench_{model_key}.csv"
    with open(out_path, "w") as f:
        f.write(csv_data)
    results_vol.commit()
    if os.path.exists(checkpoint):
        os.remove(checkpoint)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(model_key: str = ""):
    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)

    keys = [model_key] if model_key else list(MODEL_REGISTRY.keys())
    print(f"Spawning independently (2x H100 each): {keys}")
    print("Each model is isolated — cancelling one will NOT affect others.")
    print()

    # Spawn each model as a fully independent function call.
    # Small models (7-8B) get 1x H100; large models get 2x H100.
    handles = {
        k: (run_model_small if k in SMALL_MODELS else run_model).spawn(k)
        for k in keys
    }

    print("All spawned. Waiting for results...")
    for k, handle in handles.items():
        try:
            handle.get()
            print(f"  {k}: done")
        except Exception as e:
            print(f"  {k}: FAILED — {e}")

    # Collect CSVs written to the results volume
    print("\nCollecting results from volume...")
    all_rows = []
    header = None
    for k in keys:
        local_path = out_dir / f"repobench_{k}.csv"
        try:
            # Download from volume via modal volume get
            import subprocess
            subprocess.run(
                ["python3", "-m", "modal", "volume", "get",
                 "rsce-repobench-results", f"repobench_{k}.csv",
                 str(local_path)],
                check=True, capture_output=True
            )
            with open(local_path) as f:
                reader = csv.DictReader(f)
                if header is None:
                    header = reader.fieldnames
                rows = list(reader)
                all_rows.extend(rows)
                print(f"  {k}: {len(rows)} rows")
        except Exception as e:
            print(f"  {k}: could not retrieve — {e}")

    if not all_rows:
        print("No results collected.")
        return

    out_path = out_dir / "rsce_repobench_full.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows → {out_path}")
