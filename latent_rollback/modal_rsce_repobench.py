"""
Modal job: RSCE RepoBench-C benchmark (full scale, 4 models).

Runs all 4 models on RepoBench-C cross_file_first split:
  n_per_model = 200 (stratified random, seed=42)

Conditions per example:
  baseline  — full cross-file context + in-file context in prompt
  vec       — cross-file context compressed to vector; in-file in prompt
  vec_f     — vec + BM25 code-fact F block

Models:
  llama3-8b     meta-llama/Meta-Llama-3-8B-Instruct      A10G
  qwen25-7b     Qwen/Qwen2.5-7B-Instruct                 A10G
  deepseek-14b  deepseek-ai/DeepSeek-R1-Distill-Qwen-14B A100-40GB
  mistral-24b   mistralai/Mistral-Small-24B-Instruct-2501 A100-40GB

Results saved to benchmark_results/rsce_repobench_full.csv

Usage:
  modal run modal_rsce_repobench.py
  modal run modal_rsce_repobench.py --model-key llama3-8b
"""

from __future__ import annotations

import csv
import difflib
import io
import json
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

# ---------------------------------------------------------------------------
# Model registry — mirrors benchmark_runner.py MODEL_MATRIX
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "llama3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "injection_layer": 14,   # 44% depth, calibrated
        "gpu": "A10G",
    },
    "qwen25-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "injection_layer": 17,   # 61% depth
        "gpu": "A10G",
    },
    "deepseek-14b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "injection_layer": 29,   # 60% depth (code calibration)
        "gpu": "A100-40GB",
    },
    "mistral-24b": {
        "hf_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "injection_layer": 10,   # 25% depth
        "gpu": "A100-40GB",
    },
}

N_PER_MODEL = 200
SEED = 42
MAX_NEW_TOKENS = 50


# ---------------------------------------------------------------------------
# Core worker
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/weights": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=81920,
)
def run_repobench_model(model_key: str) -> str:
    """Runs one model. Returns CSV string."""
    import random
    import torch
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_REGISTRY[model_key]
    hf_id = cfg["hf_id"]
    layer = cfg["injection_layer"]
    _EPS = 1e-8

    # ---- inline helpers ---------------------------------------------------

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

    def extract_facts_code(ctx: str, query: str, top_k: int = 5) -> str:
        """BM25-ranked function signatures, class defs, imports from ctx."""
        candidates: list[str] = []

        for m in re.finditer(
            r'^\s*(?:async\s+)?def\s+(\w+\s*\([^)]*\)(?:\s*->\s*[^\n:]+)?)',
            ctx, re.MULTILINE
        ):
            candidates.append(m.group(1).strip())

        for m in re.finditer(
            r'^\s*class\s+(\w+(?:\s*\([^)]*\))?)',
            ctx, re.MULTILINE
        ):
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
        selected = [c for _, c in ranked[:top_k]]
        return "Facts: " + "; ".join(selected)

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

    # ---- load model -------------------------------------------------------

    print(f"Loading {hf_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir="/weights")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir="/weights",
    )
    model.eval()
    print(f"Loaded. Layers: {model.config.num_hidden_layers}")

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
        return out[0][input_ids.shape[1]:].tolist()

    def extract_vector(text: str) -> tuple[torch.Tensor, int]:
        ids = encode(text)
        inp = torch.tensor([ids], dtype=torch.long).to(model.device)
        with torch.no_grad():
            out = model(inp, output_hidden_states=True)
        h = out.hidden_states[layer + 1][0].mean(dim=0).float().cpu()
        return h, len(ids)

    def gen_with_injection(prompt: str, ctx_vec: torch.Tensor,
                           scale: float = 1.0) -> str:
        norm = ctx_vec.norm().item()
        if norm < _EPS:
            return ""
        v = (ctx_vec / norm).to(model.device, dtype=torch.bfloat16)

        def hook(module, inp, output):
            if isinstance(output, tuple):
                h = output[0] + scale * v[None, None, :]
                return (h,) + output[1:]
            else:
                return output + scale * v[None, None, :]

        handle = model.model.layers[layer].register_forward_hook(hook)
        ids = encode(prompt)
        inp = torch.tensor([ids], dtype=torch.long).to(model.device)
        try:
            new_ids = gen(inp)
        finally:
            handle.remove()
        return truncate_stop(decode(new_ids))

    # ---- load dataset -----------------------------------------------------

    print("Loading RepoBench-C ...")
    ds = load_dataset("tianyang/repobench_python_v1.1", split="cross_file_first", trust_remote_code=True)
    rows = [r for r in ds if r.get("next_line", "").strip()]
    rng = random.Random(SEED)
    selected = rng.sample(rows, min(N_PER_MODEL, len(rows)))
    print(f"  Selected {len(selected)} examples")

    # ---- run conditions ---------------------------------------------------

    results = []
    for i, row in enumerate(selected):
        cross_file = format_cross_file(row)
        in_file    = format_in_file(row)
        next_line  = row.get("next_line", "").strip()
        ex_id      = f"{row.get('repo_name', '')}#{i}"
        query      = in_file[-200:] if len(in_file) > 200 else in_file

        baseline_prompt = (
            f"# Cross-file context:\n{cross_file}\n\n"
            f"# Complete the next line:\n{in_file}"
        )
        query_prompt = f"# Complete the next line:\n{in_file}"

        n_baseline = len(encode(baseline_prompt))
        n_query    = len(encode(query_prompt))
        tok_red    = round(1.0 - n_query / n_baseline, 4) if n_baseline > 0 else 0.0

        t0 = time.time()

        # baseline
        b_ids = gen(torch.tensor([encode(baseline_prompt)], dtype=torch.long).to(model.device))
        b_text = truncate_stop(decode(b_ids))
        b_es = edit_sim(b_text, next_line)

        # vec
        ctx_vec, n_ctx = extract_vector(cross_file)
        v_text = gen_with_injection(query_prompt, ctx_vec)
        v_es = edit_sim(v_text, next_line)

        # vec+f
        facts = extract_facts_code(cross_file, query)
        vf_prompt = f"{facts}\n\n# Complete the next line:\n{in_file}" if facts else query_prompt
        vf_text = gen_with_injection(vf_prompt, ctx_vec)
        vf_es = edit_sim(vf_text, next_line)

        elapsed = time.time() - t0

        results.append({
            "model": model_key,
            "example_id": ex_id,
            "n_context_tokens": n_ctx,
            "n_baseline_tokens": n_baseline,
            "token_reduction": tok_red,
            "baseline_editsim": round(b_es, 4),
            "vec_editsim": round(v_es, 4),
            "vecf_editsim": round(vf_es, 4),
            "baseline_output": b_text[:150],
            "vec_output": v_text[:150],
            "vecf_output": vf_text[:150],
            "gold": next_line[:150],
            "elapsed": round(elapsed, 2),
        })

        print(
            f"[{model_key}] {i+1}/{len(selected)} "
            f"base={b_es:.3f} vec={v_es:.3f} vecf={vf_es:.3f} "
            f"tokred={tok_red:.1%}"
        )

    # ---- summary ----------------------------------------------------------

    def avg(key):
        return sum(r[key] for r in results) / len(results) if results else 0.0

    print(f"\n=== {model_key} RESULTS ===")
    print(f"  baseline  EditSim={avg('baseline_editsim'):.3f}")
    print(f"  vec       EditSim={avg('vec_editsim'):.3f}  delta={avg('vec_editsim')-avg('baseline_editsim'):+.3f}")
    print(f"  vec+f     EditSim={avg('vecf_editsim'):.3f}  delta={avg('vecf_editsim')-avg('baseline_editsim'):+.3f}")
    print(f"  avg token reduction: {avg('token_reduction'):.1%}")

    buf = io.StringIO()
    if results:
        writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(model_key: str = ""):
    """
    Run one model (--model-key llama3-8b) or all four in parallel.
    Results merged into a single CSV.
    """
    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "rsce_repobench_full.csv"

    keys = [model_key] if model_key else list(MODEL_REGISTRY.keys())
    print(f"Running models: {keys}")

    # Dispatch all in parallel
    all_csv = list(run_repobench_model.map(keys))

    # Merge CSVs
    all_rows = []
    header = None
    for csv_data in all_csv:
        reader = csv.DictReader(io.StringIO(csv_data))
        if header is None:
            header = reader.fieldnames
        all_rows.extend(reader)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to {out_path}")
