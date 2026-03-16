"""
Modal job: RSCE QA benchmark (full LongBench splits).

Runs LLaMA-3 8B on the full LongBench test splits:
  - HotpotQA:        200 examples
  - 2WikiMultiHopQA: 200 examples
  Total: 400 examples

Conditions per example:
  baseline  — full context + question in prompt
  q_only    — question only, no context, no vector
  vec       — question + vector injection, no fact block
  vec_f     — question + vector injection + NER fact block

Results saved to /results/rsce_qa_full.csv

Usage (from your Mac):
  modal run modal_rsce_qa.py
  modal run modal_rsce_qa.py --model-id meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal image and app
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
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App("rsce-qa-benchmark", image=image)

# Persistent volume to cache HF model weights across runs
vol = modal.Volume.from_name("rsce-hf-weights", create_if_missing=True)

HF_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
INJECTION_LAYER = 14   # calibrated f(M) for LLaMA-3 8B
N_PER_TASK = 200       # full LongBench test split per task
SEED = 42
MAX_NEW_TOKENS = 80


# ---------------------------------------------------------------------------
# Core worker — runs on GPU
# ---------------------------------------------------------------------------

@app.function(
    gpu="A100",
    timeout=7200,
    volumes={"/weights": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=32768,
)
def run_qa_benchmark(model_id: str = HF_MODEL_ID) -> str:
    """Returns CSV string of all results."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- inline helpers (no local imports on Modal) -----------------------

    QA_STOP = ("\n", "\nQuestion:", "\nFacts:", "\nContext:")
    _EPS = 1e-8

    def truncate_stop(text, stops=QA_STOP):
        best = len(text)
        for s in stops:
            idx = text.find(s)
            if 0 <= idx < best:
                best = idx
        return text[:best]

    def grade_qa(generated, golds):
        gen_norm = generated.lower().strip()
        gen_toks = set(re.sub(r"[^\w\s]", "", gen_norm).split())
        best_f1, exact = 0.0, False
        for gold in golds:
            g = gold.lower().strip()
            if g in gen_norm:
                exact = True
            gt = set(re.sub(r"[^\w\s]", "", g).split())
            if gt and gen_toks:
                p = len(gen_toks & gt) / len(gen_toks)
                r = len(gen_toks & gt) / len(gt)
                f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
            else:
                f1 = 0.0
            best_f1 = max(best_f1, f1)
        return {"exact_match": exact, "f1": round(best_f1, 4)}

    def extract_ner_facts(text: str, max_facts: int = 15) -> str:
        """Rule-based NER: pull capitalised phrases + numbers."""
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # proper nouns
            r'\b\d{4}\b',                              # years
            r'\b\d+(?:\.\d+)?(?:\s*(?:km|m|kg|mph|lb|ft))?\b',  # numbers/units
        ]
        facts = []
        seen = set()
        for pat in patterns:
            for m in re.finditer(pat, text):
                f = m.group().strip()
                if f.lower() not in seen and len(f) > 1:
                    seen.add(f.lower())
                    facts.append(f)
        return "Facts: " + "; ".join(facts[:max_facts])

    # ---- load model -------------------------------------------------------

    cache_dir = "/weights"
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded. Layers: {n_layers}  d_model: {model.config.hidden_size}")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    def gen(input_ids, max_new=MAX_NEW_TOKENS):
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        return out[0][input_ids.shape[1]:].tolist()

    def extract_vector(context_text: str, layer: int) -> tuple[torch.Tensor, int]:
        ids = encode(context_text)
        inp = torch.tensor([ids], dtype=torch.long).to(model.device)
        with torch.no_grad():
            out = model(inp, output_hidden_states=True)
        h = out.hidden_states[layer + 1][0].mean(dim=0).float().cpu()
        return h, len(ids)

    def gen_with_injection(question_text: str, ctx_vec: torch.Tensor,
                           layer: int, scale: float = 1.0) -> tuple[str, int]:
        norm = ctx_vec.norm().item()
        if norm < _EPS:
            raise ValueError("zero vector")
        v = (ctx_vec / norm).to(model.device, dtype=torch.bfloat16)

        def hook(module, inp, output):
            if isinstance(output, tuple):
                h = output[0] + scale * v[None, None, :]
                return (h,) + output[1:]
            else:
                return output + scale * v[None, None, :]

        handle = model.model.layers[layer].register_forward_hook(hook)
        ids = encode(question_text)
        inp = torch.tensor([ids], dtype=torch.long).to(model.device)
        try:
            new_ids = gen(inp)
        finally:
            handle.remove()
        text = truncate_stop(decode(new_ids))
        return text, len(ids)

    # ---- load dataset -----------------------------------------------------

    print("Loading LongBench ...")
    results = []

    for task in ("hotpotqa", "2wikimqa"):
        ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
        import random
        rng = random.Random(SEED)
        rows = list(ds)
        # filter by context length (200-4000 words)
        rows = [r for r in rows
                if 200 <= len(r.get("context", "").split()) <= 12000]
        selected = rng.sample(rows, min(N_PER_TASK, len(rows)))
        print(f"  {task}: {len(selected)} examples after filtering")

        for i, row in enumerate(selected):
            context = row.get("context", "")
            question = row.get("input", "")
            answers = row.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]
            ex_id = row.get("_id", str(i))

            baseline_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            qonly_prompt    = f"Question: {question}\nAnswer:"

            n_baseline_toks = len(encode(baseline_prompt))
            tok_red = round(1.0 - (len(encode(qonly_prompt)) / n_baseline_toks), 4)

            t0 = time.time()

            # 1. Baseline
            b_ids = gen(torch.tensor([encode(baseline_prompt)], dtype=torch.long).to(model.device))
            b_text = truncate_stop(decode(b_ids))
            b_grades = grade_qa(b_text, answers)

            # 2. Q-only
            q_ids = gen(torch.tensor([encode(qonly_prompt)], dtype=torch.long).to(model.device))
            q_text = truncate_stop(decode(q_ids))
            q_grades = grade_qa(q_text, answers)

            # 3. Vec-only
            ctx_vec, n_ctx_toks = extract_vector(context, INJECTION_LAYER)
            v_text, _ = gen_with_injection(qonly_prompt, ctx_vec, INJECTION_LAYER)
            v_grades = grade_qa(v_text, answers)

            # 4. Vec+F (NER fact block)
            facts = extract_ner_facts(context)
            vf_prompt = f"{facts}\n\nQuestion: {question}\nAnswer:"
            vf_text, _ = gen_with_injection(vf_prompt, ctx_vec, INJECTION_LAYER)
            vf_grades = grade_qa(vf_text, answers)

            elapsed = time.time() - t0

            results.append({
                "task": task,
                "example_id": ex_id,
                "n_context_tokens": n_ctx_toks,
                "n_baseline_tokens": n_baseline_toks,
                "token_reduction": tok_red,
                # baseline
                "baseline_f1": b_grades["f1"],
                "baseline_em": int(b_grades["exact_match"]),
                "baseline_output": b_text[:200],
                # q_only
                "qonly_f1": q_grades["f1"],
                "qonly_em": int(q_grades["exact_match"]),
                "qonly_output": q_text[:200],
                # vec
                "vec_f1": v_grades["f1"],
                "vec_em": int(v_grades["exact_match"]),
                "vec_output": v_text[:200],
                # vec+f
                "vecf_f1": vf_grades["f1"],
                "vecf_em": int(vf_grades["exact_match"]),
                "vecf_output": vf_text[:200],
                "gold": json.dumps(answers),
                "elapsed": round(elapsed, 2),
            })

            print(
                f"[{task}] {i+1}/{len(selected)} "
                f"base_F1={b_grades['f1']:.3f} "
                f"qonly_F1={q_grades['f1']:.3f} "
                f"vec_F1={v_grades['f1']:.3f} "
                f"vecf_F1={vf_grades['f1']:.3f} "
                f"| {question[:50]!r}"
            )

    # ---- summary ----------------------------------------------------------

    def avg(key, task_filter=None):
        rows = [r for r in results if task_filter is None or r["task"] == task_filter]
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    print("\n=== RESULTS ===")
    for cond in ("baseline", "qonly", "vec", "vecf"):
        f1_k = f"{cond}_f1"
        em_k = f"{cond}_em"
        print(
            f"  {cond:<10} F1={avg(f1_k):.3f} EM={avg(em_k):.3f}"
            f"  | hotpot F1={avg(f1_k,'hotpotqa'):.3f}"
            f"  | 2wiki F1={avg(f1_k,'2wikimqa'):.3f}"
        )

    # ---- serialise to CSV -------------------------------------------------

    buf = io.StringIO()
    if results:
        writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Local entrypoint — downloads CSV to results/
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(model_id: str = HF_MODEL_ID):
    csv_data = run_qa_benchmark.remote(model_id)

    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "rsce_qa_full.csv"
    out_path.write_text(csv_data)
    print(f"\nSaved {len(csv_data.splitlines())-1} rows to {out_path}")
