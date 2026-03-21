"""
Modal job: RSCE QA evaluation across all models.

Runs each model on the same LongBench QA splits used by
EHPC and LongLLMLingua benchmarks (108 examples: 17 HotpotQA + 91 2WikiMQA).

Conditions per example:
  baseline  -- full context + question in prompt
  q_only    -- question only (parametric memory)
  vec       -- question + vector injection, no fact block
  vec_f     -- question + vector injection + NER fact block

Usage:
  modal run modal_rsce_qa_eval.py
  modal run modal_rsce_qa_eval.py --models llama3-8b qwen25-7b
  modal run modal_rsce_qa_eval.py --models deepseek-14b --n 50
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
# Modal image
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

app = modal.App("rsce-qa-eval", image=image)
vol = modal.Volume.from_name("rsce-hf-weights", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("rsce-qa-checkpoints", create_if_missing=True)

# ---------------------------------------------------------------------------
# Model configs -- calibrated injection layers from calibration_results.json
# ---------------------------------------------------------------------------

MODELS = {
    "qwen25-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "injection_layer": 17,
        "n_layers": 28,
        "gpu": "H100",
    },
    "deepseek-14b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "injection_layer": 12,
        "n_layers": 48,
        "gpu": "H100:2",
    },
    "mistral-24b": {
        "hf_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "injection_layer": 10,
        "n_layers": 40,
        "gpu": "H100:2",
    },
    "llama31-70b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "injection_layer": 50,
        "n_layers": 80,
        "gpu": "H100:4",
    },
}

N_PER_TASK = 200  # will be capped by available examples after filtering
SEED = 42
MAX_NEW_TOKENS = 80
CHECKPOINT_EVERY = 10  # save partial results every N examples

CHECKPOINT_DIR = Path("/checkpoints")


# ---------------------------------------------------------------------------
# Checkpointing helpers (run inside GPU worker, write to Modal volume)
# ---------------------------------------------------------------------------

def _save_checkpoint(results: list[dict], model_key: str, run_id: str) -> None:
    """Write partial CSV to checkpoint volume."""
    if not results:
        return
    ckpt_path = CHECKPOINT_DIR / f"rsce_qa_{model_key}_{run_id}_partial.csv"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
    ckpt_path.write_text(buf.getvalue())
    checkpoint_vol.commit()
    print(f"[{model_key}] Checkpoint saved: {len(results)} rows -> {ckpt_path.name}")


def _load_checkpoint(model_key: str, run_id: str) -> list[dict]:
    """Load partial results from a previous run, if any."""
    ckpt_path = CHECKPOINT_DIR / f"rsce_qa_{model_key}_{run_id}_partial.csv"
    if not ckpt_path.exists():
        return []
    text = ckpt_path.read_text()
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    print(f"[{model_key}] Resumed from checkpoint: {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# GPU-tier worker functions
# ---------------------------------------------------------------------------

def _run_rsce_qa(model_key: str, model_cfg: dict, n_per_task: int, run_id: str) -> str:
    """Core RSCE QA logic. Returns CSV string."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = model_cfg["hf_id"]
    injection_layer = model_cfg["injection_layer"]

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
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            else:
                f1 = 0.0
            best_f1 = max(best_f1, f1)
        return {"exact_match": exact, "f1": round(best_f1, 4)}

    def extract_ner_facts(text: str, max_facts: int = 15) -> str:
        patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
            r"\b\d{4}\b",
            r"\b\d+(?:\.\d+)?(?:\s*(?:km|m|kg|mph|lb|ft))?\b",
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
    print(f"[{model_key}] Loading {hf_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()
    print(
        f"[{model_key}] Loaded. Layers: {model.config.num_hidden_layers}  "
        f"d_model: {model.config.hidden_size}  injection_layer: {injection_layer}"
    )

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
        return out[0][input_ids.shape[1] :].tolist()

    def extract_vector(context_text: str) -> tuple[torch.Tensor, int]:
        ids = encode(context_text)
        inp = torch.tensor([ids], dtype=torch.long).to(model.device)
        with torch.no_grad():
            out = model(inp, output_hidden_states=True)
        h = out.hidden_states[injection_layer + 1][0].mean(dim=0).float().cpu()
        return h, len(ids)

    def gen_with_injection(
        question_text: str, ctx_vec: torch.Tensor, scale: float = 1.0
    ) -> tuple[str, int]:
        norm = ctx_vec.norm().item()
        if norm < _EPS:
            raise ValueError("zero vector")
        v = (ctx_vec / norm).to(model.device, dtype=torch.bfloat16)

        def hook(module, inp, output):
            if isinstance(output, tuple):
                h = output[0] + scale * v[None, None, :]
                return (h,) + output[1:]
            return output + scale * v[None, None, :]

        handle = model.model.layers[injection_layer].register_forward_hook(hook)
        ids = encode(question_text)
        inp = torch.tensor([ids], dtype=torch.long).to(model.device)
        try:
            new_ids = gen(inp)
        finally:
            handle.remove()
        text = truncate_stop(decode(new_ids))
        return text, len(ids)

    # ---- load checkpoint / dataset ----------------------------------------

    results = _load_checkpoint(model_key, run_id)
    done_ids = {(r["task"], r["example_id"]) for r in results}
    if done_ids:
        print(f"[{model_key}] Skipping {len(done_ids)} already-completed examples")

    print(f"[{model_key}] Loading LongBench ...")
    total_done = len(results)

    for task in ("hotpotqa", "2wikimqa"):
        ds = load_dataset(
            "THUDM/LongBench", task, split="test", trust_remote_code=True
        )
        import random

        rng = random.Random(SEED)
        rows = list(ds)
        rows = [
            r for r in rows if 200 <= len(r.get("context", "").split()) <= 12000
        ]
        selected = rng.sample(rows, min(n_per_task, len(rows)))
        print(f"[{model_key}]   {task}: {len(selected)} examples")

        for i, row in enumerate(selected):
            ex_id = row.get("_id", str(i))

            # Skip if already checkpointed
            if (task, ex_id) in done_ids:
                continue

            context = row.get("context", "")
            question = row.get("input", "")
            answers = row.get("answers", [])
            if isinstance(answers, str):
                answers = [answers]

            baseline_prompt = (
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            qonly_prompt = f"Question: {question}\nAnswer:"

            n_baseline_toks = len(encode(baseline_prompt))
            tok_red = round(
                1.0 - (len(encode(qonly_prompt)) / n_baseline_toks), 4
            )

            t0 = time.time()

            try:
                # 1. Baseline
                b_ids = gen(
                    torch.tensor(
                        [encode(baseline_prompt)], dtype=torch.long
                    ).to(model.device)
                )
                b_text = truncate_stop(decode(b_ids))
                b_grades = grade_qa(b_text, answers)

                # 2. Q-only
                q_ids = gen(
                    torch.tensor(
                        [encode(qonly_prompt)], dtype=torch.long
                    ).to(model.device)
                )
                q_text = truncate_stop(decode(q_ids))
                q_grades = grade_qa(q_text, answers)

                # 3. Vec-only
                ctx_vec, n_ctx_toks = extract_vector(context)
                v_text, _ = gen_with_injection(qonly_prompt, ctx_vec)
                v_grades = grade_qa(v_text, answers)

                # 4. Vec+F (NER fact block)
                facts = extract_ner_facts(context)
                vf_prompt = f"{facts}\n\nQuestion: {question}\nAnswer:"
                vf_text, _ = gen_with_injection(vf_prompt, ctx_vec)
                vf_grades = grade_qa(vf_text, answers)

                status = "ok"
                error_type = ""
                error_message = ""
            except Exception as exc:
                print(f"[{model_key}]   ERROR on {ex_id}: {exc}")
                b_text = q_text = v_text = vf_text = ""
                b_grades = q_grades = v_grades = vf_grades = {
                    "exact_match": False,
                    "f1": 0.0,
                }
                n_ctx_toks = 0
                status = "error"
                error_type = type(exc).__name__
                error_message = str(exc)

            elapsed = time.time() - t0

            results.append(
                {
                    "model_key": model_key,
                    "model_hf_id": hf_id,
                    "task": task,
                    "example_id": ex_id,
                    "n_context_tokens": n_ctx_toks,
                    "n_baseline_tokens": n_baseline_toks,
                    "token_reduction": tok_red,
                    "baseline_f1": b_grades["f1"],
                    "baseline_em": int(b_grades["exact_match"]),
                    "baseline_output": b_text[:200],
                    "qonly_f1": q_grades["f1"],
                    "qonly_em": int(q_grades["exact_match"]),
                    "qonly_output": q_text[:200],
                    "vec_f1": v_grades["f1"],
                    "vec_em": int(v_grades["exact_match"]),
                    "vec_output": v_text[:200],
                    "vecf_f1": vf_grades["f1"],
                    "vecf_em": int(vf_grades["exact_match"]),
                    "vecf_output": vf_text[:200],
                    "gold": json.dumps(answers),
                    "elapsed": round(elapsed, 2),
                    "status": status,
                    "error_type": error_type,
                    "error_message": error_message,
                }
            )
            total_done += 1

            if status == "ok":
                print(
                    f"[{model_key}] [{task}] {i + 1}/{len(selected)} "
                    f"base={b_grades['f1']:.3f} q={q_grades['f1']:.3f} "
                    f"vec={v_grades['f1']:.3f} vecf={vf_grades['f1']:.3f} "
                    f"| {question[:50]!r}"
                )

            # Checkpoint every N examples
            if total_done % CHECKPOINT_EVERY == 0:
                _save_checkpoint(results, model_key, run_id)

    # ---- final checkpoint + summary ---------------------------------------

    _save_checkpoint(results, model_key, run_id)

    ok_rows = [r for r in results if r["status"] == "ok"]
    n = len(ok_rows)
    if n:
        print(f"\n[{model_key}] === RESULTS (n={n}) ===")
        for cond in ("baseline", "qonly", "vec", "vecf"):
            f1 = sum(r[f"{cond}_f1"] for r in ok_rows) / n
            em = sum(r[f"{cond}_em"] for r in ok_rows) / n
            print(f"  {cond:<10} F1={f1:.3f} EM={em:.1%}")

    buf = io.StringIO()
    if results:
        writer = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GPU-specific Modal functions (GPU is fixed at decoration time)
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    timeout=7200,
    volumes={"/weights": vol, "/checkpoints": checkpoint_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=81920,
)
def run_h100(model_key: str, model_cfg: dict, n_per_task: int, run_id: str) -> str:
    return _run_rsce_qa(model_key, model_cfg, n_per_task, run_id)


@app.function(
    gpu="H100:2",
    timeout=10800,
    volumes={"/weights": vol, "/checkpoints": checkpoint_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=163840,
)
def run_h100x2(model_key: str, model_cfg: dict, n_per_task: int, run_id: str) -> str:
    return _run_rsce_qa(model_key, model_cfg, n_per_task, run_id)


@app.function(
    gpu="H100:4",
    timeout=14400,
    volumes={"/weights": vol, "/checkpoints": checkpoint_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=327680,
)
def run_h100x4(model_key: str, model_cfg: dict, n_per_task: int, run_id: str) -> str:
    return _run_rsce_qa(model_key, model_cfg, n_per_task, run_id)


GPU_DISPATCH = {
    "H100": run_h100,
    "H100:2": run_h100x2,
    "H100:4": run_h100x4,
}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    models: list[str] = None,
    n: int = N_PER_TASK,
    run_id: str = None,
):
    selected_models = models or list(MODELS.keys())
    n_per_task = n
    rid = run_id or f"rsce_qa_eval_{int(time.time())}"

    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)

    print(f"RSCE QA Evaluation")
    print(f"  Models : {selected_models}")
    print(f"  N/task : {n_per_task}")
    print(f"  Run ID : {rid}")
    print(f"  Checkpoint every : {CHECKPOINT_EVERY} examples")
    print()

    # Spawn all models in parallel
    handles = {}
    for model_key in selected_models:
        cfg = MODELS[model_key]
        gpu = cfg["gpu"]
        print(f"Spawning {model_key} on {gpu} ...")
        fn = GPU_DISPATCH[gpu]
        handles[model_key] = fn.spawn(model_key, cfg, n_per_task, rid)

    # Collect results -- each model is isolated, failures don't affect others
    all_csv_parts = []
    failed = []
    for model_key in selected_models:
        print(f"Waiting for {model_key} ...")
        try:
            csv_data = handles[model_key].get()
        except Exception as exc:
            print(f"  FAILED {model_key}: {exc}")
            failed.append(model_key)
            continue

        model_path = out_dir / f"rsce_qa_{model_key}_{rid}.csv"
        model_path.write_text(csv_data)
        n_rows = len(csv_data.strip().splitlines()) - 1
        print(f"  Saved {n_rows} rows to {model_path}")

        all_csv_parts.append(csv_data)

    if failed:
        print(f"\nFailed models: {failed}")
        print("Partial results from checkpoints may be on the rsce-qa-checkpoints volume.")
        print("Re-run with same --run-id to resume.")

    # Combine successful results into single CSV
    if all_csv_parts:
        combined_path = out_dir / f"rsce_qa_all_{rid}.csv"
        import csv as csv_mod

        all_rows = []
        fieldnames = None
        for part in all_csv_parts:
            reader = csv_mod.DictReader(io.StringIO(part))
            if fieldnames is None:
                fieldnames = reader.fieldnames
            all_rows.extend(reader)

        if fieldnames and all_rows:
            buf = io.StringIO()
            writer = csv_mod.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
            combined_path.write_text(buf.getvalue())
            print(f"\nCombined: {len(all_rows)} rows to {combined_path}")
