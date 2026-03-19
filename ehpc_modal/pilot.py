from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

from .config import (
    DEFAULT_MAX_ATTENTION_GB,
    DEFAULT_PILOT_PROBES,
    DEFAULT_PILOT_TARGET_LENGTH,
    DEFAULT_TOP_K_HEADS,
    MODEL_MATRIX,
    head_config_path,
    packaged_head_config_path,
    resolve_model_key,
)
from .compress import estimate_attention_gb
from .model_utils import clear_bundle, load_bundle, run_attention_capture

console = Console()

HAYSTACK_SENTENCES = [
    "The sun rose slowly over the distant mountains, casting long shadows across the valley below.",
    "Scientists have discovered a new species of beetle in the Amazon rainforest.",
    "The quarterly earnings report showed a moderate increase in revenue.",
    "Local residents gathered at the community center to discuss upcoming road repairs.",
    "A new cafe opened on the corner of Fifth and Main, specializing in artisan coffee.",
    "The library announced extended hours for the upcoming exam period.",
    "Weather forecasters predict mild temperatures throughout the coming week.",
    "The museum's new exhibit on ancient Rome opens to the public next Saturday.",
    "Engineers completed the final inspection of the new suspension bridge.",
    "The city council voted unanimously to approve the new zoning regulations.",
    "Researchers published findings on the migratory patterns of Arctic terns.",
    "The annual harvest festival will take place in the town square this weekend.",
    "Local schools reported improved test scores following the new curriculum changes.",
    "The hiking trail through the national park was temporarily closed for maintenance.",
    "A spokesperson for the company declined to comment on the merger rumors.",
]

NEEDLE_TEMPLATES = [
    ("The secret code is {value}.", "What is the secret code?", "{value}"),
    ("The meeting password is {value}.", "What is the meeting password?", "{value}"),
    ("The answer to the riddle is {value}.", "What is the answer to the riddle?", "{value}"),
    ("The special keyword is {value}.", "What is the special keyword?", "{value}"),
    ("The verification phrase is {value}.", "What is the verification phrase?", "{value}"),
]

NEEDLE_VALUES = [
    "BLUE-WHALE-42",
    "CRIMSON-TIDE-99",
    "SILVER-MOON-17",
    "GOLDEN-EAGLE-55",
    "PURPLE-RAIN-33",
    "EMERALD-FOREST-71",
    "COPPER-DAWN-88",
    "AMBER-WAVE-24",
    "VIOLET-STORM-66",
    "IVORY-PEAK-13",
    "BRONZE-RIVER-47",
    "SCARLET-FLAME-92",
]


def build_niah_probe(tokenizer, target_length_tokens: int, needle_position_frac: float, rng: random.Random):
    template, question, answer_template = rng.choice(NEEDLE_TEMPLATES)
    value = rng.choice(NEEDLE_VALUES)
    needle_text = template.format(value=value)
    answer = answer_template.format(value=value)

    haystack_parts: list[str] = []
    while len(tokenizer.encode(" ".join(haystack_parts), add_special_tokens=False)) < target_length_tokens:
        haystack_parts.append(rng.choice(HAYSTACK_SENTENCES))
    haystack = " ".join(haystack_parts)

    insert_pos = max(0, min(len(haystack), int(len(haystack) * needle_position_frac)))
    while insert_pos < len(haystack) and haystack[insert_pos] != " ":
        insert_pos += 1

    full_context = haystack[:insert_pos] + " " + needle_text + " " + haystack[insert_pos:]
    prompt = (
        "Please answer the question based on the following text.\n\n"
        f"Text: {full_context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    pre_needle = (
        "Please answer the question based on the following text.\n\n"
        f"Text: {haystack[:insert_pos]} "
    )
    needle_start = len(tokenizer.encode(pre_needle, add_special_tokens=False))
    needle_end = len(tokenizer.encode(pre_needle + needle_text, add_special_tokens=False))
    return prompt, needle_start, needle_end, answer


def run_pilot_experiment(
    model_key: str,
    n_probes: int = DEFAULT_PILOT_PROBES,
    top_k: int = DEFAULT_TOP_K_HEADS,
    target_length: int = DEFAULT_PILOT_TARGET_LENGTH,
    max_attention_gb: float | None = DEFAULT_MAX_ATTENTION_GB,
    seed: int = 42,
) -> dict:
    resolved_key = resolve_model_key(model_key)
    if resolved_key not in MODEL_MATRIX:
        raise KeyError(f"Unknown model key: {model_key}")

    bundle = load_bundle(resolved_key, MODEL_MATRIX[resolved_key]["hf_id"], eager_attention=True)
    score_matrix = np.zeros((bundle.n_heads, bundle.n_layers), dtype=np.float32)
    completed = 0
    rng = random.Random(seed)

    console.rule(f"[bold magenta]EHPC Pilot | {resolved_key}[/bold magenta]")
    console.print(f"  HF ID    : {bundle.model_name}")
    console.print(f"  Layers   : {bundle.n_layers}")
    console.print(f"  Heads    : {bundle.n_heads}")
    console.print(f"  Probes   : {n_probes}")
    console.print(f"  Target T : {target_length}")
    if max_attention_gb is not None:
        console.print(f"  Attn GB  : {max_attention_gb}")

    try:
        for probe_idx in range(n_probes):
            prompt, needle_start, needle_end, answer = build_niah_probe(
                bundle.tokenizer,
                target_length_tokens=target_length,
                needle_position_frac=(probe_idx + 1) / (n_probes + 1),
                rng=rng,
            )
            inputs = bundle.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=target_length + 128,
            ).to(bundle.device)
            seq_len = inputs["input_ids"].shape[1]
            estimated_gb = estimate_attention_gb(bundle.model, seq_len, layer_count=1)
            if max_attention_gb is not None and estimated_gb > max_attention_gb:
                console.print(
                    f"  [yellow]Skipping probe {probe_idx + 1}: "
                    f"estimated attention {estimated_gb:.2f}GiB exceeds budget "
                    f"{max_attention_gb:.2f}GiB[/yellow]"
                )
                continue
            evidence_indices = list(range(needle_start, min(needle_end, seq_len)))
            if not evidence_indices:
                continue

            def make_callback(layer_idx: int):
                def capture_layer(layer_attn: torch.Tensor) -> None:
                    attn = layer_attn[0]
                    last_row = attn[:, seq_len - 1, :].float()
                    evidence_scores = last_row[:, evidence_indices].sum(dim=1).detach().cpu().numpy()
                    score_matrix[: len(evidence_scores), layer_idx] += evidence_scores

                return capture_layer

            run_attention_capture(
                bundle.model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                layer_callbacks={
                    layer_idx: make_callback(layer_idx)
                    for layer_idx in range(bundle.n_layers)
                },
            )

            completed += 1
            if (probe_idx + 1) % 10 == 0:
                console.print(f"  Completed {probe_idx + 1}/{n_probes} probes")
    finally:
        clear_bundle(bundle)

    if completed == 0:
        raise RuntimeError("Pilot experiment completed zero valid probes")

    score_matrix /= completed
    layer_scores = score_matrix.sum(axis=0)
    evaluator_layer = int(np.argmax(layer_scores))
    top_heads = np.argsort(score_matrix[:, evaluator_layer])[::-1][:top_k].tolist()
    result = {
        "model_key": resolved_key,
        "model_name": MODEL_MATRIX[resolved_key]["hf_id"],
        "n_layers": bundle.n_layers,
        "n_heads": bundle.n_heads,
        "n_probes": completed,
        "evaluator_layer": evaluator_layer,
        "evaluator_depth_frac": float(evaluator_layer / max(bundle.n_layers, 1)),
        "evaluator_heads": top_heads,
        "top_k": top_k,
        "target_length": target_length,
        "layer_scores": layer_scores.tolist(),
        "evidence_score_matrix": score_matrix.tolist(),
    }

    output_paths = [head_config_path(resolved_key), packaged_head_config_path(resolved_key)]
    for out_path in output_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(result, handle, indent=2)

    console.print(f"\nSaved evaluator heads to {output_paths[0]}")
    console.print(f"  Layer : {evaluator_layer}")
    console.print(f"  Heads : {top_heads}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EHPC evaluator-head pilot experiment")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-probes", type=int, default=DEFAULT_PILOT_PROBES, dest="n_probes")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K_HEADS, dest="top_k")
    parser.add_argument("--target-length", type=int, default=DEFAULT_PILOT_TARGET_LENGTH, dest="target_length")
    parser.add_argument("--max-attention-gb", type=float, default=DEFAULT_MAX_ATTENTION_GB, dest="max_attention_gb")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_pilot_experiment(
        model_key=args.model,
        n_probes=args.n_probes,
        top_k=args.top_k,
        target_length=args.target_length,
        max_attention_gb=args.max_attention_gb,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
