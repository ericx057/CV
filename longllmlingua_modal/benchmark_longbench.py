from __future__ import annotations

import argparse
import csv
import gc
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from latent_rollback_modal.backend_torch import clear_backend_cache, load_model
from latent_rollback_modal.benchmark_datasets import BenchmarkExample, grade_qa, load_benchmark
from latent_rollback_modal.context_injector import generate_baseline_qa

from .compat import (
    extract_compression_payload,
    infer_compressor_compat,
    normalize_compress_kwargs,
    patch_prompt_compressor_cache_compat,
)
from .config import DEFAULT_COMPRESSOR_DEVICE, DEFAULT_COMPRESSOR_MODEL, MODEL_MATRIX, results_path
from .reporting import record_to_dict, summarize_records, write_manifest

console = Console()
RESULTS_DIR = results_path("longllmlingua_longbench_results")

QA_RATIO_PRESETS = {
    "4x": {"rate": 0.25},
    "10x": {"rate": 0.10},
    "matched": {"target_token": 52},
}


@dataclass
class LongBenchRecord:
    model_key: str
    model_hf_id: str
    task: str
    example_id: str
    ratio_name: str
    compressor_model: str
    original_input_tokens: int
    compressed_input_tokens: int
    output_tokens: int
    input_token_reduction: float
    exact_match: bool
    f1: float
    prediction: str
    best_answer: str
    elapsed_s: float
    status: str
    error_type: str
    error_message: str
    compression_meta: str


def _import_prompt_compressor():
    try:
        from llmlingua import PromptCompressor
    except ImportError as exc:
        raise RuntimeError(
            "llmlingua is required for this benchmark. Install dependencies from "
            "longllmlingua_modal/requirements.txt."
        ) from exc
    return PromptCompressor


def _load_compressor(model_name: str, device_map: str):
    PromptCompressor = _import_prompt_compressor()
    console.print(
        f"[bold cyan]Loading LongLLMLingua compressor:[/bold cyan] {model_name} "
        f"(device={device_map})"
    )
    compressor = PromptCompressor(model_name=model_name, device_map=device_map)
    patch_prompt_compressor_cache_compat(compressor)
    return compressor, infer_compressor_compat(compressor.compress_prompt)


def _split_qa_context(context: str) -> list[str]:
    docs = [chunk.strip() for chunk in context.split("\n\n") if chunk.strip()]
    return docs if docs else [context]


def _merge_docs(docs: list[str]) -> str:
    return "\n\n".join(doc for doc in docs if doc.strip())


def _baseline_attempt_kwargs(question: str, compress_kwargs: dict) -> list[tuple[str, dict]]:
    primary = {
        "question": question,
        "condition_in_question": "after_condition",
        "reorder_context": "sort",
        "dynamic_context_compression_ratio": 0.3,
        "condition_compare": True,
        "context_budget": "+100",
        "rank_method": "longllmlingua",
        **compress_kwargs,
    }
    relaxed = {
        "question": question,
        "condition_in_question": "after_condition",
        "condition_compare": True,
        "rank_method": "longllmlingua",
        **compress_kwargs,
    }
    minimal = {
        "question": question,
        **compress_kwargs,
    }
    return [
        ("primary", primary),
        ("relaxed", relaxed),
        ("minimal", minimal),
    ]


def _compress_with_fallbacks(
    compressor,
    compat,
    docs: list[str] | str,
    attempts: list[tuple[str, dict]],
) -> tuple[dict, dict]:
    original_docs = docs if isinstance(docs, list) else [docs]
    merged_docs = _merge_docs(original_docs)
    candidates = [
        ("split_docs", docs, len(original_docs), False),
        ("merged_context", merged_docs, 1, True),
    ]

    errors: list[str] = []
    for mode_label, candidate_docs, input_doc_count, used_merged_context in candidates:
        if used_merged_context and not merged_docs:
            continue
        for attempt_label, kwargs in attempts:
            try:
                normalized = normalize_compress_kwargs(compat, kwargs)
                result = compressor.compress_prompt(candidate_docs, **normalized)
                payload = extract_compression_payload(result)
                return {
                    "compressed_prompt": payload.compressed_prompt,
                    "origin_tokens": payload.origin_tokens,
                    "compressed_tokens": payload.compressed_tokens,
                    "ratio": payload.ratio,
                    "raw": payload.raw,
                    "attempt_label": f"{attempt_label}_{mode_label}",
                    "input_doc_count": input_doc_count,
                    "used_merged_context": used_merged_context,
                }, normalized
            except Exception as exc:  # pragma: no cover - depends on upstream library behavior
                tb = traceback.format_exc(limit=30)
                errors.append(
                    f"{attempt_label}_{mode_label}: {type(exc).__name__}: {exc}\n{tb}"
                )
    joined = " | ".join(error.splitlines()[0] for error in errors[-6:])
    detail = errors[-1] if errors else "unknown compression failure"
    raise RuntimeError(
        "LongLLMLingua compression failed after retries: "
        f"{joined}\nLast traceback:\n{detail}"
    )


def _build_prompt(example: BenchmarkExample, compressed_context: str) -> str:
    return "\n".join(
        [
            f"Context:\n{compressed_context}",
            f"\nQuestion: {example.question}",
            "\nAnswer:",
        ]
    )


def run_model(
    model_key: str,
    model_cfg: dict,
    examples: list[BenchmarkExample],
    ratio_names: list[str],
    compressor_model: str,
    compressor_device: str,
    max_new_tokens: int,
) -> list[LongBenchRecord]:
    compressor, compat = _load_compressor(compressor_model, compressor_device)
    wrapper = load_model(model_cfg["hf_id"])
    records: list[LongBenchRecord] = []

    try:
        for ratio_name in ratio_names:
            compress_kwargs = QA_RATIO_PRESETS[ratio_name]
            console.rule(f"[bold cyan]{model_key} | {ratio_name}[/bold cyan]")

            for idx, example in enumerate(examples, start=1):
                console.print(
                    f"  [{idx}/{len(examples)}] {example.task}/{example.id[:24]} "
                    f"({example.context_word_len}w ctx)"
                )
                docs = _split_qa_context(example.context)
                attempts = _baseline_attempt_kwargs(example.question, compress_kwargs)

                try:
                    t0 = time.time()
                    compression_result, used_kwargs = _compress_with_fallbacks(
                        compressor, compat, docs, attempts
                    )
                    compressed_context = compression_result["compressed_prompt"]
                    prompt = _build_prompt(example, compressed_context)
                    prediction, compressed_input_tokens = generate_baseline_qa(
                        wrapper, prompt, max_new_tokens=max_new_tokens
                    )
                    elapsed = time.time() - t0
                    output_tokens = len(wrapper.encode(prediction))
                    original_prompt = example.full_prompt()
                    original_input_tokens = len(wrapper.encode(original_prompt))
                    grade = grade_qa(prediction, example.gold_answers)
                    status = (
                        "fallback"
                        if compression_result.get("used_merged_context")
                        else "ok"
                    )
                    error_type = ""
                    error_message = ""
                except Exception as exc:
                    console.print(f"  [red]ERROR: {exc}[/red]")
                    prediction = ""
                    original_input_tokens = len(wrapper.encode(example.full_prompt()))
                    compressed_input_tokens = original_input_tokens
                    output_tokens = 0
                    elapsed = 0.0
                    grade = {"exact_match": False, "f1": 0.0, "best_answer": ""}
                    compression_result = {"error": str(exc)}
                    used_kwargs = compress_kwargs
                    status = "error"
                    error_type = type(exc).__name__
                    error_message = str(exc)

                reduction = 1.0 - (
                    compressed_input_tokens / max(original_input_tokens, 1)
                )
                records.append(
                    LongBenchRecord(
                        model_key=model_key,
                        model_hf_id=model_cfg["hf_id"],
                        task=example.task,
                        example_id=example.id,
                        ratio_name=ratio_name,
                        compressor_model=compressor_model,
                        original_input_tokens=original_input_tokens,
                        compressed_input_tokens=compressed_input_tokens,
                        output_tokens=output_tokens,
                        input_token_reduction=round(reduction, 4),
                        exact_match=bool(grade["exact_match"]),
                        f1=float(grade["f1"]),
                        prediction=prediction.strip(),
                        best_answer=str(grade["best_answer"]),
                        elapsed_s=round(elapsed, 2),
                        status=status,
                        error_type=error_type,
                        error_message=error_message,
                        compression_meta=json.dumps(
                            {
                                "kwargs": used_kwargs,
                                "origin_tokens": compression_result.get("origin_tokens"),
                                "compressed_tokens": compression_result.get("compressed_tokens"),
                                "ratio": compression_result.get("ratio"),
                                "attempt_label": compression_result.get("attempt_label"),
                                "input_doc_count": compression_result.get("input_doc_count"),
                                "used_merged_context": compression_result.get(
                                    "used_merged_context", False
                                ),
                            }
                        ),
                    )
                )
    finally:
        del wrapper
        gc.collect()
        clear_backend_cache()

    return records


def print_summary(records: list[LongBenchRecord]) -> None:
    if not records:
        return

    grouped: dict[tuple[str, str], list[LongBenchRecord]] = {}
    for record in records:
        grouped.setdefault((record.model_key, record.ratio_name), []).append(record)

    table = Table(
        title="LongLLMLingua LongBench Summary",
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    table.add_column("Model", style="cyan")
    table.add_column("Ratio", style="magenta")
    table.add_column("N", justify="right")
    table.add_column("OK", justify="right")
    table.add_column("Fb", justify="right")
    table.add_column("Err", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("TokRed", justify="right")

    for (model_key, ratio_name), recs in sorted(grouped.items()):
        summary = summarize_records(recs, ["exact_match", "f1"])
        table.add_row(
            model_key,
            ratio_name,
            str(summary["n_total"]),
            str(summary["n_ok"]),
            str(summary["n_fallback"]),
            str(summary["n_error"]),
            f"{float(summary['exact_match']):.1%}",
            f"{float(summary['f1']):.3f}",
            f"{float(summary['input_token_reduction']):.1%}",
        )
    console.print(table)


def save_results(records: list[LongBenchRecord], run_id: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{run_id}.csv"
    json_path = RESULTS_DIR / f"{run_id}.json"

    if records:
        fields = list(asdict(records[0]).keys())
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(record_to_dict(record) for record in records)

    with open(json_path, "w") as handle:
        json.dump([record_to_dict(record) for record in records], handle, indent=2)
    write_manifest(
        records=records,
        run_id=run_id,
        benchmark="longbench",
        results_dir=RESULTS_DIR,
    )

    console.print(f"\n[dim]Saved:[/dim]\n  {csv_path}\n  {json_path}")
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone LongLLMLingua LongBench runner"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_MATRIX.keys()),
        choices=list(MODEL_MATRIX.keys()),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["hotpotqa", "2wikimqa"],
        choices=["hotpotqa", "2wikimqa"],
    )
    parser.add_argument("--n", type=int, default=10, help="Examples per task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=80, dest="max_tokens")
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=list(QA_RATIO_PRESETS.keys()),
        choices=list(QA_RATIO_PRESETS.keys()),
    )
    parser.add_argument("--compressor-model", default=DEFAULT_COMPRESSOR_MODEL)
    parser.add_argument("--compressor-device", default=DEFAULT_COMPRESSOR_DEVICE)
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or f"longllmlingua_longbench_{int(time.time())}"

    console.rule("[bold magenta]LongLLMLingua LongBench[/bold magenta]")
    console.print(f"  Models : {args.models}")
    console.print(f"  Tasks  : {args.tasks}")
    console.print(f"  N/task : {args.n}")
    console.print(f"  Ratios : {args.ratios}")
    console.print(f"  Run ID : {run_id}")

    examples = load_benchmark(
        tasks=tuple(args.tasks),
        n_per_task=args.n,
        seed=args.seed,
    )
    if not examples:
        console.print("[red]No LongBench examples available after loading/filtering.[/red]")
        return 1

    all_records: list[LongBenchRecord] = []
    for model_key in args.models:
        model_records = run_model(
            model_key=model_key,
            model_cfg=MODEL_MATRIX[model_key],
            examples=examples,
            ratio_names=args.ratios,
            compressor_model=args.compressor_model,
            compressor_device=args.compressor_device,
            max_new_tokens=args.max_tokens,
        )
        all_records.extend(model_records)
        save_results(all_records, f"{run_id}_partial")

    print_summary(all_records)
    save_results(all_records, run_id)
    return 0 if all_records else 1


if __name__ == "__main__":
    raise SystemExit(main())
