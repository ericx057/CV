from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from latent_rollback_modal.benchmark_repobench import RepoBenchExample, grade_completion, load_repobench

from .compress import compress_code_prompt, load_head_config
from .config import (
    DEFAULT_MAX_ATTENTION_GB,
    DEFAULT_POOL_KERNEL,
    MODEL_MATRIX,
    resolve_model_key,
    results_path,
)
from .model_utils import clear_bundle, generate_from_ids, load_bundle
from .reporting import record_to_dict, summarize_records, write_manifest

console = Console()
RESULTS_DIR = results_path("ehpc_repobench_results")

CODE_SETTING_PRESETS = {
    "6x": {"compression_ratio": 0.17},
    "matched": {"target_tokens": 963},
}


@dataclass
class RepoBenchRecord:
    model_key: str
    model_hf_id: str
    repo_name: str
    file_path: str
    example_id: str
    setting_name: str
    evaluator_layer: int
    evaluator_heads: list[int]
    original_input_tokens: int
    compressed_input_tokens: int
    original_context_tokens: int
    compressed_context_tokens: int
    observation_window_tokens: int
    input_token_reduction: float
    exact_match: bool
    edit_sim: float
    prefix_match: bool
    prediction: str
    gold_next_line: str
    elapsed_s: float
    status: str
    error_type: str
    error_message: str
    attention_estimated_gb: float


def run_model(
    model_key: str,
    model_cfg: dict,
    examples: list[RepoBenchExample],
    setting_names: list[str],
    pool_kernel: int,
    max_new_tokens: int,
    max_attention_gb: float | None,
) -> list[RepoBenchRecord]:
    resolved_key = resolve_model_key(model_key)
    head_config = load_head_config(resolved_key)
    bundle = load_bundle(resolved_key, model_cfg["hf_id"], eager_attention=True)
    records: list[RepoBenchRecord] = []

    try:
        for setting_name in setting_names:
            compress_kwargs = CODE_SETTING_PRESETS[setting_name]
            console.rule(f"[bold cyan]{resolved_key} | {setting_name}[/bold cyan]")
            for idx, example in enumerate(examples, start=1):
                console.print(f"  [{idx}/{len(examples)}] {example.repo_name}/{example.file_path}")
                try:
                    t0 = time.time()
                    compression = compress_code_prompt(
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        cross_file_context=example.cross_file_context,
                        local_context=example.in_file_context,
                        head_config=head_config,
                        pool_kernel=pool_kernel,
                        max_attention_gb=max_attention_gb,
                        device=bundle.device,
                        **compress_kwargs,
                    )
                    prediction = generate_from_ids(
                        bundle,
                        compression.compressed_input_ids,
                        max_new_tokens=max_new_tokens,
                        stop_strings=("\n",),
                    )
                    prediction = prediction.splitlines()[0].rstrip() if prediction else ""
                    elapsed = time.time() - t0
                    grade = grade_completion(prediction, example.next_line)
                    status = "ok"
                    error_type = ""
                    error_message = ""
                except Exception as exc:
                    console.print(f"  [red]ERROR: {exc}[/red]")
                    compression = None
                    prediction = ""
                    elapsed = 0.0
                    grade = {"exact_match": False, "edit_sim": 0.0, "prefix_match": False}
                    status = "error"
                    error_type = type(exc).__name__
                    error_message = str(exc)

                if compression is None:
                    full_prompt = f"# Cross-file context:\n{example.cross_file_context}\n\n# Current file:\n{example.in_file_context}\n\n# Complete the next line:\n"
                    orig_tokens = len(bundle.tokenizer.encode(full_prompt, add_special_tokens=False))
                    compressed_tokens = orig_tokens
                    original_context_tokens = len(bundle.tokenizer.encode(example.cross_file_context, add_special_tokens=False))
                    compressed_context_tokens = original_context_tokens
                    observation_window = 0
                    attention_estimated_gb = 0.0
                else:
                    orig_tokens = compression.original_input_tokens
                    compressed_tokens = compression.compressed_input_tokens
                    original_context_tokens = compression.original_context_tokens
                    compressed_context_tokens = compression.compressed_context_tokens
                    observation_window = compression.observation_window_tokens
                    attention_estimated_gb = compression.attention_estimated_gb

                reduction = 1.0 - (compressed_tokens / max(orig_tokens, 1))
                records.append(
                    RepoBenchRecord(
                        model_key=resolved_key,
                        model_hf_id=model_cfg["hf_id"],
                        repo_name=example.repo_name,
                        file_path=example.file_path,
                        example_id=example.id,
                        setting_name=setting_name,
                        evaluator_layer=int(head_config["evaluator_layer"]),
                        evaluator_heads=list(head_config["evaluator_heads"]),
                        original_input_tokens=orig_tokens,
                        compressed_input_tokens=compressed_tokens,
                        original_context_tokens=original_context_tokens,
                        compressed_context_tokens=compressed_context_tokens,
                        observation_window_tokens=observation_window,
                        input_token_reduction=round(reduction, 4),
                        exact_match=bool(grade["exact_match"]),
                        edit_sim=float(grade["edit_sim"]),
                        prefix_match=bool(grade["prefix_match"]),
                        prediction=prediction,
                        gold_next_line=example.next_line,
                        elapsed_s=round(elapsed, 2),
                        status=status,
                        error_type=error_type,
                        error_message=error_message,
                        attention_estimated_gb=round(attention_estimated_gb, 3),
                    )
                )
    finally:
        clear_bundle(bundle)
        gc.collect()

    return records


def print_summary(records: list[RepoBenchRecord]) -> None:
    if not records:
        return
    grouped: dict[tuple[str, str], list[RepoBenchRecord]] = {}
    for record in records:
        grouped.setdefault((record.model_key, record.setting_name), []).append(record)

    table = Table(title="EHPC RepoBench Summary", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Model", style="cyan")
    table.add_column("Setting", style="magenta")
    table.add_column("N", justify="right")
    table.add_column("OK", justify="right")
    table.add_column("Err", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("EditSim", justify="right")
    table.add_column("TokRed", justify="right")
    for (model_key, setting_name), recs in sorted(grouped.items()):
        summary = summarize_records(recs, ["exact_match", "edit_sim"])
        table.add_row(
            model_key,
            setting_name,
            str(summary["n_total"]),
            str(summary["n_ok"]),
            str(summary["n_error"]),
            f"{float(summary['exact_match']):.1%}",
            f"{float(summary['edit_sim']):.3f}",
            f"{float(summary['input_token_reduction']):.1%}",
        )
    console.print(table)


def save_results(records: list[RepoBenchRecord], run_id: str) -> tuple[Path, Path]:
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
        benchmark="repobench",
        results_dir=RESULTS_DIR,
    )
    console.print(f"\n[dim]Saved:[/dim]\n  {csv_path}\n  {json_path}")
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EHPC RepoBench benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_MATRIX.keys()),
        choices=list(MODEL_MATRIX.keys()),
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="cross_file_first")
    parser.add_argument(
        "--level",
        default="all",
        help='RepoBench level bucket (e.g. "2k") or "all" for no filter',
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=list(CODE_SETTING_PRESETS.keys()),
        choices=list(CODE_SETTING_PRESETS.keys()),
    )
    parser.add_argument("--pool-kernel", type=int, default=DEFAULT_POOL_KERNEL, dest="pool_kernel")
    parser.add_argument("--max-attention-gb", type=float, default=DEFAULT_MAX_ATTENTION_GB, dest="max_attention_gb")
    parser.add_argument("--max-tokens", type=int, default=60, dest="max_tokens")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or f"ehpc_repobench_{int(time.time())}"
    console.rule("[bold magenta]EHPC RepoBench[/bold magenta]")
    console.print(f"  Models   : {args.models}")
    console.print(f"  Split    : {args.split}")
    console.print(f"  Level    : {args.level}")
    console.print(f"  N        : {args.n}")
    console.print(f"  Settings : {args.settings}")
    console.print(f"  Run ID   : {run_id}")

    examples = load_repobench(
        split=args.split,
        level=None if args.level == "all" else args.level,
        n=args.n,
        seed=args.seed,
    )
    all_records: list[RepoBenchRecord] = []
    for model_key in args.models:
        model_records = run_model(
            model_key=model_key,
            model_cfg=MODEL_MATRIX[model_key],
            examples=examples,
            setting_names=args.settings,
            pool_kernel=args.pool_kernel,
            max_new_tokens=args.max_tokens,
            max_attention_gb=args.max_attention_gb,
        )
        all_records.extend(model_records)
        save_results(all_records, f"{run_id}_partial")

    print_summary(all_records)
    save_results(all_records, run_id)
    return 0 if all_records else 1


if __name__ == "__main__":
    raise SystemExit(main())
