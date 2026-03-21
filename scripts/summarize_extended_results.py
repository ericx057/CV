#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None
    Table = None
    box = None

console = Console() if Console is not None else None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_manifest(path: Path) -> dict[str, Any]:
    data = load_json(path)
    return {
        "source": "manifest",
        "path": str(path),
        "run_id": data.get("run_id", path.stem.replace(".manifest", "")),
        "benchmark": data.get("benchmark", infer_benchmark(path)),
        "total": data.get("total_records"),
        "success": data.get("success_records"),
        "errors": data.get("error_records", 0),
        "fallbacks": data.get("fallback_records", data.get("fb_records", 0)),
        "status_counts": data.get("status_counts", {}),
        "error_type_counts": data.get("error_type_counts", {}),
        "conditions": data.get("conditions", []),
        "model_keys": data.get("model_keys", []),
        "partial": "_partial" in path.name,
    }


def summarize_result_json(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if isinstance(data, dict) and "total_records" in data:
        return summarize_manifest(path)

    if not isinstance(data, list):
        return {
            "source": "unknown-json",
            "path": str(path),
            "run_id": path.stem,
            "benchmark": infer_benchmark(path),
            "total": None,
            "success": None,
            "errors": None,
            "fallbacks": None,
            "status_counts": {},
            "error_type_counts": {},
            "conditions": [],
            "model_keys": [],
            "partial": "_partial" in path.name,
        }

    status_counts = Counter()
    error_type_counts = Counter()
    conditions = set()
    models = set()

    for row in data:
        if not isinstance(row, dict):
            continue
        status = row.get("status", "unknown")
        status_counts[status] += 1
        if row.get("error_type"):
            error_type_counts[row["error_type"]] += 1
        for key in ("ratio_name", "setting_name"):
            if row.get(key):
                conditions.add(str(row[key]))
        if row.get("model_key"):
            models.add(str(row["model_key"]))

    return {
        "source": "json",
        "path": str(path),
        "run_id": path.stem,
        "benchmark": infer_benchmark(path),
        "total": len(data),
        "success": status_counts.get("ok", 0) + status_counts.get("fallback", 0),
        "errors": status_counts.get("error", 0),
        "fallbacks": status_counts.get("fallback", 0),
        "status_counts": dict(status_counts),
        "error_type_counts": dict(error_type_counts),
        "conditions": sorted(conditions),
        "model_keys": sorted(models),
        "partial": "_partial" in path.name,
    }


def summarize_metric_json_rows(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list) or not data:
        return []
    if not isinstance(data[0], dict):
        return []

    benchmark = infer_benchmark(path)
    if benchmark not in {"longbench", "repobench"}:
        return []

    sample = data[0]
    method = infer_method(path)
    if method not in {"LongLLMLingua", "EHPC"}:
        return []
    model_key = sample.get("model_key", infer_model_from_name(path.name))
    condition_key = "ratio_name" if "ratio_name" in sample else "setting_name"
    rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data:
        condition = str(row.get(condition_key, ""))
        rows_by_condition[condition].append(row)

    summaries: list[dict[str, Any]] = []
    for condition, cond_rows in sorted(rows_by_condition.items()):
        usable_statuses = {"ok", "fallback"}
        usable = [row for row in cond_rows if row.get("status") in usable_statuses]
        errors = [row for row in cond_rows if row.get("status") == "error"]
        task_values = sorted({str(row.get("task", "")) for row in cond_rows if row.get("task", "")})

        summary: dict[str, Any] = {
            "method": method,
            "benchmark": benchmark,
            "path": str(path),
            "run_id": path.stem,
            "model_key": model_key,
            "condition": condition or "-",
            "tasks": task_values,
            "n_total": len(cond_rows),
            "n_ok": sum(1 for row in cond_rows if row.get("status") == "ok"),
            "n_fallback": sum(1 for row in cond_rows if row.get("status") == "fallback"),
            "n_error": len(errors),
            "partial": "_partial" in path.name,
            "paper_candidate": is_paper_candidate(path),
        }

        if benchmark == "longbench":
            summary["exact_match"] = avg_metric(usable, "exact_match")
            summary["f1"] = avg_metric(usable, "f1")
        elif benchmark == "repobench":
            summary["exact_match"] = avg_metric(usable, "exact_match")
            summary["edit_sim"] = avg_metric(usable, "edit_sim")
            summary["prefix_match"] = avg_metric(usable, "prefix_match")

        summary["input_token_reduction"] = avg_metric(usable, "input_token_reduction")
        summaries.append(summary)
    return summaries


def infer_benchmark(path: Path) -> str:
    name = path.as_posix()
    if "longbench" in name:
        return "longbench"
    if "repobench" in name:
        return "repobench"
    if "head_configs" in name:
        return "head_configs"
    if "integration_results" in name:
        return "integration"
    return "unknown"


def infer_method(path: Path) -> str:
    name = path.as_posix()
    if "longllmlingua" in name:
        return "LongLLMLingua"
    if "ehpc" in name:
        return "EHPC"
    if "repobench_results" in name or "integration_results" in name:
        return "RSCE/Other"
    return "Unknown"


def infer_model_from_name(name: str) -> str:
    for key in ("qwen25-7b", "deepseek-14b", "mistral-24b", "llama3-8b"):
        if key in name:
            return key
    return "-"


def avg_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    vals = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            vals.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            vals.append(float(value))
    return (sum(vals) / len(vals)) if vals else 0.0


def is_paper_candidate(path: Path) -> bool:
    name = path.name
    if "_partial" in name:
        return False
    if "smoke" in name or "suite" in name:
        return False
    return path.suffix == ".json" and not name.endswith(".manifest.json")


def collect_runs(root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.manifest.json")):
        summaries.append(summarize_manifest(path))

    manifest_run_ids = {item["run_id"] for item in summaries}
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".manifest.json"):
            continue
        if "ehpc_head_configs" in path.as_posix():
            continue
        if path.stem in manifest_run_ids:
            continue
        summaries.append(summarize_result_json(path))
    return summaries


def collect_metric_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".manifest.json"):
            continue
        runs.extend(summarize_metric_json_rows(path))
    return runs


def collect_head_configs(root: Path) -> list[dict[str, Any]]:
    configs = []
    config_dir = root / "ehpc_head_configs"
    if not config_dir.exists():
        return configs
    for path in sorted(config_dir.glob("*.json")):
        data = load_json(path)
        configs.append(
            {
                "model_key": data.get("model_key", path.stem),
                "layer": data.get("evaluator_layer"),
                "heads": data.get("evaluator_heads", []),
            }
        )
    return configs


def print_runs(runs: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["benchmark"]].append(run)

    for benchmark in sorted(grouped):
        print(f"\n== {benchmark.upper()} ==")
        for run in sorted(grouped[benchmark], key=lambda x: (x["partial"], x["run_id"])):
            models = ",".join(run["model_keys"]) or "-"
            conditions = ",".join(run["conditions"]) or "-"
            status_counts = ",".join(f"{k}:{v}" for k, v in sorted(run["status_counts"].items())) or "-"
            error_counts = ",".join(f"{k}:{v}" for k, v in sorted(run["error_type_counts"].items())) or "-"
            partial = " partial" if run["partial"] else ""
            print(
                f"{run['run_id']}{partial}\n"
                f"  models={models} conditions={conditions}\n"
                f"  total={run['total']} success={run['success']} errors={run['errors']} fallbacks={run['fallbacks']}\n"
                f"  statuses={status_counts}\n"
                f"  error_types={error_counts}\n"
                f"  file={run['path']}"
            )


def print_missing_finals(runs: list[dict[str, Any]]) -> None:
    finals = {run["run_id"] for run in runs if not run["partial"]}
    partials = sorted(
        (run for run in runs if run["partial"]),
        key=lambda x: x["run_id"],
    )
    missing = [run for run in partials if run["run_id"].replace("_partial", "") not in finals]
    if not missing:
        return

    print("\n== PARTIALS WITHOUT MATCHING FINAL ==")
    for run in missing:
        print(f"{run['run_id']}  file={run['path']}")


def print_head_configs(configs: list[dict[str, Any]]) -> None:
    if not configs:
        return
    print("\n== EHPC HEAD CONFIGS ==")
    for item in configs:
        print(
            f"{item['model_key']}: layer={item['layer']} heads={item['heads']}"
        )


def write_csv(runs: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "run_id",
        "partial",
        "models",
        "conditions",
        "total",
        "success",
        "errors",
        "fallbacks",
        "status_counts",
        "error_type_counts",
        "path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "benchmark": run["benchmark"],
                    "run_id": run["run_id"],
                    "partial": run["partial"],
                    "models": ",".join(run["model_keys"]),
                    "conditions": ",".join(run["conditions"]),
                    "total": run["total"],
                    "success": run["success"],
                    "errors": run["errors"],
                    "fallbacks": run["fallbacks"],
                    "status_counts": json.dumps(run["status_counts"], sort_keys=True),
                    "error_type_counts": json.dumps(run["error_type_counts"], sort_keys=True),
                    "path": run["path"],
                }
            )


def print_paper_tables(metric_runs: list[dict[str, Any]], paper_only: bool = True) -> None:
    rows = [run for run in metric_runs if (run["paper_candidate"] if paper_only else True)]
    if not rows:
        return

    longbench_rows = [run for run in rows if run["benchmark"] == "longbench"]
    repobench_rows = [run for run in rows if run["benchmark"] == "repobench"]

    if longbench_rows:
        render_table(
            title="LongBench Paper Summary",
            rows=sorted(longbench_rows, key=lambda r: (r["method"], r["model_key"], r["condition"])),
            benchmark="longbench",
        )
    if repobench_rows:
        render_table(
            title="RepoBench Paper Summary",
            rows=sorted(repobench_rows, key=lambda r: (r["method"], r["model_key"], r["condition"])),
            benchmark="repobench",
        )


def render_table(*, title: str, rows: list[dict[str, Any]], benchmark: str) -> None:
    if console is None or Table is None or box is None:
        print(f"\n== {title.upper()} ==")
        for row in rows:
            metrics = f"EM={row['exact_match']:.1%} TokRed={row['input_token_reduction']:.1%}"
            if benchmark == "longbench":
                metrics += f" F1={row['f1']:.3f}"
            else:
                metrics += f" EditSim={row['edit_sim']:.3f}"
            print(
                f"{row['method']} {row['model_key']} {row['condition']} "
                f"N={row['n_total']} OK={row['n_ok']} Err={row['n_error']} {metrics}"
            )
        return

    table = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Method", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Cond", style="magenta")
    table.add_column("N", justify="right")
    table.add_column("OK", justify="right")
    table.add_column("Err", justify="right")
    if benchmark == "longbench":
        table.add_column("EM", justify="right")
        table.add_column("F1", justify="right")
    else:
        table.add_column("EM%", justify="right")
        table.add_column("EditSim", justify="right")
    table.add_column("TokRed", justify="right")

    for row in rows:
        cond = row["condition"] or "-"
        tok_red = f"{row['input_token_reduction']:.1%}"
        em = f"{row['exact_match']:.1%}"
        if benchmark == "longbench":
            metric = f"{row['f1']:.3f}"
            table.add_row(
                row["method"],
                row["model_key"],
                cond,
                str(row["n_total"]),
                str(row["n_ok"] + row["n_fallback"]),
                str(row["n_error"]),
                em,
                metric,
                tok_red,
            )
        else:
            metric = f"{row['edit_sim']:.3f}"
            table.add_row(
                row["method"],
                row["model_key"],
                cond,
                str(row["n_total"]),
                str(row["n_ok"] + row["n_fallback"]),
                str(row["n_error"]),
                em,
                metric,
                tok_red,
            )
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Modal benchmark result trees.")
    parser.add_argument(
        "root",
        nargs="?",
        default="modal_results_full_20260319/all_results",
        help="Root directory containing downloaded Modal results",
    )
    parser.add_argument(
        "--csv",
        default="modal_results_full_20260319/results_summary.csv",
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--no-paper-tables",
        action="store_true",
        help="Disable compact paper-oriented metric tables",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke and partial runs in the paper-oriented metric tables",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Results root not found: {root}")

    runs = collect_runs(root)
    metric_runs = collect_metric_runs(root)
    configs = collect_head_configs(root)

    print(f"Scanned: {root}")
    print(f"Found {len(runs)} run summaries")
    if not args.no_paper_tables:
        print_paper_tables(metric_runs, paper_only=not args.include_smoke)
    print_runs(runs)
    print_missing_finals(runs)
    print_head_configs(configs)

    if args.csv:
        write_csv(runs, Path(args.csv))
        print(f"\nWrote CSV summary to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
