#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Results root not found: {root}")

    runs = collect_runs(root)
    configs = collect_head_configs(root)

    print(f"Scanned: {root}")
    print(f"Found {len(runs)} run summaries")
    print_runs(runs)
    print_missing_finals(runs)
    print_head_configs(configs)

    if args.csv:
        write_csv(runs, Path(args.csv))
        print(f"\nWrote CSV summary to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
