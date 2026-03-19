from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any


def write_manifest(
    *,
    records: list[Any],
    run_id: str,
    benchmark: str,
    results_dir: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = results_dir / f"{run_id}.manifest.json"

    status_counter = Counter(getattr(record, "status", "unknown") for record in records)
    error_counter = Counter(
        getattr(record, "error_type", "")
        for record in records
        if getattr(record, "status", "unknown") == "error"
    )
    non_error_records = len(records) - status_counter.get("error", 0)

    payload = {
        "run_id": run_id,
        "benchmark": benchmark,
        "total_records": len(records),
        "success_records": non_error_records,
        "ok_records": status_counter.get("ok", 0),
        "fallback_records": status_counter.get("fallback", 0),
        "error_records": status_counter.get("error", 0),
        "status_counts": dict(status_counter),
        "error_type_counts": {k: v for k, v in error_counter.items() if k},
        "model_keys": sorted({getattr(record, "model_key", "") for record in records if getattr(record, "model_key", "")}),
        "conditions": sorted(
            {
                getattr(record, "ratio_name", getattr(record, "setting_name", ""))
                for record in records
                if getattr(record, "ratio_name", getattr(record, "setting_name", ""))
            }
        ),
        "example_ids": [
            getattr(record, "example_id", "")
            for record in records
            if getattr(record, "example_id", "")
        ],
        "extra": extra or {},
    }

    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def summarize_records(records: list[Any], metric_fields: list[str]) -> dict[str, float | int]:
    usable_records = [
        record
        for record in records
        if getattr(record, "status", "unknown") in {"ok", "fallback"}
    ]
    summary: dict[str, float | int] = {
        "n_total": len(records),
        "n_ok": sum(1 for record in records if getattr(record, "status", "unknown") == "ok"),
        "n_fallback": sum(
            1 for record in records if getattr(record, "status", "unknown") == "fallback"
        ),
        "n_error": sum(1 for record in records if getattr(record, "status", "unknown") == "error"),
    }
    if not usable_records:
        for field in metric_fields:
            summary[field] = 0.0
        summary["input_token_reduction"] = 0.0
        return summary

    for field in metric_fields:
        summary[field] = (
            sum(float(getattr(record, field)) for record in usable_records)
            / len(usable_records)
        )
    summary["input_token_reduction"] = (
        sum(float(getattr(record, "input_token_reduction")) for record in usable_records)
        / len(usable_records)
    )
    return summary


def record_to_dict(record: Any) -> dict[str, Any]:
    return asdict(record)
