from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from .config import runtime_paths

DEFAULT_GPU = "A100-80GB"
DEFAULT_FUNCTION_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60 * 60

BENCHMARK_ENTRYPOINTS = {
    "repobench": "latent_rollback_modal.benchmark_repobench",
    "longbench": "latent_rollback_modal.benchmark_runner",
    "matrix": "latent_rollback_modal.benchmark_matrix_runner",
    "ablation": "latent_rollback_modal.benchmark_ablation",
    "refactor": "latent_rollback_modal.benchmark_code_refactor",
    "integration": "latent_rollback_modal.benchmark_integration_runner",
}

try:
    import modal
except Exception:  # pragma: no cover - allows local import without Modal installed
    modal = None


def _build_env() -> dict[str, str]:
    paths = runtime_paths()
    return {
        "HF_HOME": str(paths.hf_home),
        "HF_DATASETS_CACHE": str(paths.datasets_cache),
        "TRANSFORMERS_CACHE": str(paths.hf_home / "transformers"),
        "LATENT_ROLLBACK_RESULTS_ROOT": str(paths.results_root),
    }


def _run_entrypoint(module_name: str, argv: list[str]) -> int:
    old_argv = sys.argv[:]
    old_env = os.environ.copy()
    os.environ.update(_build_env())
    sys.argv = [module_name, *argv]
    try:
        runpy.run_module(module_name, run_name="__main__")
        return 0
    finally:
        sys.argv = old_argv
        os.environ.clear()
        os.environ.update(old_env)


def snapshot_result_files(results_root: Path) -> dict[str, int]:
    if not results_root.exists():
        return {}
    return {
        str(path.relative_to(results_root)): path.stat().st_mtime_ns
        for path in results_root.rglob("*")
        if path.is_file()
    }


def collect_result_payload(results_root: Path, before: dict[str, int]) -> dict:
    after = snapshot_result_files(results_root)
    changed = {
        rel_path: (results_root / rel_path).read_text()
        for rel_path, mtime in after.items()
        if before.get(rel_path) != mtime
    }
    return {"files": changed}


def _modal_objects():
    if modal is None:
        return None, None

    paths = runtime_paths()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install_from_requirements(str(Path(__file__).with_name("requirements.txt")))
    )
    volumes = {
        "/vol/hf-cache": modal.Volume.from_name(paths.hf_volume_name, create_if_missing=True),
        "/vol/datasets-cache": modal.Volume.from_name(paths.datasets_volume_name, create_if_missing=True),
        "/vol/results": modal.Volume.from_name(paths.results_volume_name, create_if_missing=True),
    }
    app = modal.App("latent-rollback-modal")
    return app, {
        "image": image,
        "volumes": volumes,
        "gpu": DEFAULT_GPU,
        "timeout": DEFAULT_FUNCTION_TIMEOUT_SECONDS,
        "startup_timeout": DEFAULT_STARTUP_TIMEOUT_SECONDS,
    }


def _run_benchmark_payload(benchmark: str, args: list[str] | None = None) -> dict:
    results_root = runtime_paths().results_root
    before = snapshot_result_files(results_root)
    exit_code = _run_entrypoint(BENCHMARK_ENTRYPOINTS[benchmark], args or [])
    payload = collect_result_payload(results_root, before)
    payload["exit_code"] = exit_code
    payload["benchmark"] = benchmark
    return payload


app, _APP_KWARGS = _modal_objects()

if app is not None:
    @app.function(**_APP_KWARGS)
    def run_repobench(args: list[str] | None = None):
        return _run_benchmark_payload("repobench", args)

    @app.function(**_APP_KWARGS)
    def run_longbench(args: list[str] | None = None):
        return _run_benchmark_payload("longbench", args)

    @app.function(**_APP_KWARGS)
    def run_matrix(args: list[str] | None = None):
        return _run_benchmark_payload("matrix", args)

    @app.function(**_APP_KWARGS)
    def run_ablation(args: list[str] | None = None):
        return _run_benchmark_payload("ablation", args)

    @app.function(**_APP_KWARGS)
    def run_refactor(args: list[str] | None = None):
        return _run_benchmark_payload("refactor", args)

    @app.function(**_APP_KWARGS)
    def run_integration(args: list[str] | None = None):
        return _run_benchmark_payload("integration", args)
else:
    run_repobench = None
    run_longbench = None
    run_matrix = None
    run_ablation = None
    run_refactor = None
    run_integration = None
