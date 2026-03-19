from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from .config import (
    DEFAULT_FUNCTION_TIMEOUT_SECONDS,
    DEFAULT_GPU,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    build_env,
)

BENCHMARK_ENTRYPOINTS = {
    "longbench": "longllmlingua_modal.benchmark_longbench",
    "repobench": "longllmlingua_modal.benchmark_repobench",
    "suite": "longllmlingua_modal.benchmark_suite",
}

try:
    import modal
except Exception:  # pragma: no cover
    modal = None


def _run_entrypoint(module_name: str, argv: list[str]) -> int:
    old_argv = sys.argv[:]
    old_env = os.environ.copy()
    os.environ.update(build_env())
    sys.argv = [module_name, *argv]
    try:
        try:
            runpy.run_module(module_name, run_name="__main__")
            return 0
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            if isinstance(code, int):
                return code
            raise
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

    from latent_rollback_modal.config import runtime_paths

    paths = runtime_paths()
    base_dir = Path(__file__).resolve().parent

    def build_image(requirements_name: str):
        return (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install_from_requirements(str(base_dir / requirements_name))
            .add_local_python_source("latent_rollback_modal", "longllmlingua_modal")
        )

    volumes = {
        "/vol/hf-cache": modal.Volume.from_name(paths.hf_volume_name, create_if_missing=True),
        "/vol/datasets-cache": modal.Volume.from_name(
            paths.datasets_volume_name, create_if_missing=True
        ),
        "/vol/results": modal.Volume.from_name(
            paths.results_volume_name, create_if_missing=True
        ),
    }
    app = modal.App("longllmlingua-modal")
    shared_kwargs = {
        "volumes": volumes,
        "gpu": DEFAULT_GPU,
        "timeout": DEFAULT_FUNCTION_TIMEOUT_SECONDS,
        "startup_timeout": DEFAULT_STARTUP_TIMEOUT_SECONDS,
    }
    return app, {
        "longbench": {
            "image": build_image("requirements_longbench.txt"),
            **shared_kwargs,
        },
        "repobench": {
            "image": build_image("requirements_repobench.txt"),
            **shared_kwargs,
        },
    }


def _run_benchmark_payload(benchmark: str, args: list[str] | None = None) -> dict:
    from latent_rollback_modal.config import runtime_paths

    results_root = runtime_paths().results_root
    before = snapshot_result_files(results_root)
    exit_code = _run_entrypoint(BENCHMARK_ENTRYPOINTS[benchmark], args or [])
    payload = collect_result_payload(results_root, before)
    payload["exit_code"] = exit_code
    payload["benchmark"] = benchmark
    return payload


app, _APP_KWARGS = _modal_objects()
run_suite = None

if app is not None:
    @app.function(**_APP_KWARGS["longbench"])
    def run_longbench(args: list[str] | None = None):
        return _run_benchmark_payload("longbench", args)

    @app.function(**_APP_KWARGS["repobench"])
    def run_repobench(args: list[str] | None = None):
        return _run_benchmark_payload("repobench", args)
else:
    run_longbench = None
    run_repobench = None
    run_suite = None
