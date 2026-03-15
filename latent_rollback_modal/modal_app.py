from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from .config import runtime_paths

BENCHMARK_ENTRYPOINTS = {
    "repobench": "latent_rollback_modal.benchmark_repobench",
    "longbench": "latent_rollback_modal.benchmark_runner",
    "matrix": "latent_rollback_modal.benchmark_matrix_runner",
    "ablation": "latent_rollback_modal.benchmark_ablation",
    "refactor": "latent_rollback_modal.benchmark_code_refactor",
    "integration": "latent_rollback_modal.test_bench_integration",
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
    return app, {"image": image, "volumes": volumes, "gpu": "A10G"}


def create_modal_app():
    app, kwargs = _modal_objects()
    if app is None:
        return None

    for benchmark, module_name in BENCHMARK_ENTRYPOINTS.items():
        def _factory(name: str):
            @app.function(**kwargs)
            def _runner(args: list[str] | None = None):
                return _run_entrypoint(BENCHMARK_ENTRYPOINTS[name], args or [])

            return _runner

        globals()[f"run_{benchmark}"] = _factory(benchmark)

    return app


app = create_modal_app()
