from pathlib import Path

import pytest


def test_modal_model_matrix_exposes_expected_keys():
    from latent_rollback_modal.config import MODEL_MATRIX

    assert {"llama3-8b", "qwen25-7b", "deepseek-14b", "mistral-24b"} <= set(MODEL_MATRIX)


def test_modal_runtime_paths_live_under_modal_roots():
    from latent_rollback_modal.config import runtime_paths

    paths = runtime_paths()
    assert str(paths.results_root).startswith("/vol/results")
    assert str(paths.hf_home).startswith("/vol/hf-cache")
    assert str(paths.datasets_cache).startswith("/vol/datasets-cache")


def test_modal_benchmark_registry_covers_all_benchmark_families():
    from latent_rollback_modal.modal_app import BENCHMARK_ENTRYPOINTS

    assert {
        "repobench",
        "longbench",
        "matrix",
        "ablation",
        "refactor",
        "integration",
    } <= set(BENCHMARK_ENTRYPOINTS)


def test_modal_cli_targets_known_benchmark_module():
    from latent_rollback_modal.modal_cli import resolve_benchmark_module

    assert resolve_benchmark_module("repobench").endswith("benchmark_repobench")
    with pytest.raises(KeyError):
        resolve_benchmark_module("missing")


def test_modal_package_readme_exists():
    readme = Path("/Users/akamel/Code/CV/latent_rollback_modal/README.md")
    assert readme.exists()
