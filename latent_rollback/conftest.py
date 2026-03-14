"""
Pytest configuration for the latent rollback test bench.

Adds CLI options used by test_bench_*.py files.  All options have safe defaults
so existing tests (test_code_fblock.py, test_code_refactor_benchmark.py) are
unaffected.

Isolation usage:

  # Unit tests only — no model, instant:
  pytest test_bench_corpus.py test_bench_fblocks.py test_bench_metrics.py test_bench_conditions.py

  # Single condition, specific model:
  pytest test_bench_integration.py \\
      --run-live \\
      --bench-model llama3-8b \\
      --bench-injection vec \\
      --bench-fblock bm25_single \\
      --bench-task-type single_hop

  # Full matrix, all models:
  pytest test_bench_integration.py --run-live

  # Isolate by pytest -k (condition id is injection__fblock):
  pytest test_bench_integration.py --run-live -k "vec__bm25_double_seq"
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# CLI option registration
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help=(
            "Actually load local models and run inference. "
            "Without this flag all integration tests are skipped."
        ),
    )
    parser.addoption(
        "--bench-model",
        default=None,
        metavar="MODEL_KEY",
        help=(
            "Model key to use for live tests "
            "(e.g. llama3-8b, qwen25-7b, deepseek-14b). "
            "Defaults to all models in MODEL_MATRIX."
        ),
    )
    parser.addoption(
        "--bench-injection",
        default=None,
        choices=["baseline", "vec", "matrix", "all"],
        help="Injection strategy to test. Defaults to all.",
    )
    parser.addoption(
        "--bench-fblock",
        default=None,
        choices=[
            "none", "ner", "bm25_single",
            "bm25_double_seq", "bm25_double_entity",
            "model_summary", "all",
        ],
        help="F block strategy to test. Defaults to all.",
    )
    parser.addoption(
        "--bench-task-type",
        default=None,
        choices=[
            "multifile_refactor", "cross_file_ref", "single_hop",
            "double_hop", "short_ctx", "long_ctx", "all",
        ],
        help="Task type to run. Defaults to all.",
    )
    parser.addoption(
        "--bench-n-passes",
        type=int,
        default=5,
        metavar="N",
        help="Number of repeated query passes for amortization tracking (default: 5).",
    )
    parser.addoption(
        "--bench-layer",
        type=int,
        default=None,
        metavar="LAYER",
        help="Override injection layer for all models (skips heuristic).",
    )


# ---------------------------------------------------------------------------
# Session-scoped fixtures (safe defaults for unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def run_live(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-live"))


@pytest.fixture(scope="session")
def bench_model_key(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--bench-model")


@pytest.fixture(scope="session")
def bench_injection(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--bench-injection") or "all"


@pytest.fixture(scope="session")
def bench_fblock(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--bench-fblock") or "all"


@pytest.fixture(scope="session")
def bench_task_type(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--bench-task-type") or "all"


@pytest.fixture(scope="session")
def bench_n_passes(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--bench-n-passes"))


@pytest.fixture(scope="session")
def bench_layer(request: pytest.FixtureRequest) -> int | None:
    return request.config.getoption("--bench-layer")


# ---------------------------------------------------------------------------
# Live model fixture — skips when --run-live is absent
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def live_wrapper(run_live: bool, bench_model_key: str | None):
    """
    Load the model wrapper for live integration tests.

    Skips automatically if --run-live is not set.
    Cleans up (GC + Metal cache clear) after the session.

    Yields (wrapper, model_key) tuple.
    """
    if not run_live:
        pytest.skip("Pass --run-live to load a model for integration tests.")

    from benchmark_runner import MODEL_MATRIX
    from backend_mlx import load_model
    import gc

    model_key = bench_model_key or "llama3-8b"
    if model_key not in MODEL_MATRIX:
        pytest.skip(f"Unknown model key: {model_key!r}. "
                    f"Available: {list(MODEL_MATRIX.keys())}")

    cfg = MODEL_MATRIX[model_key]
    wrapper = load_model(cfg["hf_id"])

    yield wrapper, model_key

    del wrapper
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
