from __future__ import annotations

import argparse

from . import modal_app
from .modal_app import BENCHMARK_ENTRYPOINTS, _run_entrypoint


def resolve_benchmark_module(name: str) -> str:
    if name not in BENCHMARK_ENTRYPOINTS:
        raise KeyError(name)
    return BENCHMARK_ENTRYPOINTS[name]


def main() -> int:
    parser = argparse.ArgumentParser(description="Modal-first CLI for latent rollback benchmarks")
    parser.add_argument("benchmark", choices=sorted(BENCHMARK_ENTRYPOINTS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    remote_fn = getattr(modal_app, f"run_{args.benchmark}", None)
    if remote_fn is not None and hasattr(remote_fn, "remote"):
        return remote_fn.remote(args.args)
    return _run_entrypoint(resolve_benchmark_module(args.benchmark), args.args)


if __name__ == "__main__":
    raise SystemExit(main())
