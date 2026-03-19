from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

from . import modal_app
from .modal_app import BENCHMARK_ENTRYPOINTS, _run_entrypoint


def resolve_benchmark_module(name: str) -> str:
    if name not in BENCHMARK_ENTRYPOINTS:
        raise KeyError(name)
    return BENCHMARK_ENTRYPOINTS[name]


def local_results_root() -> Path:
    return Path(
        os.environ.get(
            "LONGLLMLINGUA_LOCAL_RESULTS_ROOT",
            str(Path(__file__).resolve().parent / "local_results"),
        )
    )


def write_local_result_payload(payload: dict, root: Path | None = None) -> list[Path]:
    target_root = root or local_results_root()
    written: list[Path] = []
    for rel_path, content in payload.get("files", {}).items():
        path = target_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        written.append(path)
    return written


def launch_remote(remote_fn, args: list[str], detach: bool, results_root: Path | None = None, app=None) -> int:
    run_ctx = nullcontext()
    if app is not None and hasattr(app, "run"):
        output_ctx = (
            modal_app.modal.enable_output()
            if getattr(modal_app, "modal", None) is not None
            else nullcontext()
        )
        run_ctx = output_ctx
        app_ctx = app.run(detach=detach)
    else:
        app_ctx = nullcontext()

    with run_ctx:
        with app_ctx:
            if detach:
                if not hasattr(remote_fn, "spawn"):
                    raise RuntimeError("Detached mode requires Modal spawn() support")
                call = remote_fn.spawn(args)
                call_id = getattr(call, "object_id", None) or getattr(call, "id", None) or "<unknown>"
                print("Launched detached Modal job.")
                print(f"Call ID: {call_id}")
                print("The job will continue even if this terminal closes.")
                return 0

            payload = remote_fn.remote(args)
            written = write_local_result_payload(payload, root=results_root)
            if written:
                print("Saved local copies:")
                for path in written:
                    print(f"  {path}")
            return int(payload.get("exit_code", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Modal-first CLI for LongLLMLingua baselines")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("benchmark", choices=sorted(BENCHMARK_ENTRYPOINTS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.detach and args.benchmark == "suite":
        raise RuntimeError(
            "Detached `suite` is disabled because LongBench and RepoBench require "
            "different Modal images. Launch `longbench` and `repobench` separately."
        )

    remote_fn = getattr(modal_app, f"run_{args.benchmark}", None)
    if remote_fn is not None and hasattr(remote_fn, "remote"):
        return launch_remote(remote_fn, args.args, detach=args.detach, app=modal_app.app)
    if args.detach:
        raise RuntimeError(f"Detached Modal entrypoint is not available for {args.benchmark!r}.")
    return _run_entrypoint(resolve_benchmark_module(args.benchmark), args.args)


if __name__ == "__main__":
    raise SystemExit(main())
