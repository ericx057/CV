from __future__ import annotations

import argparse
import gc
import time

from rich.console import Console

from .backend_torch import clear_backend_cache, load_model
from .bench_tasks import BENCH_TASKS, TASK_TYPES, get_task, get_tasks_by_type
from .benchmark_runner import MODEL_MATRIX
from .layer_selector import select_layer_heuristic
from .test_bench_conditions import _filter_conditions, _cond_id
from .test_bench_integration import _grade_output, _save_result, run_condition

console = Console()

TASK_IDS = [task.id for task in BENCH_TASKS]
TASK_TYPE_CHOICES = sorted(TASK_TYPES)
INJECTION_CHOICES = ["baseline", "vec", "matrix", "all"]
FBLOCK_CHOICES = [
    "none",
    "ner",
    "bm25_single",
    "bm25_double_seq",
    "bm25_double_entity",
    "model_summary",
    "all",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic 17-task integration benchmark runner"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_MATRIX.keys()),
        choices=list(MODEL_MATRIX.keys()),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_IDS),
        choices=TASK_IDS,
        help="Specific task ids to run. Defaults to all 17 tasks.",
    )
    parser.add_argument(
        "--task-type",
        default="all",
        choices=[*TASK_TYPE_CHOICES, "all"],
        help="Filter tasks by task family before applying --tasks.",
    )
    parser.add_argument(
        "--injection",
        default="all",
        choices=INJECTION_CHOICES,
    )
    parser.add_argument(
        "--fblock",
        default="all",
        choices=FBLOCK_CHOICES,
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=5,
        help="Number of repeated query passes for amortization tracking.",
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=80, dest="max_tokens")
    parser.add_argument("--run-id", default=None, dest="run_id")
    return parser


def _selected_tasks(task_ids: list[str], task_type: str):
    allowed = set(task_ids)
    tasks = BENCH_TASKS if task_type == "all" else get_tasks_by_type(task_type)
    return [task for task in tasks if task.id in allowed]


def _print_summary(model_key: str, completed: int, total: int, avg_score: float) -> None:
    console.print(
        f"[bold]{model_key}[/bold] completed {completed}/{total} conditions "
        f"(avg score={avg_score:.3f})"
    )


def _run_for_model(model_key: str, args, run_id: str) -> None:
    cfg = MODEL_MATRIX[model_key]
    wrapper = load_model(cfg["hf_id"])
    wrapper._model_key = model_key
    try:
        tasks = _selected_tasks(args.tasks, args.task_type)
        conditions = _filter_conditions(args.injection, args.fblock)
        if args.layer is not None:
            injection_layer = args.layer
        else:
            injection_layer = select_layer_heuristic(wrapper, cfg["hf_id"])

        total = len(tasks) * len(conditions)
        completed = 0
        score_sum = 0.0
        console.rule(f"[bold cyan]Integration Bench | {model_key}[/bold cyan]")
        console.print(
            f"  Tasks      : {[task.id for task in tasks]}\n"
            f"  Conditions : {[ _cond_id(i, f) for i, f in conditions ]}\n"
            f"  Passes     : {args.passes}\n"
            f"  Layer      : {injection_layer}"
        )
        for task in tasks:
            for injection, fblock in conditions:
                completed += 1
                console.print(
                    f"  [{completed}/{total}] {task.id} :: {_cond_id(injection, fblock)}"
                )
                passes, outputs, report = run_condition(
                    wrapper=wrapper,
                    task=task,
                    injection=injection,
                    fblock=fblock,
                    injection_layer=injection_layer,
                    n_query_passes=args.passes,
                    scale=args.scale,
                    max_new_tokens=args.max_tokens,
                )
                grades = [_grade_output(out, task) for out in outputs]
                avg_score = (
                    sum(g.get("score", g.get("f1", 0.0)) for g in grades) / len(grades)
                    if grades
                    else 0.0
                )
                score_sum += avg_score
                _save_result(
                    model_key=model_key,
                    task=task,
                    injection=injection,
                    fblock=fblock,
                    outputs=outputs,
                    passes=passes,
                    report=report,
                    grades=grades,
                    run_id=run_id,
                )
        _print_summary(model_key, completed, total, score_sum / max(total, 1))
    finally:
        del wrapper
        gc.collect()
        clear_backend_cache()


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or f"integration_{int(time.time())}"
    console.rule("[bold magenta]Synthetic Integration Benchmark[/bold magenta]")
    console.print(f"  Run ID : {run_id}")
    for model_key in args.models:
        _run_for_model(model_key, args, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
