"""
Multi-file refactor benchmark — 3-condition comparison.

Conditions:
  baseline       — full context + instruction in prompt (standard model call)
  vec_f_ner      — vector injection + NER/BM25 F block (existing approach)
  vec_f_summary  — vector injection + model-written summary as F block
                   (one-time cost: model reads codebase once, writes structured
                   notes; those notes become the F block for all subsequent calls)

The summary condition is the primary research interest: it models the "drift
correction every N queries" use case where a single summary generation amortizes
across many requests, saving ~40k tokens × N calls.

3 synthetic multi-file refactor tasks:
  rename_function    — rename update_user -> patch_user across 2 files
  add_parameter      — add timeout param to fetch() used across 2 files
  change_return_type — change find_by_id return from dict -> Optional[User]

Usage:
  source .venv/bin/activate
  python benchmark_code_refactor.py
  python benchmark_code_refactor.py --models llama3-8b --layer 14
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import mlx.core as mx
from rich.console import Console
from rich import box
from rich.table import Table

from backend_mlx import load_model, MLXModelWrapper, _ids_to_str
from context_injector import (
    extract_context_state,
    generate_baseline_qa,
    generate_with_context_injection,
    truncate_at_stop,
)
from layer_selector import select_layer_heuristic
from benchmark_runner import MODEL_MATRIX
from benchmark_ablation import extract_facts_code

console = Console()
RESULTS_DIR = Path(__file__).parent / "refactor_results"

# Stop sequences for refactor output — stop at double newline or new file marker
REFACTOR_STOP = ("\n\n", "# File:", "// File:")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

@dataclass
class RefactorExample:
    id: str
    context: str           # multi-file code context
    instruction: str       # refactor instruction
    must_contain: list[str]     # strings that should appear in a correct output
    must_not_contain: list[str] # strings that should NOT appear (old identifiers)


REFACTOR_TASKS: list[RefactorExample] = [
    RefactorExample(
        id="rename_function",
        context="""\
# File: repository.py
from typing import Optional
from dataclasses import dataclass

@dataclass
class UserDelta:
    name: str
    email: str

@dataclass
class User:
    id: int
    name: str
    email: str

class UserRepository:
    def find_by_id(self, user_id: int) -> Optional[User]:
        ...

    def update_user(self, user_id: int, delta: UserDelta) -> bool:
        ...

    def delete_user(self, user_id: int) -> bool:
        ...

# File: service.py
from repository import UserRepository
from dataclasses import dataclass

def apply_update(repo: UserRepository, user_id: int, name: str, email: str) -> bool:
    delta = UserDelta(name=name, email=email)
    return repo.update_user(user_id, delta)

def remove_user(repo: UserRepository, user_id: int) -> bool:
    return repo.delete_user(user_id)
""",
        instruction=(
            "Rename `update_user` to `patch_user` everywhere it appears. "
            "Update both the method definition in repository.py and all call sites in service.py."
        ),
        must_contain=["patch_user"],
        # Only flag if old name appears as actual code (def or call site),
        # not in explanation text like "rename `update_user` to `patch_user`"
        must_not_contain=["def update_user", ".update_user("],
    ),

    RefactorExample(
        id="add_parameter",
        context="""\
# File: config.py
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
BASE_URL = "https://api.example.com"

# File: client.py
import json
import urllib.request
from config import DEFAULT_TIMEOUT

def fetch(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())

def post(url: str, body: dict) -> dict:
    ...

# File: api.py
from client import fetch
from config import BASE_URL

def get_user(user_id: int) -> dict:
    return fetch(f"{BASE_URL}/users/{user_id}")

def list_users() -> list:
    return fetch(f"{BASE_URL}/users")
""",
        instruction=(
            "Add a `timeout: int = DEFAULT_TIMEOUT` parameter to the `fetch` function in client.py. "
            "Update the function signature and the urllib call to use the timeout. "
            "Update call sites in api.py to pass timeout where needed."
        ),
        must_contain=["timeout", "fetch"],
        must_not_contain=["def fetch(url: str) -> dict"],
    ),

    RefactorExample(
        id="change_return_type",
        context="""\
# File: models.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    id: int
    name: str
    email: str

# File: repository.py
from models import User
from typing import Optional

class UserRepository:
    def find_by_id(self, user_id: int) -> dict:
        ...

    def find_by_email(self, email: str) -> dict:
        ...

    def save(self, user: User) -> bool:
        ...

# File: service.py
from repository import UserRepository

def get_user_name(repo: UserRepository, user_id: int) -> str:
    user = repo.find_by_id(user_id)
    return user["name"]

def get_user_email(repo: UserRepository, user_id: int) -> str:
    user = repo.find_by_id(user_id)
    return user["email"]
""",
        instruction=(
            "Change `find_by_id` in repository.py to return `Optional[User]` instead of `dict`. "
            "Update service.py to use attribute access (`user.name`, `user.email`) "
            "instead of dict key access (`user['name']`, `user['email']`)."
        ),
        must_contain=["Optional[User]", "user.name"],
        # Only flag if old return type still appears in a function definition
        must_not_contain=["find_by_id(self, user_id: int) -> dict"],
    ),
]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

@dataclass
class RefactorRecord:
    task_id: str
    condition: str      # "baseline" | "vec_f_ner" | "vec_f_summary"
    output: str
    score: float        # fraction of must_contain items found (0-1), penalized if any_forbidden
    all_present: bool
    any_forbidden: bool
    f_block: str
    elapsed_s: float


def grade_refactor(output: str, task: RefactorExample) -> dict:
    """
    Grade a refactor output against must_contain and must_not_contain checks.

    score = (# must_contain found / total must_contain) * (0.5 if any_forbidden else 1.0)
    Perfect score = 1.0: all required identifiers present, no forbidden ones.
    """
    present = [s for s in task.must_contain if s in output]
    forbidden = [s for s in task.must_not_contain if s in output]

    all_present = len(present) == len(task.must_contain)
    any_forbidden = len(forbidden) > 0

    frac = len(present) / len(task.must_contain) if task.must_contain else 1.0
    score = round(frac * (0.5 if any_forbidden else 1.0), 4)

    return {
        "score": score,
        "all_present": all_present,
        "any_forbidden": any_forbidden,
        "present": present,
        "forbidden": forbidden,
    }


# ---------------------------------------------------------------------------
# Summary prompt
# ---------------------------------------------------------------------------

def build_summary_prompt(context: str) -> str:
    """
    Prompt that asks the model to write structured notes about a codebase.
    The output becomes the F block for subsequent injection-based calls.
    """
    return (
        f"Context:\n{context}\n\n"
        "Read the code above and write a concise structured reference covering:\n"
        "- Function signatures (name, parameters with types, return type)\n"
        "- Class names and their key methods\n"
        "- Import relationships between files\n"
        "- Key constants and type aliases\n\n"
        "Be precise with names and types. This will be used as a reference for a future code edit.\n\n"
        "Summary:"
    )


# ---------------------------------------------------------------------------
# Per-example runner
# ---------------------------------------------------------------------------

def run_refactor_example(
    wrapper: MLXModelWrapper,
    task: RefactorExample,
    injection_layer: int,
    scale: float = 1.0,
    max_new_tokens: int = 150,
) -> list[RefactorRecord]:
    """
    Run all 3 conditions for one refactor task. Returns 3 RefactorRecords.
    """
    instruction_prompt = (
        f"Instruction: {task.instruction}\n\n"
        "Describe the changes needed (file by file):\n"
    )
    full_prompt = f"Context:\n{task.context}\n\n{instruction_prompt}"

    # --- Condition 1: Baseline ---
    t0 = time.time()
    baseline_text, _ = generate_baseline_qa(
        wrapper, full_prompt, max_new_tokens=max_new_tokens,
        stop_strings=REFACTOR_STOP,
    )
    baseline_grades = grade_refactor(baseline_text, task)
    baseline_rec = RefactorRecord(
        task_id=task.id, condition="baseline",
        output=baseline_text.strip()[:300],
        score=baseline_grades["score"],
        all_present=baseline_grades["all_present"],
        any_forbidden=baseline_grades["any_forbidden"],
        f_block="",
        elapsed_s=round(time.time() - t0, 2),
    )

    # --- Extract context vector (shared by vec_f_ner and vec_f_summary) ---
    ctx_v, _ = extract_context_state(wrapper, task.context, injection_layer)

    # --- Condition 2: Vec + F (NER/BM25) ---
    t0 = time.time()
    ner_f_block = extract_facts_code(task.context, task.instruction)
    ner_prompt = f"{ner_f_block}\n{instruction_prompt}" if ner_f_block else instruction_prompt
    ner_text, _ = generate_with_context_injection(
        wrapper, ner_prompt, ctx_v,
        layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
        stop_strings=REFACTOR_STOP,
    )
    ner_grades = grade_refactor(ner_text, task)
    ner_rec = RefactorRecord(
        task_id=task.id, condition="vec_f_ner",
        output=ner_text.strip()[:300],
        score=ner_grades["score"],
        all_present=ner_grades["all_present"],
        any_forbidden=ner_grades["any_forbidden"],
        f_block=ner_f_block[:200],
        elapsed_s=round(time.time() - t0, 2),
    )

    # --- Condition 3: Vec + F (model summary) ---
    # One-time cost: call model to generate structured notes from full context.
    # These notes become the F block — analogous to the "drift correction" pass.
    t0 = time.time()
    summary_prompt = build_summary_prompt(task.context)
    summary_text, _ = generate_baseline_qa(
        wrapper, summary_prompt, max_new_tokens=200,
        stop_strings=("\n\n\n",),  # let it run longer for a full summary
    )
    summary_f_block = f"Codebase summary:\n{summary_text.strip()}"
    summary_instr_prompt = f"{summary_f_block}\n\n{instruction_prompt}"
    summary_out, _ = generate_with_context_injection(
        wrapper, summary_instr_prompt, ctx_v,
        layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
        stop_strings=REFACTOR_STOP,
    )
    summary_grades = grade_refactor(summary_out, task)
    summary_rec = RefactorRecord(
        task_id=task.id, condition="vec_f_summary",
        output=summary_out.strip()[:300],
        score=summary_grades["score"],
        all_present=summary_grades["all_present"],
        any_forbidden=summary_grades["any_forbidden"],
        f_block=summary_text.strip()[:200],
        elapsed_s=round(time.time() - t0, 2),
    )

    return [baseline_rec, ner_rec, summary_rec]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(records: list[RefactorRecord]) -> None:
    from collections import defaultdict
    console.rule("[bold magenta]Multi-File Refactor Benchmark[/bold magenta]")

    by_task: dict[str, list[RefactorRecord]] = defaultdict(list)
    for r in records:
        by_task[r.task_id].append(r)

    table = Table(
        title="Baseline vs Vec+F(NER) vs Vec+F(Summary)",
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    table.add_column("Task", style="cyan")
    table.add_column("Condition")
    table.add_column("Score", justify="right")
    table.add_column("All present?", justify="center")
    table.add_column("Forbidden?", justify="center")
    table.add_column("Output snippet", max_width=50)

    for task_id, recs in sorted(by_task.items()):
        for r in sorted(recs, key=lambda x: x.condition):
            score_color = "green" if r.score >= 0.8 else "yellow" if r.score >= 0.5 else "red"
            table.add_row(
                task_id, r.condition,
                f"[{score_color}]{r.score:.2f}[/{score_color}]",
                "[green]yes[/green]" if r.all_present else "[red]no[/red]",
                "[red]yes[/red]" if r.any_forbidden else "[green]no[/green]",
                r.output[:50].replace("\n", " "),
            )

    console.print(table)
    console.print()

    # Summary: which condition wins
    for task_id, recs in sorted(by_task.items()):
        best = max(recs, key=lambda r: r.score)
        console.print(f"  [bold]{task_id}[/bold]: best = [cyan]{best.condition}[/cyan] (score={best.score:.2f})")
        for r in sorted(recs, key=lambda x: x.condition):
            console.print(f"    {r.condition:20s}  score={r.score:.2f}  f_block={r.f_block[:60]!r}")
        console.print()


def save_results(records: list[RefactorRecord], run_id: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"{run_id}.csv"
    json_path = RESULTS_DIR / f"{run_id}.json"

    if records:
        fields = list(asdict(records[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(asdict(r) for r in records)

    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    console.print(f"\n[dim]Saved:\n  {csv_path}\n  {json_path}[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-file refactor benchmark: baseline vs vec+F(NER) vs vec+F(summary)"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["llama3-8b"],
        choices=list(MODEL_MATRIX.keys()),
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=150, dest="max_tokens")
    parser.add_argument("--run-id", default=None, dest="run_id")
    args = parser.parse_args()

    run_id = args.run_id or f"refactor_{int(time.time())}"

    console.rule("[bold magenta]Multi-File Refactor Benchmark[/bold magenta]")
    console.print(f"  Tasks      : {[t.id for t in REFACTOR_TASKS]}")
    console.print(f"  Models     : {args.models}")
    console.print(f"  Run ID     : {run_id}")
    console.print()
    console.print("  Conditions:")
    console.print("    1. baseline      — full context + instruction in prompt")
    console.print("    2. vec_f_ner     — vector injection + code NER/BM25 F block")
    console.print("    3. vec_f_summary — vector injection + model-written summary F block")

    all_records: list[RefactorRecord] = []

    for model_key in args.models:
        model_cfg = MODEL_MATRIX[model_key]
        console.rule(f"[bold cyan]{model_key}[/bold cyan]")

        try:
            wrapper = load_model(model_cfg["hf_id"])
        except Exception as exc:
            console.print(f"  [red]Load failed: {exc}[/red]")
            continue

        layer = args.layer or select_layer_heuristic(wrapper, model_cfg["hf_id"])
        console.print(f"  Injection layer: {layer} / {wrapper.n_layers}")

        for task in REFACTOR_TASKS:
            console.print(f"\n  Task: [bold]{task.id}[/bold]")
            try:
                recs = run_refactor_example(
                    wrapper, task,
                    injection_layer=layer,
                    scale=args.scale,
                    max_new_tokens=args.max_tokens,
                )
                all_records.extend(recs)
                for r in recs:
                    score_color = "green" if r.score >= 0.8 else "yellow" if r.score >= 0.5 else "red"
                    console.print(
                        f"    {r.condition:20s}  "
                        f"score=[{score_color}]{r.score:.2f}[/{score_color}]  "
                        f"present={r.all_present}  forbidden={r.any_forbidden}"
                    )
                    console.print(f"      output: {r.output[:80].replace(chr(10), ' ')}")
            except Exception as exc:
                console.print(f"  [red]ERROR on {task.id}: {exc}[/red]")

        del wrapper
        gc.collect()
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

    print_results(all_records)
    save_results(all_records, run_id)
    return 0 if all_records else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
