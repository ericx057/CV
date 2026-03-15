"""
Full integration test bench — requires a local model.

Skipped by default.  Pass --run-live to load the model and run inference.

Isolation:
  # Run a specific condition on a specific model and task type:
  pytest test_bench_integration.py \\
      --run-live \\
      --bench-model llama3-8b \\
      --bench-injection vec \\
      --bench-fblock bm25_single \\
      --bench-task-type single_hop

  # Run all conditions on all tasks:
  pytest test_bench_integration.py --run-live --bench-model llama3-8b

  # Use pytest -k to isolate by condition ID:
  pytest test_bench_integration.py --run-live -k "vec__bm25_double_seq"
  pytest test_bench_integration.py --run-live -k "matrix__model_summary"
  pytest test_bench_integration.py --run-live -k "rename_3file"

  # Run just double-hop tasks across vec conditions:
  pytest test_bench_integration.py --run-live \\
      --bench-injection vec \\
      --bench-task-type double_hop

For each (injection, fblock, task) triple the test:
  1. Runs the condition
  2. Records token counts (input + output) for setup pass + 5 query passes
  3. Computes amortization report
  4. Asserts score >= 0.0 (output is graded)
  5. Logs results for downstream analysis

Results are written to integration_results/<model>_<run_id>.jsonl
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Generator

import pytest

from .bench_tasks import (
    BENCH_TASKS,
    BenchTask,
    get_task,
    get_tasks_by_type,
    TASK_TYPES,
)
from .bench_metrics import (
    PassBudget,
    AmortizationReport,
    compute_amortization,
    approx_tokens,
)
from .test_bench_conditions import (
    ALL_CONDITIONS,
    _filter_conditions,
    _cond_id,
)
from .config import results_path

RESULTS_DIR = results_path("integration_results")

# ---------------------------------------------------------------------------
# Condition + task parametrization helpers
# ---------------------------------------------------------------------------

def _active_conditions(config) -> list[tuple[str, str]]:
    """Return conditions to test based on CLI options."""
    inj = config.getoption("--bench-injection") or "all"
    fb = config.getoption("--bench-fblock") or "all"
    return _filter_conditions(inj, fb)


def _active_tasks(config) -> list[BenchTask]:
    """Return tasks to test based on CLI options."""
    tt = config.getoption("--bench-task-type") or "all"
    if tt == "all":
        return BENCH_TASKS
    return get_tasks_by_type(tt)


# ---------------------------------------------------------------------------
# F block factory: returns the F block string for a (fblock, context, question)
# ---------------------------------------------------------------------------

def _build_fblock(
    fblock: str,
    context: str,
    question: str,
    wrapper=None,
    summary_prompt: str = "",
    max_new_tokens: int = 150,
) -> tuple[str, int]:
    """
    Build the F block string for a given strategy.

    Returns (fblock_str, setup_extra_tokens) where setup_extra_tokens is
    the token cost of building the F block (0 for static strategies,
    >0 for model_summary).
    """
    from .benchmark_ablation import (
        extract_facts_ner,
        extract_facts_bm25,
        extract_facts_bm25_double_seq,
        extract_facts_bm25_double_entity,
    )

    if fblock == "none":
        return "", 0

    if fblock == "ner":
        return extract_facts_ner(context), 0

    if fblock == "bm25_single":
        return extract_facts_bm25(context, question), 0

    if fblock == "bm25_double_seq":
        return extract_facts_bm25_double_seq(context, question), 0

    if fblock == "bm25_double_entity":
        return extract_facts_bm25_double_entity(context, question), 0

    if fblock == "model_summary":
        if wrapper is None:
            return "", 0
        from .benchmark_code_refactor import build_summary_prompt
        from .context_injector import generate_baseline_qa
        prompt = build_summary_prompt(context)
        summary, n_summary_toks = generate_baseline_qa(wrapper, prompt, max_new_tokens)
        return f"Summary: {summary}", n_summary_toks

    return "", 0


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------

def run_condition(
    wrapper,
    task: BenchTask,
    injection: str,
    fblock: str,
    injection_layer: int,
    n_query_passes: int = 5,
    scale: float = 1.0,
    max_new_tokens: int = 80,
) -> tuple[list[PassBudget], list[str], AmortizationReport]:
    """
    Run one (injection, fblock, task) condition for n_query_passes.

    Returns:
        passes:  PassBudget for setup + each query pass
        outputs: model output strings (one per query pass)
        report:  AmortizationReport
    """
    from .context_injector import (
        extract_context_state,
        generate_baseline_qa,
        generate_with_context_injection,
        truncate_at_stop,
    )
    from .benchmark_datasets import grade_qa

    passes: list[PassBudget] = []
    outputs: list[str] = []
    condition_key = f"{injection}__{fblock}"

    # --- Setup pass ---
    setup_extra = 0

    if injection == "baseline":
        # No setup: context is always in the prompt
        ctx_vector = None
        fblock_str = ""
        setup_input_toks = 0
    else:
        # Extract context vector (vec) or matrix (matrix)
        # Both use extract_context_state for now; matrix path adds SVD
        ctx_vector, n_ctx_toks = extract_context_state(
            wrapper, task.context, injection_layer
        )
        setup_input_toks = n_ctx_toks

        # Build F block (model_summary adds extra setup tokens)
        fblock_str, setup_extra = _build_fblock(
            fblock=fblock,
            context=task.context,
            question=task.question,
            wrapper=wrapper,
            max_new_tokens=max_new_tokens,
        )

    passes.append(PassBudget(
        pass_num=0,
        condition=condition_key,
        fblock=fblock,
        input_tokens=setup_input_toks + setup_extra,
        output_tokens=0,
        is_setup=True,
        task_id=task.id,
        model_key=getattr(wrapper, "_model_key", "unknown"),
    ))

    baseline_per_query = len(wrapper.encode(
        f"Context: {task.context}\nQuestion: {task.question}\nAnswer:"
    ))

    # --- Query passes ---
    for pass_num in range(1, n_query_passes + 1):
        t_start = time.perf_counter()

        if injection == "baseline":
            prompt = f"Context: {task.context}\nQuestion: {task.question}\nAnswer:"
            text, n_input_toks = generate_baseline_qa(wrapper, prompt, max_new_tokens)
        else:
            q_prompt = task.question + "\nAnswer:"
            if fblock_str:
                q_prompt = fblock_str + "\n" + q_prompt
            text, n_input_toks = generate_with_context_injection(
                wrapper, q_prompt, ctx_vector,
                layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
            )

        text = truncate_at_stop(text)
        n_out_toks = len(wrapper.encode(text)) if text else 0

        passes.append(PassBudget(
            pass_num=pass_num,
            condition=condition_key,
            fblock=fblock,
            input_tokens=n_input_toks,
            output_tokens=n_out_toks,
            is_setup=False,
            task_id=task.id,
            model_key=getattr(wrapper, "_model_key", "unknown"),
        ))
        outputs.append(text)

    report = compute_amortization(passes, baseline_per_query_tokens=baseline_per_query)
    return passes, outputs, report


# ---------------------------------------------------------------------------
# Grading helper
# ---------------------------------------------------------------------------

def _grade_output(output: str, task: BenchTask) -> dict:
    """Grade a model output against the task gold answers."""
    from .benchmark_datasets import grade_qa, grade_code_qa

    if task.task_type in ("multifile_refactor",):
        # For refactor tasks, use must_contain / must_not_contain scoring
        present = [p for p in task.must_contain if p in output]
        forbidden = [p for p in task.must_not_contain if p in output]
        all_present = len(present) == len(task.must_contain)
        score = len(present) / max(len(task.must_contain), 1)
        if forbidden:
            score *= 0.5
        return {
            "score": score,
            "all_present": all_present,
            "any_forbidden": len(forbidden) > 0,
        }
    else:
        return grade_code_qa(output, task.gold_answers)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save_result(
    model_key: str,
    task: BenchTask,
    injection: str,
    fblock: str,
    outputs: list[str],
    passes: list[PassBudget],
    report: AmortizationReport,
    grades: list[dict],
    run_id: str,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{model_key}_{run_id}.jsonl"

    record = {
        "run_id": run_id,
        "model_key": model_key,
        "task_id": task.id,
        "task_type": task.task_type,
        "injection": injection,
        "fblock": fblock,
        "condition_id": _cond_id(injection, fblock),
        "n_files": task.n_files,
        "hop_count": task.hop_count,
        "context_tokens_approx": task.context_tokens_approx,
        "setup_tokens": report.setup_tokens,
        "per_query_tokens": report.per_query_tokens,
        "baseline_per_query_tokens": report.baseline_per_query_tokens,
        "break_even_n": (
            report.break_even_n
            if report.break_even_n != float("inf") else None
        ),
        "is_amortized": report.is_amortized,
        "savings_pct": report.savings_pct,
        "total_injection_cost": report.total_injection_cost,
        "total_baseline_cost": report.total_baseline_cost,
        "avg_score": sum(g.get("score", g.get("f1", 0.0)) for g in grades) / len(grades) if grades else 0.0,
        "outputs": outputs,
        "grades": grades,
    }

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Integration test class
# ---------------------------------------------------------------------------

# Parametrize over all 13 conditions × all 17 tasks = 221 combinations.
# CLI options --bench-injection / --bench-fblock / --bench-task-type filter this.
# Use pytest -k to narrow further by condition_id or task_id.

@pytest.mark.slow
class TestFullMatrix:
    """
    Full (injection × fblock × task) matrix.

    Each test method is independently runnable by pytest -k.
    The --bench-* CLI options filter which conditions and tasks are active.
    """

    @pytest.fixture(autouse=True)
    def _require_live(self, run_live):
        if not run_live:
            pytest.skip("Pass --run-live to execute integration tests.")

    @pytest.fixture(autouse=True)
    def _setup(self, live_wrapper, bench_n_passes, bench_layer, request):
        self.wrapper, self.model_key = live_wrapper
        self.n_passes = bench_n_passes
        self.run_id = str(int(time.time()))

        # Determine injection layer
        if bench_layer is not None:
            self.injection_layer = bench_layer
        else:
            from .layer_selector import select_layer_heuristic
            from .benchmark_runner import MODEL_MATRIX
            cfg = MODEL_MATRIX.get(self.model_key, {})
            self.injection_layer = select_layer_heuristic(
                self.wrapper, cfg.get("hf_id", "")
            )

    # -----------------------------------------------------------------------
    # Single-hop tasks
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_single_hop_return_type(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("return_type_q", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_single_hop_param_type(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("param_type_q", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_single_hop_constant(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("constant_q", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    # -----------------------------------------------------------------------
    # Double-hop tasks
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_double_hop_transitive_return(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("transitive_return", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_double_hop_inherited_method(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("inherited_method", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_double_hop_field_access(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("field_access", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    # -----------------------------------------------------------------------
    # Multifile refactor tasks
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_refactor_rename_3file(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("rename_3file", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_refactor_add_param_chain(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("add_param_chain", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_refactor_interface_change(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("interface_change", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_refactor_move_function(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("move_function", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    # -----------------------------------------------------------------------
    # Cross-file reference tasks
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_cross_file_who_calls(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("who_calls", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_cross_file_type_provenance(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("type_provenance", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_cross_file_import_chain(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("import_chain", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    # -----------------------------------------------------------------------
    # Short context tasks
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_short_ctx_return(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("short_return", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_short_ctx_constant(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("short_constant", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    # -----------------------------------------------------------------------
    # Long context tasks
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_long_ctx_refactor(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("long_refactor", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    @pytest.mark.parametrize("injection,fblock", ALL_CONDITIONS,
                             ids=[_cond_id(i, f) for i, f in ALL_CONDITIONS])
    def test_long_ctx_double_hop(self, injection, fblock, bench_injection, bench_fblock, bench_task_type):
        self._run_task_condition("long_double_hop", injection, fblock,
                                 bench_injection, bench_fblock, bench_task_type)

    # -----------------------------------------------------------------------
    # Core runner (shared by all test methods)
    # -----------------------------------------------------------------------

    def _run_task_condition(
        self,
        task_id: str,
        injection: str,
        fblock: str,
        bench_injection: str,
        bench_fblock: str,
        bench_task_type: str,
    ) -> None:
        """
        Run a single (task, injection, fblock) combination.

        Skips if filtered by CLI options.
        Records token budgets, grades outputs, asserts basic sanity,
        and persists results to JSONL.
        """
        # Apply CLI filters
        if bench_injection != "all" and injection != bench_injection:
            pytest.skip(f"Filtered: --bench-injection={bench_injection}")
        if bench_fblock != "all" and fblock != bench_fblock:
            pytest.skip(f"Filtered: --bench-fblock={bench_fblock}")

        task = get_task(task_id)
        if bench_task_type != "all" and task.task_type != bench_task_type:
            pytest.skip(f"Filtered: --bench-task-type={bench_task_type}")

        # Run condition
        passes, outputs, report = run_condition(
            wrapper=self.wrapper,
            task=task,
            injection=injection,
            fblock=fblock,
            injection_layer=self.injection_layer,
            n_query_passes=self.n_passes,
        )

        # Grade each output
        grades = [_grade_output(out, task) for out in outputs]

        # Sanity assertions
        assert len(passes) == self.n_passes + 1, (
            f"Expected {self.n_passes + 1} passes (1 setup + {self.n_passes} queries), "
            f"got {len(passes)}"
        )
        assert len(outputs) == self.n_passes
        assert report.n_passes == self.n_passes

        for g in grades:
            score = g.get("score", g.get("f1", 0.0))
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

        # Token reduction for non-baseline should be > 0
        if injection != "baseline":
            assert report.per_query_tokens < report.baseline_per_query_tokens, (
                f"{injection}/{fblock}: per_query ({report.per_query_tokens}) "
                f">= baseline_per_query ({report.baseline_per_query_tokens})"
            )

        # Persist
        _save_result(
            model_key=self.model_key,
            task=task,
            injection=injection,
            fblock=fblock,
            outputs=outputs,
            passes=passes,
            report=report,
            grades=grades,
            run_id=self.run_id,
        )


# ---------------------------------------------------------------------------
# Amortization summary test (runs after the full matrix)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAmortizationSummary:
    """
    After running the full matrix, verify that injection conditions
    amortize within 5 passes for the majority of tasks.
    This test is run separately from the per-condition tests.
    """

    @pytest.fixture(autouse=True)
    def _require_live(self, run_live):
        if not run_live:
            pytest.skip("Pass --run-live to execute integration tests.")

    def test_vec_bm25_single_amortizes_on_medium_context(
        self, live_wrapper, bench_n_passes, bench_layer
    ):
        wrapper, model_key = live_wrapper
        layer = bench_layer or 14

        task = get_task("rename_3file")
        passes, outputs, report = run_condition(
            wrapper=wrapper, task=task,
            injection="vec", fblock="bm25_single",
            injection_layer=layer, n_query_passes=bench_n_passes,
        )
        assert report.is_amortized or report.break_even_n <= bench_n_passes + 2, (
            f"vec/bm25_single did not amortize on rename_3file at n={bench_n_passes}: "
            f"break_even={report.break_even_n:.1f}"
        )

    def test_token_reduction_fraction_for_long_ctx(
        self, live_wrapper, bench_n_passes, bench_layer
    ):
        wrapper, model_key = live_wrapper
        layer = bench_layer or 14

        task = get_task("long_refactor")
        passes, outputs, report = run_condition(
            wrapper=wrapper, task=task,
            injection="vec", fblock="ner",
            injection_layer=layer, n_query_passes=bench_n_passes,
        )
        # Long context should show meaningful token reduction per query
        reduction = 1.0 - report.per_query_tokens / report.baseline_per_query_tokens
        assert reduction > 0.5, (
            f"Expected >50% per-query token reduction on long_ctx, "
            f"got {reduction:.1%}"
        )
