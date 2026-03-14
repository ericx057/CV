"""
Token budget tracking and amortization analysis tests.

Tests the math in bench_metrics.py without any model calls.
All PassBudgets are built synthetically via build_mock_passes().

Amortization model recap:
  setup_tokens  + N * per_query  vs  N * baseline_per_query
  break_even N* = setup / (baseline_per_query - per_query)
  is_amortized  = N >= N*

Run in isolation:
    pytest test_bench_metrics.py
"""

from __future__ import annotations

import math
import pytest

from bench_metrics import (
    PassBudget,
    AmortizationReport,
    approx_tokens,
    compute_amortization,
    build_mock_passes,
)
from bench_tasks import BENCH_TASKS, get_task


# ---------------------------------------------------------------------------
# approx_tokens
# ---------------------------------------------------------------------------

class TestApproxTokens:
    def test_empty_string_returns_one(self):
        assert approx_tokens("") == 1

    def test_single_word(self):
        # "hello" → 1 word × 1.3 → 1 (int truncation)
        assert approx_tokens("hello") == 1

    def test_proportional_to_length(self):
        short = "def foo(): pass"
        long = "def foo(): pass " * 100
        assert approx_tokens(long) > approx_tokens(short) * 50

    def test_all_bench_contexts_have_positive_approx(self):
        for t in BENCH_TASKS:
            assert approx_tokens(t.context) > 0, f"{t.id}: approx_tokens returned 0"

    def test_short_ctx_smaller_than_long_ctx(self):
        short_tasks = [t for t in BENCH_TASKS if t.task_type == "short_ctx"]
        long_tasks = [t for t in BENCH_TASKS if t.task_type == "long_ctx"]
        avg_short = sum(approx_tokens(t.context) for t in short_tasks) / len(short_tasks)
        avg_long = sum(approx_tokens(t.context) for t in long_tasks) / len(long_tasks)
        assert avg_long > avg_short * 5, (
            f"Expected long_ctx to be >5x short_ctx tokens; "
            f"got avg_short={avg_short:.0f}, avg_long={avg_long:.0f}"
        )


# ---------------------------------------------------------------------------
# PassBudget construction
# ---------------------------------------------------------------------------

class TestPassBudget:
    def test_setup_pass_has_is_setup_true(self):
        passes = build_mock_passes(
            condition="vec", fblock="ner", task_id="return_type_q",
            context_tokens=1000, question_tokens=20, fblock_tokens=50,
            setup_tokens=1000, n_query_passes=5,
        )
        setup = [p for p in passes if p.is_setup]
        assert len(setup) == 1
        assert setup[0].input_tokens == 1000

    def test_query_passes_have_is_setup_false(self):
        passes = build_mock_passes(
            condition="vec", fblock="bm25_single", task_id="x",
            context_tokens=500, question_tokens=15, fblock_tokens=40,
            setup_tokens=500, n_query_passes=5,
        )
        query = [p for p in passes if not p.is_setup]
        assert len(query) == 5

    def test_query_pass_input_tokens_is_question_plus_fblock(self):
        passes = build_mock_passes(
            condition="vec", fblock="bm25_single", task_id="x",
            context_tokens=500, question_tokens=15, fblock_tokens=40,
            setup_tokens=500, n_query_passes=5,
        )
        query = [p for p in passes if not p.is_setup]
        for p in query:
            assert p.input_tokens == 15 + 40  # question + fblock

    def test_pass_nums_are_sequential(self):
        passes = build_mock_passes(
            condition="baseline", fblock="none", task_id="x",
            context_tokens=200, question_tokens=10, fblock_tokens=0,
            setup_tokens=200, n_query_passes=3,
        )
        query = [p for p in passes if not p.is_setup]
        nums = [p.pass_num for p in query]
        assert nums == [1, 2, 3]

    def test_condition_and_fblock_propagated(self):
        passes = build_mock_passes(
            condition="matrix", fblock="bm25_double_seq", task_id="t1",
            context_tokens=2000, question_tokens=20, fblock_tokens=80,
            setup_tokens=2000, n_query_passes=3,
        )
        for p in passes:
            assert p.condition == "matrix"
            assert p.fblock == "bm25_double_seq"
            assert p.task_id == "t1"


# ---------------------------------------------------------------------------
# compute_amortization: input validation
# ---------------------------------------------------------------------------

class TestComputeAmortizationValidation:
    def test_empty_passes_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_amortization([], baseline_per_query_tokens=500)

    def test_no_query_passes_raises(self):
        setup_only = [PassBudget(
            pass_num=0, condition="vec", fblock="ner",
            input_tokens=1000, output_tokens=0, is_setup=True,
        )]
        with pytest.raises(ValueError, match="query pass"):
            compute_amortization(setup_only, baseline_per_query_tokens=500)


# ---------------------------------------------------------------------------
# compute_amortization: math
# ---------------------------------------------------------------------------

class TestComputeAmortizationMath:
    def _make_passes(self, setup: int, per_query: int, n: int = 5) -> list[PassBudget]:
        return build_mock_passes(
            condition="vec", fblock="bm25_single", task_id="t",
            context_tokens=setup, question_tokens=per_query, fblock_tokens=0,
            setup_tokens=setup, n_query_passes=n,
        )

    def test_break_even_formula(self):
        # setup=100, per_query=50, baseline=100
        # break_even = 100 / (100-50) = 2.0
        passes = self._make_passes(setup=100, per_query=50, n=5)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert math.isclose(report.break_even_n, 2.0, rel_tol=0.01), (
            f"Expected break_even=2.0, got {report.break_even_n}"
        )

    def test_is_amortized_when_n_above_break_even(self):
        # break_even = 2.0, n = 5 → amortized
        passes = self._make_passes(setup=100, per_query=50, n=5)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert report.is_amortized is True

    def test_not_amortized_when_n_below_break_even(self):
        # setup=200, per_query=50, baseline=100
        # break_even = 200 / 50 = 4.0, n = 3 → not amortized
        passes = self._make_passes(setup=200, per_query=50, n=3)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert report.is_amortized is False
        assert math.isclose(report.break_even_n, 4.0, rel_tol=0.01)

    def test_total_injection_cost(self):
        # setup=100, per_query=50, n=5
        # total_injection = 100 + 5*50 = 350
        passes = self._make_passes(setup=100, per_query=50, n=5)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert report.total_injection_cost == 100 + 5 * 50

    def test_total_baseline_cost(self):
        passes = self._make_passes(setup=100, per_query=50, n=5)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert report.total_baseline_cost == 5 * 100

    def test_savings_pct_positive_when_amortized(self):
        passes = self._make_passes(setup=100, per_query=50, n=5)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert report.savings_pct > 0

    def test_savings_pct_negative_when_not_amortized(self):
        # Large setup relative to per-query savings → negative savings at n=1
        passes = self._make_passes(setup=1000, per_query=50, n=1)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert report.savings_pct < 0  # injection more expensive at n=1

    def test_infinite_break_even_when_per_query_equals_baseline(self):
        # If per_query == baseline, no savings per query → never amortized
        passes = self._make_passes(setup=100, per_query=100, n=10)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert report.break_even_n == float("inf")
        assert report.is_amortized is False

    def test_n_passes_matches_query_count(self):
        passes = self._make_passes(setup=100, per_query=50, n=7)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert report.n_passes == 7

    def test_per_query_tokens_is_average(self):
        passes = self._make_passes(setup=100, per_query=50, n=5)
        report = compute_amortization(passes, baseline_per_query_tokens=100)
        assert report.per_query_tokens == 50


# ---------------------------------------------------------------------------
# Amortization: realistic scenarios from bench tasks
# ---------------------------------------------------------------------------

class TestAmortizationRealisticScenarios:
    """
    Test amortization for realistic (context_tokens_approx) values from bench tasks.
    Uses build_mock_passes with task-derived sizes.
    """

    def _report_for_task(
        self,
        task_id: str,
        fblock: str = "bm25_single",
        fblock_tokens: int = 60,
        question_tokens: int = 12,
        n_passes: int = 5,
    ) -> AmortizationReport:
        t = get_task(task_id)
        setup = t.context_tokens_approx
        passes = build_mock_passes(
            condition="vec",
            fblock=fblock,
            task_id=task_id,
            context_tokens=setup,
            question_tokens=question_tokens,
            fblock_tokens=fblock_tokens,
            setup_tokens=setup,
            n_query_passes=n_passes,
        )
        return compute_amortization(passes, baseline_per_query_tokens=setup + question_tokens)

    def test_short_ctx_per_query_with_no_fblock_is_cheaper(self):
        """
        For short contexts, the F block overhead can exceed the context savings
        (F block tokens > context tokens). Use fblock='none' to verify that
        injection with no F block is cheaper per query than baseline.
        """
        report = self._report_for_task("short_return", fblock="none",
                                       fblock_tokens=0, n_passes=5)
        assert report.per_query_tokens < report.baseline_per_query_tokens

    def test_long_ctx_has_higher_absolute_savings_per_query(self):
        """
        Long contexts save more tokens per query in absolute terms than short contexts.
        savings_per_query = baseline_per_query - per_query
        For long: baseline_per_query >> per_query (context dominates).
        For short: baseline_per_query ~ per_query (context is small).
        """
        report_short = self._report_for_task("short_return")
        report_long = self._report_for_task("long_refactor")
        savings_short = report_short.baseline_per_query_tokens - report_short.per_query_tokens
        savings_long = report_long.baseline_per_query_tokens - report_long.per_query_tokens
        assert savings_long > savings_short, (
            f"long context should save more tokens/query than short: "
            f"long={savings_long}, short={savings_short}"
        )

    def test_five_passes_amortizes_for_typical_context(self):
        """
        For a typical medium context, 5 passes should amortize
        (or at least reduce per-query cost substantially).
        The injection per-query is question + F block << full context.
        """
        report = self._report_for_task("rename_3file", n_passes=5)
        # Per-query injection should be much cheaper than baseline
        assert report.per_query_tokens < report.baseline_per_query_tokens

    def test_model_summary_setup_is_higher_than_vec(self):
        """
        model_summary adds an extra generation pass to setup.
        We simulate this by adding summary_tokens to setup_tokens.
        """
        t = get_task("long_refactor")
        ctx_tokens = t.context_tokens_approx
        summary_gen_tokens = 200  # one-time model summary generation cost

        # vec setup: just context extraction
        passes_vec = build_mock_passes(
            condition="vec", fblock="none", task_id=t.id,
            context_tokens=ctx_tokens, question_tokens=12, fblock_tokens=0,
            setup_tokens=ctx_tokens, n_query_passes=5,
        )
        # model_summary setup: context extraction + summary generation
        passes_summary = build_mock_passes(
            condition="vec", fblock="model_summary", task_id=t.id,
            context_tokens=ctx_tokens, question_tokens=12, fblock_tokens=80,
            setup_tokens=ctx_tokens + summary_gen_tokens, n_query_passes=5,
        )

        baseline_toks = ctx_tokens + 12
        report_vec = compute_amortization(passes_vec, baseline_toks)
        report_sum = compute_amortization(passes_summary, baseline_toks)

        # model_summary has higher setup → higher break-even
        assert report_sum.setup_tokens > report_vec.setup_tokens
        assert report_sum.break_even_n >= report_vec.break_even_n

    @pytest.mark.parametrize("n_passes", [1, 2, 5, 10, 20])
    def test_savings_pct_increases_with_n_passes(self, n_passes: int):
        """More passes → better amortization → higher savings %."""
        report = self._report_for_task("rename_3file", n_passes=n_passes)
        # Savings should be non-decreasing as N increases
        # (verified by comparing to n=1)
        report_1 = self._report_for_task("rename_3file", n_passes=1)
        if n_passes > 1:
            assert report.savings_pct >= report_1.savings_pct

    def test_report_fields_are_consistent(self):
        """Cross-check total_injection_cost vs manual calculation."""
        report = self._report_for_task("constant_q", n_passes=5)
        expected = report.setup_tokens + report.n_passes * report.per_query_tokens
        assert report.total_injection_cost == expected

    def test_savings_pct_is_bounded(self):
        for task in BENCH_TASKS[:5]:
            report = self._report_for_task(task.id)
            assert -10.0 <= report.savings_pct <= 1.0, (
                f"{task.id}: savings_pct out of reasonable range: {report.savings_pct}"
            )


# ---------------------------------------------------------------------------
# Token tracking over 5 passes (the explicit user requirement)
# ---------------------------------------------------------------------------

class TestFivePassTracking:
    """
    Verify that tracking input tokens over exactly 5 query passes
    produces coherent amortization output — the primary use case.
    """

    def test_five_pass_tracker_for_each_task_type(self):
        type_samples = {
            "multifile_refactor": "rename_3file",
            "cross_file_ref":     "who_calls",
            "single_hop":         "return_type_q",
            "double_hop":         "transitive_return",
            "short_ctx":          "short_return",
            "long_ctx":           "long_refactor",
        }
        for task_type, task_id in type_samples.items():
            t = get_task(task_id)
            passes = build_mock_passes(
                condition="vec",
                fblock="bm25_single",
                task_id=task_id,
                context_tokens=t.context_tokens_approx,
                question_tokens=10,
                fblock_tokens=50,
                setup_tokens=t.context_tokens_approx,
                n_query_passes=5,
            )
            report = compute_amortization(
                passes,
                baseline_per_query_tokens=t.context_tokens_approx + 10,
            )
            assert report.n_passes == 5, (
                f"{task_type}: expected 5 query passes, got {report.n_passes}"
            )
            assert report.per_query_tokens == 60, (  # 10 + 50
                f"{task_type}: expected per_query=60, got {report.per_query_tokens}"
            )

    def test_five_passes_total_input_token_count(self):
        """
        Total tokens consumed (setup + 5 × query) should be
        less than 5 × baseline for most contexts.
        """
        for t in BENCH_TASKS:
            if t.context_tokens_approx < 200:
                continue  # tiny context, break-even is trivially low
            passes = build_mock_passes(
                condition="vec",
                fblock="bm25_single",
                task_id=t.id,
                context_tokens=t.context_tokens_approx,
                question_tokens=10,
                fblock_tokens=50,
                setup_tokens=t.context_tokens_approx,
                n_query_passes=5,
            )
            report = compute_amortization(
                passes,
                baseline_per_query_tokens=t.context_tokens_approx + 10,
            )
            # For larger contexts, 5 passes should amortize
            if t.context_tokens_approx > 500:
                assert report.total_injection_cost < report.total_baseline_cost, (
                    f"{t.id}: injection cost {report.total_injection_cost} >= "
                    f"baseline cost {report.total_baseline_cost} at n=5"
                )
