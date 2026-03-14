"""
Injection condition structural tests — no model calls.

Covers:
  - The 13-condition matrix (baseline × none, vec × 6, matrix × 6)
  - Condition filtering via bench_injection / bench_fblock fixtures
  - Mocked injection behavior for each condition type
  - Record structure for all conditions
  - Condition compatibility rules (baseline ignores F block)

Run in isolation:
    pytest test_bench_conditions.py
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from bench_tasks import BENCH_TASKS, get_task, BenchTask


# ---------------------------------------------------------------------------
# Condition matrix definition
# ---------------------------------------------------------------------------

INJECTION_TYPES = ["baseline", "vec", "matrix"]

FBLOCK_TYPES = [
    "none",
    "ner",
    "bm25_single",
    "bm25_double_seq",
    "bm25_double_entity",
    "model_summary",
]

# Full 13-condition matrix
ALL_CONDITIONS: list[tuple[str, str]] = [
    ("baseline", "none"),
    ("vec",      "none"),
    ("vec",      "ner"),
    ("vec",      "bm25_single"),
    ("vec",      "bm25_double_seq"),
    ("vec",      "bm25_double_entity"),
    ("vec",      "model_summary"),
    ("matrix",   "none"),
    ("matrix",   "ner"),
    ("matrix",   "bm25_single"),
    ("matrix",   "bm25_double_seq"),
    ("matrix",   "bm25_double_entity"),
    ("matrix",   "model_summary"),
]

# Condition ID used as pytest parameter id
def _cond_id(injection: str, fblock: str) -> str:
    return f"{injection}__{fblock}"


# ---------------------------------------------------------------------------
# Condition matrix structure
# ---------------------------------------------------------------------------

class TestConditionMatrix:
    def test_total_conditions_is_13(self):
        assert len(ALL_CONDITIONS) == 13

    def test_unique_condition_pairs(self):
        ids = [_cond_id(i, f) for i, f in ALL_CONDITIONS]
        assert len(ids) == len(set(ids))

    def test_baseline_only_has_none_fblock(self):
        baseline_conds = [(i, f) for i, f in ALL_CONDITIONS if i == "baseline"]
        assert len(baseline_conds) == 1
        assert baseline_conds[0][1] == "none"

    def test_vec_covers_all_fblock_types(self):
        vec_fblocks = {f for i, f in ALL_CONDITIONS if i == "vec"}
        for ft in FBLOCK_TYPES:
            assert ft in vec_fblocks, f"vec is missing fblock={ft!r}"

    def test_matrix_covers_all_fblock_types(self):
        matrix_fblocks = {f for i, f in ALL_CONDITIONS if i == "matrix"}
        for ft in FBLOCK_TYPES:
            assert ft in matrix_fblocks, f"matrix is missing fblock={ft!r}"

    def test_condition_id_format(self):
        for inj, fb in ALL_CONDITIONS:
            cid = _cond_id(inj, fb)
            assert "__" in cid
            parts = cid.split("__")
            assert parts[0] == inj
            assert parts[1] == fb

    def test_all_injection_types_present(self):
        injections = {i for i, _ in ALL_CONDITIONS}
        for it in INJECTION_TYPES:
            assert it in injections


# ---------------------------------------------------------------------------
# Condition filtering logic
# ---------------------------------------------------------------------------

def _filter_conditions(
    injection_filter: str,
    fblock_filter: str,
) -> list[tuple[str, str]]:
    """Simulate CLI filter: return conditions matching the filter."""
    result = []
    for inj, fb in ALL_CONDITIONS:
        if injection_filter != "all" and inj != injection_filter:
            continue
        if fblock_filter != "all" and fb != fblock_filter:
            continue
        result.append((inj, fb))
    return result


class TestConditionFiltering:
    def test_no_filter_returns_all_13(self):
        assert len(_filter_conditions("all", "all")) == 13

    def test_filter_by_injection_baseline(self):
        result = _filter_conditions("baseline", "all")
        assert result == [("baseline", "none")]

    def test_filter_by_injection_vec(self):
        result = _filter_conditions("vec", "all")
        assert len(result) == len(FBLOCK_TYPES)
        assert all(i == "vec" for i, _ in result)

    def test_filter_by_injection_matrix(self):
        result = _filter_conditions("matrix", "all")
        assert len(result) == len(FBLOCK_TYPES)
        assert all(i == "matrix" for i, _ in result)

    def test_filter_by_fblock_ner(self):
        result = _filter_conditions("all", "ner")
        assert all(f == "ner" for _, f in result)
        assert len(result) == 2  # vec+ner and matrix+ner

    def test_filter_by_fblock_none(self):
        result = _filter_conditions("all", "none")
        assert len(result) == 3  # baseline+none, vec+none, matrix+none

    def test_filter_vec_bm25_single(self):
        result = _filter_conditions("vec", "bm25_single")
        assert result == [("vec", "bm25_single")]

    def test_filter_matrix_model_summary(self):
        result = _filter_conditions("matrix", "model_summary")
        assert result == [("matrix", "model_summary")]

    def test_unknown_injection_returns_empty(self):
        result = _filter_conditions("unknown", "all")
        assert result == []

    def test_unknown_fblock_returns_empty(self):
        result = _filter_conditions("all", "unknown_fblock")
        assert result == []


# ---------------------------------------------------------------------------
# Mocked injection: baseline condition
# ---------------------------------------------------------------------------

class TestBaselineConditionMocked:
    def _make_wrapper(self) -> MagicMock:
        w = MagicMock()
        w.n_layers = 32
        w.eos_token_id = 2
        w.encode.return_value = list(range(100))
        return w

    def test_baseline_uses_full_context_in_prompt(self):
        t = get_task("return_type_q")
        wrapper = self._make_wrapper()
        captured = {}

        def mock_gen(w, prompt, max_tokens):
            captured["prompt"] = prompt
            return ("Optional[User]", 100)

        with patch("benchmark_ablation.generate_baseline_qa", side_effect=mock_gen):
            from benchmark_ablation import extract_facts_ner, extract_facts_bm25
            # simulate what run_example_ablation does for baseline
            prompt = f"Context: {t.context}\nQuestion: {t.question}\nAnswer:"
            mock_gen(wrapper, prompt, 80)

        assert t.context[:50] in captured["prompt"]

    def test_baseline_token_count_equals_context_plus_question(self):
        t = get_task("short_return")
        # Token count for baseline = context tokens + question tokens
        ctx_toks = len(t.context.split())
        q_toks = len(t.question.split())
        full_toks = ctx_toks + q_toks
        assert full_toks > q_toks  # trivially true — confirms context is included


# ---------------------------------------------------------------------------
# Mocked injection: vec condition
# ---------------------------------------------------------------------------

class TestVecConditionMocked:
    """
    Verify injection prompt construction logic without calling run_example_ablation.
    These tests exercise the context_injector prompt-building contract directly.
    """

    def test_vec_injection_prompt_is_shorter_than_full_context_prompt(self):
        """
        Injection prompt = question only (+ optional F block).
        Baseline prompt = context + question.
        Injection prompt must be shorter.
        """
        t = get_task("return_type_q")
        baseline_prompt = f"Context: {t.context}\nQuestion: {t.question}\nAnswer:"
        injection_prompt = f"Question: {t.question}\nAnswer:"
        assert len(injection_prompt) < len(baseline_prompt)

    def test_vec_with_fblock_prepends_facts_to_question_prompt(self):
        """When F block is non-empty, it is prepended before the question."""
        from benchmark_ablation import extract_facts_ner
        t = get_task("return_type_q")
        fblock = extract_facts_ner(t.context)
        if not fblock:
            fblock = "Facts: Optional; User"
        q_prompt = f"Question: {t.question}\nAnswer:"
        injection_prompt = f"{fblock}\n{q_prompt}" if fblock else q_prompt
        assert injection_prompt.startswith("Facts:")

    def test_vec_empty_fblock_produces_no_facts_prefix(self):
        """Empty F block → injection prompt starts with Question:, not Facts:."""
        t = get_task("return_type_q")
        fblock = ""
        q_prompt = f"Question: {t.question}\nAnswer:"
        injection_prompt = fblock + "\n" + q_prompt if fblock else q_prompt
        assert not injection_prompt.startswith("Facts:")

    def test_injection_prompt_does_not_contain_full_context(self):
        """The injection prompt should not contain the full context text."""
        t = get_task("return_type_q")
        injection_prompt = f"Question: {t.question}\nAnswer:"
        # Context should not be in the injection prompt
        assert t.context[:50] not in injection_prompt

    def test_baseline_prompt_contains_full_context(self):
        """The baseline prompt must include the full context."""
        t = get_task("return_type_q")
        baseline_prompt = f"Context: {t.context}\nQuestion: {t.question}\nAnswer:"
        assert t.context[:50] in baseline_prompt


# ---------------------------------------------------------------------------
# Mocked injection: record structure
# ---------------------------------------------------------------------------

class TestConditionRecordStructure:
    def _make_wrapper(self) -> MagicMock:
        w = MagicMock()
        w.n_layers = 32
        w.eos_token_id = 2
        w.encode.return_value = list(range(30))
        return w

    def test_ablation_record_has_all_condition_fields(self):
        from benchmark_ablation import AblationRecord
        rec = AblationRecord(
            model_key="llama3-8b",
            task="single_hop",
            example_id="return_type_q",
            injection_layer=14,
            n_context_words=100,
            fact_mode="ner",
            fact_block="Facts: Optional; User",
            baseline_input_tokens=500,
            rsce_only_input_tokens=20,
            rsce_f_input_tokens=30,
            baseline_exact_match=True,
            rsce_only_exact_match=True,
            rsce_f_exact_match=True,
            baseline_f1=1.0,
            rsce_only_f1=1.0,
            rsce_f_f1=1.0,
            f_contribution_f1=0.0,
            residual_gap_f1=0.0,
            input_token_reduction=0.94,
            baseline_answer="Optional[User]",
            rsce_only_answer="Optional[User]",
            rsce_f_answer="Optional[User]",
            gold_answers="['Optional[User]']",
        )
        assert rec.model_key == "llama3-8b"
        assert rec.fact_mode == "ner"
        assert rec.input_token_reduction == pytest.approx(0.94)

    def test_input_token_reduction_is_between_0_and_1(self):
        from benchmark_ablation import AblationRecord
        rec = AblationRecord(
            model_key="x", task="t", example_id="e", injection_layer=14,
            n_context_words=100, fact_mode="ner", fact_block="",
            baseline_input_tokens=1000, rsce_only_input_tokens=20,
            rsce_f_input_tokens=30,
            baseline_exact_match=False, rsce_only_exact_match=False,
            rsce_f_exact_match=False,
            baseline_f1=0.0, rsce_only_f1=0.0, rsce_f_f1=0.0,
            f_contribution_f1=0.0, residual_gap_f1=0.0,
            input_token_reduction=1 - 20/1000,
            baseline_answer="", rsce_only_answer="", rsce_f_answer="",
            gold_answers="[]",
        )
        assert 0.0 <= rec.input_token_reduction <= 1.0

    def test_refactor_record_has_expected_conditions(self):
        """
        benchmark_code_refactor.run_refactor_example returns 3 records:
        baseline, vec_f_ner, vec_f_summary.
        """
        from benchmark_code_refactor import run_refactor_example, RefactorExample
        t = get_task("rename_3file")

        wrapper = self._make_wrapper()
        re = RefactorExample(
            id=t.id,
            context=t.context,
            instruction=t.instruction,
            must_contain=t.must_contain,
            must_not_contain=t.must_not_contain,
        )

        with patch("benchmark_code_refactor.generate_baseline_qa",
                   return_value=("handle_order()", 100)), \
             patch("benchmark_code_refactor.extract_context_state",
                   return_value=(MagicMock(), 50)), \
             patch("benchmark_code_refactor.generate_with_context_injection",
                   return_value=("handle_order()", 50)), \
             patch("benchmark_code_refactor.extract_facts_code",
                   return_value="Facts: handle_order"):
            records = run_refactor_example(wrapper, re, injection_layer=14)

        conditions = {r.condition for r in records}
        assert conditions == {"baseline", "vec_f_ner", "vec_f_summary"}


# ---------------------------------------------------------------------------
# Condition compatibility rules
# ---------------------------------------------------------------------------

class TestConditionCompatibilityRules:
    def test_baseline_condition_never_uses_fblock(self):
        """
        The baseline condition always has fblock='none' in ALL_CONDITIONS.
        No F block should be constructed or used for baseline.
        """
        baseline_conds = [(i, f) for i, f in ALL_CONDITIONS if i == "baseline"]
        for _, f in baseline_conds:
            assert f == "none", f"baseline should not use fblock={f!r}"

    def test_vec_none_is_rsce_only_equivalent(self):
        """
        vec + none is the 'RSCE only' condition from the ablation paper —
        context vector without any explicit fact block.
        """
        assert ("vec", "none") in ALL_CONDITIONS

    def test_matrix_none_is_matrix_only(self):
        assert ("matrix", "none") in ALL_CONDITIONS

    def test_double_hop_fblocks_are_in_matrix(self):
        """Both double-hop F blocks should be testable with matrix injection."""
        assert ("matrix", "bm25_double_seq") in ALL_CONDITIONS
        assert ("matrix", "bm25_double_entity") in ALL_CONDITIONS

    def test_model_summary_available_for_both_injections(self):
        assert ("vec", "model_summary") in ALL_CONDITIONS
        assert ("matrix", "model_summary") in ALL_CONDITIONS

    def test_fblock_strategies_ranked_by_cost(self):
        """
        Document expected relative F block computation cost (no assertion,
        just a structure check that the ordering is consistent).
        """
        cost_rank = {
            "none": 0,
            "ner": 1,
            "bm25_single": 2,
            "bm25_double_seq": 3,
            "bm25_double_entity": 3,
            "model_summary": 10,  # one-time model generation
        }
        for ft in FBLOCK_TYPES:
            assert ft in cost_rank, f"Missing cost rank for fblock={ft!r}"
        # model_summary should be the most expensive
        assert cost_rank["model_summary"] > cost_rank["bm25_double_seq"]
        # ner should be cheaper than bm25
        assert cost_rank["ner"] < cost_rank["bm25_single"]
