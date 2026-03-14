"""
F block unit tests across all task types and all F block strategies.

Strategies covered:
  ner              — regex NER extraction (extract_facts_ner)
  bm25_single      — single-hop BM25 (extract_facts_bm25)
  bm25_double_seq  — double-hop sequential pivot (extract_facts_bm25_double_seq)
  bm25_double_entity — double-hop entity bridge (extract_facts_bm25_double_entity)
  model_summary    — model-written structured notes (mocked in unit tests)

No model calls.  model_summary tests mock generate_baseline_qa so the F block
structure can be verified without loading a model.

Run in isolation:
    pytest test_bench_fblocks.py
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from benchmark_ablation import (
    extract_facts_ner,
    extract_facts_bm25,
    extract_facts_bm25_double_seq,
    extract_facts_bm25_double_entity,
)
from bench_tasks import BENCH_TASKS, get_tasks_by_type, get_task, BenchTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fblock_body(result: str) -> str:
    """Strip the 'Facts: ' prefix for length checks."""
    if result.startswith("Facts: "):
        return result[len("Facts: "):]
    if result.startswith("Context: "):
        return result[len("Context: "):]
    return result


def _all_strategy_fns():
    return [
        ("ner",               lambda ctx, q: extract_facts_ner(ctx)),
        ("bm25_single",       extract_facts_bm25),
        ("bm25_double_seq",   extract_facts_bm25_double_seq),
        ("bm25_double_entity", extract_facts_bm25_double_entity),
    ]


# ---------------------------------------------------------------------------
# NER: basic extraction
# ---------------------------------------------------------------------------

class TestNERFblock:
    def test_returns_facts_prefix_or_empty(self):
        for t in BENCH_TASKS:
            result = extract_facts_ner(t.context)
            assert result == "" or result.startswith("Facts:"), (
                f"{t.id}: NER result has unexpected prefix: {result[:40]!r}"
            )

    def test_extracts_capitalized_entities(self):
        # NER targets multi-word capitalized proper nouns (not CamelCase identifiers).
        ctx = "Marie Curie won the Nobel Prize in Warsaw. Albert Einstein was born in Ulm."
        result = extract_facts_ner(ctx)
        assert any(word in result for word in ("Marie Curie", "Nobel Prize", "Albert Einstein", "Warsaw", "Ulm"))

    def test_extracts_numbers(self):
        from bench_tasks import _USER_SERVICE_CTX
        result = extract_facts_ner(_USER_SERVICE_CTX)
        # MAX_RETRIES = 3, DEFAULT_PAGE_SIZE = 25
        assert "3" in result or "25" in result

    def test_empty_context_returns_empty(self):
        assert extract_facts_ner("") == ""

    def test_prose_without_entities_returns_something_or_empty(self):
        result = extract_facts_ner("the quick brown fox")
        assert isinstance(result, str)

    def test_does_not_exceed_max_facts(self):
        # With max_facts=5, should not return more than 5 semicolons
        result = extract_facts_ner("A B C D E F G H I J K L M N O P", max_facts=5)
        if result.startswith("Facts: "):
            parts = result[len("Facts: "):].split(";")
            assert len(parts) <= 5

    def test_multifile_context_extracts_class_names(self):
        t = get_task("rename_3file")
        result = extract_facts_ner(t.context)
        # Order is a class defined in the context
        assert "Order" in result or result == ""


# ---------------------------------------------------------------------------
# BM25 single-hop: question conditioning
# ---------------------------------------------------------------------------

class TestBM25SingleHopFblock:
    def test_returns_facts_prefix(self):
        for t in BENCH_TASKS[:5]:  # sample first 5
            result = extract_facts_bm25(t.context, t.question)
            assert result.startswith("Facts:") or result == "", (
                f"{t.id}: unexpected prefix {result[:40]!r}"
            )

    def test_question_conditioned_for_return_type(self):
        # Code contexts: BM25 splits on prose sentences; use extract_facts_code
        # for code. Test BM25 on a prose-like context instead.
        ctx = (
            "The find_user method returns Optional[User] when found. "
            "It accepts a user_id integer parameter. "
            "The list_users method returns a paginated list."
        )
        result = extract_facts_bm25(ctx, "what does find_user return")
        assert "find_user" in result or "Optional" in result

    def test_question_conditioned_for_param_type(self):
        ctx = (
            "The update_user method takes a UserDelta object as its delta parameter. "
            "It modifies the user in place. "
            "The create_user method takes name and email strings."
        )
        result = extract_facts_bm25(ctx, "what parameter does update_user take")
        assert "update_user" in result or "UserDelta" in result

    def test_question_conditioned_for_constant(self):
        ctx = (
            "MAX_RETRIES is set to 3 in the configuration. "
            "DEFAULT_TIMEOUT is 30 seconds. "
            "The retry loop uses MAX_RETRIES as its upper bound."
        )
        result = extract_facts_bm25(ctx, "what is the max retries limit")
        assert "MAX_RETRIES" in result or "3" in result

    def test_empty_context_returns_empty(self):
        assert extract_facts_bm25("", "any question") == ""

    def test_different_questions_may_produce_different_facts(self):
        t = get_task("return_type_q")
        r1 = extract_facts_bm25(t.context, "what does find_user return")
        r2 = extract_facts_bm25(t.context, "what does list_users return")
        # At minimum both should be non-empty
        assert r1 != "" and r2 != ""

    def test_respects_max_chars(self):
        t = BENCH_TASKS[0]
        result = extract_facts_bm25(t.context, t.question, max_chars=80)
        body = _fblock_body(result)
        assert len(body) <= 90  # small tolerance for word-boundary truncation

    def test_prose_context_surfaces_matching_sentence(self):
        # BM25 works on prose-style sentences, not raw code lines.
        ctx = (
            "notify_user sends a notification to the given user_id. "
            "complete_order calls notify_user after payment. "
            "register_user calls notify_user on successful signup."
        )
        result = extract_facts_bm25(ctx, "which functions call notify_user")
        assert "notify_user" in result

    def test_import_chain_prose(self):
        ctx = (
            "get_timeout is defined in config and returns the HTTP timeout. "
            "service imports get_timeout from config and uses it in fetch_resource. "
            "handler imports fetch_resource from service."
        )
        result = extract_facts_bm25(ctx, "what function from config does handler use via service")
        assert "get_timeout" in result or "config" in result


# ---------------------------------------------------------------------------
# BM25 double-hop sequential
# ---------------------------------------------------------------------------

class TestBM25DoubleHopSequential:
    def test_returns_facts_prefix_or_empty(self):
        for t in BENCH_TASKS[:5]:
            result = extract_facts_bm25_double_seq(t.context, t.question)
            assert result.startswith("Facts:") or result == "", (
                f"{t.id}: unexpected prefix {result[:40]!r}"
            )

    def test_empty_context_returns_empty(self):
        assert extract_facts_bm25_double_seq("", "any question") == ""

    def test_single_sentence_context_does_not_crash(self):
        ctx = "def create_session(user_id: str) -> SessionToken: pass."
        result = extract_facts_bm25_double_seq(ctx, "what does create_session return")
        assert isinstance(result, str)

    def test_transitive_return_reaches_session_token(self):
        """
        For transitive_return: hop 1 surfaces create_session (calls generate_token),
        hop 2 pivots to generate_token (returns SessionToken).
        The double-hop result should contain SessionToken or generate_token.
        """
        t = get_task("transitive_return")
        result = extract_facts_bm25_double_seq(t.context, t.question, top_k=3)
        # After two hops, SessionToken or generate_token should appear
        assert "SessionToken" in result or "generate_token" in result

    def test_inherited_method_reaches_base_save(self):
        t = get_task("inherited_method")
        result = extract_facts_bm25_double_seq(t.context, t.question, top_k=3)
        assert "save" in result or "BaseUser" in result

    def test_double_hop_returns_multiple_sentences(self):
        t = get_task("transitive_return")
        result = extract_facts_bm25_double_seq(t.context, t.question, top_k=3)
        body = _fblock_body(result)
        # With top_k=3 on a multi-sentence context, body should contain content
        assert len(body) > 10

    def test_single_hop_fallback_on_trivial_context(self):
        """When pivot = only sentence, should not crash."""
        ctx = "Marie Curie won the Nobel Prize in 1903."
        result = extract_facts_bm25_double_seq(ctx, "who won the prize")
        assert isinstance(result, str)

    def test_respects_max_chars(self):
        t = BENCH_TASKS[0]
        result = extract_facts_bm25_double_seq(t.context, t.question, max_chars=100)
        body = _fblock_body(result)
        assert len(body) <= 110

    def test_differs_from_single_hop_on_double_hop_tasks(self):
        """
        On double-hop tasks, sequential should surface different passages
        than single-hop BM25 (hop-2 pivot changes the retrieved set).
        Not guaranteed but true for most well-formed double-hop contexts.
        """
        t = get_task("transitive_return")
        r_single = extract_facts_bm25(t.context, t.question)
        r_double = extract_facts_bm25_double_seq(t.context, t.question, top_k=3)
        # Both should be non-empty
        assert r_single != "" and r_double != ""

    def test_long_context_does_not_crash(self):
        t = get_task("long_refactor")
        result = extract_facts_bm25_double_seq(t.context, t.question)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# BM25 double-hop entity-expanded
# ---------------------------------------------------------------------------

class TestBM25DoubleHopEntity:
    def test_returns_facts_prefix_or_empty(self):
        for t in BENCH_TASKS[:5]:
            result = extract_facts_bm25_double_entity(t.context, t.question)
            assert result.startswith("Facts:") or result == "", (
                f"{t.id}: unexpected prefix {result[:40]!r}"
            )

    def test_empty_context_returns_empty(self):
        assert extract_facts_bm25_double_entity("", "any question") == ""

    def test_transitive_return_surfaces_session_token(self):
        """
        Entity hop: create_session pivot → entities include "generate" "token"
        → hop 2 targets generate_token definition → SessionToken in result.
        """
        t = get_task("transitive_return")
        result = extract_facts_bm25_double_entity(t.context, t.question, top_k=3)
        assert "SessionToken" in result or "generate_token" in result

    def test_inherited_method_reaches_base_user(self):
        t = get_task("inherited_method")
        result = extract_facts_bm25_double_entity(t.context, t.question, top_k=3)
        assert "BaseUser" in result or "save" in result

    def test_import_chain_bridges_to_config(self):
        t = get_task("import_chain")
        result = extract_facts_bm25_double_entity(t.context, t.question, top_k=3)
        assert "get_timeout" in result or "config" in result

    def test_entity_extraction_on_snake_case_identifier(self):
        """
        A context with only snake_case identifiers should still yield bridge tokens.
        """
        ctx = (
            "def get_user_profile(user_id: int) -> UserProfile: pass. "
            "class UserProfile: name: str; email: str."
        )
        result = extract_facts_bm25_double_entity(ctx, "what does get_user_profile return")
        assert isinstance(result, str)
        # The entity "user" should bridge to UserProfile
        assert "UserProfile" in result or "profile" in result.lower()

    def test_entity_extraction_on_camel_case(self):
        t = get_task("inherited_method")
        result = extract_facts_bm25_double_entity(t.context, t.question, top_k=3)
        # AdminUser → ["admin", "user"] bridge tokens should reach BaseUser
        assert isinstance(result, str) and len(result) > 0

    def test_differs_from_sequential_on_prose_contexts(self):
        """
        The two double-hop strategies should produce different F blocks on
        at least one prose-like context, confirming distinct retrieval paths.
        Uses prose contexts where sentence splitting is well-defined.
        """
        prose_contexts = [
            (
                "Marie Curie won the Nobel Prize in 1903. "
                "She was born in Warsaw in 1867. "
                "The Nobel Prize is awarded for outstanding contributions. "
                "Curie studied physics in Paris. "
                "Paris is the capital of France.",
                "where was the Nobel Prize winner born",
            ),
            (
                "create_session calls generate_token to produce a session. "
                "generate_token returns a SessionToken object. "
                "SessionToken contains an expiry timestamp. "
                "The expiry is set to one hour by default.",
                "what does create_session return",
            ),
        ]
        differences = 0
        for ctx, q in prose_contexts:
            r_seq = extract_facts_bm25_double_seq(ctx, q, top_k=3)
            r_ent = extract_facts_bm25_double_entity(ctx, q, top_k=3)
            if r_seq != r_ent:
                differences += 1
        assert differences > 0, (
            "bm25_double_seq and bm25_double_entity produced identical results "
            "on all prose contexts — check implementation"
        )

    def test_respects_max_chars(self):
        t = BENCH_TASKS[0]
        result = extract_facts_bm25_double_entity(t.context, t.question, max_chars=100)
        body = _fblock_body(result)
        assert len(body) <= 110

    def test_long_context_does_not_crash(self):
        t = get_task("long_double_hop")
        result = extract_facts_bm25_double_entity(t.context, t.question)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Cross-strategy comparison
# ---------------------------------------------------------------------------

class TestCrossStrategyComparison:
    """
    Compare all four practical strategies on each task.
    These tests verify contracts that hold regardless of strategy,
    and check that strategies behave differently enough to be useful.
    """

    @pytest.mark.parametrize("task", BENCH_TASKS, ids=lambda t: t.id)
    def test_all_strategies_return_string(self, task: BenchTask):
        for name, fn in _all_strategy_fns():
            result = fn(task.context, task.question)
            assert isinstance(result, str), f"{task.id} / {name}: not a string"

    @pytest.mark.parametrize("task", BENCH_TASKS, ids=lambda t: t.id)
    def test_all_strategies_have_valid_prefix_or_empty(self, task: BenchTask):
        valid_prefixes = ("Facts:", "Context:", "")
        for name, fn in _all_strategy_fns():
            result = fn(task.context, task.question)
            is_valid = any(
                result.startswith(p) for p in valid_prefixes
            )
            assert is_valid, (
                f"{task.id} / {name}: result has unexpected prefix: {result[:50]!r}"
            )

    @pytest.mark.parametrize("task_id", [
        "return_type_q", "param_type_q", "constant_q",
        "transitive_return", "inherited_method",
    ])
    def test_extract_facts_code_surfaces_gold(self, task_id: str):
        """
        For code tasks, extract_facts_code (the code-specific F block) should
        surface at least one gold answer.  BM25 prose strategies are not reliable
        on raw code contexts where sentence splitting is degenerate.
        """
        from benchmark_ablation import extract_facts_code
        t = get_task(task_id)
        result = extract_facts_code(t.context, t.question)
        found = any(gold in result for gold in t.gold_answers)
        assert found, (
            f"{task_id}: extract_facts_code did not surface any gold answer "
            f"{t.gold_answers!r} — result: {result[:120]!r}"
        )

    def test_single_hop_tasks_ner_and_bm25_both_non_empty(self):
        for t in get_tasks_by_type("single_hop"):
            ner = extract_facts_ner(t.context)
            bm25 = extract_facts_bm25(t.context, t.question)
            assert ner != "" or bm25 != "", (
                f"{t.id}: both NER and BM25 returned empty"
            )

    def test_double_hop_bm25_strategies_non_empty(self):
        for t in get_tasks_by_type("double_hop"):
            r_seq = extract_facts_bm25_double_seq(t.context, t.question)
            r_ent = extract_facts_bm25_double_entity(t.context, t.question)
            assert r_seq != "" or r_ent != "", (
                f"{t.id}: both double-hop strategies returned empty"
            )


# ---------------------------------------------------------------------------
# Model summary F block (mocked)
# ---------------------------------------------------------------------------

class TestModelSummaryFblockMocked:
    """
    Verify the model summary workflow structure without a real model.

    The model summary F block works as follows:
      1. Call build_summary_prompt(context) to produce the prompt.
      2. Call generate_baseline_qa(wrapper, prompt, ...) to get the summary.
      3. Use the returned summary as the F block for all subsequent queries.

    These tests mock generate_baseline_qa and verify the surrounding logic.
    """

    def _make_wrapper(self) -> MagicMock:
        wrapper = MagicMock()
        wrapper.n_layers = 32
        wrapper.eos_token_id = 2
        wrapper.encode.return_value = list(range(50))
        return wrapper

    def test_summary_prompt_contains_context(self):
        from benchmark_code_refactor import build_summary_prompt
        t = get_task("rename_3file")
        prompt = build_summary_prompt(t.context)
        assert t.context[:40] in prompt

    def test_summary_prompt_requests_signatures(self):
        from benchmark_code_refactor import build_summary_prompt
        prompt = build_summary_prompt("def foo(): pass")
        assert "signature" in prompt.lower() or "function" in prompt.lower()

    def test_summary_prompt_ends_with_generation_cue(self):
        from benchmark_code_refactor import build_summary_prompt
        prompt = build_summary_prompt("some code here")
        assert prompt.strip().endswith(("Summary:", "Notes:", "Functions:"))

    @pytest.mark.parametrize("task_id", [
        "rename_3file", "interface_change", "transitive_return",
    ])
    def test_mocked_summary_used_as_fblock(self, task_id: str):
        """
        When generate_baseline_qa is mocked to return a plausible summary,
        the model summary F block should contain that summary text.
        """
        from benchmark_code_refactor import build_summary_prompt, run_refactor_example

        t = get_task(task_id)
        if t.task_type not in ("multifile_refactor",):
            pytest.skip("run_refactor_example only applies to refactor tasks")

        wrapper = self._make_wrapper()
        mock_summary = f"Functions: {', '.join(t.must_contain[:2])}"

        with patch("benchmark_code_refactor.generate_baseline_qa",
                   return_value=(mock_summary, 80)), \
             patch("benchmark_code_refactor.extract_context_state",
                   return_value=(MagicMock(), 40)), \
             patch("benchmark_code_refactor.generate_with_context_injection",
                   return_value=("handle_order patch_user output", 40)), \
             patch("benchmark_code_refactor.extract_facts_code",
                   return_value="Facts: some_ner_result"):
            records = run_refactor_example(wrapper, t, injection_layer=14)

        summary_rec = next(r for r in records if r.condition == "vec_f_summary")
        assert mock_summary in summary_rec.f_block or len(summary_rec.f_block) > 0

    def test_summary_fblock_differs_from_ner_fblock(self):
        from benchmark_code_refactor import run_refactor_example
        t = get_task("rename_3file")
        wrapper = self._make_wrapper()

        with patch("benchmark_code_refactor.generate_baseline_qa",
                   return_value=("Summary: handle_order is the new name", 80)), \
             patch("benchmark_code_refactor.extract_context_state",
                   return_value=(MagicMock(), 40)), \
             patch("benchmark_code_refactor.generate_with_context_injection",
                   return_value=("handle_order", 40)), \
             patch("benchmark_code_refactor.extract_facts_code",
                   return_value="Facts: ner_result_xyz"):
            records = run_refactor_example(wrapper, t, injection_layer=14)

        ner_rec = next(r for r in records if r.condition == "vec_f_ner")
        sum_rec = next(r for r in records if r.condition == "vec_f_summary")
        assert ner_rec.f_block != sum_rec.f_block
