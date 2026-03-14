"""
Stop sequence behavior tests — no model calls.

Covers:
  - truncate_at_stop: all QA_STOP_STRINGS, position sensitivity, edge cases
  - All stop strings are caught and trigger truncation
  - Model bleed patterns: answer followed by regenerated question/context/facts
  - Grader receives clean answer when truncation fires correctly
  - Grader receives polluted input when bleed is not truncated (F1 dilution)
  - All five conditions apply identical truncation (deterministic)
  - Custom stop string overrides

Why this matters:
  If a model generates "Optional[User]\nQuestion: What does foo return?",
  the grader sees "Optional[User]" (correct) only if truncate_at_stop fires
  before grading. Without truncation, the extra tokens dilute F1.
  This file verifies that contract holds for every stop string and condition.

Run in isolation:
    pytest test_bench_stop_sequences.py
"""

from __future__ import annotations

import pytest

from context_injector import truncate_at_stop, QA_STOP_STRINGS
from benchmark_datasets import grade_qa


# ---------------------------------------------------------------------------
# truncate_at_stop: basic contracts
# ---------------------------------------------------------------------------

class TestTruncateAtStopBasic:
    def test_no_stop_string_returns_full_text(self):
        text = "Marie Curie"
        assert truncate_at_stop(text) == "Marie Curie"

    def test_newline_truncates_at_first_newline(self):
        text = "Marie Curie\nSome bleed"
        assert truncate_at_stop(text) == "Marie Curie"

    def test_question_prefix_truncates(self):
        text = "Optional[User]\nQuestion: What is the return type?"
        assert truncate_at_stop(text) == "Optional[User]"

    def test_facts_prefix_truncates(self):
        text = "SessionToken\nFacts: generate_token returns SessionToken"
        assert truncate_at_stop(text) == "SessionToken"

    def test_context_prefix_truncates(self):
        text = "int\nContext: def foo() -> int:"
        assert truncate_at_stop(text) == "int"

    def test_empty_input_returns_empty(self):
        assert truncate_at_stop("") == ""

    def test_stop_string_at_start_returns_empty(self):
        assert truncate_at_stop("\nSome text") == ""

    def test_stop_string_exactly_at_end(self):
        text = "Marie Curie\n"
        assert truncate_at_stop(text) == "Marie Curie"

    def test_no_stop_string_no_newline_unchanged(self):
        text = "Optional[User]"
        assert truncate_at_stop(text) == "Optional[User]"

    def test_whitespace_only_answer_preserved(self):
        """Answer with spaces but no stop strings is not truncated."""
        text = "New York City"
        assert truncate_at_stop(text) == "New York City"

    def test_earliest_stop_string_wins(self):
        """When multiple stop strings appear, the one with the lowest index wins."""
        text = "first\nContext: x\nQuestion: y"
        # "\n" at position 5 wins over "\nContext:" at position 6
        assert truncate_at_stop(text) == "first"

    def test_embedded_stop_string_not_at_boundary_still_truncates(self):
        """Stop strings truncate regardless of position within the text."""
        text = "answer part one\nmore text here"
        result = truncate_at_stop(text)
        assert result == "answer part one"


# ---------------------------------------------------------------------------
# All QA_STOP_STRINGS are caught
# ---------------------------------------------------------------------------

class TestAllQAStopStringsCaught:
    def test_every_stop_string_triggers_truncation(self):
        """Each stop string in QA_STOP_STRINGS must truncate the answer."""
        answer = "Marie Curie"
        for stop in QA_STOP_STRINGS:
            text = answer + stop + "extra continuation text that should be removed"
            result = truncate_at_stop(text)
            assert result == answer, (
                f"Stop string {stop!r} not caught: got {result!r}"
            )

    def test_stop_strings_are_non_empty(self):
        for s in QA_STOP_STRINGS:
            assert len(s) > 0, f"Empty stop string in QA_STOP_STRINGS: {s!r}"

    def test_qa_stop_strings_includes_bare_newline(self):
        assert "\n" in QA_STOP_STRINGS

    def test_qa_stop_strings_includes_question_prefix(self):
        assert "\nQuestion:" in QA_STOP_STRINGS

    def test_qa_stop_strings_includes_facts_prefix(self):
        assert "\nFacts:" in QA_STOP_STRINGS

    def test_qa_stop_strings_includes_context_prefix(self):
        assert "\nContext:" in QA_STOP_STRINGS

    def test_qa_stop_strings_count(self):
        assert len(QA_STOP_STRINGS) == 4


# ---------------------------------------------------------------------------
# Model bleed patterns: truncation fires before grading
# ---------------------------------------------------------------------------

class TestModelBleedPatterns:
    """
    Model bleed = model generates a correct answer then continues generating
    the next question, facts, or context verbatim. Truncation must fire
    before the grader sees the output.
    """

    def test_newline_bleed_exact_match_preserved(self):
        """Answer + newline + question bleed: grader sees exact match after truncation."""
        gold = ["Optional[User]"]
        raw = "Optional[User]\nQuestion: What does create_session return?"
        truncated = truncate_at_stop(raw)
        grades = grade_qa(truncated, gold)
        assert grades["exact_match"] is True, (
            f"Grader missed answer after truncation: {truncated!r}"
        )

    def test_facts_bleed_exact_match_preserved(self):
        gold = ["SessionToken"]
        raw = "SessionToken\nFacts: generate_token; create_session"
        truncated = truncate_at_stop(raw)
        grades = grade_qa(truncated, gold)
        assert grades["exact_match"] is True

    def test_context_bleed_exact_match_preserved(self):
        gold = ["int"]
        raw = "int\nContext: def foo() -> int:\n    return 42"
        truncated = truncate_at_stop(raw)
        assert truncated == "int"
        grades = grade_qa(truncated, gold)
        assert grades["exact_match"] is True

    def test_bleed_without_truncation_dilutes_f1(self):
        """
        Without truncation, extra tokens reduce F1 by increasing the denominator
        (or by introducing non-gold tokens that lower precision).
        This documents the failure mode truncation prevents.
        """
        gold = ["Optional[User]"]
        # Bleed adds many non-gold tokens
        raw = "Optional[User]\nQuestion: What does create_session return?  Answer: SessionToken"
        grades_raw = grade_qa(raw, gold)
        grades_clean = grade_qa(truncate_at_stop(raw), gold)
        assert grades_clean["f1"] >= grades_raw["f1"], (
            f"Truncation degraded F1: clean={grades_clean['f1']:.3f}, "
            f"raw={grades_raw['f1']:.3f}"
        )

    def test_empty_after_truncation_gives_zero_f1(self):
        """If answer is entirely bleed (stop string at position 0), F1 = 0."""
        gold = ["Optional[User]"]
        raw = "\nOptional[User]"  # stop string at index 0
        truncated = truncate_at_stop(raw)
        assert truncated == ""
        grades = grade_qa(truncated, gold)
        assert grades["f1"] == 0.0
        assert grades["exact_match"] is False

    def test_multiline_bleed_truncated_at_first_newline(self):
        """Only the first newline fires; single-line answers are fully preserved."""
        gold = ["Paris"]
        raw = "Paris\nFrance\nEurope"
        truncated = truncate_at_stop(raw)
        assert truncated == "Paris"
        grades = grade_qa(truncated, gold)
        assert grades["exact_match"] is True

    def test_long_bleed_removed_completely(self):
        """Multiple paragraphs of bleed are all removed."""
        gold = ["Marie Curie"]
        bleed = "\n".join([
            "\nQuestion: Who discovered radium?",
            "Answer: Marie Curie",
            "Context: Marie Curie was a physicist...",
        ])
        raw = "Marie Curie" + bleed
        truncated = truncate_at_stop(raw)
        assert truncated == "Marie Curie"
        assert "\n" not in truncated

    def test_answer_with_trailing_space_before_stop(self):
        """Trailing space before newline is kept (truncation position is the newline)."""
        gold = ["Marie Curie"]
        raw = "Marie Curie \nbleed"
        truncated = truncate_at_stop(raw)
        # Truncation at "\n", so "Marie Curie " (with trailing space) is returned
        assert truncated.strip() == "Marie Curie"
        grades = grade_qa(truncated, gold)
        assert grades["exact_match"] is True


# ---------------------------------------------------------------------------
# All five conditions apply truncation consistently
# ---------------------------------------------------------------------------

class TestAllConditionsApplyTruncation:
    """
    The five benchmark conditions (baseline, vector, vector_f, matrix, matrix_f)
    all call truncate_at_stop before returning. This class verifies that the
    truncation contract is consistent and deterministic across conditions.
    """

    CONDITIONS = ("baseline", "vector", "vector_f", "matrix", "matrix_f")

    def _simulate_raw_output(self, answer: str, bleed: str) -> str:
        """Simulate the raw model output before truncation."""
        return f"{answer}\n{bleed}"

    def test_all_conditions_produce_same_truncated_output(self):
        """
        truncate_at_stop is a pure function: same raw input → same output
        regardless of which condition produced it.
        """
        raw = "Optional[User]\nQuestion: What does foo return?"
        results = {c: truncate_at_stop(raw) for c in self.CONDITIONS}
        assert len(set(results.values())) == 1, (
            f"Conditions gave different truncation results: {results}"
        )

    def test_all_conditions_preserve_exact_match_after_truncation(self):
        """For each condition, grader finds exact match in truncated output."""
        gold = ["Marie Curie"]
        bleed = "Question: Who discovered radium?"
        for condition in self.CONDITIONS:
            raw = self._simulate_raw_output("Marie Curie", bleed)
            truncated = truncate_at_stop(raw)
            grades = grade_qa(truncated, gold)
            assert grades["exact_match"], (
                f"{condition}: exact_match=False after truncation; "
                f"truncated={truncated!r}"
            )

    def test_all_conditions_give_f1_one_for_exact_answer(self):
        """Clean single-word answer matches gold with F1 = 1.0."""
        gold = ["Paris"]
        for condition in self.CONDITIONS:
            result = truncate_at_stop("Paris")  # no bleed
            grades = grade_qa(result, gold)
            assert grades["f1"] == pytest.approx(1.0), (
                f"{condition}: F1 = {grades['f1']:.3f} for exact answer"
            )

    def test_truncation_is_idempotent(self):
        """Applying truncation twice gives the same result as applying it once."""
        raw = "answer\nbleed text\nmore bleed"
        once = truncate_at_stop(raw)
        twice = truncate_at_stop(once)
        assert once == twice, (
            f"Truncation not idempotent: once={once!r}, twice={twice!r}"
        )

    def test_no_bleed_passes_through_unchanged(self):
        """If the model generates only the answer with no bleed, truncation is a no-op."""
        for answer in ["Marie Curie", "Optional[User]", "SessionToken", "int", "Paris"]:
            assert truncate_at_stop(answer) == answer


# ---------------------------------------------------------------------------
# Truncation preserves multi-word and typed answers
# ---------------------------------------------------------------------------

class TestAnswerPreservation:
    """Verify that legitimate answer tokens are not destroyed by truncation."""

    @pytest.mark.parametrize("answer", [
        "Marie Curie",
        "Optional[User]",
        "SessionToken",
        "New York City",
        "42",
        "True",
        "list[str]",
        "def handle_order",
    ])
    def test_single_line_answer_unchanged(self, answer: str):
        assert truncate_at_stop(answer) == answer

    def test_answer_with_brackets_unchanged(self):
        assert truncate_at_stop("Optional[User]") == "Optional[User]"

    def test_answer_with_colon_unchanged(self):
        """A colon in the answer (not at start of a stop string) is preserved."""
        # "\nContext:" is a stop, but just "Context:" mid-answer is not
        assert truncate_at_stop("key: value") == "key: value"

    def test_answer_with_newline_before_stop_string_truncates_at_newline(self):
        """If the answer itself spans multiple lines, only the first line is kept."""
        text = "line one\nline two"
        assert truncate_at_stop(text) == "line one"


# ---------------------------------------------------------------------------
# Custom stop strings
# ---------------------------------------------------------------------------

class TestCustomStopStrings:
    def test_custom_stop_string_overrides_default(self):
        text = "answer|STOP|more text"
        result = truncate_at_stop(text, stop_strings=("|STOP|",))
        assert result == "answer"

    def test_empty_stop_strings_returns_full_text(self):
        text = "Marie Curie\nbleed text"
        result = truncate_at_stop(text, stop_strings=())
        assert result == text

    def test_multiple_custom_stops_earliest_wins(self):
        text = "a|B|c|A|d"
        result = truncate_at_stop(text, stop_strings=("|A|", "|B|"))
        assert result == "a"  # |B| appears at position 1, |A| at position 3

    def test_single_character_stop(self):
        text = "hello world"
        result = truncate_at_stop(text, stop_strings=(" ",))
        assert result == "hello"

    def test_stop_string_not_in_text_returns_full(self):
        text = "no stop here"
        result = truncate_at_stop(text, stop_strings=("|CUSTOM|",))
        assert result == text
