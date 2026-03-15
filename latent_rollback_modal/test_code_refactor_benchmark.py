"""
Tests for multi-file refactor benchmark infrastructure.

3 conditions:
  baseline       — full context + refactor instruction in prompt
  vec_f_ner      — vector injection + NER F block
  vec_f_summary  — vector injection + model-written summary as F block

3 synthetic tasks mimicking real refactor workflows:
  rename         — rename a function across 2 files
  add_param      — add a parameter to a function used in multiple files
  change_return  — change a return type and its downstream usages

No model calls in these tests — all model-dependent functions are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from .benchmark_code_refactor import (
    REFACTOR_TASKS,
    RefactorExample,
    RefactorRecord,
    grade_refactor,
    build_summary_prompt,
    run_refactor_example,
)


# ---------------------------------------------------------------------------
# REFACTOR_TASKS: structure
# ---------------------------------------------------------------------------

class TestRefactorTasks:
    def test_has_three_tasks(self):
        assert len(REFACTOR_TASKS) == 3

    def test_all_are_refactor_examples(self):
        assert all(isinstance(t, RefactorExample) for t in REFACTOR_TASKS)

    def test_all_have_multifile_context(self):
        for task in REFACTOR_TASKS:
            # Multi-file context has at least two file markers
            file_markers = task.context.count("# File:") + task.context.count("// File:")
            assert file_markers >= 2, f"{task.id}: only {file_markers} file markers"

    def test_all_have_refactor_instruction(self):
        for task in REFACTOR_TASKS:
            assert len(task.instruction) > 0

    def test_all_have_success_checks(self):
        for task in REFACTOR_TASKS:
            assert len(task.must_contain) > 0, f"{task.id}: no must_contain"

    def test_all_have_failure_checks(self):
        for task in REFACTOR_TASKS:
            assert len(task.must_not_contain) > 0, f"{task.id}: no must_not_contain"

    def test_rename_task_checks_new_name(self):
        rename = next(t for t in REFACTOR_TASKS if t.id == "rename_function")
        assert any("patch_user" in s for s in rename.must_contain)
        # Forbidden patterns are code-specific (def or call site), not prose mentions
        assert any("def update_user" in s or ".update_user(" in s for s in rename.must_not_contain)

    def test_add_param_task_checks_new_signature(self):
        add_param = next(t for t in REFACTOR_TASKS if t.id == "add_parameter")
        assert any("timeout" in s for s in add_param.must_contain)

    def test_change_return_task_checks_type(self):
        change_ret = next(t for t in REFACTOR_TASKS if t.id == "change_return_type")
        assert any("Optional" in s or "User" in s for s in change_ret.must_contain)

    def test_forbidden_identifiers_root_in_context(self):
        # The root identifier (first word) of each forbidden pattern must exist in context
        for task in REFACTOR_TASKS:
            for pattern in task.must_not_contain:
                root = pattern.lstrip(".").split("(")[0].split(" ")[-1]
                assert root in task.context, (
                    f"{task.id}: root {root!r} of forbidden pattern not in context"
                )

    def test_unique_ids(self):
        ids = [t.id for t in REFACTOR_TASKS]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# grade_refactor
# ---------------------------------------------------------------------------

class TestGradeRefactor:
    def test_full_success_all_checks_pass(self):
        task = REFACTOR_TASKS[0]  # rename
        # Construct output that contains must_contain and avoids must_not_contain
        good_output = " ".join(task.must_contain) + " rest of output"
        result = grade_refactor(good_output, task)
        assert result["all_present"] is True

    def test_fail_when_must_contain_missing(self):
        task = REFACTOR_TASKS[0]
        result = grade_refactor("some unrelated output", task)
        assert result["all_present"] is False

    def test_flags_old_identifier_present(self):
        task = REFACTOR_TASKS[0]  # rename: must_not_contain old name
        bad_output = task.must_not_contain[0] + " is still here"
        result = grade_refactor(bad_output, task)
        assert result["any_forbidden"] is True

    def test_clean_output_has_no_forbidden(self):
        task = REFACTOR_TASKS[0]
        clean = " ".join(task.must_contain)
        result = grade_refactor(clean, task)
        assert result["any_forbidden"] is False

    def test_score_is_float_between_0_and_1(self):
        task = REFACTOR_TASKS[0]
        result = grade_refactor("anything", task)
        assert 0.0 <= result["score"] <= 1.0

    def test_perfect_output_scores_1(self):
        task = REFACTOR_TASKS[0]
        perfect = " ".join(task.must_contain)
        result = grade_refactor(perfect, task)
        assert result["score"] == 1.0

    def test_empty_output_scores_0(self):
        task = REFACTOR_TASKS[0]
        result = grade_refactor("", task)
        assert result["score"] == 0.0

    def test_partial_credit_for_partial_match(self):
        task = REFACTOR_TASKS[0]
        # Include only first must_contain item
        partial = task.must_contain[0]
        result = grade_refactor(partial, task)
        assert 0.0 < result["score"] < 1.0 or result["score"] == 1.0  # could be 1.0 if only 1 item


# ---------------------------------------------------------------------------
# build_summary_prompt
# ---------------------------------------------------------------------------

class TestBuildSummaryPrompt:
    def test_contains_context(self):
        task = REFACTOR_TASKS[0]
        prompt = build_summary_prompt(task.context)
        assert task.context[:50] in prompt

    def test_asks_for_signatures(self):
        prompt = build_summary_prompt("def foo(): pass")
        assert "signature" in prompt.lower() or "function" in prompt.lower()

    def test_asks_for_imports(self):
        prompt = build_summary_prompt("import os")
        assert "import" in prompt.lower()

    def test_ends_with_generation_cue(self):
        prompt = build_summary_prompt("some code")
        # Should end with a cue for the model to start generating
        assert prompt.strip().endswith(("Summary:", "Notes:", "Functions:"))


# ---------------------------------------------------------------------------
# RefactorRecord: structure
# ---------------------------------------------------------------------------

class TestRefactorRecord:
    def test_has_required_fields(self):
        rec = RefactorRecord(
            task_id="rename_function",
            condition="baseline",
            output="patch_user()",
            score=1.0,
            all_present=True,
            any_forbidden=False,
            f_block="",
            elapsed_s=1.0,
        )
        assert rec.task_id == "rename_function"
        assert rec.condition == "baseline"
        assert rec.score == 1.0

    def test_condition_is_one_of_three(self):
        valid = {"baseline", "vec_f_ner", "vec_f_summary"}
        rec = RefactorRecord(
            task_id="x", condition="baseline", output="",
            score=0.0, all_present=False, any_forbidden=False,
            f_block="", elapsed_s=0.0,
        )
        assert rec.condition in valid


# ---------------------------------------------------------------------------
# run_refactor_example: mocked model calls
# ---------------------------------------------------------------------------

class TestRunRefactorExampleMocked:
    def _make_wrapper(self):
        wrapper = MagicMock()
        wrapper.n_layers = 32
        wrapper.eos_token_id = 2
        wrapper.encode.return_value = [1, 2, 3, 4, 5]
        return wrapper

    def test_returns_three_records_per_example(self):
        wrapper = self._make_wrapper()
        task = REFACTOR_TASKS[0]

        with patch("benchmark_code_refactor.generate_baseline_qa", return_value=("patch_user output", 100)), \
             patch("benchmark_code_refactor.extract_context_state", return_value=(MagicMock(), 50)), \
             patch("benchmark_code_refactor.generate_with_context_injection", return_value=("patch_user output", 50)), \
             patch("benchmark_code_refactor.extract_facts_code", return_value="Facts: patch_user"):
            records = run_refactor_example(wrapper, task, injection_layer=14)

        assert len(records) == 3

    def test_record_conditions_are_all_three(self):
        wrapper = self._make_wrapper()
        task = REFACTOR_TASKS[0]

        with patch("benchmark_code_refactor.generate_baseline_qa", return_value=("patch_user", 100)), \
             patch("benchmark_code_refactor.extract_context_state", return_value=(MagicMock(), 50)), \
             patch("benchmark_code_refactor.generate_with_context_injection", return_value=("patch_user", 50)), \
             patch("benchmark_code_refactor.extract_facts_code", return_value="Facts: patch_user"):
            records = run_refactor_example(wrapper, task, injection_layer=14)

        conditions = {r.condition for r in records}
        assert conditions == {"baseline", "vec_f_ner", "vec_f_summary"}

    def test_baseline_uses_full_context(self):
        wrapper = self._make_wrapper()
        task = REFACTOR_TASKS[0]

        with patch("benchmark_code_refactor.generate_baseline_qa", return_value=("out", 100)) as mock_bl, \
             patch("benchmark_code_refactor.extract_context_state", return_value=(MagicMock(), 50)), \
             patch("benchmark_code_refactor.generate_with_context_injection", return_value=("out", 50)), \
             patch("benchmark_code_refactor.extract_facts_code", return_value="Facts:"):
            run_refactor_example(wrapper, task, injection_layer=14)

        # Baseline prompt should contain the full context
        call_args = mock_bl.call_args[0]
        assert task.context[:30] in call_args[1]

    def test_summary_f_block_differs_from_ner_f_block(self):
        wrapper = self._make_wrapper()
        task = REFACTOR_TASKS[0]

        with patch("benchmark_code_refactor.generate_baseline_qa", return_value=("model summary output", 100)), \
             patch("benchmark_code_refactor.extract_context_state", return_value=(MagicMock(), 50)), \
             patch("benchmark_code_refactor.generate_with_context_injection", return_value=("out", 50)), \
             patch("benchmark_code_refactor.extract_facts_code", return_value="Facts: ner_result"):
            records = run_refactor_example(wrapper, task, injection_layer=14)

        ner_rec = next(r for r in records if r.condition == "vec_f_ner")
        summary_rec = next(r for r in records if r.condition == "vec_f_summary")
        assert ner_rec.f_block != summary_rec.f_block
