"""
Structural tests for the bench task corpus (bench_tasks.py).

No model calls.  Verifies that every task is well-formed, contexts contain
expected code patterns, and gold answers are present in context.

Run in isolation:
    pytest test_bench_corpus.py
"""

from __future__ import annotations

import pytest

from bench_tasks import (
    BENCH_TASKS,
    BenchTask,
    TASK_TYPES,
    get_tasks_by_type,
    get_task,
)


# ---------------------------------------------------------------------------
# Corpus completeness
# ---------------------------------------------------------------------------

class TestCorpusCompleteness:
    def test_has_at_least_17_tasks(self):
        assert len(BENCH_TASKS) >= 17

    def test_all_are_bench_task_instances(self):
        assert all(isinstance(t, BenchTask) for t in BENCH_TASKS)

    def test_unique_ids(self):
        ids = [t.id for t in BENCH_TASKS]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_all_task_types_covered(self):
        types_present = {t.task_type for t in BENCH_TASKS}
        for tt in TASK_TYPES:
            assert tt in types_present, f"No tasks for task_type={tt!r}"

    def test_each_type_has_at_least_two_tasks(self):
        for tt in TASK_TYPES:
            tasks = get_tasks_by_type(tt)
            assert len(tasks) >= 2, f"Only {len(tasks)} tasks for type {tt!r}"

    def test_get_task_by_id(self):
        for task in BENCH_TASKS:
            retrieved = get_task(task.id)
            assert retrieved.id == task.id

    def test_get_task_unknown_raises(self):
        with pytest.raises(KeyError):
            get_task("nonexistent_task_id_xyz")


# ---------------------------------------------------------------------------
# Field completeness
# ---------------------------------------------------------------------------

class TestTaskFieldCompleteness:
    def test_all_have_non_empty_context(self):
        for t in BENCH_TASKS:
            assert len(t.context.strip()) > 0, f"{t.id}: empty context"

    def test_all_have_non_empty_question(self):
        for t in BENCH_TASKS:
            assert len(t.question.strip()) > 0, f"{t.id}: empty question"

    def test_all_have_at_least_one_gold_answer(self):
        for t in BENCH_TASKS:
            assert len(t.gold_answers) >= 1, f"{t.id}: no gold answers"
            assert all(len(a) > 0 for a in t.gold_answers), f"{t.id}: empty gold answer"

    def test_hop_count_is_one_or_two(self):
        for t in BENCH_TASKS:
            assert t.hop_count in (1, 2), f"{t.id}: hop_count={t.hop_count}"

    def test_n_files_is_positive(self):
        for t in BENCH_TASKS:
            assert t.n_files >= 1, f"{t.id}: n_files={t.n_files}"

    def test_context_tokens_approx_is_positive(self):
        for t in BENCH_TASKS:
            assert t.context_tokens_approx > 0, f"{t.id}: zero context_tokens_approx"


# ---------------------------------------------------------------------------
# Context structure
# ---------------------------------------------------------------------------

class TestContextStructure:
    def test_multifile_contexts_have_file_markers(self):
        for t in get_tasks_by_type("multifile_refactor"):
            markers = t.context.count("# File:")
            assert markers >= 3, (
                f"{t.id}: multifile_refactor should have >=3 file markers, got {markers}"
            )

    def test_cross_file_ref_contexts_have_multiple_files(self):
        for t in get_tasks_by_type("cross_file_ref"):
            markers = t.context.count("# File:")
            assert markers >= 2, f"{t.id}: cross_file_ref should have >=2 files"

    def test_code_contexts_contain_code_keywords(self):
        for t in BENCH_TASKS:
            has_code = any(
                kw in t.context
                for kw in ("def ", "class ", "import ", "from ", "function ", "interface ")
            )
            assert has_code, f"{t.id}: context has no recognizable code keywords"

    def test_short_ctx_is_small(self):
        for t in get_tasks_by_type("short_ctx"):
            # short contexts should be under 1000 tokens
            assert t.context_tokens_approx < 1000, (
                f"{t.id}: short_ctx has {t.context_tokens_approx} tokens (expected <1000)"
            )

    def test_long_ctx_is_large(self):
        for t in get_tasks_by_type("long_ctx"):
            # long contexts should be substantially larger than short contexts.
            # code is token-dense, so approx_tokens (word-based) underestimates;
            # threshold is set relative to short_ctx rather than absolute.
            short_max = max(
                s.context_tokens_approx for s in get_tasks_by_type("short_ctx")
            )
            assert t.context_tokens_approx > short_max * 3, (
                f"{t.id}: long_ctx ({t.context_tokens_approx} tokens) is not "
                f"substantially larger than short_ctx (max {short_max} tokens)"
            )

    def test_double_hop_tasks_have_hop_count_2(self):
        for t in get_tasks_by_type("double_hop"):
            assert t.hop_count == 2, f"{t.id}: double_hop task has hop_count={t.hop_count}"

    def test_single_hop_tasks_have_hop_count_1(self):
        for t in get_tasks_by_type("single_hop"):
            assert t.hop_count == 1, f"{t.id}: single_hop task has hop_count={t.hop_count}"


# ---------------------------------------------------------------------------
# Gold answer verifiability
# ---------------------------------------------------------------------------

class TestGoldAnswers:
    def test_gold_answers_present_in_context(self):
        """
        At least one gold answer must appear literally in the context.

        Skips refactor tasks (multifile_refactor, long_ctx refactors): their
        gold_answers describe the *target* state absent from the original context.
        """
        REFACTOR_IDS = {"rename_3file", "add_param_chain", "interface_change",
                        "move_function", "long_refactor"}
        for t in BENCH_TASKS:
            if t.task_type == "multifile_refactor" or t.id in REFACTOR_IDS:
                continue
            found = any(
                gold.lower() in t.context.lower()
                for gold in t.gold_answers
            )
            assert found, (
                f"{t.id}: none of {t.gold_answers!r} found in context"
            )

    def test_refactor_must_contain_not_in_original_context(self):
        """
        For refactor tasks: must_contain patterns describe the *target* state
        and should NOT fully exist in the original (pre-refactor) context.
        At least one must_contain pattern should be absent before the refactor.
        """
        for t in get_tasks_by_type("multifile_refactor"):
            if not t.must_contain:
                continue
            all_present = all(pat in t.context for pat in t.must_contain)
            assert not all_present, (
                f"{t.id}: all must_contain patterns already in context — "
                "task is trivially satisfied without any change"
            )

    def test_refactor_must_not_contain_present_in_context(self):
        """
        must_not_contain patterns define what the refactored output must avoid.
        These patterns MUST exist in the original context (they're the old code).
        """
        for t in get_tasks_by_type("multifile_refactor"):
            for pattern in t.must_not_contain:
                assert pattern in t.context, (
                    f"{t.id}: forbidden pattern {pattern!r} not in original context — "
                    "cannot test for removal of something that was never there"
                )

    def test_long_ctx_gold_in_context(self):
        for t in get_tasks_by_type("long_ctx"):
            if t.id == "long_refactor":
                # Refactor task: gold is the target (not in original context).
                # Verify the old name IS present instead.
                assert any(p in t.context for p in t.must_not_contain), (
                    f"{t.id}: old name patterns not found in context"
                )
            else:
                found = any(gold.lower() in t.context.lower() for gold in t.gold_answers)
                assert found, f"{t.id}: gold answer not in long context"


# ---------------------------------------------------------------------------
# Refactor task specifics
# ---------------------------------------------------------------------------

class TestRefactorTasks:
    def test_all_have_instruction(self):
        for t in get_tasks_by_type("multifile_refactor"):
            assert len(t.instruction) > 0, f"{t.id}: missing instruction"

    def test_all_have_must_contain(self):
        for t in get_tasks_by_type("multifile_refactor"):
            assert len(t.must_contain) > 0, f"{t.id}: no must_contain"

    def test_all_have_must_not_contain(self):
        for t in get_tasks_by_type("multifile_refactor"):
            assert len(t.must_not_contain) > 0, f"{t.id}: no must_not_contain"

    def test_rename_task_checks_new_name(self):
        t = get_task("rename_3file")
        assert any("handle_order" in p for p in t.must_contain)
        assert any("process_order" in p for p in t.must_not_contain)

    def test_add_param_task_checks_signature(self):
        t = get_task("add_param_chain")
        assert any("dry_run" in p for p in t.must_contain)

    def test_interface_change_checks_optional(self):
        t = get_task("interface_change")
        assert any("Optional" in p for p in t.must_contain)

    def test_move_function_checks_import(self):
        t = get_task("move_function")
        assert any("validators" in p for p in t.must_contain)
        assert any("helpers" in p for p in t.must_not_contain)


# ---------------------------------------------------------------------------
# Cross-file reference task specifics
# ---------------------------------------------------------------------------

class TestCrossFileRefTasks:
    def test_who_calls_gold_answers_are_function_names(self):
        t = get_task("who_calls")
        for gold in t.gold_answers:
            # Gold answers should be identifiers, not sentences
            assert " " not in gold.strip(), (
                f"who_calls gold answer {gold!r} looks like a sentence, "
                "expected a function name"
            )

    def test_type_provenance_gold_is_filename(self):
        t = get_task("type_provenance")
        assert any("models" in g for g in t.gold_answers)

    def test_import_chain_hop_count(self):
        t = get_task("import_chain")
        assert t.hop_count == 2

    def test_import_chain_gold_in_context(self):
        t = get_task("import_chain")
        assert any(g in t.context for g in t.gold_answers)


# ---------------------------------------------------------------------------
# Double-hop task specifics
# ---------------------------------------------------------------------------

class TestDoubleHopTasks:
    def test_transitive_return_gold_is_type(self):
        t = get_task("transitive_return")
        assert "SessionToken" in t.gold_answers

    def test_inherited_method_gold_are_param_names(self):
        t = get_task("inherited_method")
        assert "name" in t.gold_answers
        assert "email" in t.gold_answers
        assert "role" in t.gold_answers

    def test_field_access_gold_are_field_names(self):
        t = get_task("field_access")
        assert any(f in t.gold_answers for f in ("city", "country", "street"))

    def test_double_hop_chain_exists_in_context(self):
        """
        For transitive_return: create_session must call generate_token,
        and generate_token must return SessionToken.
        """
        t = get_task("transitive_return")
        assert "generate_token" in t.context
        assert "create_session" in t.context
        assert "SessionToken" in t.context

    def test_inherited_method_chain_exists(self):
        t = get_task("inherited_method")
        assert "AdminUser" in t.context
        assert "BaseUser" in t.context
        assert "def save" in t.context
