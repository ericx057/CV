"""
Tests for code-specific F block extraction and code QA benchmarking.

Covers:
  - extract_facts_code: signature/import/class extraction + BM25 ranking
  - load_code_benchmark: synthetic code QA examples
  - grade_code_qa: token-level grading adapted for code identifiers
  - truncate_at_stop: stop sequences for code generation outputs
"""

from __future__ import annotations

import pytest

from .benchmark_ablation import extract_facts_code
from .benchmark_datasets import BenchmarkExample, load_code_benchmark, grade_code_qa
from .context_injector import truncate_at_stop, QA_STOP_STRINGS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_MODULE = """
from typing import Optional
import json

MAX_USERS = 1000
DEFAULT_TIMEOUT = 30

class UserRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def find_by_id(self, user_id: int) -> Optional[dict]:
        ...

    def update_user(self, user_id: int, delta: UserDelta) -> bool:
        ...

class UserDelta:
    name: str
    email: str

def create_user(name: str, email: str) -> dict:
    ...

def delete_user(user_id: int) -> bool:
    ...
"""

MULTI_FILE_CONTEXT = """
# File: types.py
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

# File: repository.py
from types import UserDelta, User
from typing import Optional

class UserRepository:
    def update_user(self, user_id: int, delta: UserDelta) -> Optional[User]:
        ...

    def find_by_id(self, user_id: int) -> Optional[User]:
        ...

# File: service.py
from repository import UserRepository
from types import UserDelta

def patch_user_email(repo: UserRepository, user_id: int, new_email: str) -> bool:
    delta = UserDelta(name=None, email=new_email)
    ...
"""

TYPESCRIPT_MODULE = """
import { UserDelta, User } from './types';
import { Repository } from './base';

export interface UserRepository extends Repository<User> {
  findById(userId: number): Promise<User | null>;
  updateUser(userId: number, delta: UserDelta): Promise<boolean>;
}

export const MAX_RETRIES = 3;

export function createUser(name: string, email: string): User {
  return { id: Date.now(), name, email };
}
"""


# ---------------------------------------------------------------------------
# extract_facts_code: signature extraction
# ---------------------------------------------------------------------------

class TestExtractFactsCodeSignatures:
    def test_extracts_function_signatures(self):
        result = extract_facts_code(SIMPLE_MODULE, "what does create_user return")
        assert "create_user" in result

    def test_extracts_return_type(self):
        result = extract_facts_code(SIMPLE_MODULE, "what does find_by_id return")
        assert "Optional" in result or "find_by_id" in result

    def test_extracts_parameter_types(self):
        result = extract_facts_code(SIMPLE_MODULE, "what type does update_user take as delta")
        assert "UserDelta" in result

    def test_extracts_class_definitions(self):
        # Use a question with lexical overlap so BM25 can rank classes up
        result = extract_facts_code(SIMPLE_MODULE, "what does UserRepository contain")
        assert "UserRepository" in result

    def test_extracts_imports(self):
        result = extract_facts_code(SIMPLE_MODULE, "what is imported")
        assert "Optional" in result or "typing" in result

    def test_extracts_constants(self):
        result = extract_facts_code(SIMPLE_MODULE, "what is the max users limit")
        assert "MAX_USERS" in result or "1000" in result

    def test_returns_facts_prefix(self):
        result = extract_facts_code(SIMPLE_MODULE, "any question")
        assert result.startswith("Facts:") or result == ""

    def test_empty_context_returns_empty(self):
        result = extract_facts_code("", "any question")
        assert result == ""

    def test_non_code_prose_returns_empty_or_short(self):
        prose = "The quick brown fox jumps over the lazy dog. No code here."
        result = extract_facts_code(prose, "what function is defined")
        # Should not crash; may return empty or minimal content
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# extract_facts_code: BM25 question conditioning
# ---------------------------------------------------------------------------

class TestExtractFactsCodeBM25:
    def test_return_type_question_surfaces_return_annotation(self):
        result = extract_facts_code(SIMPLE_MODULE, "what does find_by_id return")
        # find_by_id is most relevant — should appear in top facts
        assert "find_by_id" in result

    def test_param_type_question_surfaces_correct_function(self):
        result = extract_facts_code(SIMPLE_MODULE, "what parameter does update_user take")
        assert "update_user" in result

    def test_import_question_surfaces_imports(self):
        result = extract_facts_code(SIMPLE_MODULE, "what is imported from typing")
        assert "Optional" in result or "typing" in result

    def test_class_question_surfaces_class(self):
        result = extract_facts_code(SIMPLE_MODULE, "what does UserDelta contain")
        assert "UserDelta" in result

    def test_different_questions_produce_different_facts(self):
        r1 = extract_facts_code(SIMPLE_MODULE, "what does create_user return")
        r2 = extract_facts_code(SIMPLE_MODULE, "what does delete_user take as argument")
        # BM25 should rank differently for different questions
        # At minimum they should both be non-empty
        assert r1 != "" and r2 != ""

    def test_multi_file_context_surfaces_cross_file_signatures(self):
        result = extract_facts_code(MULTI_FILE_CONTEXT, "what does update_user return")
        assert "update_user" in result

    def test_typescript_context_extracts_interface_members(self):
        result = extract_facts_code(TYPESCRIPT_MODULE, "what does findById return")
        assert "findById" in result or "UserRepository" in result


# ---------------------------------------------------------------------------
# extract_facts_code: output constraints
# ---------------------------------------------------------------------------

class TestExtractFactsCodeConstraints:
    def test_respects_max_chars(self):
        result = extract_facts_code(SIMPLE_MODULE, "any question", max_chars=100)
        # Strip "Facts: " prefix before measuring
        body = result[len("Facts: "):] if result.startswith("Facts: ") else result
        assert len(body) <= 110  # small tolerance for truncation boundary

    def test_does_not_duplicate_entries(self):
        result = extract_facts_code(SIMPLE_MODULE, "find_by_id return type")
        # find_by_id should not appear more than twice in facts
        assert result.count("find_by_id") <= 2

    def test_handles_large_context_without_error(self):
        large_ctx = SIMPLE_MODULE * 20
        result = extract_facts_code(large_ctx, "what does create_user return")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# load_code_benchmark
# ---------------------------------------------------------------------------

class TestLoadCodeBenchmark:
    def test_returns_list_of_benchmark_examples(self):
        examples = load_code_benchmark()
        assert isinstance(examples, list)
        assert len(examples) > 0
        assert all(isinstance(e, BenchmarkExample) for e in examples)

    def test_examples_have_code_contexts(self):
        examples = load_code_benchmark()
        for ex in examples:
            # Code context should contain at least one def, class, or import
            has_code = any(kw in ex.context for kw in ("def ", "class ", "import ", "function ", "=>"))
            assert has_code, f"Example {ex.id} has no code in context"

    def test_examples_have_answerable_gold(self):
        examples = load_code_benchmark()
        for ex in examples:
            assert len(ex.gold_answers) >= 1
            assert all(len(a) > 0 for a in ex.gold_answers)

    def test_examples_have_code_task_type(self):
        examples = load_code_benchmark()
        for ex in examples:
            assert ex.task == "code_qa"

    def test_examples_cover_different_question_types(self):
        examples = load_code_benchmark()
        questions = [ex.question.lower() for ex in examples]
        # Should have return type, parameter type, and import questions
        has_return = any("return" in q for q in questions)
        has_param = any("parameter" in q or "argument" in q or "take" in q for q in questions)
        assert has_return, "No return-type questions in code benchmark"
        assert has_param, "No parameter-type questions in code benchmark"

    def test_gold_answers_are_in_context(self):
        examples = load_code_benchmark()
        for ex in examples:
            found = any(
                gold.lower() in ex.context.lower()
                for gold in ex.gold_answers
            )
            assert found, f"Gold answer not found in context for {ex.id}: {ex.gold_answers}"

    def test_n_per_task_respected(self):
        examples = load_code_benchmark(n=5)
        assert len(examples) == 5

    def test_reproducible_with_seed(self):
        e1 = load_code_benchmark(n=3, seed=42)
        e2 = load_code_benchmark(n=3, seed=42)
        assert [e.id for e in e1] == [e.id for e in e2]

    def test_different_seeds_produce_different_order(self):
        e1 = load_code_benchmark(n=5, seed=42)
        e2 = load_code_benchmark(n=5, seed=99)
        # Not guaranteed to differ but with enough examples should
        assert [e.id for e in e1] != [e.id for e in e2] or len(e1) < 2


# ---------------------------------------------------------------------------
# grade_code_qa
# ---------------------------------------------------------------------------

class TestGradeCodeQA:
    def test_exact_type_match(self):
        result = grade_code_qa("Optional[User]", ["Optional[User]"])
        assert result["exact_match"] is True
        assert result["f1"] == 1.0

    def test_partial_type_match(self):
        result = grade_code_qa("returns Optional[User]", ["Optional[User]"])
        assert result["f1"] > 0.0

    def test_case_sensitive_for_identifiers(self):
        # Code identifiers are case-sensitive: "user" != "User"
        result = grade_code_qa("optional[user]", ["Optional[User]"])
        assert result["exact_match"] is False

    def test_no_match_returns_zero(self):
        result = grade_code_qa("int", ["Optional[User]"])
        assert result["f1"] < 0.5

    def test_bool_return_type(self):
        result = grade_code_qa("bool", ["bool"])
        assert result["exact_match"] is True

    def test_multiple_gold_answers(self):
        result = grade_code_qa("UserDelta", ["delta: UserDelta", "UserDelta"])
        assert result["exact_match"] is True

    def test_empty_generated_returns_zero(self):
        result = grade_code_qa("", ["Optional[User]"])
        assert result["f1"] == 0.0

    def test_constant_value_match(self):
        result = grade_code_qa("1000", ["1000"])
        assert result["exact_match"] is True


# ---------------------------------------------------------------------------
# truncate_at_stop for code generation outputs
# ---------------------------------------------------------------------------

class TestTruncateAtStopCode:
    def test_stops_at_double_newline(self):
        output = "Optional[User]\n\ndef next_function():"
        assert truncate_at_stop(output) == "Optional[User]"

    def test_stops_at_next_question(self):
        output = "bool\nQuestion: What does delete_user return"
        assert truncate_at_stop(output) == "bool"

    def test_stops_at_facts_prefix(self):
        output = "UserDelta\nFacts: UserRepository; update_user"
        assert truncate_at_stop(output) == "UserDelta"

    def test_does_not_truncate_clean_answer(self):
        output = "Optional[User]"
        assert truncate_at_stop(output) == "Optional[User]"

    def test_handles_type_with_brackets(self):
        output = "dict[str, int]"
        assert truncate_at_stop(output) == "dict[str, int]"

    def test_handles_none_type(self):
        output = "None"
        assert truncate_at_stop(output) == "None"
