"""
Test: Token Efficiency via Context Injection

Validates the core claims of the Latent Rollback benchmark:
  1. Input token reduction is strictly positive (injection uses fewer tokens)
  2. Context state extraction produces a non-zero vector
  3. Layer selector returns a valid layer index
  4. Injection generation produces non-empty text
  5. QA grader produces correct exact match / F1 scores
  6. End-to-end benchmark on LLaMA-3 8B with 3 hardcoded examples

Run:
  source .venv/bin/activate
  python -m pytest tests/test_token_efficiency.py -v
  # or directly:
  python tests/test_token_efficiency.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from benchmark_datasets import (
    BenchmarkExample, grade_qa, _hardcoded_fallback
)
from context_injector import compute_token_metrics
from layer_selector import select_layer_heuristic


# ---------------------------------------------------------------------------
# Unit tests (no model required)
# ---------------------------------------------------------------------------

class TestGradeQA:
    def test_exact_match(self):
        result = grade_qa("The answer is France.", ["France"])
        assert result["exact_match"] is True
        assert result["f1"] > 0.0

    def test_no_match(self):
        result = grade_qa("I don't know the answer.", ["France"])
        assert result["exact_match"] is False
        assert result["f1"] == 0.0

    def test_case_insensitive(self):
        result = grade_qa("it is france yes", ["France"])
        assert result["exact_match"] is True

    def test_multiple_gold(self):
        result = grade_qa("She won two Nobel Prizes.", ["two", "2", "2 Nobel Prizes"])
        assert result["exact_match"] is True

    def test_partial_f1(self):
        result = grade_qa("France is the country", ["France is correct"])
        assert result["f1"] > 0.0


class TestTokenMetrics:
    def test_strict_reduction(self):
        # Injection input (question only) must be less than baseline (ctx + question)
        metrics = compute_token_metrics(
            n_baseline_input=500,
            n_injection_input=50,
            n_baseline_output=30,
            n_injection_output=35,
        )
        assert metrics["input_token_reduction"] > 0.0
        assert metrics["input_token_reduction"] == pytest.approx(0.9, abs=0.01)
        assert metrics["total_cost_ratio"] < 1.0

    def test_zero_reduction(self):
        metrics = compute_token_metrics(100, 100, 50, 50)
        assert metrics["input_token_reduction"] == pytest.approx(0.0, abs=0.01)
        assert metrics["total_cost_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_cost_ratio_accounts_for_output(self):
        # Even if input is reduced, if injection generates much more output,
        # total cost ratio may exceed 1.0
        metrics = compute_token_metrics(500, 50, 10, 2000)
        assert metrics["total_cost_ratio"] > 1.0


class TestDatasetFallback:
    def test_fallback_returns_examples(self):
        examples = _hardcoded_fallback("hotpotqa")
        assert len(examples) >= 2
        for ex in examples:
            assert ex.context
            assert ex.question
            assert ex.gold_answers
            assert ex.context_word_len > 0

    def test_example_prompts(self):
        examples = _hardcoded_fallback("hotpotqa")
        ex = examples[0]
        full = ex.full_prompt()
        q = ex.question_prompt()
        assert "Context:" in full
        assert "Question:" in full
        assert "Answer:" in full
        assert "Context:" not in q
        assert "Question:" in q
        # Question prompt is always shorter than full prompt
        assert len(q) < len(full)


class TestLayerSelector:
    """Test the heuristic layer selector without loading a real model."""

    class _FakeWrapper:
        n_layers = 32

    def test_llama_family(self):
        wrapper = self._FakeWrapper()
        layer = select_layer_heuristic(wrapper, "mlx-community/Meta-Llama-3-8B-Instruct-4bit")
        assert 0 < layer < wrapper.n_layers

    def test_mistral_family(self):
        wrapper = self._FakeWrapper()
        layer = select_layer_heuristic(wrapper, "mlx-community/Mistral-Small-24B-Instruct-2501-4bit")
        assert 0 < layer < wrapper.n_layers

    def test_qwen_family(self):
        self._FakeWrapper.n_layers = 28
        wrapper = self._FakeWrapper()
        layer = select_layer_heuristic(wrapper, "mlx-community/Qwen2.5-7B-Instruct-4bit")
        assert 0 < layer < 28

    def test_unknown_family(self):
        wrapper = self._FakeWrapper()
        wrapper.n_layers = 40
        layer = select_layer_heuristic(wrapper, "some-unknown-model-xyz")
        assert 0 < layer < 40


# ---------------------------------------------------------------------------
# Integration tests (require LLaMA-3 8B loaded)
# ---------------------------------------------------------------------------

LLAMA_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"


def _model_available() -> bool:
    """Check if model is cached locally (don't trigger download in CI)."""
    try:
        from pathlib import Path
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        return any(cache.glob(f"*Meta-Llama-3-8B-Instruct*"))
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="LLaMA-3 not cached locally")
class TestEndToEnd:
    """Full end-to-end benchmark on LLaMA-3 8B using hardcoded examples."""

    @pytest.fixture(scope="class")
    def wrapper(self):
        from backend_mlx import load_model
        return load_model(LLAMA_MODEL)

    @pytest.fixture(scope="class")
    def examples(self):
        return _hardcoded_fallback("hotpotqa")

    def test_context_extraction_nonzero(self, wrapper, examples):
        from context_injector import extract_context_state
        from layer_selector import select_layer_heuristic

        layer = select_layer_heuristic(wrapper, LLAMA_MODEL)
        ex = examples[0]
        ctx_v, n_tokens = extract_context_state(wrapper, ex.context, layer)

        assert isinstance(ctx_v, torch.Tensor)
        assert ctx_v.shape[0] == wrapper.d_model
        assert ctx_v.norm().item() > 0.0
        assert n_tokens > 0

    def test_injection_produces_text(self, wrapper, examples):
        from context_injector import extract_context_state, generate_with_context_injection
        from layer_selector import select_layer_heuristic

        layer = select_layer_heuristic(wrapper, LLAMA_MODEL)
        ex = examples[0]
        ctx_v, _ = extract_context_state(wrapper, ex.context, layer)
        text, n_q_tokens = generate_with_context_injection(
            wrapper, ex.question_prompt(), ctx_v, layer,
            scale=1.0, max_new_tokens=30,
        )
        assert isinstance(text, str)
        assert len(text.strip()) > 0
        assert n_q_tokens > 0

    def test_input_token_reduction_positive(self, wrapper, examples):
        from context_injector import (
            extract_context_state, generate_with_context_injection,
            generate_baseline_qa, compute_token_metrics,
        )
        from layer_selector import select_layer_heuristic

        layer = select_layer_heuristic(wrapper, LLAMA_MODEL)
        ex = examples[0]

        _, n_baseline_input = generate_baseline_qa(wrapper, ex.full_prompt(), 30)
        ctx_v, _ = extract_context_state(wrapper, ex.context, layer)
        _, n_injection_input = generate_with_context_injection(
            wrapper, ex.question_prompt(), ctx_v, layer,
            scale=1.0, max_new_tokens=30,
        )

        metrics = compute_token_metrics(n_baseline_input, n_injection_input, 30, 30)
        # Injection input must be strictly fewer tokens (context removed)
        assert metrics["input_token_reduction"] > 0.0, (
            f"Expected positive token reduction, got {metrics['input_token_reduction']}"
        )

    def test_all_three_hardcoded_examples(self, wrapper, examples):
        """Run full benchmark on all hardcoded examples, assert token reduction."""
        from context_injector import (
            extract_context_state, generate_with_context_injection,
            generate_baseline_qa, compute_token_metrics,
        )
        from layer_selector import select_layer_heuristic

        layer = select_layer_heuristic(wrapper, LLAMA_MODEL)
        reductions = []

        for ex in examples:
            _, n_bl = generate_baseline_qa(wrapper, ex.full_prompt(), 30)
            ctx_v, _ = extract_context_state(wrapper, ex.context, layer)
            _, n_inj = generate_with_context_injection(
                wrapper, ex.question_prompt(), ctx_v, layer,
                scale=1.0, max_new_tokens=30,
            )
            m = compute_token_metrics(n_bl, n_inj, 30, 30)
            reductions.append(m["input_token_reduction"])

        avg_reduction = sum(reductions) / len(reductions)
        assert avg_reduction > 0.3, (
            f"Expected average token reduction > 30%, got {avg_reduction:.1%}"
        )


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use pytest when run directly so pytest fixtures and markers work correctly
    import subprocess
    ret = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    sys.exit(ret.returncode)
