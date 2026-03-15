"""
RSCE Ablation Benchmark — Part 2

3-way ablation to measure the contribution of the fact block F:

  Condition 1 — Baseline  : [ctx + P]       full context in token stream
  Condition 2 — RSCE only : [C + P]         context vector injection, no F
  Condition 3 — RSCE + F  : [C + F + P]     context vector + fact block

Three fact extraction strategies for h(D):
  - NER   : regex-based named entity + number extraction (practical, no extra model)
  - BM25  : question-conditioned sentence selection via BM25 ranking (practical)
  - Oracle: sentences from ctx containing a gold answer string (upper bound)

The F1 delta between conditions 2 and 3 measures how much F contributes.
The F1 delta between condition 3 and baseline measures the residual gap.

Usage:
  source .venv/bin/activate
  python benchmark_ablation.py
  python benchmark_ablation.py --models llama3-8b --n 10 --fact-mode ner
  python benchmark_ablation.py --fact-mode bm25           # question-conditioned BM25
  python benchmark_ablation.py --fact-mode oracle         # upper bound experiment
  python benchmark_ablation.py --fact-mode all            # run all three modes
  python benchmark_ablation.py --fact-mode both           # run ner + oracle (legacy)
  python benchmark_ablation.py --layer 14                 # override injection layer
  python benchmark_ablation.py --condition rsce_f         # only run RSCE+F pass
  python benchmark_ablation.py --resume ablation_123      # skip already-done examples
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from rich.console import Console
from rich import box
from rich.table import Table

from .backend_torch import load_model, MLXModelWrapper, clear_backend_cache
from .benchmark_datasets import load_benchmark, BenchmarkExample, grade_qa
from .context_injector import (
    extract_context_state,
    generate_baseline_qa,
    generate_with_context_injection,
    compute_token_metrics,
    truncate_at_stop,
    QA_STOP_STRINGS,
)
from .layer_selector import select_layer_heuristic
from .benchmark_runner import MODEL_MATRIX
from .config import results_path

console = Console()
RESULTS_DIR = results_path("ablation_results")


# ---------------------------------------------------------------------------
# h(D): Fact extraction for QA contexts
# ---------------------------------------------------------------------------

def extract_facts_ner(ctx: str, max_facts: int = 15) -> str:
    """
    NER-based h(D): extract named entities and numbers from ctx using regex.

    Targets:
      - Capitalized multi-word sequences (proper nouns / named entities)
      - 4-digit years
      - Standalone numbers
      - Quoted strings

    Returns a compact fact block string, e.g.:
      "Facts: Marie Curie; Nobel Prize; 1903; Warsaw; Paris"
    """
    facts: list[str] = []
    seen: set[str] = set()

    def add(f: str) -> None:
        f = f.strip()
        if f and f.lower() not in seen and len(f) > 1:
            seen.add(f.lower())
            facts.append(f)

    # Capitalized multi-word proper nouns (2-5 words, each capitalized)
    for m in re.finditer(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b', ctx
    ):
        add(m.group(1))

    # Single capitalized words that aren't sentence starters
    # (appear mid-sentence after lowercase word)
    for m in re.finditer(r'(?<=[a-z,;]\s)([A-Z][a-z]{2,})\b', ctx):
        add(m.group(1))

    # 4-digit years
    for m in re.finditer(r'\b(1[0-9]{3}|20[0-2][0-9])\b', ctx):
        add(m.group(1))

    # Numbers with units or standalone numbers
    for m in re.finditer(r'\b(\d+(?:\.\d+)?(?:\s*(?:km|m|kg|mph|°|percent|%))?)\b', ctx):
        val = m.group(1).strip()
        if val.isdigit() and len(val) == 1:
            continue  # skip single digits
        add(val)

    # Quoted strings
    for m in re.finditer(r'"([^"]{3,40})"', ctx):
        add(m.group(1))

    # Deduplicate while preserving order, trim to max_facts
    final = facts[:max_facts]
    if not final:
        return ""
    return "Facts: " + "; ".join(final)


def extract_facts_bm25(ctx: str, question: str, top_k: int = 3, max_chars: int = 300) -> str:
    """
    BM25 h(D): question-conditioned sentence selection.

    Scores every sentence in ctx against the question using BM25Okapi, then
    returns the top-k sentences (re-ordered by their original position for
    coherence).  No gold answers required — practical at inference time.

    Returns a compact fact block string, e.g.:
      "Facts: Marie Curie won the Nobel Prize in 1903. She was born in Warsaw."
    """
    from rank_bm25 import BM25Okapi

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', ctx) if s.strip()]
    if not sentences:
        return ""

    # Simple whitespace + punctuation tokenizer
    def tokenize(text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    tokenized_corpus = [tokenize(s) for s in sentences]
    tokenized_query = tokenize(question)

    # BM25 requires at least one query token
    if not tokenized_query:
        return extract_facts_ner(ctx)

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Take top_k by score; preserve original document order for coherence
    top_indices = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    )
    selected = [sentences[i] for i in top_indices]

    block = " ".join(selected)
    if len(block) > max_chars:
        block = block[:max_chars].rsplit(" ", 1)[0] + "..."

    return f"Facts: {block}"


def extract_facts_code(ctx: str, question: str, top_k: int = 5, max_chars: int = 400) -> str:
    """
    Code-specific h(D): extract function signatures, class definitions, imports,
    and constants from a code context, then BM25-rank against the question.

    Handles Python and TypeScript/JavaScript via regex patterns.
    Falls back to empty string if no code-like structures are found.

    Returns a compact fact block string, e.g.:
      "Facts: def find_by_id(self, user_id: int) -> Optional[User]; class UserRepository"
    """
    from rank_bm25 import BM25Okapi

    candidates: list[str] = []

    # Python / TS function signatures
    for m in re.finditer(
        r'^\s*(?:async\s+)?def\s+(\w+\s*\([^)]*\)(?:\s*->\s*[^\n:]+)?)',
        ctx, re.MULTILINE
    ):
        candidates.append(m.group(1).strip())

    # TypeScript/JS function signatures
    for m in re.finditer(
        r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+\s*\([^)]*\)(?:\s*:\s*[^\n{]+)?)',
        ctx, re.MULTILINE
    ):
        candidates.append(m.group(1).strip())

    # TypeScript interface method signatures
    for m in re.finditer(
        r'^\s+(\w+\s*\([^)]*\)\s*:\s*[^\n;]+)',
        ctx, re.MULTILINE
    ):
        sig = m.group(1).strip()
        if len(sig) < 120:
            candidates.append(sig)

    # Class definitions (Python + TS)
    for m in re.finditer(
        r'^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+(?:\([^)]*\)|\s+extends\s+\w+)?)',
        ctx, re.MULTILINE
    ):
        candidates.append(m.group(1).strip())

    # Interface definitions (TS)
    for m in re.finditer(
        r'^\s*(?:export\s+)?interface\s+(\w+(?:\s+extends\s+\w+)?)',
        ctx, re.MULTILINE
    ):
        candidates.append(m.group(1).strip())

    # Imports
    for m in re.finditer(
        r'^(?:from\s+[\w.]+\s+import\s+.+|import\s+.+)',
        ctx, re.MULTILINE
    ):
        candidates.append(m.group(0).strip())

    # TypeScript imports
    for m in re.finditer(
        r"^import\s+\{[^}]+\}\s+from\s+'[^']+'",
        ctx, re.MULTILINE
    ):
        candidates.append(m.group(0).strip())

    # Constants: plain assignment (MAX_FOO = value) or typed (MAX_FOO: type = value)
    for m in re.finditer(
        r'^\s*(?:export\s+)?(?:const\s+)?([A-Z_]{2,}(?::\s*\w+)?\s*=\s*\S+)',
        ctx, re.MULTILINE
    ):
        candidates.append(m.group(1).strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        key = c.strip()
        if key and key.lower() not in seen:
            seen.add(key.lower())
            unique.append(key)

    if not unique:
        return ""

    # BM25 rank against question
    def tokenize(text: str) -> list[str]:
        # Split on underscores so MAX_USERS -> ["max", "users"] matches query tokens
        raw = re.findall(r'[a-zA-Z_]\w*|\d+', text.lower())
        expanded: list[str] = []
        for tok in raw:
            parts = [p for p in tok.split('_') if p]
            expanded.extend(parts if len(parts) > 1 else [tok])
        return expanded

    tokenized_corpus = [tokenize(c) for c in unique]
    tokenized_query = tokenize(question)

    if tokenized_query:
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        )
    else:
        top_indices = list(range(min(top_k, len(unique))))

    selected = [unique[i] for i in top_indices]
    block = "; ".join(selected)
    if len(block) > max_chars:
        block = block[:max_chars].rsplit(";", 1)[0] + "..."

    return f"Facts: {block}"


def extract_facts_bm25_double_seq(
    ctx: str, question: str, top_k: int = 3, max_chars: int = 400
) -> str:
    """
    BM25 double-hop sequential h(D).

    Hop 1: Score all sentences against `question` via BM25 → select top-1
           pivot passage P1.
    Hop 2: Use P1 *as the new query* → score sentences again → select top-(k-1)
           additional passages.

    Intuition: P1 names an intermediate concept that is lexically distant from
    the original question but logically adjacent (e.g., the function a caller
    invokes, or a type returned by an intermediary).  Using P1 as the hop-2
    query pulls in the passage that *defines or elaborates* on that concept,
    rather than the one that merely mentions the question terms.

    All selected sentences are re-ordered by their original document position
    for coherence before returning.

    Returns "Facts: <sentences>" or "" if ctx is empty.
    """
    from rank_bm25 import BM25Okapi
    import numpy as np

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', ctx) if s.strip()]
    if not sentences:
        return ""

    def tokenize(text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    tokenized_corpus = [tokenize(s) for s in sentences]
    bm25 = BM25Okapi(tokenized_corpus)

    tq = tokenize(question)
    if not tq:
        return extract_facts_bm25(ctx, question, top_k=top_k, max_chars=max_chars)

    # Hop 1: retrieve top-1 pivot using the original question
    scores_1 = bm25.get_scores(tq)
    pivot_idx = int(np.argmax(scores_1))
    pivot = sentences[pivot_idx]

    selected: set[int] = {pivot_idx}

    # Hop 2: score using pivot as the new query; exclude the pivot itself
    if top_k > 1:
        tq2 = tokenize(pivot)
        if tq2:
            scores_2 = bm25.get_scores(tq2)
            scores_2[pivot_idx] = -1.0  # don't re-select the pivot
            hop2 = sorted(
                range(len(scores_2)), key=lambda i: scores_2[i], reverse=True
            )[: top_k - 1]
            selected.update(hop2)

    ordered = sorted(selected)
    block = " ".join(sentences[i] for i in ordered)
    if len(block) > max_chars:
        block = block[:max_chars].rsplit(" ", 1)[0] + "..."
    return f"Facts: {block}"


def extract_facts_bm25_double_entity(
    ctx: str, question: str, top_k: int = 3, max_chars: int = 400
) -> str:
    """
    BM25 double-hop entity-expanded h(D).

    Hop 1: Score all sentences against `question` via BM25 → select top-1
           pivot passage P1.
    Bridge: Extract *entity tokens* from P1 — code identifiers (snake_case,
            CamelCase) or capitalized proper nouns.  These are the conceptual
            bridge between the question and the evidence.
    Hop 2: BM25 score sentences using the bridge tokens as the new query →
           select top-(k-1) additional passages.

    Intuition: P1 names an entity (a function, class, or person) that the
    question is *about*.  The entity token hop then finds the passage that
    *defines or characterises* that entity, rather than the one that simply
    co-occurs with question terms.  This targets the definition site — typically
    what you want for code cross-reference and type-tracing tasks.

    All selected sentences are re-ordered by original document position.

    Returns "Facts: <sentences>" or "" if ctx is empty.
    """
    from rank_bm25 import BM25Okapi
    import numpy as np

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', ctx) if s.strip()]
    if not sentences:
        return ""

    def tokenize(text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def extract_entities(text: str) -> list[str]:
        """
        Extract bridge tokens: code identifiers split on _ / CamelCase boundaries
        and capitalized proper nouns.  Returns deduplicated list of lowercase tokens.
        """
        tokens: list[str] = []
        seen: set[str] = set()

        def add(t: str) -> None:
            t = t.lower()
            if t and len(t) > 1 and t not in seen:
                seen.add(t)
                tokens.append(t)

        # snake_case identifiers: split on underscore
        for m in re.finditer(r'\b[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+\b', text):
            for part in m.group().split('_'):
                add(part)

        # CamelCase identifiers: split on case boundaries
        for m in re.finditer(r'\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b', text):
            for part in re.findall(r'[A-Z][a-z0-9]+', m.group()):
                add(part)

        # Capitalized words (proper nouns in prose)
        for m in re.finditer(r'\b[A-Z][a-z]{2,}\b', text):
            add(m.group())

        return tokens

    tokenized_corpus = [tokenize(s) for s in sentences]
    bm25 = BM25Okapi(tokenized_corpus)

    tq = tokenize(question)
    if not tq:
        return extract_facts_bm25(ctx, question, top_k=top_k, max_chars=max_chars)

    # Hop 1: retrieve top-1 pivot using the original question
    scores_1 = bm25.get_scores(tq)
    pivot_idx = int(np.argmax(scores_1))
    pivot = sentences[pivot_idx]

    selected: set[int] = {pivot_idx}

    # Hop 2: score using bridge entity tokens extracted from the pivot
    if top_k > 1:
        bridge_tokens = extract_entities(pivot)
        if bridge_tokens:
            scores_2 = bm25.get_scores(bridge_tokens)
            scores_2[pivot_idx] = -1.0  # don't re-select the pivot
            hop2 = sorted(
                range(len(scores_2)), key=lambda i: scores_2[i], reverse=True
            )[: top_k - 1]
            selected.update(hop2)

    ordered = sorted(selected)
    block = " ".join(sentences[i] for i in ordered)
    if len(block) > max_chars:
        block = block[:max_chars].rsplit(" ", 1)[0] + "..."
    return f"Facts: {block}"


def extract_facts_oracle(ctx: str, gold_answers: list[str]) -> str:
    """
    Oracle h(D): extract sentences from ctx that contain a gold answer string.

    This is an upper bound — it uses knowledge of the correct answer to
    select the most relevant sentences. Shows the ceiling of what F can achieve.
    """
    sentences = re.split(r'(?<=[.!?])\s+', ctx)
    relevant: list[str] = []

    for sent in sentences:
        for gold in gold_answers:
            if gold.lower() in sent.lower():
                relevant.append(sent.strip())
                break

    if not relevant:
        # Fallback: first 2 sentences (at least some context)
        relevant = sentences[:2]

    # Keep it short — max 3 sentences
    block = " ".join(relevant[:3])
    # Trim to ~200 chars
    if len(block) > 200:
        block = block[:200].rsplit(" ", 1)[0] + "..."
    return f"Context: {block}"


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class AblationRecord:
    model_key: str
    task: str
    example_id: str
    injection_layer: int
    n_context_words: int
    fact_mode: str          # "ner" | "oracle"
    fact_block: str         # the actual F string used

    # Token counts
    baseline_input_tokens: int
    rsce_only_input_tokens: int
    rsce_f_input_tokens: int

    # Quality
    baseline_exact_match: bool
    rsce_only_exact_match: bool
    rsce_f_exact_match: bool

    baseline_f1: float
    rsce_only_f1: float
    rsce_f_f1: float

    # Derived
    f_contribution_f1: float      # rsce_f_f1 - rsce_only_f1
    residual_gap_f1: float        # baseline_f1 - rsce_f_f1
    input_token_reduction: float  # 1 - rsce_only_input / baseline_input

    baseline_answer: str
    rsce_only_answer: str
    rsce_f_answer: str
    gold_answers: str


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

ALL_CONDITIONS = ("baseline", "rsce_only", "rsce_f")


def run_example_ablation(
    wrapper: MLXModelWrapper,
    example: BenchmarkExample,
    model_key: str,
    injection_layer: int,
    fact_mode: str,
    scale: float = 1.0,
    max_new_tokens: int = 80,
    conditions: tuple[str, ...] = ALL_CONDITIONS,
    verbose: bool = False,
) -> list[AblationRecord]:
    """
    Run 3-way ablation for one example.

    If fact_mode == "both", returns two records (one NER, one oracle).
    Otherwise returns one record.
    """
    if fact_mode == "both":
        modes = ["ner", "oracle"]
    elif fact_mode == "all":
        modes = ["ner", "bm25", "oracle"]
    else:
        modes = [fact_mode]
    records = []

    run_baseline  = "baseline"  in conditions
    run_rsce_only = "rsce_only" in conditions
    run_rsce_f    = "rsce_f"    in conditions

    # --- Condition 1: Baseline ---
    if run_baseline:
        baseline_text, n_baseline = generate_baseline_qa(
            wrapper, example.full_prompt(), max_new_tokens
        )
        baseline_grades = grade_qa(baseline_text, example.gold_answers)
    else:
        baseline_text, n_baseline = "", 0
        baseline_grades = {"exact_match": False, "f1": 0.0}

    # --- Condition 2: RSCE only ---
    if run_rsce_only or run_rsce_f:
        ctx_v, _ = extract_context_state(wrapper, example.context, injection_layer)
    else:
        ctx_v = None

    if run_rsce_only:
        rsce_only_text, n_rsce_only = generate_with_context_injection(
            wrapper, example.question_prompt(), ctx_v,
            layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
        )
        rsce_only_grades = grade_qa(rsce_only_text, example.gold_answers)
        token_metrics = compute_token_metrics(
            n_baseline, n_rsce_only,
            len(wrapper.encode(baseline_text)) if baseline_text else 0,
            len(wrapper.encode(rsce_only_text)),
        )
    else:
        rsce_only_text, n_rsce_only = "", 0
        rsce_only_grades = {"exact_match": False, "f1": 0.0}
        token_metrics = {"input_token_reduction": 0.0}

    if verbose:
        if run_baseline:
            console.print(f"    [dim]baseline:   {baseline_text.strip()[:80]}[/dim]")
        if run_rsce_only:
            console.print(f"    [dim]rsce-only:  {rsce_only_text.strip()[:80]}[/dim]")
        console.print(f"    [dim]gold:       {example.gold_answers}[/dim]")

    # --- Condition 3: RSCE + F (once per mode) ---
    for mode in modes:
        if run_rsce_f:
            if mode == "ner":
                fact_block = extract_facts_ner(example.context)
            elif mode == "bm25":
                fact_block = extract_facts_bm25(example.context, example.question)
            else:
                fact_block = extract_facts_oracle(example.context, example.gold_answers)

            rsce_f_prompt = f"{fact_block}\n{example.question_prompt()}" if fact_block else example.question_prompt()
            rsce_f_text, n_rsce_f = generate_with_context_injection(
                wrapper, rsce_f_prompt, ctx_v,
                layer=injection_layer, scale=scale, max_new_tokens=max_new_tokens,
            )
            rsce_f_grades = grade_qa(rsce_f_text, example.gold_answers)

            if verbose:
                console.print(f"    [dim]rsce+F ({mode}): {rsce_f_text.strip()[:80]}[/dim]")
                console.print(f"    [dim]fact block: {fact_block[:80]}[/dim]")
        else:
            fact_block = ""
            rsce_f_text, n_rsce_f = "", 0
            rsce_f_grades = {"exact_match": False, "f1": 0.0}

        records.append(AblationRecord(
            model_key=model_key,
            task=example.task,
            example_id=example.id,
            injection_layer=injection_layer,
            n_context_words=example.context_word_len,
            fact_mode=mode,
            fact_block=fact_block[:200],
            baseline_input_tokens=n_baseline,
            rsce_only_input_tokens=n_rsce_only,
            rsce_f_input_tokens=n_rsce_f,
            baseline_exact_match=baseline_grades["exact_match"],
            rsce_only_exact_match=rsce_only_grades["exact_match"],
            rsce_f_exact_match=rsce_f_grades["exact_match"],
            baseline_f1=baseline_grades["f1"],
            rsce_only_f1=rsce_only_grades["f1"],
            rsce_f_f1=rsce_f_grades["f1"],
            f_contribution_f1=round(rsce_f_grades["f1"] - rsce_only_grades["f1"], 4),
            residual_gap_f1=round(baseline_grades["f1"] - rsce_f_grades["f1"], 4),
            input_token_reduction=token_metrics["input_token_reduction"],
            baseline_answer=baseline_text.strip()[:120],
            rsce_only_answer=rsce_only_text.strip()[:120],
            rsce_f_answer=rsce_f_text.strip()[:120],
            gold_answers=str(example.gold_answers),
        ))

    return records


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model_ablation(
    model_key: str,
    model_cfg: dict,
    examples: list[BenchmarkExample],
    fact_mode: str = "both",
    scale: float = 1.0,
    max_new_tokens: int = 80,
    layer_override: int | None = None,
    conditions: tuple[str, ...] = ALL_CONDITIONS,
    done_ids: set[str] | None = None,
    verbose: bool = False,
) -> list[AblationRecord]:
    hf_id = model_cfg["hf_id"]
    console.rule(f"[bold cyan]Ablation: {model_key}[/bold cyan]")
    console.print(f"  HF ID      : {hf_id}")
    console.print(f"  Fact mode  : {fact_mode}")
    console.print(f"  Conditions : {list(conditions)}")

    try:
        wrapper = load_model(hf_id)
    except Exception as exc:
        console.print(f"  [red]Load failed: {exc}[/red]")
        return []

    if layer_override is not None:
        injection_layer = layer_override
        console.print(f"  Layer f(M): {injection_layer} (override)")
    else:
        injection_layer = select_layer_heuristic(wrapper, hf_id)
        console.print(f"  Layer f(M): {injection_layer} / {wrapper.n_layers}")

    all_records: list[AblationRecord] = []
    skipped = 0

    if fact_mode == "both":
        modes = ["ner", "oracle"]
    elif fact_mode == "all":
        modes = ["ner", "bm25", "oracle"]
    else:
        modes = [fact_mode]
    for i, ex in enumerate(examples):
        # Check if all modes for this example are already done
        all_done = done_ids and all(
            f"{model_key}:{ex.id}:{mode}" in done_ids for mode in modes
        )
        if all_done:
            skipped += 1
            continue
        console.print(
            f"\n  [{i+1}/{len(examples)}] {ex.task}/{ex.id[:24]} "
            f"({ex.context_word_len}w)"
        )
        try:
            recs = run_example_ablation(
                wrapper, ex,
                model_key=model_key,
                injection_layer=injection_layer,
                fact_mode=fact_mode,
                scale=scale,
                max_new_tokens=max_new_tokens,
                conditions=conditions,
                verbose=verbose,
            )
            all_records.extend(recs)
        except Exception as exc:
            console.print(f"  [red]ERROR: {exc}[/red]")
    if skipped:
        console.print(f"  [dim]Skipped {skipped} already-done examples[/dim]")

    _print_model_ablation_summary(model_key, all_records)

    del wrapper
    gc.collect()
    clear_backend_cache()

    return all_records


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _print_model_ablation_summary(
    model_key: str, records: list[AblationRecord]
) -> None:
    if not records:
        return

    for mode in ["ner", "oracle"]:
        mode_recs = [r for r in records if r.fact_mode == mode]
        if not mode_recs:
            continue

        n = len(mode_recs)
        b_f1  = sum(r.baseline_f1 for r in mode_recs) / n
        r_f1  = sum(r.rsce_only_f1 for r in mode_recs) / n
        rf_f1 = sum(r.rsce_f_f1 for r in mode_recs) / n
        f_contrib = sum(r.f_contribution_f1 for r in mode_recs) / n
        gap = sum(r.residual_gap_f1 for r in mode_recs) / n
        red = sum(r.input_token_reduction for r in mode_recs) / n

        b_em  = sum(1 for r in mode_recs if r.baseline_exact_match) / n
        r_em  = sum(1 for r in mode_recs if r.rsce_only_exact_match) / n
        rf_em = sum(1 for r in mode_recs if r.rsce_f_exact_match) / n

        contrib_color = "green" if f_contrib > 0 else "yellow" if f_contrib == 0 else "red"
        gap_color = "green" if gap < 0.05 else "yellow" if gap < 0.15 else "red"

        console.rule(f"[bold]{model_key} | fact_mode={mode}[/bold]")
        console.print(f"  N examples         : {n}")
        console.print(f"  Input token reduction: {red:.1%}")
        console.print(f"  Baseline     EM/F1 : {b_em:.1%} / {b_f1:.3f}")
        console.print(f"  RSCE only    EM/F1 : {r_em:.1%} / {r_f1:.3f}")
        console.print(f"  RSCE + F     EM/F1 : {rf_em:.1%} / {rf_f1:.3f}")
        console.print(
            f"  F contribution F1  : [{contrib_color}]{f_contrib:+.3f}[/{contrib_color}]"
            f"  (RSCE+F minus RSCE-only)"
        )
        console.print(
            f"  Residual gap   F1  : [{gap_color}]{gap:+.3f}[/{gap_color}]"
            f"  (baseline minus RSCE+F)"
        )


def print_ablation_summary(all_records: list[AblationRecord]) -> None:
    from collections import defaultdict
    console.rule("[bold magenta]ABLATION SUMMARY — F Contribution[/bold magenta]")

    # Group by (model, fact_mode)
    groups: dict[tuple[str, str], list[AblationRecord]] = defaultdict(list)
    for r in all_records:
        groups[(r.model_key, r.fact_mode)].append(r)

    table = Table(
        title="RSCE 3-Way Ablation: Baseline vs RSCE-only vs RSCE+F",
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    table.add_column("Model", style="cyan")
    table.add_column("F mode")
    table.add_column("N", justify="right")
    table.add_column("BL-F1", justify="right")
    table.add_column("C-only F1", justify="right")
    table.add_column("C+F F1", justify="right")
    table.add_column("F contrib", justify="right")
    table.add_column("Gap vs BL", justify="right")
    table.add_column("Tok saved", justify="right")

    for (model_key, mode), recs in sorted(groups.items()):
        n = len(recs)
        b_f1  = sum(r.baseline_f1 for r in recs) / n
        r_f1  = sum(r.rsce_only_f1 for r in recs) / n
        rf_f1 = sum(r.rsce_f_f1 for r in recs) / n
        f_contrib = rf_f1 - r_f1
        gap = b_f1 - rf_f1
        red = sum(r.input_token_reduction for r in recs) / n

        contrib_color = "green" if f_contrib > 0.01 else "yellow" if f_contrib >= 0 else "red"
        gap_color = "green" if gap < 0.05 else "yellow" if gap < 0.15 else "red"

        table.add_row(
            model_key, mode, str(n),
            f"{b_f1:.3f}",
            f"{r_f1:.3f}",
            f"{rf_f1:.3f}",
            f"[{contrib_color}]{f_contrib:+.3f}[/{contrib_color}]",
            f"[{gap_color}]{gap:+.3f}[/{gap_color}]",
            f"{red:.1%}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_ablation(records: list[AblationRecord], run_id: str) -> None:
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
        description="RSCE Ablation: measure F contribution to quality"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_MATRIX.keys()),
        choices=list(MODEL_MATRIX.keys()),
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["hotpotqa", "2wikimqa"],
        choices=["hotpotqa", "2wikimqa"],
    )
    parser.add_argument("--n", type=int, default=10, help="Examples per task (default: 10)")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=80, dest="max_tokens")
    parser.add_argument(
        "--fact-mode", choices=["ner", "bm25", "oracle", "both", "all"], default="both",
        dest="fact_mode",
        help=(
            "ner: regex NER extraction (practical); "
            "bm25: question-conditioned sentence selection (practical); "
            "oracle: gold-answer-guided extraction (upper bound); "
            "both: run ner + oracle (legacy default); "
            "all: run all three modes"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--run-id", default=None, dest="run_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=None,
                        help="Override injection layer for all models (skips heuristic)")
    parser.add_argument(
        "--condition", nargs="+", default=list(ALL_CONDITIONS),
        choices=list(ALL_CONDITIONS), dest="conditions",
        help="Which conditions to run (default: all three)",
    )
    parser.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help="Resume a previous run: load its CSV and skip already-completed examples",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"ablation_{int(time.time())}"

    # Load already-done keys from a prior run
    done_ids: set[str] = set()
    if args.resume:
        resume_path = RESULTS_DIR / f"{args.resume}.csv"
        if not resume_path.exists():
            resume_path = RESULTS_DIR / f"{args.resume}_partial.csv"
        if resume_path.exists():
            with open(resume_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_ids.add(f"{row['model_key']}:{row['example_id']}:{row['fact_mode']}")
            console.print(f"  [dim]Resuming from {resume_path.name}: {len(done_ids)} examples already done[/dim]")
            run_id = args.resume
        else:
            console.print(f"  [yellow]Resume file not found: {resume_path}[/yellow]")

    console.rule("[bold magenta]RSCE Ablation Benchmark — Part 2[/bold magenta]")
    console.print(f"  Models    : {args.models}")
    console.print(f"  Tasks     : {args.tasks}")
    console.print(f"  N/task    : {args.n}")
    console.print(f"  Fact mode : {args.fact_mode}")
    console.print(f"  Run ID    : {run_id}")
    console.print()
    console.print("  Conditions:")
    console.print("    1. Baseline  — [ctx + P]     full context in token stream")
    console.print("    2. RSCE only — [C + P]       vector injection, no fact block")
    console.print("    3. RSCE + F  — [C + F + P]   vector injection + fact block")

    examples = load_benchmark(
        tasks=tuple(args.tasks),
        n_per_task=args.n,
        seed=args.seed,
    )
    console.print(f"\n  Total examples: {len(examples)}")

    all_records: list[AblationRecord] = []

    for model_key in args.models:
        recs = run_model_ablation(
            model_key=model_key,
            model_cfg=MODEL_MATRIX[model_key],
            examples=examples,
            fact_mode=args.fact_mode,
            scale=args.scale,
            max_new_tokens=args.max_tokens,
            layer_override=args.layer,
            conditions=tuple(args.conditions),
            done_ids=done_ids,
            verbose=args.verbose,
        )
        all_records.extend(recs)
        # Checkpoint after each model
        save_ablation(all_records, f"{run_id}_partial")

    print_ablation_summary(all_records)
    save_ablation(all_records, run_id)

    return 0 if all_records else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
