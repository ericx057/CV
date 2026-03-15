"""
Benchmark dataset loader for cross-model Latent Rollback evaluation.

Pulls subsets from LongBench (THUDM/LongBench on HuggingFace):
  hotpotqa        — Multi-document QA, answer requires combining 2 docs
  2wikimqa        — Two-hop reasoning across Wikipedia articles

Each returned example:
  {
    "id":               str,
    "task":             str,           # "hotpotqa" | "2wikimqa"
    "context":          str,           # long multi-document context
    "question":         str,           # the query
    "gold_answers":     list[str],     # acceptable answers
    "context_word_len": int,
    "question_word_len": int,
  }

Fallback: if `datasets` library is unavailable, a small hardcoded set is returned.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

LONGBENCH_REPO = "THUDM/LongBench"
SUPPORTED_TASKS = ("hotpotqa", "2wikimqa", "code_qa")
CACHE_DIR = Path(__file__).parent / ".benchmark_cache"


@dataclass
class BenchmarkExample:
    id: str
    task: str
    context: str
    question: str
    gold_answers: list[str]
    context_word_len: int
    question_word_len: int

    def full_prompt(self, system_prefix: str = "") -> str:
        """Compose baseline prompt: context + question."""
        parts = []
        if system_prefix:
            parts.append(system_prefix)
        parts.append(f"Context:\n{self.context}")
        parts.append(f"\nQuestion: {self.question}")
        parts.append("\nAnswer:")
        return "\n".join(parts)

    def question_prompt(self, system_prefix: str = "") -> str:
        """Question-only prompt used with latent injection."""
        parts = []
        if system_prefix:
            parts.append(system_prefix)
        parts.append(f"Question: {self.question}")
        parts.append("\nAnswer:")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_benchmark(
    tasks: tuple[str, ...] = SUPPORTED_TASKS,
    n_per_task: int = 10,
    seed: int = 42,
    min_context_words: int = 200,
    max_context_words: int = 4000,
    cache: bool = True,
) -> list[BenchmarkExample]:
    """
    Load n_per_task examples for each task.

    Filters to examples where context is between min/max word counts so that:
    - Context is long enough to be meaningful
    - Context fits in model memory without crashing
    """
    examples: list[BenchmarkExample] = []

    for task in tasks:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unknown task {task!r}. Supported: {SUPPORTED_TASKS}")

        task_examples = _load_task(task, seed=seed, cache=cache)
        filtered = [
            e for e in task_examples
            if min_context_words <= e.context_word_len <= max_context_words
        ]

        rng = random.Random(seed)
        selected = rng.sample(filtered, min(n_per_task, len(filtered)))
        examples.extend(selected)
        console.print(
            f"  [{task}] loaded {len(selected)} examples "
            f"(filtered from {len(task_examples)})"
        )

    return examples


def _load_task(task: str, seed: int = 42, cache: bool = True) -> list[BenchmarkExample]:
    """Try HuggingFace datasets first, then cached JSON, then hardcoded fallback."""
    cache_path = CACHE_DIR / f"{task}.json"

    # Try cache
    if cache and cache_path.exists():
        return _load_cache(cache_path, task)

    # Try HuggingFace datasets
    try:
        return _load_from_hf(task, cache_path)
    except Exception as exc:
        console.print(f"  [yellow]HF load failed for {task}: {exc}[/yellow]")
        console.print("  [yellow]Falling back to hardcoded examples.[/yellow]")
        return _hardcoded_fallback(task)


def _load_from_hf(task: str, cache_path: Path) -> list[BenchmarkExample]:
    """Download from THUDM/LongBench using the datasets library."""
    from datasets import load_dataset  # type: ignore

    console.print(f"  Downloading {LONGBENCH_REPO} / {task} from HuggingFace...")
    ds = load_dataset(LONGBENCH_REPO, task, split="test")

    examples = []
    for i, row in enumerate(ds):
        context = row.get("context", "")
        question = row.get("input", "")
        answers = row.get("answers", [])

        if isinstance(answers, str):
            answers = [answers]

        ex = BenchmarkExample(
            id=row.get("_id", str(i)),
            task=task,
            context=context,
            question=question,
            gold_answers=answers,
            context_word_len=len(context.split()),
            question_word_len=len(question.split()),
        )
        examples.append(ex)

    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            [
                {
                    "id": e.id, "task": e.task, "context": e.context,
                    "question": e.question, "gold_answers": e.gold_answers,
                    "context_word_len": e.context_word_len,
                    "question_word_len": e.question_word_len,
                }
                for e in examples
            ],
            f,
            indent=2,
        )
    console.print(f"  Cached {len(examples)} examples to {cache_path}")
    return examples


def _load_cache(cache_path: Path, task: str) -> list[BenchmarkExample]:
    with open(cache_path) as f:
        rows = json.load(f)
    return [
        BenchmarkExample(
            id=r["id"], task=task,
            context=r["context"], question=r["question"],
            gold_answers=r["gold_answers"],
            context_word_len=r["context_word_len"],
            question_word_len=r["question_word_len"],
        )
        for r in rows
    ]


def _hardcoded_fallback(task: str) -> list[BenchmarkExample]:
    """Minimal hardcoded examples for offline testing."""
    examples = [
        BenchmarkExample(
            id=f"{task}_fallback_0",
            task=task,
            context=(
                "Document 1: The Eiffel Tower is located in Paris, France. "
                "It was completed in 1889 and stands 330 meters tall. "
                "It was designed by Gustave Eiffel for the 1889 World's Fair.\n\n"
                "Document 2: Paris is the capital city of France. "
                "It is situated on the Seine river in north-central France. "
                "The city has a population of about 2.1 million people."
            ),
            question="In which country is the Eiffel Tower located?",
            gold_answers=["France"],
            context_word_len=80,
            question_word_len=10,
        ),
        BenchmarkExample(
            id=f"{task}_fallback_1",
            task=task,
            context=(
                "Document 1: Marie Curie was born in Warsaw, Poland in 1867. "
                "She later moved to Paris to study at the Sorbonne university. "
                "She became the first woman to win a Nobel Prize.\n\n"
                "Document 2: The Nobel Prize in Physics was awarded to Marie Curie "
                "in 1903 for her research on radiation. She also won the Nobel Prize "
                "in Chemistry in 1911, making her the first person to win Nobel Prizes "
                "in two different sciences."
            ),
            question="How many Nobel Prizes did Marie Curie win?",
            gold_answers=["two", "2"],
            context_word_len=90,
            question_word_len=10,
        ),
        BenchmarkExample(
            id=f"{task}_fallback_2",
            task=task,
            context=(
                "Document 1: The Amazon River is the largest river in the world by "
                "discharge volume. It flows through South America, primarily Brazil. "
                "The river is approximately 6,400 km long.\n\n"
                "Document 2: Brazil is the largest country in South America and the "
                "fifth largest in the world. Its capital city is Brasilia, but the "
                "largest city is Sao Paulo. The country borders all South American "
                "nations except Chile and Ecuador."
            ),
            question="Through which continent does the Amazon River flow?",
            gold_answers=["South America"],
            context_word_len=85,
            question_word_len=9,
        ),
    ]
    return examples


# ---------------------------------------------------------------------------
# Code QA benchmark (synthetic)
# ---------------------------------------------------------------------------

_CODE_EXAMPLES: list[dict] = [
    {
        "id": "code_qa_0",
        "question": "What does find_by_id return?",
        "gold_answers": ["Optional[User]"],
        "context": """
from typing import Optional
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str

@dataclass
class UserDelta:
    name: str
    email: str

class UserRepository:
    def find_by_id(self, user_id: int) -> Optional[User]:
        ...

    def update_user(self, user_id: int, delta: UserDelta) -> bool:
        ...

    def delete_user(self, user_id: int) -> bool:
        ...

def create_user(name: str, email: str) -> User:
    ...

MAX_USERS = 1000
""",
    },
    {
        "id": "code_qa_1",
        "question": "What parameter type does update_user take as delta?",
        "gold_answers": ["UserDelta"],
        "context": """
from typing import Optional
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

class UserRepository:
    def find_by_id(self, user_id: int) -> Optional[User]:
        ...

    def update_user(self, user_id: int, delta: UserDelta) -> bool:
        ...
""",
    },
    {
        "id": "code_qa_2",
        "question": "What is the value of MAX_RETRIES?",
        "gold_answers": ["3"],
        "context": """
import time
from typing import Callable

MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
BASE_URL = "https://api.example.com"

def retry(fn: Callable, retries: int = MAX_RETRIES) -> bool:
    for _ in range(retries):
        try:
            fn()
            return True
        except Exception:
            time.sleep(1)
    return False

def fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    ...
""",
    },
    {
        "id": "code_qa_3",
        "question": "What does create_user return?",
        "gold_answers": ["User", "dict"],
        "context": """
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str

def create_user(name: str, email: str) -> User:
    return User(id=0, name=name, email=email)

def delete_user(user_id: int) -> bool:
    ...

def list_users() -> list[User]:
    ...
""",
    },
    {
        "id": "code_qa_4",
        "question": "What is imported from typing?",
        "gold_answers": ["Optional", "Optional, List"],
        "context": """
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class Post:
    id: int
    title: str
    body: str
    author_id: int

class PostRepository:
    def find_by_author(self, author_id: int) -> List[Post]:
        ...

    def find_by_id(self, post_id: int) -> Optional[Post]:
        ...

    def create_post(self, title: str, body: str, author_id: int) -> Post:
        ...
""",
    },
    {
        "id": "code_qa_5",
        "question": "What argument does patch_user_email take as repo?",
        "gold_answers": ["UserRepository"],
        "context": """
from typing import Optional
from repository import UserRepository
from types_module import UserDelta, User

def patch_user_email(repo: UserRepository, user_id: int, new_email: str) -> bool:
    delta = UserDelta(name=None, email=new_email)
    return repo.update_user(user_id, delta)

def patch_user_name(repo: UserRepository, user_id: int, new_name: str) -> bool:
    delta = UserDelta(name=new_name, email=None)
    return repo.update_user(user_id, delta)

def deactivate_user(repo: UserRepository, user_id: int) -> bool:
    ...
""",
    },
    {
        "id": "code_qa_6",
        "question": "What does list_users return?",
        "gold_answers": ["list[User]"],
        "context": """
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    id: int
    name: str
    active: bool = True

class UserService:
    def get_user(self, user_id: int) -> Optional[User]:
        ...

    def list_users(self) -> list[User]:
        ...

    def deactivate(self, user_id: int) -> bool:
        ...

    def count_active(self) -> int:
        ...
""",
    },
    {
        "id": "code_qa_7",
        "question": "What class does UserRepository extend?",
        "gold_answers": ["Repository"],
        "context": """
from base import Repository
from typing import Optional

class User:
    id: int
    name: str

class UserRepository(Repository):
    def find_by_id(self, user_id: int) -> Optional[User]:
        ...

    def save(self, user: User) -> bool:
        ...

class PostRepository(Repository):
    def find_recent(self, limit: int) -> list:
        ...
""",
    },
    {
        "id": "code_qa_8",
        "question": "What does the fetch function return?",
        "gold_answers": ["dict"],
        "context": """
import json
import urllib.request
from typing import Optional

DEFAULT_TIMEOUT = 30

def fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())

def fetch_optional(url: str) -> Optional[dict]:
    try:
        return fetch(url)
    except Exception:
        return None

def post(url: str, body: dict) -> dict:
    ...
""",
    },
    {
        "id": "code_qa_9",
        "question": "What does count_active return?",
        "gold_answers": ["int"],
        "context": """
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    active: bool

def count_active(users: list[User]) -> int:
    return sum(1 for u in users if u.active)

def count_inactive(users: list[User]) -> int:
    return sum(1 for u in users if not u.active)

def filter_active(users: list[User]) -> list[User]:
    return [u for u in users if u.active]
""",
    },
]


def load_code_benchmark(
    n: int = len(_CODE_EXAMPLES),
    seed: int = 42,
) -> list[BenchmarkExample]:
    """
    Return synthetic code QA examples for benchmarking the code-specific F block.

    Questions are answerable from function signatures, return types, parameter
    types, imports, and constants — the exact content a code F block would carry.
    """
    import random
    rng = random.Random(seed)
    pool = list(_CODE_EXAMPLES)
    rng.shuffle(pool)
    selected = pool[:min(n, len(pool))]

    return [
        BenchmarkExample(
            id=ex["id"],
            task="code_qa",
            context=ex["context"].strip(),
            question=ex["question"],
            gold_answers=ex["gold_answers"],
            context_word_len=len(ex["context"].split()),
            question_word_len=len(ex["question"].split()),
        )
        for ex in selected
    ]


# ---------------------------------------------------------------------------
# QA grading
# ---------------------------------------------------------------------------

def grade_code_qa(generated_text: str, gold_answers: list[str]) -> dict:
    """
    Grade generated text against gold answers for code QA.

    Case-sensitive unlike grade_qa — code identifiers are case-sensitive.
    Otherwise identical scoring: exact containment + token F1.
    """
    import re
    gen_norm = generated_text.strip()
    gen_tokens = set(re.findall(r'[a-zA-Z_]\w*|\d+', gen_norm))

    best_f1 = 0.0
    best_answer = ""
    exact = False

    for gold in gold_answers:
        gold_norm = gold.strip()
        if gold_norm in gen_norm:
            exact = True

        gold_tokens = set(re.findall(r'[a-zA-Z_]\w*|\d+', gold_norm))
        if not gold_tokens or not gen_tokens:
            f1 = 0.0
        else:
            precision = len(gen_tokens & gold_tokens) / len(gen_tokens)
            recall = len(gen_tokens & gold_tokens) / len(gold_tokens)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_answer = gold

    return {
        "exact_match": exact,
        "f1": round(best_f1, 4),
        "best_answer": best_answer,
    }


def grade_qa(generated_text: str, gold_answers: list[str]) -> dict:
    """
    Grade generated text against gold answers.

    Returns:
      exact_match  : bool  — generated contains exact gold answer (case-insensitive)
      f1           : float — token F1 between best matching gold and generated
      best_answer  : str   — which gold answer matched best
    """
    import re
    gen_norm = generated_text.lower().strip()
    gen_tokens = set(re.sub(r"[^\w\s]", "", gen_norm).split())

    best_f1 = 0.0
    best_answer = ""
    exact = False

    for gold in gold_answers:
        gold_norm = gold.lower().strip()
        if gold_norm in gen_norm:
            exact = True

        gold_tokens = set(re.sub(r"[^\w\s]", "", gold_norm).split())
        if not gold_tokens or not gen_tokens:
            f1 = 0.0
        else:
            precision = len(gen_tokens & gold_tokens) / len(gen_tokens)
            recall = len(gen_tokens & gold_tokens) / len(gold_tokens)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_answer = gold

    return {
        "exact_match": exact,
        "f1": round(best_f1, 4),
        "best_answer": best_answer,
    }
