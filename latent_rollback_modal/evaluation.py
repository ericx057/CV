"""
Evaluation and result grading for the Latent Rollback experiment.

Outcome classification (per spec):
  MASSIVE_SUCCESS  — rollback output contains "5432" (expected_a)
  PARTIAL_SUCCESS  — output contains "8080" (expected_b) but not "5432"
  FAILURE          — output contains neither (gibberish / broken manifold)

Also provides pretty-printing via rich and a summary dict for ablation logging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Container for all results of one full experiment run."""
    model_name: str
    device: str
    dtype: str
    extraction_layer: int
    rollback_scale: float

    state_a_text: str = ""
    state_b_text: str = ""
    rollback_text: str = ""

    state_a_outcome: str = ""
    state_b_outcome: str = ""
    rollback_outcome: str = ""

    delta_norm: float = 0.0
    v_a_norm: float = 0.0
    v_b_norm: float = 0.0

    ablation_rows: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().replace(",", "").replace(".", "")


def grade_output(
    generated_text: str,
    expected_a: str = "5432",
    expected_b: str = "8080",
) -> tuple[bool, bool, str]:
    """
    Returns (contains_a, contains_b, outcome).
    outcome: "MASSIVE_SUCCESS" | "PARTIAL_SUCCESS" | "FAILURE"
    """
    norm = _normalize(generated_text)

    contains_a = expected_a in norm
    contains_b = expected_b in norm

    if contains_a and not contains_b:
        outcome = "MASSIVE_SUCCESS"
    elif contains_a and contains_b:
        # Both present — model is hedging; partial credit
        outcome = "PARTIAL_SUCCESS"
    elif contains_b:
        outcome = "PARTIAL_SUCCESS"
    else:
        outcome = "FAILURE"

    return contains_a, contains_b, outcome


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

OUTCOME_COLORS = {
    "MASSIVE_SUCCESS": "bold green",
    "PARTIAL_SUCCESS": "yellow",
    "FAILURE": "red",
}

OUTCOME_VERDICT_TEXT = {
    "MASSIVE_SUCCESS": "[bold green]MASSIVE SUCCESS — Latent Time Travel achieved![/bold green]",
    "PARTIAL_SUCCESS": "[yellow]PARTIAL SUCCESS — delta was not strong enough (or wrong layer).[/yellow]",
    "FAILURE": "[red]FAILURE — subtraction broke the non-linear manifold.[/red]",
}


def print_step(
    label: str,
    prompt: str,
    generated_text: str,
    outcome: str,
    extra_lines: Optional[list[str]] = None,
) -> None:
    color = OUTCOME_COLORS.get(outcome, "white")
    header = f"[bold]{label}[/bold]  [{color}]{outcome}[/{color}]"
    body = (
        f"[dim]Prompt:[/dim] {prompt[:80]}{'...' if len(prompt)>80 else ''}\n"
        f"[dim]Output:[/dim] {generated_text.strip()}"
    )
    if extra_lines:
        body += "\n" + "\n".join(extra_lines)

    console.print(Panel(body, title=header, border_style=color))


def print_vector_stats(v_a_norm: float, v_b_norm: float, delta_norm: float) -> None:
    table = Table(title="Residual Vector Stats", box=box.SIMPLE)
    table.add_column("Vector", style="cyan")
    table.add_column("L2 Norm", justify="right")
    table.add_row("v_A", f"{v_a_norm:.4f}")
    table.add_row("v_B", f"{v_b_norm:.4f}")
    table.add_row("delta = v_B - v_A", f"{delta_norm:.4f}")
    console.print(table)


def print_rollback_verdict(outcome: str) -> None:
    console.print()
    console.rule("[bold]ROLLBACK VERDICT[/bold]")
    console.print(OUTCOME_VERDICT_TEXT.get(outcome, outcome))
    console.rule()


def print_ablation_table(rows: list[dict]) -> None:
    """Print a layer x scale ablation matrix."""
    if not rows:
        return
    table = Table(title="Ablation: Layer x Scale", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Layer", style="cyan", justify="right")
    table.add_column("Scale", justify="right")
    table.add_column("Outcome", justify="center")
    table.add_column("Output snippet", max_width=50)

    for row in rows:
        color = OUTCOME_COLORS.get(row.get("outcome", ""), "white")
        table.add_row(
            str(row.get("layer", "")),
            str(row.get("scale", "")),
            f"[{color}]{row.get('outcome', '')}[/{color}]",
            row.get("snippet", "")[:50],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_result(result: ExperimentResult, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(asdict(result), f, indent=2)
    console.print(f"[dim]Results saved to {path}[/dim]")
