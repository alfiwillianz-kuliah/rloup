"""Phase 3 evaluation suite for rloup.

``rloup.eval.evaluate``
    Round-robin tournament with Wilson confidence intervals and Bradley-Terry
    Elo, 4-player scaling tables, behavioural diagnostics, the controlled
    impossible-claim card-counting probe, approximate exploitability and
    snapshot Elo progression.  Writes ``results/eval_summary.json``.
``rloup.eval.plots``
    Publication-grade figures (180 DPI) from ``results/*/metrics.jsonl`` and
    ``results/eval_summary.json`` into ``results/figures/``.

Both modules are runnable::

    python -m rloup.eval.evaluate
    python -m rloup.eval.plots

Names are re-exported lazily so that importing the package never pulls in torch
or matplotlib, and so ``python -m rloup.eval.evaluate`` works without the
"found in sys.modules" warning.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "COIN_BRACKETS",
    "Competitor",
    "ProbeState",
    "bradley_terry_elo",
    "build_impossible_claim_state",
    "coin_bracket",
    "collect_behaviour",
    "evaluate_tables",
    "exploitability",
    "impossible_claim_probe",
    "model_competitors",
    "round_robin_2p",
    "run_games",
    "wilson_interval",
    "wilson_record",
]

_LAZY = {
    "COIN_BRACKETS",
    "Competitor",
    "ProbeState",
    "bradley_terry_elo",
    "build_impossible_claim_state",
    "coin_bracket",
    "collect_behaviour",
    "evaluate_tables",
    "exploitability",
    "impossible_claim_probe",
    "model_competitors",
    "round_robin_2p",
    "run_games",
    "wilson_interval",
    "wilson_record",
}


def __getattr__(name: str) -> Any:
    """Lazy re-export of the evaluation helpers (they pull in torch)."""
    if name in _LAZY:
        from . import evaluate

        return getattr(evaluate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
