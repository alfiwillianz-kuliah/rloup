"""Structured metric logging shared by the PPO and NFSP trainers.

Every trainer appends one JSON object per line to ``<save_dir>/metrics.jsonl``
with exactly the same keys, so the two algorithms can be compared and plotted by
the same tooling::

    step, update, episodes, wall_time, steps_per_second,
    avg_vs_baselines, strong_avg,
    win_rate_random, win_rate_honest, win_rate_aggressive_bluffer,
    win_rate_cautious_skeptical, win_rate_card_counter,
    bluff_rate, challenge_rate, challenge_precision,
    impossible_claim_challenge_rate,
    policy_loss, value_loss, entropy, approx_kl, clip_fraction,
    learner_elo, snapshot_count
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["METRIC_KEYS", "MetricsWriter"]

METRIC_KEYS: tuple[str, ...] = (
    "step",
    "update",
    "episodes",
    "wall_time",
    "steps_per_second",
    "avg_vs_baselines",
    "strong_avg",
    "win_rate_random",
    "win_rate_honest",
    "win_rate_aggressive_bluffer",
    "win_rate_cautious_skeptical",
    "win_rate_card_counter",
    "bluff_rate",
    "challenge_rate",
    "challenge_precision",
    "impossible_claim_challenge_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "learner_elo",
    "snapshot_count",
)


class MetricsWriter:
    """Append-only JSONL writer with the shared metric schema."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        payload = {key: record.get(key) for key in METRIC_KEYS}
        for key, value in record.items():
            payload.setdefault(key, value)
        self._file.write(json.dumps(payload, sort_keys=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
