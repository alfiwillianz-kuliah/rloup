"""Opponent pool for Prioritised Fictitious Self-Play in Coup.

The pool mixes three opponent sources, which is what makes PFSP work in a
partially-observed bluffing game:

``latest`` (25%)
    The learner's own current weights.  Keeps the ladder anchored at the
    frontier so the policy is never only training against stale copies.
``snapshot`` (60%)
    Historical frozen checkpoints, sampled with PFSP weights
    ``w_i = (1 - p_i)^2 + 0.05`` where ``p_i`` is the EMA of the learner's win
    rate against that snapshot.  Squaring concentrates effort on opponents the
    learner currently loses to, while the ``+0.05`` floor keeps every snapshot
    reachable so no forgotten exploiter gets a free ride.
``scripted`` (15%)
    The five fixed baselines.  Grounds the pool against degenerate strategies
    that self-play alone would never punish.

Before any snapshot exists the sampler falls back cleanly to
``85% latest / 15% scripted`` so training can start from step zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from rloup.agents import BASELINES, Agent
from rloup.agents.ppo import MaskedActorCritic

__all__ = [
    "OpponentSpec",
    "Snapshot",
    "OpponentPool",
    "update_elo",
    "PRIOR_WIN_RATE",
]

#: Prior win rate of the learner against any unseen opponent.
PRIOR_WIN_RATE = 0.5


def update_elo(ra: float, rb: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    """Standard Elo update.  ``score_a`` is 1.0 (win), 0.5 (draw) or 0.0 (loss)."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    delta = float(k) * (float(score_a) - expected_a)
    return ra + delta, rb - delta


@dataclass(frozen=True)
class OpponentSpec:
    """A sampled opponent: the current learner, a snapshot, or a scripted bot."""

    kind: str  # 'latest' | 'snapshot' | 'scripted'
    key: str
    index: int | None = None
    name: str | None = None

    @property
    def label(self) -> str:
        if self.kind == "latest":
            return "latest"
        if self.kind == "scripted":
            return f"scripted:{self.name}"
        return self.key


@dataclass
class Snapshot:
    """One frozen historical policy."""

    index: int
    key: str
    step: int
    elo: float
    module: MaskedActorCritic
    win_rate: float = PRIOR_WIN_RATE


class OpponentPool:
    """Historical snapshots + scripted baselines with PFSP sampling and EMA tracking."""

    LATEST_PROB = 0.25
    SNAPSHOT_PROB = 0.60
    SCRIPTED_PROB = 0.15
    WEIGHT_FLOOR = 0.05

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        device: str | torch.device = "cpu",
        max_snapshots: int = 40,
        hidden: int = 256,
        num_layers: int = 3,
        ema_alpha: float = 0.05,
        seed: int | None = None,
        scripted_classes: Sequence[type[Agent]] = BASELINES,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.device = torch.device(device)
        self.max_snapshots = int(max_snapshots)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.ema_alpha = float(ema_alpha)
        self.rng = np.random.default_rng(seed)
        self.scripted_classes = tuple(scripted_classes)
        self.snapshots: list[Snapshot] = []
        self._by_index: dict[int, Snapshot] = {}
        self._win_rates: dict[str, float] = {"latest": PRIOR_WIN_RATE}
        for cls in self.scripted_classes:
            self._win_rates[f"scripted:{cls.name}"] = PRIOR_WIN_RATE
        self.learner_elo = 1000.0
        self._next_index = 0

    # -- snapshots ------------------------------------------------------ #
    def add_snapshot(
        self,
        actor_critic: MaskedActorCritic,
        step: int = 0,
        elo: float | None = None,
        key: str | None = None,
    ) -> Snapshot:
        """Freeze a deep copy of ``actor_critic`` into the pool."""
        index = self._next_index
        self._next_index += 1
        module = MaskedActorCritic(
            self.obs_dim, self.n_actions, self.hidden, self.num_layers
        ).to(self.device)
        module.load_state_dict(
            {k: v.detach().clone() for k, v in actor_critic.state_dict().items()}
        )
        module.eval()
        module.requires_grad_(False)
        snapshot = Snapshot(
            index=index,
            key=key or f"snap_{index:04d}",
            step=int(step),
            elo=float(self.learner_elo if elo is None else elo),
            module=module,
        )
        self.snapshots.append(snapshot)
        self._by_index[snapshot.index] = snapshot
        self._win_rates[snapshot.key] = PRIOR_WIN_RATE
        if len(self.snapshots) > self.max_snapshots:
            dropped = self.snapshots.pop(0)
            self._by_index.pop(dropped.index, None)
            self._win_rates.pop(dropped.key, None)
        return snapshot

    def snapshot(self, index: int) -> Snapshot:
        """Snapshot by its stable index (list position shifts as old ones drop)."""
        return self._by_index[int(index)]

    @property
    def snapshot_count(self) -> int:
        return len(self.snapshots)

    def module_for(self, spec: OpponentSpec) -> MaskedActorCritic | None:
        """The frozen network backing ``spec`` (``None`` for scripted bots)."""
        if spec.kind != "snapshot":
            return None
        return self.snapshot(int(spec.index)).module

    def scripted_agent(self, spec: OpponentSpec, seed: int | None = None) -> Agent:
        """A fresh scripted agent instance for ``spec``."""
        for cls in self.scripted_classes:
            if cls.name == spec.name:
                return cls(seed=seed)
        raise KeyError(f"unknown scripted opponent {spec.name!r}")

    # -- PFSP sampling -------------------------------------------------- #
    def weights(self) -> np.ndarray:
        """PFSP sampling weights ``(1 - p_i)^2 + 0.05`` over the snapshots."""
        if not self.snapshots:
            return np.zeros(0, dtype=np.float64)
        w = np.array(
            [
                (1.0 - self._win_rates.get(s.key, PRIOR_WIN_RATE)) ** 2
                + self.WEIGHT_FLOOR
                for s in self.snapshots
            ],
            dtype=np.float64,
        )
        total = w.sum()
        return w / total if total > 0 else np.full(len(w), 1.0 / len(w))

    def sample(self) -> OpponentSpec:
        """Draw one opponent with the configured mixture."""
        roll = float(self.rng.random())
        has_snapshots = bool(self.snapshots)
        if roll < self.LATEST_PROB:
            return OpponentSpec(kind="latest", key="latest")
        if roll < self.LATEST_PROB + self.SNAPSHOT_PROB:
            if has_snapshots:
                weights = self.weights()
                index = int(self.rng.choice(len(self.snapshots), p=weights))
                snapshot = self.snapshots[index]
                return OpponentSpec(
                    kind="snapshot", key=snapshot.key, index=snapshot.index
                )
            return OpponentSpec(kind="latest", key="latest")
        return self._sample_scripted()

    def _sample_scripted(self) -> OpponentSpec:
        index = int(self.rng.integers(len(self.scripted_classes)))
        cls = self.scripted_classes[index]
        return OpponentSpec(kind="scripted", key=f"scripted:{cls.name}", name=cls.name)

    # -- book-keeping --------------------------------------------------- #
    def win_rate(self, key: str) -> float:
        return self._win_rates.get(key, PRIOR_WIN_RATE)

    def record_result(self, spec: OpponentSpec, learner_score: float) -> None:
        """EMA-update the learner's win rate against ``spec`` and its Elo."""
        key = spec.key if spec.kind != "scripted" else f"scripted:{spec.name}"
        old = self._win_rates.get(key, PRIOR_WIN_RATE)
        self._win_rates[key] = (1.0 - self.ema_alpha) * old + self.ema_alpha * float(
            learner_score
        )
        if spec.kind == "snapshot":
            snapshot = self.snapshot(int(spec.index))
            snapshot.win_rate = self._win_rates[key]
            self.learner_elo, new_elo = update_elo(
                self.learner_elo, snapshot.elo, float(learner_score)
            )
            snapshot.elo = new_elo

    def stats(self) -> dict[str, object]:
        """Compact pool summary for logs and metrics."""
        return {
            "snapshot_count": self.snapshot_count,
            "latest_win_rate": self.win_rate("latest"),
            "learner_elo": self.learner_elo,
            "scripted_win_rates": {
                cls.name: self.win_rate(f"scripted:{cls.name}")
                for cls in self.scripted_classes
            },
            "snapshot_win_rates": {s.key: s.win_rate for s in self.snapshots},
        }

    def elo_table(self) -> list[dict[str, float]]:
        """Snapshot Elo progression, ready for a plot."""
        return [
            {"key": s.key, "step": s.step, "elo": s.elo, "win_rate": s.win_rate}
            for s in self.snapshots
        ]
