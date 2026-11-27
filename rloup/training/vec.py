"""Vectorised execution of many :class:`~rloup.engine.CoupEngine` instances.

A Coup engine gives *one* decision at a time (AEC).  To use a GPU efficiently we
therefore keep thousands of environments alive at once and, on every tick,
gather the observations of all currently acting seats into a single batch.  The
function :func:`evaluate_vs_baselines` is the shared benchmark used by every
trainer, so the reported numbers are always produced by exactly the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from rloup.agents import BASELINES, Agent
from rloup.engine import CoupEngine

__all__ = [
    "STRONG_BASELINES",
    "EvaluationResult",
    "BehaviourStats",
    "evaluate_vs_baselines",
]

#: Baselines a model must beat to be considered strong in Coup.
STRONG_BASELINES = ("honest", "cautious_skeptical", "card_counter")


@dataclass
class BehaviourStats:
    """Pooled learner behaviour telemetry from ``CoupEngine.get_episode_stats``."""

    claims: int = 0
    bluffs: int = 0
    challenges_attempted: int = 0
    challenges_won: int = 0
    challenge_opportunities: int = 0
    impossible_claim_opportunities: int = 0
    challenges_on_impossible_claim: int = 0
    games: int = 0

    def add_player_stats(self, stats: dict) -> None:
        """Accumulate one ``players[seat]`` entry from the engine telemetry."""
        self.claims += sum(int(v) for v in stats["claims"].values())
        self.bluffs += sum(int(v) for v in stats["bluffs"].values())
        self.challenges_attempted += int(stats["challenges_attempted"])
        self.challenges_won += int(stats["challenges_won"])
        self.challenge_opportunities += int(stats["challenge_opportunities"])
        self.impossible_claim_opportunities += int(
            stats["impossible_claim_opportunities"]
        )
        self.challenges_on_impossible_claim += int(
            stats["challenges_on_impossible_claim"]
        )
        self.games += 1

    @property
    def bluff_rate(self) -> float:
        return self.bluffs / self.claims if self.claims else 0.0

    @property
    def challenge_rate(self) -> float:
        if not self.challenge_opportunities:
            return 0.0
        return self.challenges_attempted / self.challenge_opportunities

    @property
    def challenge_precision(self) -> float:
        if not self.challenges_attempted:
            return 0.0
        return self.challenges_won / self.challenges_attempted

    @property
    def impossible_claim_challenge_rate(self) -> float:
        if not self.impossible_claim_opportunities:
            return 0.0
        return self.challenges_on_impossible_claim / self.impossible_claim_opportunities

    def as_dict(self) -> dict[str, float]:
        return {
            "bluff_rate": self.bluff_rate,
            "challenge_rate": self.challenge_rate,
            "challenge_precision": self.challenge_precision,
            "impossible_claim_challenge_rate": self.impossible_claim_challenge_rate,
        }


@dataclass
class EvaluationResult:
    """Seat-balanced outcome of the periodic benchmark."""

    win_rates: dict[str, float]
    games: dict[str, int]
    behaviour: BehaviourStats = field(default_factory=BehaviourStats)

    @property
    def avg_vs_baselines(self) -> float:
        if not self.win_rates:
            return 0.0
        return float(np.mean(list(self.win_rates.values())))

    @property
    def strong_avg(self) -> float:
        """Average win rate against the three strongest scripted baselines."""
        values = [self.win_rates[name] for name in STRONG_BASELINES if name in self.win_rates]
        return float(np.mean(values)) if values else 0.0

    def as_dict(self) -> dict[str, float]:
        out = {"avg_vs_baselines": self.avg_vs_baselines, "strong_avg": self.strong_avg}
        for name, value in self.win_rates.items():
            out[f"win_rate_{name}"] = float(value)
        out.update(self.behaviour.as_dict())
        return out


def evaluate_vs_baselines(
    policy: Callable[[np.ndarray, np.ndarray], np.ndarray],
    num_players: int,
    games_per_baseline: int = 200,
    *,
    num_envs: int = 64,
    seed: int = 0,
    max_turns: int = 200,
    baseline_classes: Sequence[type[Agent]] = BASELINES,
) -> EvaluationResult:
    """Play ``games_per_baseline`` seat-balanced games against each baseline.

    ``policy`` receives a ``(batch, obs_dim)`` observation array and a matching
    ``(batch, n_actions)`` mask array and must return a batch of action ids.  The
    learner occupies each seat equally often, which removes the substantial
    first-seat advantage of Coup from the reported win rates.
    """
    games_per_baseline = int(games_per_baseline)
    num_envs = int(num_envs)
    if num_envs < 1 or games_per_baseline < 1:
        raise ValueError("num_envs and games_per_baseline must be positive")
    num_envs = min(num_envs, games_per_baseline)

    win_rates: dict[str, float] = {}
    games_by_name: dict[str, int] = {}
    behaviour = BehaviourStats()

    for offset, cls in enumerate(baseline_classes):
        engines = [
            CoupEngine(
                int(num_players),
                max_turns=max_turns,
                seed=seed * 100_003 + offset * 997 + i,
            )
            for i in range(num_envs)
        ]
        agents = [
            [
                cls(seed=seed * 65_537 + i * 31 + seat)
                for seat in range(int(num_players))
            ]
            for i in range(num_envs)
        ]
        learner_seat = [i % int(num_players) for i in range(num_envs)]
        wins = 0
        played = 0
        while played < games_per_baseline:
            active = [i for i in range(num_envs) if not engines[i].done]
            if not active:  # pragma: no cover - defensive
                break
            actions: list[int | None] = [None] * num_envs
            batch_env: list[int] = []
            batch_obs: list[np.ndarray] = []
            batch_mask: list[np.ndarray] = []
            for i in active:
                engine = engines[i]
                seat = engine.current_player
                obs = engine.get_obs(seat)
                mask = engine.legal_mask()
                if seat == learner_seat[i]:
                    batch_env.append(i)
                    batch_obs.append(obs)
                    batch_mask.append(mask)
                else:
                    actions[i] = int(agents[i][seat].select_action(obs, mask))
            if batch_env:
                predicted = policy(np.stack(batch_obs), np.stack(batch_mask))
                for index, i in enumerate(batch_env):
                    actions[i] = int(predicted[index])
            for i in active:
                engine = engines[i]
                engine.step(int(actions[i]))
                if not engine.done:
                    continue
                seat = learner_seat[i]
                if played < games_per_baseline:
                    played += 1
                    if engine.winner() == seat:
                        wins += 1
                    episode = engine.get_episode_stats()
                    behaviour.add_player_stats(episode["players"][seat])
                learner_seat[i] = (learner_seat[i] + 1) % int(num_players)
                engine.reset(seed=None)
        name = cls.name
        win_rates[name] = wins / played if played else 0.0
        games_by_name[name] = played

    return EvaluationResult(
        win_rates=win_rates, games=games_by_name, behaviour=behaviour
    )
