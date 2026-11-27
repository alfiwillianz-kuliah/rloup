"""Comprehensive evaluation suite for rloup (Phase 3).

Everything here is deterministic given a seed and writes a single, fully
self-describing JSON document::

    python -m rloup.eval.evaluate                     # full suite
    python -m rloup.eval.evaluate --skip-exploiters    # reuse saved exploiters
    python -m rloup.eval.evaluate --skip-snapshots     # skip snapshot Elo

Contents
--------
``round_robin_2p``
    1,000 seat-balanced games for every unordered pair of the eight 2-player
    competitors (28 pairs / 28,000 games), exact 95% Wilson intervals and
    Bradley-Terry Elo (MM fit, anchored at ``Random = 1000``) plus an
    independent iterative-Elo fit as a sanity check.
``four_player_evaluation``
    Seat-rotated 4-player tables: three copies of each baseline plus a mixed
    strong and a mixed diverse table.  Win rate, average finish rank, CIs.
``collect_behaviour``
    Pooled bluff / challenge / block telemetry over the strong suite and
    self-play, bluff rates per claim kind, and the main-action distribution by
    coin bracket.
``impossible_claim_probe``
    1,000 constructed, provably-impossible claim states per model, in both
    challenge phases, measuring true card-counting deduction (plus the exact
    challenge probability of the neural models).
``exploitability``
    A fresh PPO exploiter is trained against each frozen 2-player model and then
    measured over 1,000 seat-balanced games; ``approx_exploitability`` is
    ``max(0, exploiter_win_rate - 0.5)``.
``snapshot_elo_progression``
    Every saved snapshot is scored against the fixed five-baseline anchor suite
    and translated into Elo against the anchor ratings of the round robin.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from rloup.agents import Agent
from rloup.agents.baselines import (
    AggressiveBlufferAgent,
    CardCounterAgent,
    CautiousSkepticalAgent,
    HonestAgent,
    RandomAgent,
)
from rloup.agents.dqn import DoubleDQNAgent
from rloup.agents.nfsp import NFSPAgent
from rloup.agents.ppo import PPOAgent
from rloup.constants import CLAIM_KINDS, COPIES_PER_ROLE, NUM_ROLES, ActionType, Phase, Role
from rloup.engine import CoupEngine
from rloup.training.vec import evaluate_vs_baselines

__all__ = [
    "COIN_BRACKETS",
    "Competitor",
    "ProbeState",
    "bradley_terry_elo",
    "build_impossible_claim_state",
    "coin_bracket",
    "collect_behaviour",
    "elo_from_win_rates",
    "evaluate_tables",
    "exploitability",
    "finish_rank",
    "four_player_tables",
    "head_to_head",
    "impossible_claim_probe",
    "iterative_elo",
    "load_competitors",
    "main",
    "model_competitors",
    "round_robin_2p",
    "run_games",
    "snapshot_elo_progression",
    "wilson_interval",
    "wilson_record",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: 95% two-sided normal quantile used by every Wilson interval.
Z95 = 1.959963984540054

#: Coin brackets used by the behavioural action-distribution table.
COIN_BRACKETS: tuple[str, ...] = ("0-2", "3-6", "7-9", "10+")

#: Directory of the trained Phase 2 models, relative to the repository root.
MODEL_RUNS: dict[str, str] = {
    "PFSP-PPO": "results/ppo_pfsp_2p/final.pt",
    "Naive-PPO": "results/ppo_naive_2p/final.pt",
    "NFSP": "results/nfsp_2p/final.pt",
    "Double-DQN": "results/ddqn_pfsp_2p/final.pt",
    "PFSP-PPO-4P": "results/ppo_pfsp_4p/final.pt",
    "Double-DQN-4P": "results/ddqn_pfsp_4p/final.pt",
}

#: Display name -> scripted agent class, in tournament order.
SCRIPTED_COMPETITORS: tuple[tuple[str, type[Agent]], ...] = (
    ("CardCounter", CardCounterAgent),
    ("Honest", HonestAgent),
    ("Cautious", CautiousSkepticalAgent),
    ("Bluffer", AggressiveBlufferAgent),
    ("Random", RandomAgent),
)

#: Display name -> the lowercase ``Agent.name`` used by the training metrics.
AGENT_KEY: dict[str, str] = {
    "PFSP-PPO": "pfsp_ppo",
    "Naive-PPO": "naive_ppo",
    "NFSP": "nfsp",
    "Double-DQN": "double_dqn",
    "PFSP-PPO-4P": "pfsp_ppo_4p",
    "Double-DQN-4P": "double_dqn_4p",
    "CardCounter": "card_counter",
    "Honest": "honest",
    "Cautious": "cautious_skeptical",
    "Bluffer": "aggressive_bluffer",
    "Random": "random",
}

#: ``evaluate_vs_baselines`` reports lowercase class names; map them back.
BASELINE_KEY_TO_DISPLAY: dict[str, str] = {
    "card_counter": "CardCounter",
    "honest": "Honest",
    "cautious_skeptical": "Cautious",
    "aggressive_bluffer": "Bluffer",
    "random": "Random",
}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """Exact 95% Wilson score interval for a binomial proportion.

    Unlike the Wald interval it never leaves ``[0, 1]`` and stays informative at
    ``successes == 0`` or ``successes == trials``, which is exactly where a
    deterministic scripted matchup lands.
    """
    n = int(trials)
    if n <= 0:
        return (0.0, 1.0)
    k = min(max(int(successes), 0), n)
    p = k / n
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (float(z) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def wilson_record(successes: int, trials: int, z: float = Z95) -> dict[str, Any]:
    """Wilson interval plus the raw counts, ready for JSON."""
    low, high = wilson_interval(successes, trials, z)
    n = int(trials)
    return {
        "successes": int(successes),
        "trials": n,
        "rate": (float(successes) / n) if n else 0.0,
        "ci_low": low,
        "ci_high": high,
        "half_width": (high - low) / 2.0,
    }


def coin_bracket(coins: int) -> str:
    """Bucket a coin count into ``0-2`` / ``3-6`` / ``7-9`` / ``10+``."""
    c = int(coins)
    if c <= 2:
        return "0-2"
    if c <= 6:
        return "3-6"
    if c <= 9:
        return "7-9"
    return "10+"



def resolve_device(name: str = "auto") -> str:
    """``auto`` picks CUDA when available; any other string is passed through."""
    if name != "auto":
        return str(name)
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Competitors
# --------------------------------------------------------------------------- #
@dataclass
class Competitor:
    """One named tournament entrant: a scripted class or a frozen network.

    Neural entrants own a shared, frozen policy: every seat played by the same
    competitor uses the same weights, and batched inference keeps the tournament
    fast.  Scripted entrants are instantiated once per seat and game so that
    their private RNG streams stay independent.
    """

    name: str
    cls: type[Agent] | None = None
    ppo: PPOAgent | None = None
    nfsp: NFSPAgent | None = None
    ddqn: DoubleDQNAgent | None = None
    checkpoint: str | None = None

    @property
    def neural(self) -> bool:
        return self.ppo is not None or self.nfsp is not None or self.ddqn is not None

    def new_agent(self, seed: int | None = None) -> Agent:
        """A per-seat agent handle (scripted: fresh instance, neural: shared)."""
        if self.cls is not None:
            return self.cls(seed=seed)
        if self.ppo is not None:
            return self.ppo
        if self.nfsp is not None:
            return self.nfsp
        if self.ddqn is not None:
            return self.ddqn
        raise RuntimeError(f"competitor {self.name!r} has no policy")

    # -- inference ------------------------------------------------------ #
    def batched_actions(self, obs: np.ndarray, masks: np.ndarray) -> list[int]:
        """Actions for a batch of observations (neural entrants only)."""
        if self.ppo is not None:
            return [int(a) for a in self.ppo.batched_sample(obs, masks)]
        if self.nfsp is not None:
            return [int(a) for a in self.nfsp.batched_actions(obs, masks)]
        if self.ddqn is not None:
            return [int(a) for a in self.ddqn.batched_actions(obs, masks)]
        raise RuntimeError(f"competitor {self.name!r} is not batched")

    def deterministic_actions(self, obs: np.ndarray, masks: np.ndarray) -> list[int]:
        """Greedy actions for a batch of observations (neural entrants only)."""
        if self.ppo is not None:
            return [int(a) for a in self.ppo.batched_sample(obs, masks, deterministic=True)]
        if self.nfsp is not None:
            return [
                int(a)
                for a in self.nfsp.batched_actions(obs, masks, deterministic=True)
            ]
        if self.ddqn is not None:
            return [
                int(a)
                for a in self.ddqn.batched_actions(obs, masks, deterministic=True)
            ]
        raise RuntimeError(f"competitor {self.name!r} is not batched")

    def batched_probs(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Masked action probabilities over a batch (neural entrants only)."""
        if self.ppo is not None:
            return np.asarray(self.ppo.batched_probs(obs, masks), dtype=np.float64)
        if self.nfsp is not None:
            import torch

            with torch.inference_mode():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.nfsp.device)
                mask_t = torch.as_tensor(masks, dtype=torch.float32, device=self.nfsp.device)
                logits = self.nfsp.sl_network.masked_logits(obs_t, mask_t)
                return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        if self.ddqn is not None:
            return np.asarray(self.ddqn.batched_probs(obs, masks), dtype=np.float64)
        raise RuntimeError(f"competitor {self.name!r} has no distribution")

    def select_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """One action for one observation, through the shared agent interface."""
        return int(self.new_agent().select_action(obs, mask))

    def action_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Masked action probabilities, or ``None`` for scripted entrants."""
        agent = (
            self.ppo
            if self.ppo is not None
            else (self.nfsp if self.nfsp is not None else self.ddqn)
        )
        if agent is None:
            return None
        return np.asarray(agent.action_probs(obs, mask), dtype=np.float64)

    def value(self, obs: np.ndarray) -> float | None:
        """Critic estimate of the state value (PPO only)."""
        if self.ppo is None:
            return None
        import torch

        with torch.inference_mode():
            tensor = torch.as_tensor(
                np.asarray(obs, dtype=np.float32), device=self.ppo.device
            ).unsqueeze(0)
            return float(self.ppo.actor_critic.value(tensor).item())


def scripted_competitor(name: str) -> Competitor:
    """The scripted competitor with display ``name``."""
    for display, cls in SCRIPTED_COMPETITORS:
        if display == name:
            return Competitor(name=display, cls=cls)
    raise KeyError(f"unknown scripted competitor {name!r}")


def load_competitors(
    device: str = "auto", root: Path | str = ".", verbose: bool = True
) -> list[Competitor]:
    """Load trained 2-player RL models (including Double DQN if present) plus baselines."""
    root = Path(root)
    device = resolve_device(device)
    competitors: list[Competitor] = []
    for name in ("PFSP-PPO", "Naive-PPO", "NFSP", "Double-DQN"):
        path = root / MODEL_RUNS[name]
        if not path.exists():
            if name == "Double-DQN":
                continue
            raise FileNotFoundError(f"missing model checkpoint {path}")
        if name == "NFSP":
            competitors.append(
                Competitor(
                    name=name,
                    nfsp=NFSPAgent.load(path, device=device, seed=0),
                    checkpoint=str(path),
                )
            )
        elif name == "Double-DQN":
            competitors.append(
                Competitor(
                    name=name,
                    ddqn=DoubleDQNAgent.load(path, device=device, seed=0),
                    checkpoint=str(path),
                )
            )
        else:
            competitors.append(
                Competitor(
                    name=name,
                    ppo=PPOAgent.load(path, device=device, seed=0),
                    checkpoint=str(path),
                )
            )
        if verbose:
            print(f"  loaded {name:<10s} <- {path}", flush=True)
    competitors.extend(scripted_competitor(name) for name, _ in SCRIPTED_COMPETITORS)
    return competitors


def model_competitors(
    device: str = "auto", root: Path | str = ".", names: Sequence[str] | None = None
) -> dict[str, Competitor]:
    """Load a subset of the trained models, keyed by display name."""
    root = Path(root)
    device = resolve_device(device)
    wanted = (
        list(names)
        if names is not None
        else ["PFSP-PPO", "Naive-PPO", "NFSP", "Double-DQN", "PFSP-PPO-4P", "Double-DQN-4P"]
    )
    out: dict[str, Competitor] = {}
    for name in wanted:
        path = root / MODEL_RUNS[name]
        if not path.exists():
            raise FileNotFoundError(f"missing model checkpoint {path}")
        if name == "NFSP":
            out[name] = Competitor(
                name=name,
                nfsp=NFSPAgent.load(path, device=device, seed=0),
                checkpoint=str(path),
            )
        elif name.startswith("Double-DQN"):
            out[name] = Competitor(
                name=name,
                ddqn=DoubleDQNAgent.load(path, device=device, seed=0),
                checkpoint=str(path),
            )
        else:
            out[name] = Competitor(
                name=name,
                ppo=PPOAgent.load(path, device=device, seed=0),
                checkpoint=str(path),
            )
    return out


def two_player_name(competitors: Sequence[Competitor], name: str) -> Competitor:
    """Look up a competitor by display name."""
    for comp in competitors:
        if comp.name == name:
            return comp
    raise KeyError(f"no competitor named {name!r}")


# --------------------------------------------------------------------------- #
# Core batched game loop
# --------------------------------------------------------------------------- #
def _make_agents(
    comps: Sequence[Competitor], seed: int, env_index: int, game_index: int
) -> list[Agent]:
    return [
        comp.new_agent(seed=seed * 65_537 + env_index * 31 + game_index * 7 + seat)
        for seat, comp in enumerate(comps)
    ]


def run_games(
    schedule: Sequence[Sequence[Competitor]],
    *,
    num_players: int,
    num_envs: int = 64,
    seed: int = 0,
    max_turns: int = 200,
    on_decision: Callable[[CoupEngine, int, int], None] | None = None,
    on_action: Callable[[CoupEngine, int, int, int], None] | None = None,
    on_finish: Callable[[int, CoupEngine], None] | None = None,
    label: str = "",
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Play every entry of ``schedule`` once, many games in flight.

    ``schedule[g][seat]`` is the competitor occupying ``seat`` in game ``g``.
    Returned per game: the winning seat (``None`` for a truncated game), the
    elimination order, the turn count, the terminal rewards and the seat names.

    ``on_decision(engine, game_index, seat)`` fires before the action of a
    decision, ``on_action(engine, game_index, seat, action)`` once that action is
    known and still un-applied (so pre-action tableaus such as the coin count are
    readable), and ``on_finish(game_index, engine)`` right after a game ends,
    before its engine is recycled.
    """
    games = len(schedule)
    if games == 0:
        return []
    n = int(num_players)
    for entry in schedule:
        if len(entry) != n:
            raise ValueError(f"schedule entry has {len(entry)} seats, expected {n}")

    num_envs = max(1, min(int(num_envs), games))
    results: list[dict[str, Any] | None] = [None] * games

    engines: list[CoupEngine] = []
    current: list[int] = []
    agents: list[list[Agent]] = []
    for i in range(num_envs):
        engine = CoupEngine(n, max_turns=max_turns, seed=seed * 1_000_003 + i)
        engine.reset(seed=seed * 7_919 + 11 + i)
        engines.append(engine)
        current.append(i)
        agents.append(_make_agents(schedule[i], seed, i, i))

    next_game = num_envs
    finished = 0
    started = time.time()
    while finished < games:
        buckets: dict[str, tuple[Competitor, list[tuple[int, np.ndarray, np.ndarray]]]] = {}
        actions: list[int] = [0] * num_envs
        for i in range(num_envs):
            engine = engines[i]
            if engine.done:  # pragma: no cover - environments restart immediately
                continue
            game = current[i]
            seat = engine.current_player
            comp = schedule[game][seat]
            obs = engine.get_obs(seat)
            mask = engine.legal_mask()
            if on_decision is not None:
                on_decision(engine, game, seat)
            if comp.neural:
                buckets.setdefault(comp.name, (comp, []))[1].append((i, obs, mask))
            else:
                actions[i] = int(agents[i][seat].select_action(obs, mask))
        for comp, rows in buckets.values():
            chosen = comp.batched_actions(
                np.stack([row[1] for row in rows]), np.stack([row[2] for row in rows])
            )
            for k, (i, _, _) in enumerate(rows):
                actions[i] = int(chosen[k])
        for i in range(num_envs):
            engine = engines[i]
            if engine.done:  # pragma: no cover - defensive
                continue
            if on_action is not None:
                on_action(engine, current[i], engine.current_player, int(actions[i]))
            engine.step(actions[i])
            if not engine.done:
                continue
            game = current[i]
            truncated = bool(engine.truncated)
            results[game] = {
                "game": int(game),
                "seats": [c.name for c in schedule[game]],
                "winner": None if truncated else engine.winner(),
                "truncated": truncated,
                "turns": int(engine.turn),
                "elim_order": [int(s) for s in engine._elim_order],
                "rewards": [float(r) for r in engine.rewards],
            }
            if on_finish is not None:
                on_finish(game, engine)
            finished += 1
            if next_game < games:
                engines[i].reset(seed=seed * 7_919 + 11 + next_game)
                current[i] = next_game
                agents[i] = _make_agents(schedule[next_game], seed, i, next_game)
                next_game += 1
    if verbose:
        elapsed = time.time() - started
        print(
            f"    {label}: {games} games in {elapsed:.1f}s "
            f"({games / max(elapsed, 1e-9):.0f} games/s)",
            flush=True,
        )
    return [r for r in results if r is not None]


def head_to_head(
    a: Competitor,
    b: Competitor,
    *,
    games: int = 1_000,
    num_envs: int = 64,
    seed: int = 0,
    max_turns: int = 200,
    verbose: bool = False,
) -> dict[str, Any]:
    """Seat-balanced duel: the first half seats ``a`` in seat 0."""
    first = games // 2
    schedule: list[tuple[Competitor, Competitor]] = [(a, b)] * first
    schedule += [(b, a)] * (games - first)
    results = run_games(
        schedule,
        num_players=2,
        num_envs=num_envs,
        seed=seed,
        max_turns=max_turns,
        label=f"{a.name} vs {b.name}",
        verbose=verbose,
    )
    wins = 0
    draws = 0
    for res in results:
        winner = res["winner"]
        if winner is None:
            draws += 1
        elif res["seats"][winner] == a.name:
            wins += 1
    record = wilson_record(wins, len(results))
    record.update(
        {
            "a": a.name,
            "b": b.name,
            "games": len(results),
            "a_wins": wins,
            "b_wins": len(results) - wins - draws,
            "draws": draws,
        }
    )
    return record


# --------------------------------------------------------------------------- #
# Elo / Bradley-Terry
# --------------------------------------------------------------------------- #
def _records_from_pairs(
    names: Sequence[str], pairs: dict[tuple[str, str], dict[str, int]]
) -> list[tuple[int, int, int, int, int]]:
    """``(i, j, wins_i, wins_j, draws)`` per unordered pair, ``i < j``."""
    index = {name: k for k, name in enumerate(names)}
    out: list[tuple[int, int, int, int, int]] = []
    for (a, b), entry in pairs.items():
        if a == b:
            continue
        i, j = index[a], index[b]
        out.append(
            (
                i,
                j,
                int(entry.get("a_wins", 0)),
                int(entry.get("b_wins", 0)),
                int(entry.get("draws", 0)),
            )
        )
    return out


def bradley_terry_elo(
    names: Sequence[str],
    pairs: dict[tuple[str, str], dict[str, int]],
    *,
    anchor: str = "Random",
    anchor_rating: float = 1000.0,
    scale: float = 400.0,
    iterations: int = 2000,
    tol: float = 1e-12,
) -> dict[str, float]:
    """Maximum-likelihood Bradley-Terry ratings expressed on the Elo scale.

    Solves ``theta_i = W_i / sum_j n_ij / (theta_i + theta_j)`` (the standard
    MM / Zermelo iteration) and reports ``anchor_rating + scale *
    log10(theta_i / theta_anchor)``, so the anchor sits exactly at
    ``anchor_rating``.
    """
    names = list(names)
    size = len(names)
    if size == 0:
        return {}
    index = {name: k for k, name in enumerate(names)}
    theta = np.ones(size, dtype=np.float64)
    wins = np.zeros(size, dtype=np.float64)
    games = np.zeros((size, size), dtype=np.float64)
    for i, j, wi, wj, draws in _records_from_pairs(names, pairs):
        wins[i] += wi + 0.5 * draws
        wins[j] += wj + 0.5 * draws
        games[i, j] += wi + wj + draws
        games[j, i] += wi + wj + draws
    if games.sum() == 0:
        return {name: float(anchor_rating) for name in names}
    for _ in range(int(iterations)):
        denom = np.zeros(size, dtype=np.float64)
        for i in range(size):
            for j in range(size):
                if i == j or games[i, j] == 0:
                    continue
                denom[i] += games[i, j] / (theta[i] + theta[j])
        updated = np.where(denom > 0, wins / np.maximum(denom, 1e-300), theta)
        updated = np.maximum(updated, 1e-12)
        # The MM update is scale free; normalise for numerical stability.
        updated /= np.exp(np.mean(np.log(updated)))
        delta = float(np.max(np.abs(updated - theta)))
        theta = updated
        if delta < float(tol):
            break
    base = max(theta[index.get(anchor, 0)], 1e-12)
    return {
        name: float(anchor_rating + scale * math.log10(max(theta[k], 1e-12) / base))
        for k, name in enumerate(names)
    }


def iterative_elo(
    names: Sequence[str],
    pairs: dict[tuple[str, str], dict[str, int]],
    *,
    anchor: str = "Random",
    anchor_rating: float = 1000.0,
    k: float = 24.0,
    rounds: int = 300,
) -> dict[str, float]:
    """Sequential Elo fit with a decaying step size, re-anchored every round.

    Kept alongside the Bradley-Terry fit as an independent sanity check: two
    different estimators agreeing on the ranking is stronger evidence than
    either one alone.
    """
    names = list(names)
    rating = {name: float(anchor_rating) for name in names}
    records = _records_from_pairs(names, pairs)
    for step in range(int(rounds)):
        rate = float(k) / (1.0 + step / 20.0)
        for i, j, wi, wj, draws in records:
            played = wi + wj + draws
            if played == 0:
                continue
            a, b = names[i], names[j]
            expected = 1.0 / (1.0 + 10.0 ** ((rating[b] - rating[a]) / 400.0))
            score = (wi + 0.5 * draws) / played
            delta = rate * (score - expected)
            rating[a] += delta
            rating[b] -= delta
        shift = rating.get(anchor, anchor_rating) - anchor_rating
        for name in names:
            rating[name] -= shift
    return {name: float(value) for name, value in rating.items()}


def elo_from_win_rates(
    win_rates: dict[str, float],
    opponent_elos: dict[str, float],
    *,
    games_per_opponent: int,
    anchor_rating: float = 1000.0,
    iterations: int = 500,
) -> float:
    """Elo of a single player scored against opponents with known ratings.

    Used by the snapshot progression: the five baselines keep their round-robin
    ratings, the snapshot is the only free parameter, and the same MM update as
    :func:`bradley_terry_elo` is iterated to convergence.
    """
    pairs = [
        (float(rate), float(games_per_opponent), float(opponent_elos[name]))
        for name, rate in win_rates.items()
        if name in opponent_elos
    ]
    if not pairs:
        return float(anchor_rating)
    theta = 1.0
    for _ in range(int(iterations)):
        numerator = sum(rate * played for rate, played, _ in pairs)
        denominator = sum(
            played / (theta + 10.0 ** ((elo - anchor_rating) / 400.0))
            for _, played, elo in pairs
        )
        if denominator <= 0:  # pragma: no cover - defensive
            return float(anchor_rating)
        updated = max(numerator / denominator, 1e-12)
        converged = abs(updated - theta) < 1e-12
        theta = updated
        if converged:
            break
    return float(anchor_rating + 400.0 * math.log10(theta))


# --------------------------------------------------------------------------- #
# 2-player round robin
# --------------------------------------------------------------------------- #
def build_round_robin_schedule(
    competitors: Sequence[Competitor], games_per_pair: int
) -> list[tuple[Competitor, Competitor]]:
    """Every unordered pair, ``games_per_pair // 2`` in each seat assignment."""
    first = int(games_per_pair) // 2
    schedule: list[tuple[Competitor, Competitor]] = []
    for i, a in enumerate(competitors):
        for b in competitors[i + 1 :]:
            schedule.extend([(a, b)] * first)
            schedule.extend([(b, a)] * (int(games_per_pair) - first))
    return schedule


def round_robin_2p(
    competitors: Sequence[Competitor],
    *,
    games_per_pair: int = 1_000,
    num_envs: int = 96,
    seed: int = 0,
    max_turns: int = 200,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full 2-player round robin with Wilson intervals and Elo ratings."""
    schedule = build_round_robin_schedule(competitors, games_per_pair)
    names = [c.name for c in competitors]
    if verbose:
        print(
            f"2P round robin: {len(schedule)} games over "
            f"{len(names) * (len(names) - 1) // 2} pairs ({games_per_pair} each)",
            flush=True,
        )
    results = run_games(
        schedule,
        num_players=2,
        num_envs=num_envs,
        seed=seed,
        max_turns=max_turns,
        label="round-robin",
        verbose=verbose,
    )

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for res in results:
        left, right = res["seats"]
        key = (left, right) if left < right else (right, left)
        entry = pairs.setdefault(
            key,
            {
                "games": 0,
                "a_wins": 0,
                "b_wins": 0,
                "draws": 0,
                "a_seat0_games": 0,
                "a_seat0_wins": 0,
                "a_seat1_games": 0,
                "a_seat1_wins": 0,
            },
        )
        winner = res["winner"]
        a_seat = 0 if left == key[0] else 1
        entry["games"] += 1
        entry[f"a_seat{a_seat}_games"] += 1
        if winner is None:
            entry["draws"] += 1
            continue
        if res["seats"][winner] == key[0]:
            entry["a_wins"] += 1
            entry[f"a_seat{a_seat}_wins"] += 1
        else:
            entry["b_wins"] += 1

    size = len(names)
    matrix = np.full((size, size), np.nan)
    matrix_ci = np.full((size, size, 2), np.nan)
    np.fill_diagonal(matrix, 0.5)
    for k in range(size):
        matrix_ci[k, k] = (0.0, 1.0)
    head_to_head_rows: list[dict[str, Any]] = []
    for (a, b), entry in pairs.items():
        i, j = names.index(a), names.index(b)
        played = entry["games"]
        matrix[i, j] = entry["a_wins"] / played if played else float("nan")
        matrix[j, i] = entry["b_wins"] / played if played else float("nan")
        matrix_ci[i, j] = wilson_interval(entry["a_wins"], played)
        matrix_ci[j, i] = wilson_interval(entry["b_wins"], played)
        row: dict[str, Any] = dict(entry)
        row.update(wilson_record(entry["a_wins"], played))
        row.update(
            {
                "a": a,
                "b": b,
                "a_score_rate": (entry["a_wins"] + 0.5 * entry["draws"]) / played
                if played
                else 0.0,
                "a_seat0_win_rate": entry["a_seat0_wins"] / entry["a_seat0_games"]
                if entry["a_seat0_games"]
                else 0.0,
                "a_seat1_win_rate": entry["a_seat1_wins"] / entry["a_seat1_games"]
                if entry["a_seat1_games"]
                else 0.0,
            }
        )
        head_to_head_rows.append(row)

    elo = bradley_terry_elo(names, pairs, anchor="Random")
    elo_alt = iterative_elo(names, pairs, anchor="Random")
    ranking = sorted(
        (
            {
                "agent": name,
                "elo": elo[name],
                "elo_iterative": elo_alt[name],
                "win_rate": float(np.nanmean(matrix[names.index(name)])),
            }
            for name in names
        ),
        key=lambda row: -row["elo"],
    )
    return {
        "games_per_pair": int(games_per_pair),
        "total_games": len(results),
        "agents": names,
        "matrix": [[float(v) for v in row] for row in matrix],
        "matrix_ci": [
            [[float(lo), float(hi)] for lo, hi in row] for row in matrix_ci
        ],
        "head_to_head": head_to_head_rows,
        "elo": {name: float(elo[name]) for name in names},
        "elo_iterative": {name: float(elo_alt[name]) for name in names},
        "elo_anchor": {"agent": "Random", "rating": 1000.0},
        "ranking": ranking,
        "draws": sum(row["draws"] for row in head_to_head_rows) // 2,
    }


# --------------------------------------------------------------------------- #
# 4-player evaluation
# --------------------------------------------------------------------------- #
def finish_rank(
    elim_order: Sequence[int], winner: int | None, seat: int, num_players: int
) -> int | None:
    """Finish place from the elimination order (``1`` is the winner)."""
    if winner is None:
        return None
    if int(seat) == int(winner):
        return 1
    order = [int(s) for s in elim_order]
    if int(seat) in order:
        return int(num_players) - order.index(int(seat))
    return None  # pragma: no cover - defensive


def four_player_tables() -> list[dict[str, Any]]:
    """The five baseline tables plus the two mixed ones."""
    tables: list[dict[str, Any]] = []
    for display, _ in SCRIPTED_COMPETITORS:
        tables.append(
            {
                "name": f"3x{display}",
                "opponents": (display, display, display),
                "kind": "baseline",
            }
        )
    tables.append(
        {
            "name": "mixed_strong",
            "opponents": ("CardCounter", "Honest", "Cautious"),
            "kind": "mixed",
        }
    )
    tables.append(
        {
            "name": "mixed_diverse",
            "opponents": ("CardCounter", "Bluffer", "Random"),
            "kind": "mixed",
        }
    )
    return tables


def evaluate_tables(
    learner: Competitor,
    tables: Sequence[dict[str, Any]],
    *,
    games_per_table: int = 1_000,
    num_players: int = 4,
    num_envs: int = 64,
    seed: int = 0,
    max_turns: int = 200,
    verbose: bool = True,
) -> dict[str, Any]:
    """Seat-rotated games on every table; win rate and average finish rank."""
    n = int(num_players)
    schedule: list[list[Competitor]] = []
    labels: list[str] = []
    learner_seats: list[int] = []
    for table in tables:
        opponents = [scripted_competitor(name) for name in table["opponents"]]
        for k in range(int(games_per_table)):
            seat = k % n
            schedule.append(list(opponents[:seat]) + [learner] + list(opponents[seat:]))
            labels.append(table["name"])
            learner_seats.append(seat)

    per_table: dict[str, dict[str, Any]] = {
        table["name"]: {
            "games": 0,
            "wins": 0,
            "truncated": 0,
            "rank_sum": 0,
            "ranked_games": 0,
            "rank_counts": {str(r): 0 for r in range(1, n + 1)},
            "seat_games": {str(s): 0 for s in range(n)},
            "seat_wins": {str(s): 0 for s in range(n)},
        }
        for table in tables
    }

    wins = 0
    truncated_total = 0
    seat_games: dict[int, int] = {}
    seat_wins: dict[int, int] = {}
    all_ranks: list[int] = []

    results = run_games(
        schedule,
        num_players=n,
        num_envs=num_envs,
        seed=seed,
        max_turns=max_turns,
        label="4p-tables",
        verbose=verbose,
    )
    for index, res in enumerate(results):
        label = labels[index]
        entry = per_table[label]
        learner_seat = learner_seats[index]
        entry["games"] += 1
        entry["seat_games"][str(learner_seat)] += 1
        seat_games[learner_seat] = seat_games.get(learner_seat, 0) + 1
        winner = res["winner"]
        if winner is None:
            entry["truncated"] += 1
            truncated_total += 1
            continue
        if winner == learner_seat:
            entry["wins"] += 1
            entry["seat_wins"][str(learner_seat)] += 1
            seat_wins[learner_seat] = seat_wins.get(learner_seat, 0) + 1
            wins += 1
        rank = finish_rank(res["elim_order"], winner, learner_seat, n)
        if rank is not None:
            entry["rank_sum"] += rank
            entry["ranked_games"] += 1
            entry["rank_counts"][str(rank)] += 1
            all_ranks.append(rank)

    table_metrics: dict[str, Any] = {}
    for name, entry in per_table.items():
        record = wilson_record(entry["wins"], entry["games"])
        record.update(
            {
                "games": entry["games"],
                "wins": entry["wins"],
                "truncated": entry["truncated"],
                "avg_finish_rank": entry["rank_sum"] / entry["ranked_games"]
                if entry["ranked_games"]
                else 0.0,
                "ranked_games": entry["ranked_games"],
                "rank_counts": entry["rank_counts"],
                "win_rate_by_seat": {
                    seat: (entry["seat_wins"][seat] / entry["seat_games"][seat])
                    if entry["seat_games"][seat]
                    else 0.0
                    for seat in entry["seat_games"]
                },
            }
        )
        table_metrics[name] = record

    played = len(schedule) - truncated_total
    overall = wilson_record(wins, played)
    overall.update(
        {
            "games": len(schedule),
            "played": played,
            "truncated": truncated_total,
            "avg_finish_rank": float(np.mean(all_ranks)) if all_ranks else 0.0,
            "win_rate_by_seat": {
                str(seat): (seat_wins.get(seat, 0) / games)
                for seat, games in seat_games.items()
            },
        }
    )
    return {
        "chance": 1.0 / float(n),
        "games_per_table": int(games_per_table),
        "tables": table_metrics,
        "overall": overall,
    }


# --------------------------------------------------------------------------- #
# Behavioural diagnostics
# --------------------------------------------------------------------------- #
@dataclass
class BehaviourAccumulator:
    """Pooled telemetry for one learner on one table (or across all tables)."""

    games: int = 0
    claims: dict[str, int] = field(
        default_factory=lambda: {kind: 0 for kind in CLAIM_KINDS}
    )
    bluffs: dict[str, int] = field(
        default_factory=lambda: {kind: 0 for kind in CLAIM_KINDS}
    )
    action_counts: dict[str, int] = field(default_factory=dict)
    challenges_attempted: int = 0
    challenges_won: int = 0
    challenge_opportunities: int = 0
    impossible_claim_opportunities: int = 0
    challenges_on_impossible_claim: int = 0
    blocks_attempted: int = 0
    block_opportunities: int = 0
    block_bluffs: int = 0
    claims_proven: int = 0
    caught_bluffing: int = 0
    influence_lost: int = 0
    brackets: dict[str, dict[str, int]] = field(
        default_factory=lambda: {bracket: {} for bracket in COIN_BRACKETS}
    )

    def add_episode(self, player_stats: dict[str, Any]) -> None:
        """Accumulate one ``players[seat]`` telemetry entry."""
        self.games += 1
        for kind, value in player_stats["claims"].items():
            self.claims[kind] = self.claims.get(kind, 0) + int(value)
        for kind, value in player_stats["bluffs"].items():
            self.bluffs[kind] = self.bluffs.get(kind, 0) + int(value)
        for kind, value in player_stats["action_counts"].items():
            self.action_counts[kind] = self.action_counts.get(kind, 0) + int(value)
        self.challenges_attempted += int(player_stats["challenges_attempted"])
        self.challenges_won += int(player_stats["challenges_won"])
        self.challenge_opportunities += int(player_stats["challenge_opportunities"])
        self.impossible_claim_opportunities += int(
            player_stats["impossible_claim_opportunities"]
        )
        self.challenges_on_impossible_claim += int(
            player_stats["challenges_on_impossible_claim"]
        )
        self.blocks_attempted += int(player_stats["blocks_attempted"])
        self.block_bluffs += int(player_stats["block_bluffs"])
        self.claims_proven += int(player_stats["claims_proven"])
        self.caught_bluffing += int(player_stats["caught_bluffing"])
        self.influence_lost += int(player_stats["influence_lost"])

    def add_bracket_action(self, bracket: str, kind: str) -> None:
        """Record one main action taken with ``coins`` in ``bracket``."""
        counts = self.brackets.setdefault(bracket, {})
        counts[kind] = counts.get(kind, 0) + 1

    # -- derived views --------------------------------------------------- #
    @property
    def total_claims(self) -> int:
        return sum(self.claims.values())

    @property
    def total_bluffs(self) -> int:
        return sum(self.bluffs.values())

    @property
    def bluff_rate(self) -> float:
        return self.total_bluffs / self.total_claims if self.total_claims else 0.0

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
    def block_rate(self) -> float:
        if not self.block_opportunities:
            return 0.0
        return self.blocks_attempted / self.block_opportunities

    def as_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "claims": self.total_claims,
            "bluffs": self.total_bluffs,
            "bluff_rate": self.bluff_rate,
            "claims_by_kind": dict(self.claims),
            "bluffs_by_kind": dict(self.bluffs),
            "bluff_rate_by_kind": {
                kind: (
                    self.bluffs[kind] / self.claims[kind] if self.claims[kind] else 0.0
                )
                for kind in CLAIM_KINDS
            },
            "challenge_rate": self.challenge_rate,
            "challenge_precision": self.challenge_precision,
            "challenges_attempted": self.challenges_attempted,
            "challenges_won": self.challenges_won,
            "challenge_opportunities": self.challenge_opportunities,
            "impossible_claim_challenge_rate": (
                self.challenges_on_impossible_claim / self.impossible_claim_opportunities
                if self.impossible_claim_opportunities
                else 0.0
            ),
            "impossible_claim_opportunities": self.impossible_claim_opportunities,
            "challenges_on_impossible_claim": self.challenges_on_impossible_claim,
            "block_rate": self.block_rate,
            "blocks_attempted": self.blocks_attempted,
            "block_opportunities": self.block_opportunities,
            "block_bluffs": self.block_bluffs,
            "claims_proven": self.claims_proven,
            "caught_bluffing": self.caught_bluffing,
            "action_distribution": dict(self.action_counts),
            "coin_brackets": {
                bracket: dict(counts) for bracket, counts in self.brackets.items()
            },
        }


def collect_behaviour(
    learner: Competitor,
    *,
    num_players: int = 2,
    games_per_table: int = 500,
    num_envs: int = 64,
    seed: int = 0,
    max_turns: int = 200,
    verbose: bool = True,
) -> dict[str, Any]:
    """Bluff / challenge / block telemetry over the strong suite and self-play.

    Four tables - Honest, Cautious, CardCounter and the learner's own frozen
    weights (``SelfPlay``) - with ``games_per_table`` seat-rotated games each, so
    the defaults give the 2,000-game aggregate for 2 players and its 4-player
    analogue (three copies of each opponent).
    """
    n = int(num_players)
    if n < 2:
        raise ValueError("num_players must be >= 2")
    table_labels = ("Honest", "Cautious", "CardCounter", "SelfPlay")
    lookup: dict[str, Competitor] = {
        display: scripted_competitor(display) for display, _ in SCRIPTED_COMPETITORS
    }
    # Self-play reuses the learner's own policy (weights for a network, the class
    # for a scripted bot) under a distinct name so seats stay identifiable.
    lookup["SelfPlay"] = Competitor(
        name="SelfPlay", cls=learner.cls, ppo=learner.ppo, nfsp=learner.nfsp, ddqn=learner.ddqn
    )

    schedule: list[list[Competitor]] = []
    labels: list[str] = []
    learner_seats: list[int] = []
    for label in table_labels:
        opponents = [lookup[label]] * (n - 1)
        for k in range(int(games_per_table)):
            seat = k % n
            schedule.append(list(opponents[:seat]) + [learner] + list(opponents[seat:]))
            labels.append(label)
            learner_seats.append(seat)

    pooled = BehaviourAccumulator()
    per_table: dict[str, BehaviourAccumulator] = {
        label: BehaviourAccumulator() for label in table_labels
    }

    def on_decision(engine: CoupEngine, game: int, seat: int) -> None:
        if seat != learner_seats[game]:
            return
        if engine.phase == Phase.BLOCK_ACTION:
            pooled.block_opportunities += 1
            per_table[labels[game]].block_opportunities += 1

    def on_action(engine: CoupEngine, game: int, seat: int, action: int) -> None:
        if seat != learner_seats[game] or engine.phase != Phase.CHOOSE_ACTION:
            return
        bracket = coin_bracket(engine.players[seat].coins)
        kind = engine.spec.decode(action).kind
        pooled.add_bracket_action(bracket, kind)
        per_table[labels[game]].add_bracket_action(bracket, kind)

    def on_finish(game: int, engine: CoupEngine) -> None:
        seat = learner_seats[game]
        stats = engine.get_episode_stats()["players"][seat]
        pooled.add_episode(stats)
        per_table[labels[game]].add_episode(stats)

    if verbose:
        print(
            f"behavioural diagnostics ({n}P, {learner.name}): {len(schedule)} games",
            flush=True,
        )
    run_games(
        schedule,
        num_players=n,
        num_envs=num_envs,
        seed=seed,
        max_turns=max_turns,
        on_decision=on_decision,
        on_action=on_action,
        on_finish=on_finish,
        label=f"behaviour-{n}p",
        verbose=verbose,
    )

    result = pooled.as_dict()
    result["learner"] = learner.name
    result["num_players"] = n
    result["per_table"] = {label: acc.as_dict() for label, acc in per_table.items()}
    result["vs_honest"] = per_table["Honest"].as_dict()
    return result


# --------------------------------------------------------------------------- #
# Impossible-claim probe
# --------------------------------------------------------------------------- #
@dataclass
class ProbeState:
    """A constructed decision state in which the pending claim is impossible."""

    engine: CoupEngine
    viewer: int
    role: int
    claimant: int
    block_phase: bool
    obs: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    mask: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros(0, dtype=np.int8)
    )

    @property
    def challenge_id(self) -> int:
        return int(self.engine.spec.challenge)

    @property
    def pass_id(self) -> int:
        return int(self.engine.spec.pass_)


def _draw_card(counts: dict[int, int], rng: np.random.Generator) -> int | None:
    pool = [role for role, count in counts.items() for _ in range(int(count))]
    if not pool:
        return None
    card = int(pool[int(rng.integers(len(pool)))])
    counts[card] -= 1
    return card


#: Main action that claims each role (used to build realistic pending actions).
_CLAIM_ACTION_FOR_ROLE: dict[int, ActionType] = {
    int(Role.DUKE): ActionType.TAX,
    int(Role.ASSASSIN): ActionType.ASSASSINATE,
    int(Role.CAPTAIN): ActionType.STEAL,
    int(Role.AMBASSADOR): ActionType.EXCHANGE,
}

#: The action that can be blocked by each role (Contessa-style blocks).
_BLOCKABLE_ACTION_FOR_ROLE: dict[int, ActionType] = {
    int(Role.DUKE): ActionType.FOREIGN_AID,
    int(Role.CONTESSA): ActionType.ASSASSINATE,
    int(Role.CAPTAIN): ActionType.STEAL,
    int(Role.AMBASSADOR): ActionType.STEAL,
}


def build_impossible_claim_state(
    rng: np.random.Generator,
    num_players: int,
    *,
    block_phase: bool = False,
    viewer: int = 0,
    max_attempts: int = 64,
) -> ProbeState:
    """Build a legal, randomized state where the viewer knows the claim is false.

    The construction enforces

    * a legal 15-card deal (hands + revealed + deck),
    * ``own_hand[role] + revealed[role] == 3``, so the engine reports
      ``claimed_impossible == 1.0`` in the viewer's observation,
    * at least two living players and a living claimant that does **not** hold
      the claimed role, i.e. the claim is genuinely a bluff,
    * a ``CHALLENGE_ACTION`` / ``CHALLENGE_BLOCK`` decision for the viewer with
      exactly two legal actions (pass and challenge).
    """
    n = int(num_players)
    if not 2 <= n <= 6:
        raise ValueError(f"num_players must be in 2..6, got {n}")
    viewer = int(viewer) % n
    for _ in range(int(max_attempts)):
        state = _try_build_probe_state(rng, n, bool(block_phase), viewer)
        if state is not None:
            return state
    raise RuntimeError("could not build a valid impossible-claim state")


def _try_build_probe_state(
    rng: np.random.Generator, n: int, block_phase: bool, viewer: int
) -> ProbeState | None:
    blockable_roles = [int(role) for role in Role if int(role) != int(Role.ASSASSIN)]
    claimable_roles = [int(role) for role in Role if int(role) != int(Role.CONTESSA)]
    role = (
        int(rng.choice(blockable_roles))
        if block_phase
        else int(rng.choice(claimable_roles))
    )

    own = int(rng.integers(0, 3))  # copies of `role` in the viewer's hand
    role_reveals = COPIES_PER_ROLE - own  # the rest of the role, already revealed
    hand_size = int(rng.integers(max(1, own), 3))
    hand_sizes: dict[int, int] = {viewer: hand_size}
    rev_sizes: dict[int, int] = {viewer: 0}
    others = [seat for seat in range(n) if seat != viewer]
    for seat in others:
        hand_sizes[seat] = int(rng.integers(1, 3)) if rng.random() < 0.75 else 0
    if not any(hand_sizes[seat] > 0 for seat in others):
        seat = int(rng.choice(others))
        hand_sizes[seat] = int(rng.integers(1, 3))

    # Place the remaining role copies into free influence slots.
    slots: list[int] = []
    for seat in range(n):
        slots.extend([seat] * (2 - hand_sizes.get(seat, 0)))
    if len(slots) < role_reveals:
        return None
    rng.shuffle(slots)
    role_reveal_seats = slots[:role_reveals]
    for seat in role_reveal_seats:
        rev_sizes[seat] = rev_sizes.get(seat, 0) + 1
    for seat in range(n):
        capacity = 2 - hand_sizes.get(seat, 0) - rev_sizes.get(seat, 0)
        if capacity > 0:
            rev_sizes[seat] = rev_sizes.get(seat, 0) + int(rng.integers(0, capacity + 1))

    counts: dict[int, int] = {r: COPIES_PER_ROLE for r in range(NUM_ROLES)}
    counts[role] -= own
    counts[role] -= len(role_reveal_seats)
    if any(value < 0 for value in counts.values()):  # pragma: no cover - defensive
        return None

    hands: dict[int, list[int]] = {seat: [] for seat in range(n)}
    reveals: dict[int, list[int]] = {seat: [] for seat in range(n)}
    hands[viewer] = [role] * own
    for seat in role_reveal_seats:
        reveals[seat].append(role)
    for _ in range(hand_sizes[viewer] - own):
        card = _draw_card(counts, rng)
        if card is None:
            return None
        hands[viewer].append(card)
    for seat in range(n):
        while len(hands[seat]) < hand_sizes.get(seat, 0):
            card = _draw_card(counts, rng)
            if card is None:
                return None
            hands[seat].append(card)
        while len(reveals[seat]) < rev_sizes.get(seat, 0):
            card = _draw_card(counts, rng)
            if card is None:
                return None
            reveals[seat].append(card)
    deck = [r for r, count in counts.items() for _ in range(int(count))]

    engine = CoupEngine(n, max_turns=200)
    engine.reset(seed=int(rng.integers(2**31 - 1)))
    engine.debug_set_hands(
        hands={seat: sorted(cards) for seat, cards in hands.items()},
        revealed={seat: sorted(cards) for seat, cards in reveals.items()},
        coins={seat: int(rng.integers(2, 11)) for seat in range(n)},
        deck=deck,
    )
    if engine.total_cards() != 15:  # pragma: no cover - defensive
        return None

    alive_others = [
        seat for seat in range(n) if seat != viewer and engine.players[seat].alive
    ]
    if not alive_others:
        return None
    claimant = int(rng.choice(alive_others))
    if role in engine.players[claimant].hand:  # pragma: no cover - construction
        return None

    engine.turn = int(rng.integers(0, 40))
    engine.current_player = viewer
    engine._poll_order = [viewer]
    engine._poll_idx = 0
    engine.terminated = False
    engine.truncated = False
    engine._last_action = None

    if block_phase:
        action = _BLOCKABLE_ACTION_FOR_ROLE[role]
        actor_choices = [seat for seat in alive_others if seat != claimant]
        actor = int(rng.choice(actor_choices)) if actor_choices else claimant
        engine.pending_actor = actor
        engine.pending_action = action
        engine.pending_blocker = claimant
        engine.pending_block_role = Role(role)
        engine.pending_target = claimant if action == ActionType.STEAL else None
        engine.phase = Phase.CHALLENGE_BLOCK
    else:
        action = _CLAIM_ACTION_FOR_ROLE[role]
        engine.pending_actor = claimant
        engine.pending_action = action
        engine.pending_blocker = None
        engine.pending_block_role = None
        engine.pending_target = (
            claimant if action in (ActionType.STEAL, ActionType.ASSASSINATE) else None
        )
        engine.phase = Phase.CHALLENGE_ACTION

    return ProbeState(
        engine=engine,
        viewer=viewer,
        role=role,
        claimant=claimant,
        block_phase=bool(block_phase),
        obs=engine.get_obs(viewer),
        mask=engine.legal_mask(),
    )


def impossible_claim_probe(
    competitors: Sequence[Competitor],
    *,
    num_players: int = 2,
    scenarios: int = 1_000,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Measure real card-counting deduction on provably impossible claims.

    Half of the scenarios are ``CHALLENGE_ACTION`` windows and half are
    ``CHALLENGE_BLOCK`` windows.  A perfect card counter challenges 100% of the
    time; a uniform random policy challenges 50%.
    """
    rng = np.random.default_rng(seed)
    states: list[ProbeState] = []
    for i in range(int(scenarios)):
        states.append(
            build_impossible_claim_state(
                rng,
                int(num_players),
                block_phase=bool(i % 2),
                viewer=int(rng.integers(num_players)),
            )
        )
    obs_batch = np.stack([state.obs for state in states])
    mask_batch = np.stack([state.mask for state in states])
    challenge_ids = np.array([state.challenge_id for state in states], dtype=np.int64)
    block_flags = np.array([state.block_phase for state in states], dtype=bool)
    # A single pass over the constructed states covers both phase subsets.

    results: dict[str, Any] = {}
    for competitor in competitors:
        if competitor.neural:
            actions = np.asarray(
                competitor.batched_actions(obs_batch, mask_batch), dtype=np.int64
            )
            deterministic = np.asarray(
                competitor.deterministic_actions(obs_batch, mask_batch), dtype=np.int64
            )
            probs = competitor.batched_probs(obs_batch, mask_batch)
            challenge_probs = probs[np.arange(len(states)), challenge_ids]
            mean_prob: float | None = float(np.mean(challenge_probs))
        else:
            # One agent instance for the whole probe so its RNG stream is
            # reproducible from ``seed``.
            agent = competitor.new_agent(seed=seed)
            actions = np.array(
                [int(agent.select_action(state.obs, state.mask)) for state in states],
                dtype=np.int64,
            )
            deterministic = actions
            challenge_probs = np.zeros(len(states), dtype=np.float64)
            mean_prob = None
        challenged = actions == challenge_ids
        record = wilson_record(int(challenged.sum()), len(states))
        record.update(
            {
                "agent": competitor.name,
                "challenge_rate": float(challenged.mean()),
                "deterministic_challenge_rate": float(
                    (deterministic == challenge_ids).mean()
                ),
                "mean_challenge_prob": mean_prob,
                "challenge_rate_action_phase": float(challenged[~block_flags].mean())
                if (~block_flags).any()
                else 0.0,
                "challenge_rate_block_phase": float(challenged[block_flags].mean())
                if block_flags.any()
                else 0.0,
                "legal_action_count": float(mask_batch.sum(axis=1).mean()),
                "claimed_impossible_in_obs": 1.0,
            }
        )
        results[competitor.name] = record
        if verbose:
            suffix = (
                f" p_challenge={mean_prob:.3f}" if mean_prob is not None else ""
            )
            print(
                f"  probe {competitor.name:<12s} challenge={record['challenge_rate']:.3f} "
                f"[{record['ci_low']:.3f}, {record['ci_high']:.3f}]{suffix}",
                flush=True,
            )
    return {
        "scenarios": int(scenarios),
        "num_players": int(num_players),
        "agents": results,
    }


# --------------------------------------------------------------------------- #
# Approximate exploitability
# --------------------------------------------------------------------------- #
EXPLOITER_RUNS: tuple[tuple[str, str], ...] = (
    ("vs_pfsp_2p", "PFSP-PPO"),
    ("vs_naive_2p", "Naive-PPO"),
    ("vs_nfsp_2p", "NFSP"),
    ("vs_ddqn_2p", "Double-DQN"),
)


def train_exploiter(
    target_checkpoint: str | Path,
    save_dir: str | Path,
    *,
    total_steps: int = 500_000,
    seed: int = 0,
    device: str = "auto",
    num_envs: int = 128,
    verbose: bool = True,
) -> Path:
    """Train a fresh PPO learner against one frozen checkpoint."""
    from rloup.training.train_ppo import PPOTrainer, TrainConfig

    device = resolve_device(device)
    config = TrainConfig(
        num_players=2,
        mode="exploiter",
        exploiter_target=str(target_checkpoint),
        total_steps=int(total_steps),
        num_envs=int(num_envs),
        rollout_steps=8192,
        eval_interval=10**6,  # scripted baselines are irrelevant for an exploiter
        eval_games=40,
        save_dir=str(save_dir),
        seed=int(seed),
        device=device,
        num_snapshots=0,
        verbose=verbose,
    )
    PPOTrainer(config).train()
    return Path(save_dir) / "final.pt"


def exploitability(
    competitors: dict[str, Competitor],
    *,
    exploiter_dir: str | Path = "results/exploiters",
    games: int = 1_000,
    total_steps: int = 500_000,
    skip_training: bool = False,
    num_envs: int = 64,
    seed: int = 0,
    device: str = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Train (or reuse) one exploiter per 2-player model and measure it."""
    exploiter_root = Path(exploiter_dir)
    out: dict[str, Any] = {
        "games": int(games),
        "learner_steps": int(total_steps),
        "models": {},
    }
    for run_name, model_name in EXPLOITER_RUNS:
        target = competitors.get(model_name)
        if target is None or target.checkpoint is None:  # pragma: no cover
            continue
        run_dir = exploiter_root / run_name
        checkpoint = run_dir / "final.pt"
        trained_now = False
        if skip_training and checkpoint.exists():
            if verbose:
                print(f"  reusing exploiter checkpoint {checkpoint}", flush=True)
        else:
            if verbose:
                print(
                    f"  training exploiter {run_name} vs {model_name} "
                    f"({total_steps} learner steps)",
                    flush=True,
                )
            started = time.time()
            checkpoint = train_exploiter(
                target.checkpoint,
                run_dir,
                total_steps=total_steps,
                seed=seed + 1,
                device=device,
                verbose=verbose,
            )
            trained_now = True
            if verbose:
                print(f"    trained in {time.time() - started:.1f}s", flush=True)

        exploiter = Competitor(
            name=f"Exploiter-{model_name}",
            ppo=PPOAgent.load(checkpoint, device=device, seed=0),
            checkpoint=str(checkpoint),
        )
        duel = head_to_head(
            exploiter,
            target,
            games=int(games),
            num_envs=num_envs,
            seed=seed + 2,
            verbose=verbose,
        )
        win_rate = duel["rate"]
        out["models"][model_name] = {
            "target": str(target.checkpoint),
            "exploiter_checkpoint": str(checkpoint),
            "trained_now": trained_now,
            "learner_steps": int(total_steps),
            "games": duel["games"],
            "exploiter_wins": duel["a_wins"],
            "target_wins": duel["b_wins"],
            "draws": duel["draws"],
            "exploiter_win_rate": win_rate,
            "ci95": [duel["ci_low"], duel["ci_high"]],
            "ci95_low": duel["ci_low"],
            "ci95_high": duel["ci_high"],
            "approx_exploitability": max(0.0, win_rate - 0.5),
        }
        if verbose:
            print(
                f"  {model_name}: exploiter win rate {win_rate:.3f} "
                f"[{duel['ci_low']:.3f}, {duel['ci_high']:.3f}] "
                f"-> approx_exploitability {max(0.0, win_rate - 0.5):.3f}",
                flush=True,
            )
    return out


# --------------------------------------------------------------------------- #
# Snapshot Elo progression
# --------------------------------------------------------------------------- #
def snapshot_elo_progression(
    run_dir: str | Path,
    baseline_elos: dict[str, float],
    *,
    games_per_baseline: int = 200,
    num_envs: int = 64,
    seed: int = 0,
    max_turns: int = 200,
    device: str = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    """Score every saved snapshot against the fixed five-baseline anchors."""
    run_dir = Path(run_dir)
    device = resolve_device(device)
    snapshot_dir = run_dir / "snapshots"
    if not snapshot_dir.exists():
        return {"run": str(run_dir), "snapshots": [], "note": "no snapshots directory"}
    paths = sorted(snapshot_dir.glob("*.pt"))
    if not paths:
        return {"run": str(run_dir), "snapshots": [], "note": "no snapshots"}
    anchor_elos = {
        display: float(baseline_elos[display])
        for display in BASELINE_KEY_TO_DISPLAY.values()
        if display in baseline_elos
    }
    rows: list[dict[str, Any]] = []
    for path in paths:
        agent = PPOAgent.load(path, device=device, seed=0)
        meta = dict(agent.metadata or {})
        result = evaluate_vs_baselines(
            agent.batched_sample,
            2,
            games_per_baseline=int(games_per_baseline),
            num_envs=num_envs,
            seed=seed,
            max_turns=max_turns,
        )
        elo = elo_from_win_rates(
            {
                BASELINE_KEY_TO_DISPLAY.get(key, key): float(value)
                for key, value in result.win_rates.items()
            },
            anchor_elos,
            games_per_opponent=int(games_per_baseline),
        )
        rows.append(
            {
                "snapshot": path.name,
                "step": int(meta.get("step", 0)),
                "elo": elo,
                "avg_vs_baselines": float(result.avg_vs_baselines),
                "strong_avg": float(result.strong_avg),
                "win_rates": {
                    BASELINE_KEY_TO_DISPLAY.get(key, key): float(value)
                    for key, value in result.win_rates.items()
                },
            }
        )
        if verbose:
            print(
                f"    {path.name}: step={rows[-1]['step']:>9d} "
                f"elo={elo:7.1f} avg={result.avg_vs_baselines:.3f}",
                flush=True,
            )
    rows.sort(key=lambda row: row["step"])
    return {
        "run": str(run_dir),
        "games_per_baseline": int(games_per_baseline),
        "anchor_elos": anchor_elos,
        "snapshots": rows,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rloup.eval.evaluate",
        description="Full Phase 3 evaluation suite for the trained rloup models.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--games-per-pair", type=int, default=1_000)
    parser.add_argument("--games-per-table", type=int, default=1_000)
    parser.add_argument(
        "--behaviour-games",
        type=int,
        default=500,
        help="games per behavioural table (4 tables)",
    )
    parser.add_argument("--probe-scenarios", type=int, default=1_000)
    parser.add_argument("--exploiter-games", type=int, default=1_000)
    parser.add_argument("--exploiter-steps", type=int, default=500_000)
    parser.add_argument("--exploiter-dir", default="results/exploiters")
    parser.add_argument("--snapshot-games", type=int, default=200)
    parser.add_argument(
        "--skip-exploiters", action="store_true", help="reuse saved exploiter checkpoints"
    )
    parser.add_argument("--skip-snapshots", action="store_true")
    parser.add_argument("--skip-round-robin", action="store_true")
    parser.add_argument("--skip-4p", action="store_true")
    parser.add_argument("--skip-behaviour", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--output", default="results/eval_summary.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    started = time.time()
    results_root = Path(args.results_root)
    section_timing: dict[str, float] = {}

    def stamp(name: str, section_start: float) -> None:
        section_timing[name] = time.time() - section_start
        if verbose:
            print(f"  [{name}] {section_timing[name]:.1f}s", flush=True)

    if verbose:
        print("loading 2-player models", flush=True)
    two_player = load_competitors(device=device, verbose=verbose)
    summary: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "device": str(device),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "max_turns": args.max_turns,
            "games_per_pair": args.games_per_pair,
            "games_per_table": args.games_per_table,
            "behaviour_games_per_table": args.behaviour_games,
            "probe_scenarios": args.probe_scenarios,
            "exploiter_games": args.exploiter_games,
            "exploiter_steps": args.exploiter_steps,
        },
        "models": {
            comp.name: comp.checkpoint for comp in two_player if comp.checkpoint is not None
        },
    }

    # -- 2-player round robin ------------------------------------------- #
    if not args.skip_round_robin:
        if verbose:
            print("\n[1/6] 2-player round robin", flush=True)
        section_start = time.time()
        summary["round_robin_2p"] = round_robin_2p(
            two_player,
            games_per_pair=args.games_per_pair,
            num_envs=args.num_envs,
            seed=args.seed,
            max_turns=args.max_turns,
            verbose=verbose,
        )
        stamp("round_robin_2p", section_start)
        if verbose:
            for row in summary["round_robin_2p"]["ranking"]:
                print(
                    f"    {row['agent']:<12s} elo={row['elo']:7.1f} "
                    f"avg_win_rate={row['win_rate']:.3f}",
                    flush=True,
                )
    else:
        summary["round_robin_2p"] = {}

    baseline_elos: dict[str, float] = {
        name: 1000.0 for name in BASELINE_KEY_TO_DISPLAY.values()
    }
    baseline_elos.update(summary["round_robin_2p"].get("elo", {}))

    # -- 4-player ------------------------------------------------------- #
    if not args.skip_4p:
        if verbose:
            print("\n[2/6] 4-player tables", flush=True)
        section_start = time.time()
        four_p = model_competitors(device=device, names=["PFSP-PPO-4P"])["PFSP-PPO-4P"]
        summary["four_player"] = evaluate_tables(
            four_p,
            four_player_tables(),
            games_per_table=args.games_per_table,
            num_players=4,
            num_envs=max(32, args.num_envs // 2),
            seed=args.seed,
            max_turns=args.max_turns,
            verbose=verbose,
        )
        stamp("four_player", section_start)
        if verbose:
            for name, record in summary["four_player"]["tables"].items():
                print(
                    f"    {name:<16s} win={record['rate']:.3f} "
                    f"[{record['ci_low']:.3f}, {record['ci_high']:.3f}] "
                    f"rank={record['avg_finish_rank']:.2f}",
                    flush=True,
                )
    else:
        summary["four_player"] = {}

    # -- behaviour ------------------------------------------------------ #
    if not args.skip_behaviour:
        if verbose:
            print("\n[3/6] behavioural diagnostics", flush=True)
        section_start = time.time()
        behaviour: dict[str, Any] = {}
        for comp in two_player:
            if not comp.neural:
                continue
            behaviour[comp.name] = collect_behaviour(
                comp,
                num_players=2,
                games_per_table=args.behaviour_games,
                num_envs=args.num_envs,
                seed=args.seed,
                max_turns=args.max_turns,
                verbose=verbose,
            )
        four_p = model_competitors(device=device, names=["PFSP-PPO-4P"])["PFSP-PPO-4P"]
        behaviour["PFSP-PPO-4P"] = collect_behaviour(
            four_p,
            num_players=4,
            games_per_table=args.behaviour_games,
            num_envs=max(32, args.num_envs // 2),
            seed=args.seed,
            max_turns=args.max_turns,
            verbose=verbose,
        )
        if Path(MODEL_RUNS["Double-DQN-4P"]).exists():
            ddqn_4p = model_competitors(device=device, names=["Double-DQN-4P"])["Double-DQN-4P"]
            behaviour["Double-DQN-4P"] = collect_behaviour(
                ddqn_4p,
                num_players=4,
                games_per_table=args.behaviour_games,
                num_envs=max(32, args.num_envs // 2),
                seed=args.seed,
                max_turns=args.max_turns,
                verbose=verbose,
            )
        summary["behaviour"] = behaviour
        stamp("behaviour", section_start)
    else:
        summary["behaviour"] = {}

    # -- impossible-claim probe ----------------------------------------- #
    if not args.skip_probe:
        if verbose:
            print("\n[4/6] impossible-claim card-counting probe", flush=True)
        section_start = time.time()
        probe_agents: list[Competitor] = []
        seen: set[str] = set()
        for comp in list(two_player) + [
            scripted_competitor("CardCounter"),
            scripted_competitor("Honest"),
            scripted_competitor("Random"),
        ]:
            if comp.name in seen:
                continue
            seen.add(comp.name)
            probe_agents.append(comp)
        four_p = model_competitors(device=device, names=["PFSP-PPO-4P"])["PFSP-PPO-4P"]
        four_p_probe_agents = [four_p]
        if Path(MODEL_RUNS["Double-DQN-4P"]).exists():
            four_p_probe_agents.append(
                model_competitors(device=device, names=["Double-DQN-4P"])["Double-DQN-4P"]
            )
        four_p_probe_agents.extend(
            [
                scripted_competitor("CardCounter"),
                scripted_competitor("Honest"),
                scripted_competitor("Random"),
            ]
        )
        summary["impossible_claim_probe"] = {
            "two_player": impossible_claim_probe(
                probe_agents,
                num_players=2,
                scenarios=args.probe_scenarios,
                seed=args.seed,
                verbose=verbose,
            ),
            "four_player": impossible_claim_probe(
                four_p_probe_agents,
                num_players=4,
                scenarios=args.probe_scenarios,
                seed=args.seed + 1,
                verbose=verbose,
            ),
        }
        stamp("impossible_claim_probe", section_start)
    else:
        summary["impossible_claim_probe"] = {}

    # -- exploitability -------------------------------------------------- #
    if verbose:
        print("\n[5/6] approximate exploitability", flush=True)
    section_start = time.time()
    summary["exploitability_2p"] = exploitability(
        {comp.name: comp for comp in two_player},
        exploiter_dir=args.exploiter_dir,
        games=args.exploiter_games,
        total_steps=args.exploiter_steps,
        skip_training=args.skip_exploiters,
        num_envs=args.num_envs,
        seed=args.seed,
        device=device,
        verbose=verbose,
    )
    stamp("exploitability", section_start)

    # -- snapshots ------------------------------------------------------- #
    if not args.skip_snapshots:
        if verbose:
            print("\n[6/6] snapshot Elo progression", flush=True)
        section_start = time.time()
        snapshot_progress: dict[str, Any] = {}
        for name, relative in (("PFSP-PPO", "ppo_pfsp_2p"), ("Naive-PPO", "ppo_naive_2p")):
            if verbose:
                print(f"  {name} ({relative})", flush=True)
            snapshot_progress[name] = snapshot_elo_progression(
                results_root / relative,
                baseline_elos,
                games_per_baseline=args.snapshot_games,
                num_envs=max(32, args.num_envs // 2),
                seed=args.seed,
                max_turns=args.max_turns,
                device=device,
                verbose=verbose,
            )
        summary["snapshot_elo"] = snapshot_progress
        stamp("snapshot_elo", section_start)
    else:
        summary["snapshot_elo"] = {}

    summary["timing"] = {
        "sections": section_timing,
        "total_seconds": time.time() - started,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if verbose:
        print(
            f"\nwrote {output} in {summary['timing']['total_seconds']:.1f}s", flush=True
        )
    return summary


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
