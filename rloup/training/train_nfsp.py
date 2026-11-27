"""Vectorised self-play NFSP trainer for Coup.

Same collection machinery as the PPO trainer: dozens of engines advance in
lock-step and every decision of a tick is batched onto the GPU.  Each seat of
each environment flips a coin at the start of an episode and then plays either
the masked best response (``QNetwork``) or the masked average policy
(``SLNetwork``) for that whole episode, which is what the anticipatory parameter
``eta`` prescribes.

Metrics are written to ``<save_dir>/metrics.jsonl`` with the *same* schema as
:mod:`rloup.training.train_ppo`, so the two algorithms are directly comparable.

    python -m rloup.training.train_nfsp --num-players 2 --total-steps 1000000 \\
        --save-dir results/nfsp_2p
"""

from __future__ import annotations

import argparse
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rloup.agents.nfsp import NFSPAgent, NFSPHyperparams, PerSeatTransitionRecorder
from rloup.constants import num_actions_for
from rloup.engine import CoupEngine, obs_dim

from .metrics import MetricsWriter
from .vec import BehaviourStats, EvaluationResult, evaluate_vs_baselines

__all__ = ["NFSPConfig", "NFSPTrainer", "build_parser", "main"]


@dataclass
class NFSPConfig:
    """Trainer configuration (mirrored to ``config.json``)."""

    num_players: int = 2
    total_steps: int = 1_000_000
    num_envs: int = 64
    iteration_steps: int = 8192
    eval_interval: int = 20
    eval_games: int = 200
    save_dir: str = "results/nfsp"
    seed: int = 0
    device: str = "auto"
    hidden: int = 256
    num_layers: int = 3
    max_turns: int = 200
    reward_mode: str = "auto"
    eta: float = 0.1
    epsilon_start: float = 0.15
    epsilon_end: float = 0.02
    gamma: float = 0.99
    lr_q: float = 2e-4
    lr_sl: float = 5e-4
    batch_size: int = 128
    replay_capacity: int = 150_000
    reservoir_capacity: int = 300_000
    target_update: int = 500
    q_updates_per_tick: int = 8
    sl_updates_per_tick: int = 4
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.num_players != 2:
            raise ValueError("NFSP in rloup is implemented for 2-player Coup")
        if self.num_envs < 1 or self.iteration_steps < 1:
            raise ValueError("num_envs and iteration_steps must be positive")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class NFSPTrainer:
    """Action-masked NFSP self-play with batched GPU inference."""

    def __init__(self, config: NFSPConfig) -> None:
        self.cfg = config
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else torch.device(config.device)
        )
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        self.num_players = int(config.num_players)
        self.obs_dim = obs_dim(self.num_players)
        self.n_actions = num_actions_for(self.num_players)
        self.hyper = NFSPHyperparams(
            eta=config.eta,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            gamma=config.gamma,
            lr_q=config.lr_q,
            lr_sl=config.lr_sl,
            batch_size=config.batch_size,
            replay_capacity=config.replay_capacity,
            reservoir_capacity=config.reservoir_capacity,
            target_update=config.target_update,
        )
        self.agent = NFSPAgent(
            self.obs_dim,
            self.n_actions,
            hidden=config.hidden,
            num_layers=config.num_layers,
            device=self.device,
            seed=config.seed,
            hyper=self.hyper,
        )
        self.engines = [
            CoupEngine(
                self.num_players,
                max_turns=config.max_turns,
                reward_mode=config.reward_mode,
                seed=config.seed * 1_000_003 + i * 101,
            )
            for i in range(config.num_envs)
        ]
        for i, engine in enumerate(self.engines):
            engine.reset(seed=config.seed * 7_919 + i)
        self.recorder = PerSeatTransitionRecorder()
        #: ``(env, seat) -> plays the best response this episode?``
        self.best_response: dict[tuple[int, int], bool] = {}
        #: ``(env, seat) -> [(obs, mask, action)]`` for best-response episodes only.
        self.br_history: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, int]]] = {}
        for i in range(config.num_envs):
            self._sample_modes(i)

        self.save_dir = Path(config.save_dir)
        self.steps_taken = 0
        self.iterations = 0
        self.episodes = 0
        self.best_strong_avg = -1.0
        self._loss_window: dict[str, list[float]] = {}
        self._start = time.time()
        self._steps_at_last_eval = 0
        self._time_at_last_eval = self._start
        self.rollout_behaviour = BehaviourStats()
        self.last_eval: EvaluationResult | None = None
        self.history: list[dict[str, Any]] = []

    # -- episode bookkeeping --------------------------------------------- #
    def _sample_modes(self, index: int) -> None:
        for seat in range(self.num_players):
            key = (index, seat)
            is_br = bool(self.rng.random() < self.hyper.eta)
            self.best_response[key] = is_br
            if is_br:
                self.br_history[key] = []
            else:
                self.br_history.pop(key, None)

    # -- rollout ---------------------------------------------------------- #
    def collect(self, steps: int) -> int:
        """Play until ``steps`` decisions have been taken (and learned from)."""
        agent = self.agent
        batch_size = self.hyper.batch_size
        warmup = max(self.hyper.warmup, batch_size)
        start = self.steps_taken
        while self.steps_taken - start < steps:
            rows: list[tuple[int, int, np.ndarray, np.ndarray]] = []
            for i, engine in enumerate(self.engines):
                seat = engine.current_player
                rows.append((i, seat, engine.get_obs(seat), engine.legal_mask()))
            epsilon = NFSPAgent.epsilon_at(
                self.steps_taken / max(1, self.cfg.total_steps), self.hyper
            )
            acting: list[int] = [0] * len(rows)
            br_rows = [k for k, row in enumerate(rows) if self.best_response[(row[0], row[1])]]
            avg_rows = [k for k, row in enumerate(rows) if not self.best_response[(row[0], row[1])]]
            if br_rows:
                chosen = agent.batched_actions(
                    np.stack([rows[k][2] for k in br_rows]),
                    np.stack([rows[k][3] for k in br_rows]),
                    best_response=True,
                    epsilon=epsilon,
                )
                for slot, k in enumerate(br_rows):
                    acting[k] = int(chosen[slot])
            if avg_rows:
                chosen = agent.batched_actions(
                    np.stack([rows[k][2] for k in avg_rows]),
                    np.stack([rows[k][3] for k in avg_rows]),
                    best_response=False,
                )
                for slot, k in enumerate(avg_rows):
                    acting[k] = int(chosen[slot])
            for k, (i, seat, obs, mask) in enumerate(rows):
                action = acting[k]
                transition = self.recorder.add_decision(seat, obs, mask, action, scope=i)
                if transition is not None:
                    agent.replay.push(transition)
                if self.best_response[(i, seat)]:
                    self.br_history[(i, seat)].append((obs, mask, action))
                self.engines[i].step(action)
                self.steps_taken += 1
                if self.engines[i].done:
                    self._finish_episode(i)
            if len(agent.replay) >= warmup:
                for _ in range(self.cfg.q_updates_per_tick):
                    self._record("value_loss", agent.q_update())
            if len(agent.reservoir) >= batch_size:
                for _ in range(self.cfg.sl_updates_per_tick):
                    loss, entropy = agent.sl_update()
                    self._record("policy_loss", loss)
                    self._record("entropy", entropy)
        return self.steps_taken - start

    def _finish_episode(self, index: int) -> None:
        engine = self.engines[index]
        for transition in self.recorder.finish_episode(engine.rewards, scope=index):
            self.agent.replay.push(transition)
        for seat in range(self.num_players):
            history = self.br_history.pop((index, seat), None)
            if history:
                for obs, mask, action in history:
                    self.agent.reservoir.push(obs, mask, action)
        stats = engine.get_episode_stats()
        for seat in range(self.num_players):
            self.rollout_behaviour.add_player_stats(stats["players"][seat])
        self.episodes += 1
        engine.reset(seed=None)
        self._sample_modes(index)

    # -- metric plumbing -------------------------------------------------- #
    def _record(self, key: str, value: float) -> None:
        self._loss_window.setdefault(key, []).append(float(value))

    def _drain_loss_window(self) -> dict[str, float]:
        out = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in self._loss_window.items()
        }
        self._loss_window = {}
        return out

    def eval_policy(self):
        """Stochastic batched average-policy closure used by the benchmark."""
        return lambda obs, mask: self.agent.batched_actions(
            obs, mask, best_response=False
        )

    def evaluate(self) -> EvaluationResult:
        result = evaluate_vs_baselines(
            self.eval_policy(),
            self.num_players,
            games_per_baseline=self.cfg.eval_games,
            num_envs=min(self.cfg.num_envs * 2, 128),
            seed=self.cfg.seed,
            max_turns=self.cfg.max_turns,
        )
        self.last_eval = result
        return result

    # -- checkpoints ------------------------------------------------------ #
    def _write_json(self, name: str, payload: Any) -> None:
        path = self.save_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _checkpoint(self, name: str, evaluation: EvaluationResult | None = None) -> None:
        self.agent.save(
            self.save_dir / name,
            {
                "step": self.steps_taken,
                "iteration": self.iterations,
                "num_players": self.num_players,
                "strong_avg": None if evaluation is None else evaluation.strong_avg,
                "avg_vs_baselines": None
                if evaluation is None
                else evaluation.avg_vs_baselines,
                "win_rates": None if evaluation is None else evaluation.win_rates,
            },
        )

    def save_best(self, evaluation: EvaluationResult) -> bool:
        if evaluation.strong_avg <= self.best_strong_avg:
            return False
        self.best_strong_avg = evaluation.strong_avg
        self._checkpoint("best.pt", evaluation)
        return True

    # -- logging ---------------------------------------------------------- #
    def _log_metrics(
        self,
        writer: MetricsWriter,
        evaluation: EvaluationResult,
        losses: dict[str, float],
    ) -> dict[str, Any]:
        wall = time.time() - self._start
        window_steps = self.steps_taken - self._steps_at_last_eval
        window_time = time.time() - self._time_at_last_eval
        record: dict[str, Any] = {
            "step": self.steps_taken,
            "update": self.iterations,
            "episodes": self.episodes,
            "wall_time": wall,
            "steps_per_second": window_steps / window_time if window_time > 0 else 0.0,
            "learner_elo": None,
            "snapshot_count": 0,
        }
        record.update(evaluation.as_dict())
        record.update(
            {
                "policy_loss": losses.get("policy_loss", 0.0),
                "value_loss": losses.get("value_loss", 0.0),
                "entropy": losses.get("entropy", 0.0),
                # NFSP has no importance ratio; the fields exist so the schema
                # stays identical to the PPO runs.
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
            }
        )
        writer.write(record)
        self.history.append(record)
        self._steps_at_last_eval = self.steps_taken
        self._time_at_last_eval = time.time()
        if self.cfg.verbose:
            rates = " ".join(
                f"{name}={value:.3f}" for name, value in evaluation.win_rates.items()
            )
            print(
                f"[nfsp {self.num_players}p] step={self.steps_taken:>9d} "
                f"it={self.iterations:>4d} avg={evaluation.avg_vs_baselines:.3f} "
                f"strong={evaluation.strong_avg:.3f} {rates} "
                f"b={evaluation.behaviour.bluff_rate:.2f} "
                f"cp={evaluation.behaviour.challenge_precision:.2f} "
                f"fps={record['steps_per_second']:.0f}",
                flush=True,
            )
        return record

    # -- main loop -------------------------------------------------------- #
    def train(self) -> dict[str, Any]:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        config_payload = self.cfg.as_dict()
        config_payload["device"] = str(self.device)
        config_payload["obs_dim"] = self.obs_dim
        config_payload["n_actions"] = self.n_actions
        self._write_json("config.json", config_payload)
        writer = MetricsWriter(self.save_dir / "metrics.jsonl")
        try:
            if self.cfg.verbose:
                print(
                    f"[nfsp {self.num_players}p] device={self.device} "
                    f"obs_dim={self.obs_dim} n_actions={self.n_actions} "
                    f"envs={self.cfg.num_envs} target_steps={self.cfg.total_steps}",
                    flush=True,
                )
            evaluation = self.evaluate()
            self.save_best(evaluation)
            self._log_metrics(writer, evaluation, {})
            while self.steps_taken < self.cfg.total_steps:
                self.collect(self.cfg.iteration_steps)
                self.iterations += 1
                if self.iterations % max(1, self.cfg.eval_interval) == 0:
                    evaluation = self.evaluate()
                    self.save_best(evaluation)
                    self._log_metrics(writer, evaluation, self._drain_loss_window())
            evaluation = self.evaluate()
            self.save_best(evaluation)
            self._log_metrics(writer, evaluation, self._drain_loss_window())
            self._checkpoint("final.pt", evaluation)
        finally:
            writer.close()
        summary = {
            "config": config_payload,
            "wall_time": time.time() - self._start,
            "steps": self.steps_taken,
            "iterations": self.iterations,
            "episodes": self.episodes,
            "steps_per_second": self.steps_taken / max(1e-9, time.time() - self._start),
            "final_eval": evaluation.as_dict(),
            "best_strong_avg": self.best_strong_avg,
            "q_updates": self.agent.q_updates,
            "sl_updates": self.agent.sl_updates,
            "history": self.history,
        }
        self._write_json("summary.json", summary)
        return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rloup.training.train_nfsp",
        description="Vectorised action-masked NFSP for 2-player Coup.",
    )
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iteration-steps", type=int, default=8192)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--save-dir", type=str, default="results/nfsp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--reward-mode", choices=("auto", "win", "rank"), default="auto")
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--epsilon-start", type=float, default=0.15)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr-q", type=float, default=2e-4)
    parser.add_argument("--lr-sl", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=150_000)
    parser.add_argument("--reservoir-capacity", type=int, default=300_000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--q-updates-per-tick", type=int, default=8)
    parser.add_argument("--sl-updates-per-tick", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    payload = vars(args)
    verbose = not payload.pop("quiet")
    config = NFSPConfig(verbose=verbose, **payload)
    trainer = NFSPTrainer(config)
    return trainer.train()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
