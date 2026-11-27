"""Vectorised self-play / PFSP Action-Masked Double DQN trainer for Coup.

Supports 2-6 players and two training modes:

``pfsp`` (default)
    Prioritised Fictitious Self-Play opponent pool: 25% current Q-network,
    60% win-rate-prioritised frozen Q-network snapshots, 15% scripted
    baselines.  Transitions are recorded from the learner seat only.
``naive``
    Pure self-play: every seat uses the current Q-network and contributes its
    own per-seat transition chain to the replay buffer.

Metrics are written to ``<save_dir>/metrics.jsonl`` using the exact same schema
as :mod:`rloup.training.train_ppo` and :mod:`rloup.training.train_nfsp`.

Example::

    python -m rloup.training.train_dqn --num-players 2 --mode pfsp \\
        --total-steps 1000000 --num-envs 64 --seed 42 --save-dir results/ddqn_pfsp_2p
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from rloup.agents import BASELINES, Agent
from rloup.agents.dqn import DoubleDQNAgent, DoubleDQNHyperparams
from rloup.agents.nfsp import PerSeatTransitionRecorder, QNetwork
from rloup.constants import num_actions_for
from rloup.engine import CoupEngine, obs_dim

from .metrics import MetricsWriter
from .pool import PRIOR_WIN_RATE, OpponentSpec, update_elo
from .vec import BehaviourStats, EvaluationResult, evaluate_vs_baselines

__all__ = ["DoubleDQNConfig", "DoubleDQNTrainer", "QOpponentPool", "build_parser", "main"]


@dataclass
class _QSnapshot:
    index: int
    key: str
    step: int
    elo: float
    module: QNetwork
    win_rate: float = PRIOR_WIN_RATE


class QOpponentPool:
    """PFSP opponent pool storing frozen :class:`QNetwork` snapshots + scripted bots."""

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
        max_snapshots: int = 32,
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
        self.snapshots: list[_QSnapshot] = []
        self._by_index: dict[int, _QSnapshot] = {}
        self._win_rates: dict[str, float] = {"latest": PRIOR_WIN_RATE}
        for cls in self.scripted_classes:
            self._win_rates[f"scripted:{cls.name}"] = PRIOR_WIN_RATE
        self.learner_elo = 1000.0
        self._next_index = 0

    def add_snapshot(self, q_network: QNetwork, step: int = 0) -> _QSnapshot:
        index = self._next_index
        self._next_index += 1
        module = QNetwork(self.obs_dim, self.n_actions, self.hidden, self.num_layers).to(
            self.device
        )
        module.load_state_dict(
            {k: v.detach().clone() for k, v in q_network.state_dict().items()}
        )
        module.eval()
        module.requires_grad_(False)
        snap = _QSnapshot(
            index=index,
            key=f"snap_{index:04d}",
            step=int(step),
            elo=float(self.learner_elo),
            module=module,
        )
        self.snapshots.append(snap)
        self._by_index[index] = snap
        self._win_rates[snap.key] = PRIOR_WIN_RATE
        if len(self.snapshots) > self.max_snapshots:
            dropped = self.snapshots.pop(0)
            self._by_index.pop(dropped.index, None)
            self._win_rates.pop(dropped.key, None)
        return snap

    @property
    def snapshot_count(self) -> int:
        return len(self.snapshots)

    def module_for(self, spec: OpponentSpec) -> QNetwork | None:
        if spec.kind != "snapshot":
            return None
        return self._by_index[int(spec.index)].module

    def scripted_agent(self, spec: OpponentSpec, seed: int | None = None) -> Agent:
        for cls in self.scripted_classes:
            if cls.name == spec.name:
                return cls(seed=seed)
        raise KeyError(f"unknown scripted opponent {spec.name!r}")

    def weights(self) -> np.ndarray:
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
        roll = float(self.rng.random())
        if roll < self.LATEST_PROB:
            return OpponentSpec(kind="latest", key="latest")
        if roll < self.LATEST_PROB + self.SNAPSHOT_PROB and self.snapshots:
            idx = int(self.rng.choice(len(self.snapshots), p=self.weights()))
            snap = self.snapshots[idx]
            return OpponentSpec(kind="snapshot", key=snap.key, index=snap.index)
        if roll < self.LATEST_PROB + self.SNAPSHOT_PROB:
            return OpponentSpec(kind="latest", key="latest")
        cls = self.scripted_classes[int(self.rng.integers(len(self.scripted_classes)))]
        return OpponentSpec(kind="scripted", key=f"scripted:{cls.name}", name=cls.name)

    def record_result(self, spec: OpponentSpec, learner_score: float) -> None:
        key = spec.key if spec.kind != "scripted" else f"scripted:{spec.name}"
        old = self._win_rates.get(key, PRIOR_WIN_RATE)
        self._win_rates[key] = (1.0 - self.ema_alpha) * old + self.ema_alpha * float(
            learner_score
        )
        if spec.kind == "snapshot" and int(spec.index) in self._by_index:
            snap = self._by_index[int(spec.index)]
            snap.win_rate = self._win_rates[key]
            self.learner_elo, snap.elo = update_elo(
                self.learner_elo, snap.elo, float(learner_score)
            )


@dataclass
class DoubleDQNConfig:
    """Trainer configuration (mirrored to ``config.json``)."""

    num_players: int = 2
    mode: str = "pfsp"  # "pfsp" | "naive"
    total_steps: int = 1_000_000
    num_envs: int = 64
    iteration_steps: int = 8192
    eval_interval: int = 12
    snapshot_interval: int = 12
    eval_games: int = 200
    save_dir: str = "results/ddqn_pfsp_2p"
    seed: int = 42
    device: str = "auto"
    hidden: int = 256
    num_layers: int = 3
    max_turns: int = 200
    reward_mode: str = "auto"
    epsilon_start: float = 0.25
    epsilon_end: float = 0.02
    epsilon_decay_frac: float = 0.70
    gamma: float = 0.99
    lr: float = 2e-4
    batch_size: int = 256
    replay_capacity: int = 200_000
    warmup: int = 4_000
    target_update: int = 500
    tau: float = 0.0
    updates_per_tick: int = 8
    verbose: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.num_players <= 6:
            raise ValueError(f"num_players must be in 2..6, got {self.num_players}")
        if self.mode not in ("pfsp", "naive"):
            raise ValueError(f"mode must be 'pfsp' or 'naive', got {self.mode!r}")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class DoubleDQNTrainer:
    """Vectorised Action-Masked Double DQN trainer with optional PFSP pool."""

    def __init__(self, config: DoubleDQNConfig) -> None:
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
        self.hyper = DoubleDQNHyperparams(
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay_frac=config.epsilon_decay_frac,
            gamma=config.gamma,
            lr=config.lr,
            batch_size=config.batch_size,
            replay_capacity=config.replay_capacity,
            warmup=config.warmup,
            target_update=config.target_update,
            tau=config.tau,
        )
        self.agent = DoubleDQNAgent(
            self.obs_dim,
            self.n_actions,
            hidden=config.hidden,
            num_layers=config.num_layers,
            device=self.device,
            seed=config.seed,
            hyper=self.hyper,
        )
        self.pool = (
            QOpponentPool(
                self.obs_dim,
                self.n_actions,
                device=self.device,
                hidden=config.hidden,
                num_layers=config.num_layers,
                seed=config.seed + 17,
            )
            if config.mode == "pfsp"
            else None
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
        for i, eng in enumerate(self.engines):
            eng.reset(seed=config.seed * 7_919 + i)

        self.recorder = PerSeatTransitionRecorder()
        self.learner_seats: list[int] = [0] * config.num_envs
        self.opp_specs: list[list[OpponentSpec | None]] = []
        self.scripted_handles: list[dict[int, Agent]] = [{} for _ in range(config.num_envs)]
        for i in range(config.num_envs):
            self.opp_specs.append([None] * self.num_players)
            self._assign_opponents(i)

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

    def _assign_opponents(self, env_idx: int) -> None:
        self.scripted_handles[env_idx] = {}
        if self.cfg.mode == "naive" or self.pool is None:
            self.learner_seats[env_idx] = -1  # all seats learn
            self.opp_specs[env_idx] = [None] * self.num_players
            return
        learner = int(self.rng.integers(self.num_players))
        self.learner_seats[env_idx] = learner
        specs: list[OpponentSpec | None] = [None] * self.num_players
        for seat in range(self.num_players):
            if seat == learner:
                continue
            spec = self.pool.sample()
            specs[seat] = spec
            if spec.kind == "scripted":
                self.scripted_handles[env_idx][seat] = self.pool.scripted_agent(
                    spec, seed=int(self.rng.integers(2**31 - 1))
                )
        self.opp_specs[env_idx] = specs

    def collect(self, steps: int) -> int:
        agent = self.agent
        start = self.steps_taken
        while self.steps_taken - start < steps:
            epsilon = DoubleDQNAgent.epsilon_at(
                self.steps_taken / max(1, self.cfg.total_steps), self.hyper
            )
            learner_rows: list[tuple[int, int, np.ndarray, np.ndarray]] = []
            latest_opp_rows: list[tuple[int, int, np.ndarray, np.ndarray]] = []
            snap_buckets: dict[int, list[tuple[int, int, np.ndarray, np.ndarray]]] = {}
            actions = [0] * len(self.engines)
            acting_meta: list[tuple[int, int, np.ndarray, np.ndarray, bool]] = []

            for i, eng in enumerate(self.engines):
                seat = eng.current_player
                obs = eng.get_obs(seat)
                mask = eng.legal_mask()
                lseat = self.learner_seats[i]
                if lseat == -1 or seat == lseat:
                    learner_rows.append((i, seat, obs, mask))
                    acting_meta.append((i, seat, obs, mask, True))
                else:
                    spec = self.opp_specs[i][seat]
                    assert spec is not None
                    if spec.kind == "scripted":
                        actions[i] = int(
                            self.scripted_handles[i][seat].select_action(obs, mask)
                        )
                    elif spec.kind == "latest":
                        latest_opp_rows.append((i, seat, obs, mask))
                    else:
                        snap_buckets.setdefault(int(spec.index), []).append(
                            (i, seat, obs, mask)
                        )
                    acting_meta.append((i, seat, obs, mask, False))

            if learner_rows:
                chosen = agent.batched_actions(
                    np.stack([r[2] for r in learner_rows]),
                    np.stack([r[3] for r in learner_rows]),
                    epsilon=epsilon,
                    deterministic=False,
                )
                for (i, _, _, _), act in zip(learner_rows, chosen):
                    actions[i] = int(act)

            if latest_opp_rows:
                chosen = agent.batched_actions(
                    np.stack([r[2] for r in latest_opp_rows]),
                    np.stack([r[3] for r in latest_opp_rows]),
                    epsilon=0.02,
                    deterministic=False,
                )
                for (i, _, _, _), act in zip(latest_opp_rows, chosen):
                    actions[i] = int(act)

            if snap_buckets and self.pool is not None:
                with torch.inference_mode():
                    for snap_idx, rows in snap_buckets.items():
                        mod = self.pool._by_index[snap_idx].module
                        obs_t = torch.as_tensor(
                            np.stack([r[2] for r in rows]),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        mask_t = torch.as_tensor(
                            np.stack([r[3] for r in rows]), device=self.device
                        )
                        chosen = mod.masked(obs_t, mask_t).argmax(dim=-1).cpu().numpy()
                        for (i, _, _, _), act in zip(rows, chosen):
                            actions[i] = int(act)

            for i, seat, obs, mask, is_learner in acting_meta:
                act = actions[i]
                if is_learner:
                    trans = self.recorder.add_decision(seat, obs, mask, act, scope=i)
                    if trans is not None:
                        agent.replay.push(trans)
                    self.steps_taken += 1
                self.engines[i].step(act)
                if self.engines[i].done:
                    self._finish_episode(i)

            if agent.ready:
                for _ in range(self.cfg.updates_per_tick):
                    td_loss, mean_q = agent.update()
                    self._record("value_loss", td_loss)
                    self._record("mean_q", mean_q)
        return self.steps_taken - start

    def _finish_episode(self, env_idx: int) -> None:
        eng = self.engines[env_idx]
        for trans in self.recorder.finish_episode(eng.rewards, scope=env_idx):
            self.agent.replay.push(trans)

        lseat = self.learner_seats[env_idx]
        stats = eng.get_episode_stats()["players"]
        if lseat == -1:
            for s in range(self.num_players):
                self.rollout_behaviour.add_player_stats(stats[s])
        else:
            self.rollout_behaviour.add_player_stats(stats[lseat])
            if self.pool is not None:
                winner = None if eng.truncated else eng.winner()
                score = 0.5 if winner is None else (1.0 if winner == lseat else 0.0)
                for seat, spec in enumerate(self.opp_specs[env_idx]):
                    if spec is not None and seat != lseat:
                        self.pool.record_result(spec, score)

        self.episodes += 1
        eng.reset(seed=None)
        self._assign_opponents(env_idx)

    # -- metrics & eval -------------------------------------------------- #
    def _record(self, key: str, value: float) -> None:
        self._loss_window.setdefault(key, []).append(float(value))

    def _drain_loss_window(self) -> dict[str, float]:
        out = {
            k: float(np.mean(v)) if v else 0.0 for k, v in self._loss_window.items()
        }
        self._loss_window = {}
        return out

    def eval_policy(self):
        return lambda obs, mask: self.agent.batched_actions(
            obs, mask, epsilon=0.0, deterministic=True
        )

    def evaluate(self) -> EvaluationResult:
        res = evaluate_vs_baselines(
            self.eval_policy(),
            self.num_players,
            games_per_baseline=self.cfg.eval_games,
            num_envs=min(self.cfg.num_envs * 2, 128),
            seed=self.cfg.seed,
            max_turns=self.cfg.max_turns,
        )
        self.last_eval = res
        return res

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
                "mode": self.cfg.mode,
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
            "learner_elo": self.pool.learner_elo if self.pool is not None else None,
            "snapshot_count": self.pool.snapshot_count if self.pool is not None else 0,
        }
        record.update(evaluation.as_dict())
        record.update(
            {
                "policy_loss": 0.0,
                "value_loss": losses.get("value_loss", 0.0),
                "entropy": 0.0,
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
                f"[ddqn-{self.cfg.mode} {self.num_players}p] step={self.steps_taken:>9d} "
                f"it={self.iterations:>4d} avg={evaluation.avg_vs_baselines:.3f} "
                f"strong={evaluation.strong_avg:.3f} {rates} "
                f"td={losses.get('value_loss', 0.0):.4f} "
                f"fps={record['steps_per_second']:.0f}",
                flush=True,
            )
        return record

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
                    f"[ddqn-{self.cfg.mode} {self.num_players}p] device={self.device} "
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
                if (
                    self.pool is not None
                    and self.iterations % max(1, self.cfg.snapshot_interval) == 0
                ):
                    snap = self.pool.add_snapshot(self.agent.q_network, step=self.steps_taken)
                    self._checkpoint(f"snapshots/{snap.key}.pt")
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
            "updates": self.agent.updates,
            "history": self.history,
        }
        self._write_json("summary.json", summary)
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rloup.training.train_dqn",
        description="Vectorised Action-Masked Double DQN trainer for Coup.",
    )
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--mode", choices=("pfsp", "naive"), default="pfsp")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iteration-steps", type=int, default=8192)
    parser.add_argument("--eval-interval", type=int, default=12)
    parser.add_argument("--snapshot-interval", type=int, default=12)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--save-dir", type=str, default="results/ddqn_pfsp_2p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--reward-mode", choices=("auto", "win", "rank"), default="auto")
    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-frac", type=float, default=0.70)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=200_000)
    parser.add_argument("--warmup", type=int, default=4_000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--updates-per-tick", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    payload = vars(args)
    verbose = not payload.pop("quiet")
    config = DoubleDQNConfig(verbose=verbose, **payload)
    trainer = DoubleDQNTrainer(config)
    return trainer.train()


if __name__ == "__main__":  # pragma: no cover
    main()
