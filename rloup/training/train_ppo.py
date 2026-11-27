"""Vectorised, batched-inference PPO trainer for Coup.

One trainer process drives ``--num-envs`` independent :class:`CoupEngine`
instances.  Because the engine is an agent-environment-cycle simulator, the
collector works in *ticks*: on every tick the observation of every currently
acting seat is gathered into one batch, a single GPU forward pass produces all
actions, and all engines advance together.  Opponents are grouped by identity,
so a whole rollout needs at most a handful of forward passes per tick.

Three training modes:

``naive``
    Pure self-play: every seat plays the latest learner policy.  Each seat keeps
    its own trajectory and its own GAE, so the AEC structure is preserved.
``pfsp``
    Prioritised Fictitious Self-Play: one seat per environment is the learner and
    the rest are played by :class:`~rloup.training.pool.OpponentPool`
    opponents (25% latest weights, 60% PFSP-weighted snapshots, 15% scripted
    baselines).
``exploiter``
    A frozen checkpoint (``--exploiter-target``) plays every non-learner seat
    while a fresh learner maximises its win rate against it.  This is the
    approximate exploitability measurement of the current best policy.  The
    target may be a PPO *or* an NFSP checkpoint: PPO targets contribute their
    actor-critic head, NFSP targets their supervised average policy.

Example::

    python -m rloup.training.train_ppo --num-players 2 --mode pfsp \\
        --total-steps 2000000 --save-dir results/ppo_pfsp_2p
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

from rloup.agents.base import Agent
from rloup.agents.ppo import MaskedActorCritic, PPOAgent, PerAgentRolloutBuffer
from rloup.constants import num_actions_for
from rloup.engine import CoupEngine, obs_dim

from .metrics import MetricsWriter
from .pool import OpponentPool, OpponentSpec
from .vec import BehaviourStats, EvaluationResult, evaluate_vs_baselines

__all__ = [
    "TrainConfig",
    "EnvSlot",
    "PPOTrainer",
    "FrozenOpponent",
    "load_frozen_opponent",
    "main",
    "build_parser",
]

#: Distinct opponents kept in the per-rollout menu (bounds forwards per tick).
OPPONENT_MENU_SIZE = 6


# --------------------------------------------------------------------------- #
# Frozen exploiter targets
# --------------------------------------------------------------------------- #
class FrozenOpponent:
    """A frozen checkpoint behind the ``sample_action(obs, mask)`` interface.

    The exploiter mode only ever *samples* from its target, so a PPO target
    contributes its actor-critic head while an NFSP target contributes the
    supervised average policy - the same head ``NFSPAgent`` plays.  Putting both
    behind this one object keeps the rollout loop itself unchanged.
    """

    def __init__(self, module: Any, kind: str) -> None:
        self.module = module
        self.kind = kind

    def sample_action(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample action ids for a batch of decisions."""
        if self.kind == "actor_critic":
            return self.module.sample_action(
                obs, mask, deterministic=deterministic, generator=generator
            )
        if self.kind == "q":
            return self.module.masked(obs, mask).argmax(dim=-1)
        logits = self.module.masked_logits(obs, mask)
        return MaskedActorCritic._sample(logits, deterministic, generator)


def load_frozen_opponent(path: str | Path, device: str | torch.device) -> FrozenOpponent:
    """Load any rloup checkpoint (PPO, NFSP, or Double DQN) as a frozen sampling policy."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    fmt = payload.get("format")
    if fmt == "rloup.ppo.v1":
        agent = PPOAgent.load(path, device=device)
        return FrozenOpponent(agent.clone_actor_critic(device), "actor_critic")
    if fmt == "rloup.nfsp.v1":
        from rloup.agents.nfsp import NFSPAgent

        agent = NFSPAgent.load(path, device=device)
        network = agent.sl_network.eval()
        network.requires_grad_(False)
        return FrozenOpponent(network, "sl")
    if fmt == "rloup.ddqn.v1":
        from rloup.agents.dqn import DoubleDQNAgent

        agent = DoubleDQNAgent.load(path, device=device)
        network = agent.q_network.eval()
        network.requires_grad_(False)
        return FrozenOpponent(network, "q")
    raise ValueError(f"{path} is not a rloup checkpoint (format={fmt!r})")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Everything the trainer needs; also the contents of ``config.json``."""

    num_players: int = 2
    mode: str = "pfsp"
    exploiter_target: str | None = None
    total_steps: int = 2_000_000
    num_envs: int = 64
    rollout_steps: int = 8192
    lr: float = 3e-4
    min_lr: float = 5e-5
    ent_coef_start: float = 0.025
    ent_coef_end: float = 0.005
    eval_interval: int = 25
    eval_games: int = 200
    save_dir: str = "results/ppo"
    seed: int = 0
    device: str = "auto"
    hidden: int = 256
    num_layers: int = 3
    minibatch_size: int = 1024
    epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    max_turns: int = 200
    reward_mode: str = "auto"
    num_snapshots: int = 8
    max_snapshots: int = 40
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("pfsp", "naive", "exploiter"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if not 2 <= self.num_players <= 6:
            raise ValueError("num_players must be in 2..6")
        if self.mode == "exploiter" and not self.exploiter_target:
            raise ValueError("--mode exploiter requires --exploiter-target")
        if self.num_envs < 1 or self.rollout_steps < 1:
            raise ValueError("num_envs and rollout_steps must be positive")
        if self.reward_mode not in ("auto", "win", "rank"):
            raise ValueError(f"unknown reward_mode {self.reward_mode!r}")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# Environment slot
# --------------------------------------------------------------------------- #
@dataclass
class EnvSlot:
    """One engine plus the policy assignment for its current episode."""

    engine: CoupEngine
    learner_seat: int
    opponent: OpponentSpec
    opponent_module: MaskedActorCritic | FrozenOpponent | None = None
    opponent_agent: Agent | None = None


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class PPOTrainer:
    """Batched-inference self-play PPO with per-seat GAE."""

    def __init__(self, config: TrainConfig) -> None:
        self.cfg = config
        self.device = self._resolve_device(config.device)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":  # pragma: no cover - GPU only
            torch.cuda.manual_seed_all(config.seed)

        n = int(config.num_players)
        self.num_players = n
        self.obs_dim = obs_dim(n)
        self.n_actions = num_actions_for(n)
        self.agent = PPOAgent(
            self.obs_dim,
            self.n_actions,
            hidden=config.hidden,
            num_layers=config.num_layers,
            device=self.device,
            seed=config.seed,
        )
        self.ac = self.agent.actor_critic
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.lr, eps=1e-5)
        self.buffer = PerAgentRolloutBuffer(
            n, gamma=config.gamma, gae_lambda=config.gae_lambda
        )
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(self.rng.integers(2**31 - 1)))
        self._opp_gen = torch.Generator(device=self.device)
        self._opp_gen.manual_seed(int(self.rng.integers(2**31 - 1)))

        self.pool = OpponentPool(
            self.obs_dim,
            self.n_actions,
            device=self.device,
            max_snapshots=config.max_snapshots,
            hidden=config.hidden,
            num_layers=config.num_layers,
            seed=config.seed,
        )
        self.exploiter: OpponentSpec | None = None
        self.exploiter_module: FrozenOpponent | None = None
        if config.mode == "exploiter":
            self.exploiter_module = load_frozen_opponent(
                config.exploiter_target, self.device
            )
            self.exploiter = OpponentSpec(kind="frozen", key="exploiter")

        self.slots: list[EnvSlot] = []
        self._menu: list[OpponentSpec] = []
        self._menu_probs = np.ones(1, dtype=np.float64)
        self._init_slots()

        self.steps_taken = 0
        self.updates = 0
        self.episodes = 0
        self.best_strong_avg = -1.0
        self.save_dir = Path(config.save_dir)
        self.snapshot_dir = self.save_dir / "snapshots"
        self._loss_window: dict[str, list[float]] = {}
        self._start = time.time()
        self._steps_at_last_eval = 0
        self._time_at_last_eval = self._start
        self.rollout_behaviour = BehaviourStats()
        self.last_eval: EvaluationResult | None = None
        self._next_snapshot_at = (
            max(1, config.total_steps // config.num_snapshots)
            if config.num_snapshots > 0
            else None
        )
        self.snapshot_paths: list[str] = []
        self.history: list[dict[str, Any]] = []

    # -- setup ---------------------------------------------------------- #
    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    def _init_slots(self) -> None:
        self.slots = []
        for i in range(self.cfg.num_envs):
            engine = CoupEngine(
                self.num_players,
                max_turns=self.cfg.max_turns,
                reward_mode=self.cfg.reward_mode,
                seed=self.cfg.seed * 1_000_003 + i * 101,
            )
            engine.reset(seed=self.cfg.seed * 7_919 + i)
            slot = EnvSlot(
                engine=engine,
                learner_seat=int(self.rng.integers(self.num_players)),
                opponent=OpponentSpec(kind="latest", key="latest"),
            )
            self.slots.append(slot)
        self._refresh_menu()
        for slot in self.slots:
            self._assign_opponent(slot)

    def _refresh_menu(self) -> None:
        """Sample a small opponent menu so each tick needs few forward passes.

        ``OpponentPool.sample`` is applied repeatedly and deduplicated; the
        per-spec draw frequencies become the assignment weights, so the marginal
        opponent distribution of a game stays equal to the pool mixture.
        """
        if self.cfg.mode != "pfsp":
            self._menu = [OpponentSpec(kind="latest", key="latest")]
            self._menu_probs = np.ones(1, dtype=np.float64)
            return
        counts: dict[str, int] = {}
        specs: dict[str, OpponentSpec] = {}
        for _ in range(OPPONENT_MENU_SIZE * 8):
            spec = self.pool.sample()
            key = spec.label
            if key not in specs:
                specs[key] = spec
                counts[key] = 0
                if len(specs) >= OPPONENT_MENU_SIZE:
                    break
            counts[key] += 1
        if not specs:  # pragma: no cover - defensive
            specs = {"latest": OpponentSpec(kind="latest", key="latest")}
            counts = {"latest": 1}
        self._menu = list(specs.values())
        weights = np.array([counts[s.label] for s in self._menu], dtype=np.float64)
        self._menu_probs = weights / weights.sum()

    def _assign_opponent(self, slot: EnvSlot) -> None:
        if self.cfg.mode == "naive":
            slot.opponent = OpponentSpec(kind="learner", key="learner")
            slot.opponent_module = None
            slot.opponent_agent = None
            return
        if self.cfg.mode == "exploiter":
            slot.opponent = self.exploiter
            slot.opponent_module = self.exploiter_module
            slot.opponent_agent = None
            return
        index = int(self.rng.choice(len(self._menu), p=self._menu_probs))
        spec = self._menu[index]
        slot.opponent = spec
        if spec.kind == "scripted":
            slot.opponent_module = None
            slot.opponent_agent = self.pool.scripted_agent(
                spec, seed=int(self.rng.integers(2**31 - 1))
            )
        elif spec.kind == "snapshot":
            slot.opponent_module = self.pool.module_for(spec)
            slot.opponent_agent = None
        else:
            slot.opponent_module = self.ac
            slot.opponent_agent = None

    # -- rollout collection --------------------------------------------- #
    def collect(self, target_steps: int) -> int:
        """Run environments until ``target_steps`` completed learner steps exist."""
        buffer = self.buffer
        planner = self.ac
        device = self.device
        is_naive = self.cfg.mode == "naive"
        start_completed = buffer.completed_steps
        actions: list[int] = [0] * len(self.slots)
        while buffer.completed_steps - start_completed < target_steps:
            learner_rows: list[tuple[int, int, np.ndarray, np.ndarray]] = []
            scripted_rows: list[tuple[int, int, np.ndarray, np.ndarray]] = []
            grouped: dict[int, tuple[MaskedActorCritic, list[tuple[int, int, np.ndarray, np.ndarray]]]] = {}
            for i, slot in enumerate(self.slots):
                engine = slot.engine
                seat = engine.current_player
                obs = engine.get_obs(seat)
                mask = engine.legal_mask()
                if is_naive or seat == slot.learner_seat:
                    learner_rows.append((i, seat, obs, mask))
                elif slot.opponent_agent is not None:
                    scripted_rows.append((i, seat, obs, mask))
                else:
                    module = slot.opponent_module if slot.opponent_module is not None else planner
                    grouped.setdefault(id(module), (module, []))[1].append(
                        (i, seat, obs, mask)
                    )
            if learner_rows:
                obs_b = torch.as_tensor(
                    np.stack([row[2] for row in learner_rows]),
                    dtype=torch.float32,
                    device=device,
                )
                mask_b = torch.as_tensor(
                    np.stack([row[3] for row in learner_rows]),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.inference_mode():
                    chosen, log_prob, _, value = planner.get_action_and_value(
                        obs_b, mask_b, generator=self._gen
                    )
                chosen_np = chosen.cpu().numpy()
                log_prob_np = log_prob.cpu().numpy()
                value_np = value.cpu().numpy()
                for k, (i, seat, obs, mask) in enumerate(learner_rows):
                    buffer.add_step(
                        seat,
                        obs,
                        mask,
                        int(chosen_np[k]),
                        float(log_prob_np[k]),
                        float(value_np[k]),
                        scope=i,
                    )
                    actions[i] = int(chosen_np[k])
            for i, seat, obs, mask in scripted_rows:
                actions[i] = int(self.slots[i].opponent_agent.select_action(obs, mask))
            for module, rows in grouped.values():
                obs_b = torch.as_tensor(
                    np.stack([row[2] for row in rows]), dtype=torch.float32, device=device
                )
                mask_b = torch.as_tensor(
                    np.stack([row[3] for row in rows]), dtype=torch.float32, device=device
                )
                with torch.inference_mode():
                    chosen = module.sample_action(
                        obs_b, mask_b, generator=self._opp_gen
                    )
                chosen_np = chosen.cpu().numpy()
                for k, (i, seat, obs, mask) in enumerate(rows):
                    actions[i] = int(chosen_np[k])
            for i, slot in enumerate(self.slots):
                slot.engine.step(actions[i])
                if slot.engine.done:
                    self._finish_episode(slot, i)
        return buffer.completed_steps - start_completed

    def _finish_episode(self, slot: EnvSlot, index: int) -> None:
        engine = slot.engine
        self.buffer.finish_episode(engine.rewards, scope=index)
        self.episodes += 1
        stats = engine.get_episode_stats()
        if self.cfg.mode == "naive":
            for seat in range(self.num_players):
                self.rollout_behaviour.add_player_stats(stats["players"][seat])
        else:
            self.rollout_behaviour.add_player_stats(stats["players"][slot.learner_seat])
            if self.cfg.mode == "pfsp" and slot.opponent.kind != "learner":
                winner = engine.winner()
                score = 0.0 if winner is None else float(winner == slot.learner_seat)
                self.pool.record_result(slot.opponent, score)
        engine.reset(seed=None)
        slot.learner_seat = int(self.rng.integers(self.num_players))
        self._assign_opponent(slot)

    # -- optimisation ---------------------------------------------------- #
    def _schedules(self) -> tuple[float, float]:
        progress = min(1.0, self.steps_taken / max(1, self.cfg.total_steps))
        lr = self.cfg.lr + (self.cfg.min_lr - self.cfg.lr) * progress
        ent = self.cfg.ent_coef_start + (
            self.cfg.ent_coef_end - self.cfg.ent_coef_start
        ) * progress
        return lr, ent

    def update(self) -> dict[str, float]:
        """One PPO epoch sweep over the flushed rollout batch."""
        batch = self.buffer.flush()
        tensors = batch.to_tensors(self.device)
        n = len(batch)
        lr, ent_coef = self._schedules()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        clip = float(self.cfg.clip_coef)
        acc: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        num_mb = 0
        for _ in range(int(self.cfg.epochs)):
            perm = torch.randperm(n, device=self.device, generator=self._gen)
            for start in range(0, n, int(self.cfg.minibatch_size)):
                idx = perm[start : start + int(self.cfg.minibatch_size)]
                obs = tensors["obs"][idx]
                masks = tensors["masks"][idx]
                acts = tensors["actions"][idx]
                log_probs_old = tensors["log_probs"][idx]
                advantages = tensors["advantages"][idx]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )
                _, log_prob, entropy, value = self.ac.get_action_and_value(
                    obs, masks, acts
                )
                log_ratio = log_prob - log_probs_old
                ratio = log_ratio.exp()
                surrogate = torch.max(
                    -advantages * ratio,
                    -advantages * ratio.clamp(1.0 - clip, 1.0 + clip),
                ).mean()
                value_loss = 0.5 * ((value - tensors["returns"][idx]) ** 2).mean()
                entropy_mean = entropy.mean()
                loss = (
                    surrogate
                    + float(self.cfg.value_coef) * value_loss
                    - ent_coef * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ac.parameters(), float(self.cfg.max_grad_norm)
                )
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip).float().mean()
                acc["policy_loss"] += float(surrogate.detach())
                acc["value_loss"] += float(value_loss.detach())
                acc["entropy"] += float(entropy_mean.detach())
                acc["approx_kl"] += float(approx_kl)
                acc["clip_fraction"] += float(clip_fraction)
                num_mb += 1
        for key in acc:
            acc[key] /= max(1, num_mb)
        acc["lr"] = lr
        acc["ent_coef"] = ent_coef
        acc["samples"] = float(n)
        self.updates += 1
        self.steps_taken += n
        self._record_loss(acc)
        return acc

    def _record_loss(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._loss_window.setdefault(key, []).append(float(value))

    def _drain_loss_window(self) -> dict[str, float]:
        out = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in self._loss_window.items()
        }
        self._loss_window = {}
        return out

    # -- evaluation ------------------------------------------------------ #
    def eval_policy(self):
        """Stochastic batched policy closure used by the benchmark."""
        return lambda obs, mask: self.agent.batched_sample(obs, mask)

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

    # -- snapshots / checkpoints ----------------------------------------- #
    def _write_json(self, name: str, payload: Any) -> None:
        path = self.save_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def save_snapshot(self, suffix: str | None = None) -> str:
        """Freeze the current weights into the pool and onto disk."""
        name = suffix or f"step_{self.steps_taken:09d}"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"{name}.pt"
        self.agent.save(
            path,
            {
                "step": self.steps_taken,
                "update": self.updates,
                "num_players": self.num_players,
                "mode": self.cfg.mode,
                "elo": self.pool.learner_elo,
            },
        )
        self.pool.add_snapshot(self.ac, step=self.steps_taken, key=f"snap_{name}")
        self.snapshot_paths.append(str(path))
        return str(path)

    def save_best(self, evaluation: EvaluationResult) -> bool:
        if evaluation.strong_avg <= self.best_strong_avg:
            return False
        self.best_strong_avg = evaluation.strong_avg
        self.agent.save(
            self.save_dir / "best.pt",
            {
                "step": self.steps_taken,
                "update": self.updates,
                "num_players": self.num_players,
                "mode": self.cfg.mode,
                "strong_avg": evaluation.strong_avg,
                "avg_vs_baselines": evaluation.avg_vs_baselines,
                "win_rates": evaluation.win_rates,
            },
        )
        return True

    def save_final(self) -> None:
        self.agent.save(
            self.save_dir / "final.pt",
            {
                "step": self.steps_taken,
                "update": self.updates,
                "num_players": self.num_players,
                "mode": self.cfg.mode,
                "strong_avg": self.best_strong_avg,
            },
        )

    # -- main loop -------------------------------------------------------- #
    def _log_metrics(
        self, writer: MetricsWriter, evaluation: EvaluationResult, losses: dict[str, float]
    ) -> dict[str, Any]:
        wall = time.time() - self._start
        window_steps = self.steps_taken - self._steps_at_last_eval
        window_time = time.time() - self._time_at_last_eval
        record: dict[str, Any] = {
            "step": self.steps_taken,
            "update": self.updates,
            "episodes": self.episodes,
            "wall_time": wall,
            "steps_per_second": (
                window_steps / window_time if window_time > 0 else 0.0
            ),
            "learner_elo": self.pool.learner_elo,
            "snapshot_count": self.pool.snapshot_count,
        }
        record.update(evaluation.as_dict())
        record.update(
            {
                "policy_loss": losses.get("policy_loss", 0.0),
                "value_loss": losses.get("value_loss", 0.0),
                "entropy": losses.get("entropy", 0.0),
                "approx_kl": losses.get("approx_kl", 0.0),
                "clip_fraction": losses.get("clip_fraction", 0.0),
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
                f"[{self.cfg.mode} {self.num_players}p] step={self.steps_taken:>9d} "
                f"upd={self.updates:>4d} avg={evaluation.avg_vs_baselines:.3f} "
                f"strong={evaluation.strong_avg:.3f} {rates} "
                f"b={evaluation.behaviour.bluff_rate:.2f} "
                f"cp={evaluation.behaviour.challenge_precision:.2f} "
                f"fps={record['steps_per_second']:.0f}",
                flush=True,
            )
        return record

    def train(self) -> dict[str, Any]:
        """Run the whole training loop and return the trainer summary."""
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
                    f"[{self.cfg.mode} {self.num_players}p] device={self.device} "
                    f"obs_dim={self.obs_dim} n_actions={self.n_actions} "
                    f"envs={self.cfg.num_envs} target_steps={self.cfg.total_steps}",
                    flush=True,
                )
            evaluation = self.evaluate()
            self.save_best(evaluation)
            self._log_metrics(writer, evaluation, {})
            while self.steps_taken < self.cfg.total_steps:
                self._refresh_menu()
                self.collect(self.cfg.rollout_steps)
                self.update()
                if (
                    self._next_snapshot_at is not None
                    and self.steps_taken >= self._next_snapshot_at
                ):
                    self.save_snapshot()
                    self._next_snapshot_at += max(
                        1, self.cfg.total_steps // self.cfg.num_snapshots
                    )
                due = self.updates % max(1, self.cfg.eval_interval) == 0
                if due:
                    evaluation = self.evaluate()
                    self.save_best(evaluation)
                    self._log_metrics(writer, evaluation, self._drain_loss_window())
            evaluation = self.evaluate()
            self.save_best(evaluation)
            self._log_metrics(writer, evaluation, self._drain_loss_window())
            self.save_final()
        finally:
            writer.close()
        summary = {
            "config": config_payload,
            "wall_time": time.time() - self._start,
            "steps": self.steps_taken,
            "updates": self.updates,
            "episodes": self.episodes,
            "steps_per_second": self.steps_taken / max(1e-9, time.time() - self._start),
            "final_eval": evaluation.as_dict(),
            "best_strong_avg": self.best_strong_avg,
            "snapshots": self.snapshot_paths,
            "pool": self.pool.stats(),
            "elo": self.pool.elo_table(),
            "history": self.history,
        }
        self._write_json("summary.json", summary)
        return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rloup.training.train_ppo",
        description="Vectorised batched-inference self-play PPO for Coup.",
    )
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--mode", choices=("pfsp", "naive", "exploiter"), default="pfsp")
    parser.add_argument("--exploiter-target", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--ent-coef-start", type=float, default=0.025)
    parser.add_argument("--ent-coef-end", type=float, default=0.005)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--save-dir", type=str, default="results/ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--reward-mode", choices=("auto", "win", "rank"), default="auto")
    parser.add_argument("--num-snapshots", type=int, default=8)
    parser.add_argument("--max-snapshots", type=int, default=40)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    payload = vars(args)
    verbose = not payload.pop("quiet")
    config = TrainConfig(verbose=verbose, **payload)
    trainer = PPOTrainer(config)
    return trainer.train()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
