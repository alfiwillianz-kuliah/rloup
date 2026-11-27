"""Standalone Action-Masked Double DQN agent for Coup (van Hasselt et al., 2016).

Uses two copies of :class:`~rloup.agents.nfsp.QNetwork` (online ``q_network``
and frozen ``target_network``) over the flat discrete Coup action space:

* Online network selects the greedy legal next action
  ``a* = argmax_{a legal} Q_online(s', a)``.
* Target network evaluates that action ``Q_target(s', a*)``.
* Illegal actions are masked to ``-1e8`` before every ``argmax`` and softmax so
  they can never be chosen or bootstrap into TD targets.

Works for any seat count (2-6 players) and supports both hard periodic target
syncs (``tau == 0``) and Polyak soft updates (``tau > 0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import Agent
from .nfsp import QNetwork, ReplayBuffer, masked_double_dqn_loss
from .ppo import MASKED_LOGIT_VALUE, MaskedActorCritic

__all__ = [
    "DoubleDQNHyperparams",
    "DoubleDQNAgent",
]


@dataclass
class DoubleDQNHyperparams:
    """Algorithm hyperparameters for Action-Masked Double DQN."""

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
    grad_clip: float = 10.0
    eval_temperature: float = 0.0


class DoubleDQNAgent(Agent):
    """Action-masked Double DQN agent for 2-6 player Coup."""

    name = "double_dqn"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 256,
        num_layers: int = 3,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        hyper: DoubleDQNHyperparams | None = None,
    ) -> None:
        super().__init__(seed)
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.hyper = hyper or DoubleDQNHyperparams()

        self.q_network = QNetwork(obs_dim, n_actions, hidden, num_layers).to(self.device)
        self.target_network = QNetwork(obs_dim, n_actions, hidden, num_layers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.target_network.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.hyper.lr)
        seed_int = None if seed is None else int(seed)
        self.replay = ReplayBuffer(
            self.hyper.replay_capacity, obs_dim, n_actions, seed=seed_int
        )
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(self.rng.integers(2**31 - 1)))
        self.updates = 0
        self.metadata: dict[str, Any] = {}

    # -- schedules ------------------------------------------------------- #
    @staticmethod
    def epsilon_at(progress: float, hyper: DoubleDQNHyperparams | None = None) -> float:
        """Linear epsilon decay reaching ``epsilon_end`` at ``epsilon_decay_frac``."""
        hyper = hyper or DoubleDQNHyperparams()
        frac = max(1e-6, float(hyper.epsilon_decay_frac))
        p = min(1.0, max(0.0, float(progress) / frac))
        return hyper.epsilon_start + (hyper.epsilon_end - hyper.epsilon_start) * p

    # -- action selection ------------------------------------------------ #
    def _tensors(
        self, obs: np.ndarray, mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device),
            torch.as_tensor(np.asarray(mask), device=self.device),
        )

    def q_values(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """Masked Q-values for one observation (illegal ids at ``-1e8``)."""
        obs_t, mask_t = self._tensors(obs, action_mask)
        with torch.inference_mode():
            q = self.q_network.masked(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
        return q.squeeze(0).cpu().numpy()

    def max_q(self, obs: np.ndarray, action_mask: np.ndarray) -> float:
        """Estimated state value ``max_{a legal} Q(s, a)``."""
        return float(self.q_values(obs, action_mask).max())

    def action_probs(
        self, obs: np.ndarray, action_mask: np.ndarray, temperature: float = 0.25
    ) -> np.ndarray:
        """Boltzmann softmax distribution over legal Q-values (illegal = 0.0)."""
        obs_t, mask_t = self._tensors(obs, action_mask)
        legal = MaskedActorCritic.legal(mask_t.unsqueeze(0))
        temp = max(1e-3, float(temperature))
        with torch.inference_mode():
            raw_q = self.q_network(obs_t.unsqueeze(0)) / temp
            logits = raw_q.masked_fill(~legal, MASKED_LOGIT_VALUE)
            probs = torch.softmax(logits, dim=-1)
            probs = probs.masked_fill(~legal, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return probs.squeeze(0).cpu().numpy()

    def batched_probs(
        self, obs: np.ndarray, mask: np.ndarray, temperature: float = 0.25
    ) -> np.ndarray:
        """Boltzmann softmax probabilities for a batch of observations."""
        obs_t, mask_t = self._tensors(obs, mask)
        legal = MaskedActorCritic.legal(mask_t)
        temp = max(1e-3, float(temperature))
        with torch.inference_mode():
            logits = (self.q_network(obs_t) / temp).masked_fill(~legal, MASKED_LOGIT_VALUE)
            probs = torch.softmax(logits, dim=-1).masked_fill(~legal, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return probs.cpu().numpy()

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        engine=None,
        player_id: int | None = None,
        deterministic: bool = True,
    ) -> int:
        """Pick greedy legal action (or Boltzmann sample if ``eval_temperature > 0``)."""
        del engine, player_id
        obs_t, mask_t = self._tensors(obs, action_mask)
        with torch.inference_mode():
            q = self.q_network.masked(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
            if deterministic or self.hyper.eval_temperature <= 0.0:
                action = q.argmax(dim=-1)
            else:
                legal = MaskedActorCritic.legal(mask_t.unsqueeze(0))
                logits = (self.q_network(obs_t.unsqueeze(0)) / self.hyper.eval_temperature).masked_fill(
                    ~legal, MASKED_LOGIT_VALUE
                )
                action = MaskedActorCritic._sample(logits, False, self._generator)
        return int(action.item())

    def batched_actions(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        epsilon: float = 0.0,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Vectorised greedy or epsilon-greedy legal actions for a batch."""
        obs_t, mask_t = self._tensors(obs, mask)
        legal = MaskedActorCritic.legal(mask_t)
        with torch.inference_mode():
            q_values = self.q_network.masked(obs_t, mask_t)
            greedy = q_values.argmax(dim=-1)
            if deterministic and epsilon <= 0.0:
                return greedy.cpu().numpy()
            if epsilon > 0.0:
                # Uniform random choice strictly over legal actions.
                noise = torch.rand(
                    q_values.shape, generator=self._generator, device=self.device
                )
                rand_legal = noise.masked_fill(~legal, -1.0).argmax(dim=-1)
                rolls = torch.rand(
                    greedy.shape, generator=self._generator, device=self.device
                )
                actions = torch.where(rolls < float(epsilon), rand_legal, greedy)
                return actions.cpu().numpy()
            if self.hyper.eval_temperature > 0.0:
                logits = (
                    self.q_network(obs_t) / self.hyper.eval_temperature
                ).masked_fill(~legal, MASKED_LOGIT_VALUE)
                return MaskedActorCritic._sample(logits, False, self._generator).cpu().numpy()
            return greedy.cpu().numpy()

    # -- learning -------------------------------------------------------- #
    @property
    def ready(self) -> bool:
        return len(self.replay) >= max(self.hyper.warmup, self.hyper.batch_size)

    def _sync_target(self) -> None:
        tau = float(self.hyper.tau)
        if tau > 0.0:
            with torch.no_grad():
                for tgt, src in zip(
                    self.target_network.parameters(), self.q_network.parameters()
                ):
                    tgt.mul_(1.0 - tau).add_(src, alpha=tau)
        elif self.updates % max(1, int(self.hyper.target_update)) == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def update(self) -> tuple[float, float]:
        """One masked Double-DQN gradient step; returns ``(td_loss, mean_q)``."""
        batch = self.replay.sample(self.hyper.batch_size, self.device)
        loss = masked_double_dqn_loss(
            self.q_network,
            self.target_network,
            batch["obs"],
            batch["masks"],
            batch["actions"],
            batch["rewards"],
            batch["next_obs"],
            batch["next_masks"],
            batch["dones"],
            gamma=self.hyper.gamma,
        )
        with torch.no_grad():
            mean_q = (
                self.q_network.masked(batch["obs"], batch["masks"])
                .gather(-1, batch["actions"].unsqueeze(-1))
                .mean()
            )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.hyper.grad_clip)
        self.optimizer.step()
        self.updates += 1
        self._sync_target()
        return float(loss.detach()), float(mean_q.detach())

    # -- persistence ----------------------------------------------------- #
    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "rloup.ddqn.v1",
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "num_layers": self.num_layers,
            "hyper": dict(self.hyper.__dict__),
            "q_state_dict": {
                k: v.detach().cpu() for k, v in self.q_network.state_dict().items()
            },
            "target_state_dict": {
                k: v.detach().cpu() for k, v in self.target_network.state_dict().items()
            },
            "metadata": dict(self.metadata if metadata is None else metadata),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> "DoubleDQNAgent":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != "rloup.ddqn.v1":
            raise ValueError(f"{path} is not a rloup Double DQN checkpoint")
        hyper = DoubleDQNHyperparams(**payload["hyper"])
        agent = cls(
            obs_dim=int(payload["obs_dim"]),
            n_actions=int(payload["n_actions"]),
            hidden=int(payload["hidden"]),
            num_layers=int(payload["num_layers"]),
            device=device,
            seed=seed,
            hyper=hyper,
        )
        agent.q_network.load_state_dict(payload["q_state_dict"])
        agent.target_network.load_state_dict(payload["target_state_dict"])
        agent.q_network.eval()
        agent.target_network.eval()
        agent.metadata = dict(payload.get("metadata") or {})
        return agent
