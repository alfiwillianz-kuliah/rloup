"""Action-masked Neural Fictitious Self-Play (Heinrich & Silver, 2016) for Coup.

Two networks, both trained only on legal actions:

``QNetwork``
    Action-value best response, trained with **masked Double DQN** on a circular
    replay buffer of ``(obs, mask, action, reward, next_obs, next_mask, done)``
    transitions.  Illegal ids are masked to ``-1e8`` so they can never be the
    greedy action and never contribute gradient.
``SLNetwork``
    Supervised average policy, trained with masked cross-entropy on a reservoir
    buffer (Algorithm R) of ``(obs, mask, action)`` triples collected **only**
    while the agent was following its best-response policy.  Its job is to
    approximate the average of the whole sequence of best responses, which is the
    Nash-relevant object for a two-player zero-sum game.

During play each seat picks, at the start of every episode, between the best
response and the average policy with the anticipatory parameter ``eta``
(default ``0.1``).  The best response explores ``epsilon``-greedily
(``0.15 -> 0.02`` over training).

AEC bookkeeping matters here: the terminal reward of an episode is credited to
each seat's **last recorded decision**, because a Coup episode usually ends
during an opponent's turn.  :class:`PerSeatTransitionRecorder` builds the
``(s, a, r, s', a', done)`` chains per seat and never mixes seats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .base import Agent
from .ppo import MASKED_LOGIT_VALUE, MLP, MaskedActorCritic, masked_entropy, masked_log_prob

__all__ = [
    "QNetwork",
    "SLNetwork",
    "ReplayBuffer",
    "ReservoirBuffer",
    "PerSeatTransitionRecorder",
    "NFSPAgent",
    "masked_q_values",
    "masked_double_dqn_loss",
]

#: ``(obs, mask, action, reward, next_obs, next_mask, done)`` as stored.
Transition = tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, bool]


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
class QNetwork(nn.Module):
    """Masked action-value head over the flat Coup action space."""

    def __init__(
        self, obs_dim: int, n_actions: int, hidden: int = 256, num_layers: int = 3
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.net = MLP(obs_dim, n_actions, hidden, num_layers, out_gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(obs)

    def masked(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Q-values with illegal actions at ``-1e8``."""
        legal = MaskedActorCritic.legal(mask)
        return self.net(obs).masked_fill(~legal, MASKED_LOGIT_VALUE)


class SLNetwork(nn.Module):
    """Supervised average policy producing masked logits."""

    def __init__(
        self, obs_dim: int, n_actions: int, hidden: int = 256, num_layers: int = 3
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.net = MLP(obs_dim, n_actions, hidden, num_layers, out_gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(obs)

    def masked_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        legal = MaskedActorCritic.legal(mask)
        return self.net(obs).masked_fill(~legal, MASKED_LOGIT_VALUE)


def masked_q_values(
    q_network: QNetwork, obs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Legal-only Q values (helper kept free so tests can call it directly)."""
    return q_network.masked(obs, mask)


def masked_double_dqn_loss(
    q_network: QNetwork,
    target_network: QNetwork,
    obs: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    next_masks: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Double-DQN temporal-difference loss with strict action masking."""
    q_values = masked_q_values(q_network, obs, masks)
    q_taken = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_online = masked_q_values(q_network, next_obs, next_masks)
        next_actions = next_online.argmax(dim=-1)
        next_values = target_network.masked(next_obs, next_masks).gather(
            -1, next_actions.unsqueeze(-1)
        ).squeeze(-1)
        targets = rewards + float(gamma) * (1.0 - dones) * next_values
    return nn.functional.smooth_l1_loss(q_taken, targets)


# --------------------------------------------------------------------------- #
# Buffers
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Circular replay buffer of masked transitions."""

    def __init__(
        self, capacity: int, obs_dim: int, n_actions: int, seed: int | None = None
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, n_actions), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_masks = np.zeros((capacity, n_actions), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(self, transition: Transition) -> None:
        obs, mask, action, reward, next_obs, next_mask, done = transition
        i = self.position
        self.obs[i] = obs
        self.masks[i] = mask
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = next_obs
        self.next_masks[i] = next_mask
        self.dones[i] = 1.0 if done else 0.0
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        """Uniform minibatch as device tensors."""
        if self.size == 0:  # pragma: no cover - defensive
            raise RuntimeError("cannot sample an empty replay buffer")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        dev = torch.device(device)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=dev),
            "masks": torch.as_tensor(self.masks[idx], dtype=torch.float32, device=dev),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.long, device=dev),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=dev),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=dev),
            "next_masks": torch.as_tensor(
                self.next_masks[idx], dtype=torch.float32, device=dev
            ),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=dev),
        }


class ReservoirBuffer:
    """Reservoir sample (Algorithm R) of the best-response action history."""

    def __init__(
        self, capacity: int, obs_dim: int, n_actions: int, seed: int | None = None
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, n_actions), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(self, obs: np.ndarray, mask: np.ndarray, action: int) -> None:
        """Algorithm R: every item is kept with probability ``capacity / seen``."""
        self.seen += 1
        if self.size < self.capacity:
            index = self.size
            self.size += 1
        else:
            index = int(self.rng.integers(0, self.seen))
            if index >= self.capacity:
                return
        self.obs[index] = obs
        self.masks[index] = mask
        self.actions[index] = int(action)

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if self.size == 0:  # pragma: no cover - defensive
            raise RuntimeError("cannot sample an empty reservoir buffer")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        dev = torch.device(device)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=dev),
            "masks": torch.as_tensor(self.masks[idx], dtype=torch.float32, device=dev),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.long, device=dev),
        }


class PerSeatTransitionRecorder:
    """Turn AEC decisions into per-seat ``(s, a, r, s', a', done)`` chains.

    A seat only decides on its own turns, so its successor observation is only
    known at the seat's *next* decision.  The recorder therefore holds the last
    decision of every ``(scope, seat)`` and emits the transition when the
    successor arrives - or, at episode end, emits the terminal transition with
    ``engine.rewards[seat]`` attached to that seat's last decision.
    """

    def __init__(self) -> None:
        self.pending: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.pending)

    def add_decision(
        self,
        seat: int,
        obs: np.ndarray,
        mask: np.ndarray,
        action: int,
        scope: int = 0,
    ) -> Transition | None:
        """Record a decision; returns the previous decision's transition if any."""
        key = (int(scope), int(seat))
        previous = self.pending.get(key)
        self.pending[key] = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(mask, dtype=np.float32),
            int(action),
        )
        if previous is None:
            return None
        prev_obs, prev_mask, prev_action = previous
        return (prev_obs, prev_mask, prev_action, 0.0, self.pending[key][0], self.pending[key][1], False)

    def finish_episode(self, rewards: Sequence[float], scope: int = 0) -> list[Transition]:
        """Close the ``scope`` chains, crediting each seat its terminal reward."""
        scope = int(scope)
        keys = [key for key in self.pending if key[0] == scope]
        out: list[Transition] = []
        for key in keys:
            obs, mask, action = self.pending.pop(key)
            out.append((obs, mask, action, float(rewards[key[1]]), obs, mask, True))
        return out


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@dataclass
class NFSPHyperparams:
    """The knobs of the algorithm itself (mirrored in ``config.json``)."""

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
    grad_clip: float = 10.0
    warmup: int = 2_000


class NFSPAgent(Agent):
    """Action-masked NFSP agent for Coup (self-play over every seat)."""

    name = "nfsp"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 256,
        num_layers: int = 3,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        hyper: NFSPHyperparams | None = None,
    ) -> None:
        super().__init__(seed)
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.hyper = hyper or NFSPHyperparams()
        self.q_network = QNetwork(obs_dim, n_actions, hidden, num_layers).to(self.device)
        self.target_network = QNetwork(obs_dim, n_actions, hidden, num_layers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.target_network.requires_grad_(False)
        self.sl_network = SLNetwork(obs_dim, n_actions, hidden, num_layers).to(self.device)
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.hyper.lr_q
        )
        self.sl_optimizer = torch.optim.Adam(
            self.sl_network.parameters(), lr=self.hyper.lr_sl
        )
        seed_int = None if seed is None else int(seed)
        self.replay = ReplayBuffer(
            self.hyper.replay_capacity, obs_dim, n_actions, seed=seed_int
        )
        self.reservoir = ReservoirBuffer(
            self.hyper.reservoir_capacity, obs_dim, n_actions, seed=seed_int
        )
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(self.rng.integers(2**31 - 1)))
        self.q_updates = 0
        self.sl_updates = 0
        self.metadata: dict[str, Any] = {}

    # -- exploration schedule ------------------------------------------- #
    @staticmethod
    def epsilon_at(progress: float, hyper: NFSPHyperparams | None = None) -> float:
        """Linear ``epsilon`` decay from ``epsilon_start`` to ``epsilon_end``."""
        hyper = hyper or NFSPHyperparams()
        p = min(1.0, max(0.0, float(progress)))
        return hyper.epsilon_start + (hyper.epsilon_end - hyper.epsilon_start) * p

    # -- action selection ------------------------------------------------ #
    def _tensors(
        self, obs: np.ndarray, mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device),
            torch.as_tensor(np.asarray(mask), device=self.device),
        )

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        engine=None,
        player_id: int | None = None,
        deterministic: bool = False,
    ) -> int:
        """Play the average (SL) policy, which is how NFSP evaluates."""
        del engine, player_id
        obs_t, mask_t = self._tensors(obs, action_mask)
        with torch.inference_mode():
            logits = self.sl_network.masked_logits(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
            action = MaskedActorCritic._sample(logits, deterministic, self._generator)
        return int(action.item())

    def action_probs(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        """Average-policy probabilities over legal actions."""
        obs_t, mask_t = self._tensors(obs, action_mask)
        with torch.inference_mode():
            logits = self.sl_network.masked_logits(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy()

    def batched_actions(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        best_response: bool = False,
        epsilon: float = 0.0,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Actions for a batch of decisions, from either policy head."""
        obs_t, mask_t = self._tensors(obs, mask)
        with torch.inference_mode():
            if best_response:
                q_values = self.q_network.masked(obs_t, mask_t)
                if deterministic or epsilon <= 0.0:
                    actions = q_values.argmax(dim=-1)
                else:
                    greedy = q_values.argmax(dim=-1)
                    explored = MaskedActorCritic._sample(q_values, False, self._generator)
                    rolls = torch.rand(
                        greedy.shape, generator=self._generator, device=self.device
                    )
                    actions = torch.where(rolls < float(epsilon), explored, greedy)
            else:
                logits = self.sl_network.masked_logits(obs_t, mask_t)
                actions = MaskedActorCritic._sample(logits, deterministic, self._generator)
        return actions.cpu().numpy()

    # -- learning -------------------------------------------------------- #
    @property
    def ready(self) -> bool:
        """True once both buffers can serve a full minibatch."""
        return (
            len(self.replay) >= max(self.hyper.warmup, self.hyper.batch_size)
            and len(self.reservoir) >= self.hyper.batch_size
        )

    def q_update(self) -> float:
        """One masked Double-DQN gradient step; returns the TD loss."""
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
        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.hyper.grad_clip)
        self.q_optimizer.step()
        self.q_updates += 1
        if self.q_updates % max(1, self.hyper.target_update) == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return float(loss.detach())

    def sl_update(self) -> tuple[float, float]:
        """One masked cross-entropy step on the reservoir sample.

        Returns ``(loss, entropy)`` of the average policy on the batch.
        """
        batch = self.reservoir.sample(self.hyper.batch_size, self.device)
        logits = self.sl_network.masked_logits(batch["obs"], batch["masks"])
        loss = -masked_log_prob(logits, batch["masks"], batch["actions"]).mean()
        entropy = masked_entropy(logits, batch["masks"]).mean()
        self.sl_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sl_network.parameters(), self.hyper.grad_clip)
        self.sl_optimizer.step()
        self.sl_updates += 1
        return float(loss.detach()), float(entropy.detach())

    # -- persistence ----------------------------------------------------- #
    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "rloup.nfsp.v1",
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "num_layers": self.num_layers,
            "hyper": dict(self.hyper.__dict__),
            "q_state_dict": {k: v.detach().cpu() for k, v in self.q_network.state_dict().items()},
            "target_state_dict": {
                k: v.detach().cpu() for k, v in self.target_network.state_dict().items()
            },
            "sl_state_dict": {k: v.detach().cpu() for k, v in self.sl_network.state_dict().items()},
            "metadata": dict(self.metadata if metadata is None else metadata),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu", seed: int | None = None
    ) -> "NFSPAgent":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != "rloup.nfsp.v1":
            raise ValueError(f"{path} is not a rloup NFSP checkpoint")
        hyper = NFSPHyperparams(**payload["hyper"])
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
        agent.sl_network.load_state_dict(payload["sl_state_dict"])
        agent.q_network.eval()
        agent.sl_network.eval()
        agent.metadata = dict(payload.get("metadata") or {})
        return agent
