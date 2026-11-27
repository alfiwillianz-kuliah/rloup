"""PettingZoo AEC wrapper around :class:`rloup.engine.CoupEngine`.

Every agent observes a ``Dict`` with

``observation``
    ``Box(0, 1, (obs_dim,), float32)`` - the ego-centric engine observation.
``action_mask``
    ``Box(0, 1, (n_actions,), int8)`` - legal action ids for the observing
    agent's *next* decision (all ones fall back to the pending "pass" slot when
    the agent is not the one to move).

The action space is ``Discrete(21 + 3 * (N - 1))``.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete, Dict as DictSpace
from pettingzoo import AECEnv

from .constants import Phase
from .engine import CoupEngine


class CoupAECEnv(AECEnv):
    """Agent-environment-cycle wrapper for Coup."""

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "coup_v0",
        "is_parallelizable": False,
        "render_fps": 4,
    }

    def __init__(
        self,
        num_players: int = 2,
        max_turns: int = 200,
        render_mode: str | None = None,
        two_player_coin_rule: bool = True,
        reward_mode: str = "auto",
        engine: CoupEngine | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in ("human", "ansi"):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.engine = engine if engine is not None else CoupEngine(
            num_players=int(num_players),
            max_turns=max_turns,
            two_player_coin_rule=two_player_coin_rule,
            reward_mode=reward_mode,
        )
        engine = self.engine
        self.num_players = engine.num_players
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]

        n_actions = engine.n_actions
        obs_space = DictSpace(
            {
                "observation": Box(
                    low=0.0, high=1.0, shape=(engine.obs_dim,), dtype=np.float32
                ),
                "action_mask": Box(low=0, high=1, shape=(n_actions,), dtype=np.int8),
            }
        )
        act_space = Discrete(n_actions)
        self.observation_spaces = dict.fromkeys(self.possible_agents, obs_space)
        self.action_spaces = dict.fromkeys(self.possible_agents, act_space)

        self.agent_selection = self.possible_agents[0]
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.reset()

    # ------------------------------------------------------------------ #
    # Agent / seat mapping
    # ------------------------------------------------------------------ #
    def _seat(self, agent: str) -> int:
        return int(agent.rsplit("_", 1)[1])

    def _agent(self, seat: int) -> str:
        return self.possible_agents[seat]

    # ------------------------------------------------------------------ #
    # Spaces
    # ------------------------------------------------------------------ #
    @functools.lru_cache(maxsize=None)  # noqa: B019 - one instance per agent
    def observation_space(self, agent: str) -> DictSpace:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)  # noqa: B019 - one instance per agent
    def action_space(self, agent: str) -> Discrete:
        return self.action_spaces[agent]

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self.agents = self.possible_agents[:]
        self.engine.reset(seed=seed, randomize_start=True)
        self.agent_selection = self._agent(self.engine.current_player)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        seat = self._seat(agent)
        if seat == self.engine.current_player:
            mask = self.engine.legal_mask()
        else:
            mask = np.zeros(self.engine.n_actions, dtype=np.int8)
            mask[self.engine.spec.pass_] = 1
        return {
            "observation": self.engine.get_obs(seat),
            "action_mask": mask,
        }

    def step(self, action) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        seat = self._seat(agent)
        if seat != self.engine.current_player:
            raise RuntimeError(
                f"agent {agent} acted out of turn (engine expects seat "
                f"{self.engine.current_player})"
            )
        if action is None:
            raise ValueError("a live agent must provide an action")
        self.engine.step(int(action))

        self.rewards = dict.fromkeys(self.agents, 0.0)
        if self.engine.done:
            self.terminations = dict.fromkeys(self.agents, bool(self.engine.terminated))
            self.truncations = dict.fromkeys(self.agents, bool(self.engine.truncated))
            self.rewards = {
                a: float(self.engine.rewards[self._seat(a)]) for a in self.agents
            }
        else:
            self.agent_selection = self._agent(self.engine.current_player)
        self._cumulative_rewards[agent] = 0.0
        self._accumulate_rewards()

    def render(self) -> str | None:
        text = self.engine.render()
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        """No resources are held."""

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def phase(self) -> Phase:
        return Phase(self.engine.phase)

    def legal_mask(self) -> np.ndarray:
        return self.engine.legal_mask()

    def get_episode_stats(self) -> dict[str, Any]:
        return self.engine.get_episode_stats()


def env(**kwargs) -> CoupAECEnv:
    """Factory used by ``pettingzoo.make`` style tooling."""
    return CoupAECEnv(**kwargs)


def raw_env(**kwargs) -> CoupAECEnv:
    return CoupAECEnv(**kwargs)


__all__ = ["CoupAECEnv", "env", "raw_env"]
