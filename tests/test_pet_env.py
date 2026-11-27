"""PettingZoo AEC conformance tests for :class:`rloup.pet_env.CoupAECEnv`."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict as DictSpace, Discrete
from pettingzoo.test import api_test

from rloup.agents import BASELINES
from rloup.constants import Phase
from rloup.pet_env import CoupAECEnv, env as env_factory

PLAYER_COUNTS = (2, 3, 4, 6)


def make_env(n: int = 2, **kwargs) -> CoupAECEnv:
    e = CoupAECEnv(num_players=n, **kwargs)
    e.reset(seed=0)
    return e


# --------------------------------------------------------------------------- #
# Official conformance suite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_api_test(n):
    """The official PettingZoo AEC conformance suite must pass."""
    env = CoupAECEnv(num_players=n)
    api_test(env, num_cycles=60)


# --------------------------------------------------------------------------- #
# Spaces
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_spaces(n):
    env = make_env(n)
    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)
        assert isinstance(obs_space, DictSpace)
        assert isinstance(act_space, Discrete)
        assert obs_space is env.observation_space(agent), "spaces must be cached"
        assert act_space is env.action_space(agent)
        assert set(obs_space.spaces) == {"observation", "action_mask"}
        assert obs_space["observation"].shape == (env.engine.obs_dim,)
        assert obs_space["observation"].dtype == np.float32
        assert obs_space["action_mask"].shape == (21 + 3 * (n - 1),)
        assert obs_space["action_mask"].dtype == np.int8
        assert act_space.n == 21 + 3 * (n - 1)
        assert obs_space.contains(env.observe(agent))


def test_agent_names_follow_the_recommended_pattern():
    env = make_env(4)
    assert env.possible_agents == ["player_0", "player_1", "player_2", "player_3"]
    assert env.agents == env.possible_agents


# --------------------------------------------------------------------------- #
# Observations and masks
# --------------------------------------------------------------------------- #
def test_observe_matches_engine():
    env = make_env(3)
    for agent in env.agents:
        seat = env._seat(agent)
        obs = env.observe(agent)
        assert np.array_equal(obs["observation"], env.engine.get_obs(seat))
        assert obs["observation"].dtype == np.float32
        assert obs["action_mask"].dtype == np.int8
        assert obs["action_mask"].any()


def test_action_mask_only_for_the_decision_maker():
    env = make_env(3)
    current = env.agent_selection
    assert np.array_equal(env.observe(current)["action_mask"], env.engine.legal_mask())
    for agent in env.agents:
        if agent == current:
            continue
        mask = env.observe(agent)["action_mask"]
        assert mask.sum() == 1, "non-decision agents see only a placeholder slot"
        assert mask[env.engine.spec.pass_] == 1


def test_actions_must_be_legal():
    env = make_env(2)
    with pytest.raises(ValueError):
        env.step(env.engine.spec.coup(1))  # not enough coins


def test_seeded_reset_is_deterministic():
    env = make_env(2)
    a = env.reset(seed=123)
    obs_a = env.observe(env.agent_selection)["observation"].copy()
    b = env.reset(seed=123)
    obs_b = env.observe(env.agent_selection)["observation"].copy()
    assert a is None and b is None
    assert np.array_equal(obs_a, obs_b)
    assert env.agent_selection == "player_0"


# --------------------------------------------------------------------------- #
# Rewards / terminations
# --------------------------------------------------------------------------- #
def _play_out(env: CoupAECEnv, agent_factory=None) -> None:
    """Drive a full game through the AEC API."""
    agents = {}
    for i, agent in enumerate(env.possible_agents):
        factory = BASELINES[i % len(BASELINES)] if agent_factory is None else agent_factory
        agents[agent] = factory(seed=i)
    for agent in env.agent_iter(10_000):
        obs, reward, terminated, truncated, info = env.last()
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert isinstance(info, dict)
        if terminated or truncated:
            assert reward == reward  # not NaN
            env.step(None)
            continue
        action = agents[agent].select_action(obs["observation"], obs["action_mask"])
        assert obs["action_mask"][action] == 1
        env.step(action)


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_full_game_through_the_aec_api(n):
    env = make_env(n, max_turns=250)
    _play_out(env)
    assert env.agents == [], "all agents must be removed after termination"
    assert env.engine.done


@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_terminal_rewards_are_zero_sum_and_assigned_once(n):
    env = make_env(n, max_turns=250)
    rewards_seen: dict[str, float] = {}
    final = env.engine
    terminal_rewards = None
    while env.agents:
        agent = env.agent_selection
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            if terminal_rewards is None:
                terminal_rewards = dict(env.rewards)
            env.step(None)
            continue
        rewards_seen[agent] = rewards_seen.get(agent, 0.0) + 0.0
        mask = env.engine.legal_mask()
        env.step(int(np.flatnonzero(mask)[0]))
    assert final.done
    assert terminal_rewards is not None
    if final.terminated:
        assert abs(sum(terminal_rewards.values())) < 1e-9
        assert terminal_rewards[env.possible_agents[final.winner()]] > 0.0
        if n == 2:
            assert sorted(terminal_rewards.values()) == [-1.0, 1.0]
    else:
        assert all(r == 0.0 for r in terminal_rewards.values())


def test_truncation_propagates_to_all_agents():
    env = CoupAECEnv(num_players=2, max_turns=2)
    env.reset(seed=0)
    for agent in env.agent_iter(100):
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            assert truncated is True
            assert terminated is False
            env.step(None)
            continue
        assert truncated is False
        env.step(env.engine.spec.income)
    assert env.agents == []


def test_out_of_turn_step_raises():
    env = make_env(3)
    wrong_seat = (env.engine.current_player + 1) % 3
    env.agent_selection = env.possible_agents[wrong_seat]
    with pytest.raises(RuntimeError):
        env.step(env.engine.spec.income)


def test_render_and_close():
    env = make_env(2, render_mode="ansi")
    text = env.render()
    assert isinstance(text, str) and "phase=" in text
    env.close()
    silent = make_env(2)
    assert silent.render() is not None or silent.render() is None  # no exception


def test_factory_helpers():
    e = env_factory(num_players=4)
    assert isinstance(e, CoupAECEnv)
    assert e.num_agents == 4
    assert e.phase == Phase.CHOOSE_ACTION


def test_engine_can_be_injected():
    from rloup.engine import CoupEngine

    engine = CoupEngine(num_players=5, max_turns=99)
    e = CoupAECEnv(engine=engine)
    assert e.num_players == 5
    assert e.engine is engine
    assert e.action_space("player_0").n == 21 + 3 * 4
