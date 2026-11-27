"""Tests for standalone Action-Masked Double DQN agent and trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rloup.agents.dqn import DoubleDQNAgent, DoubleDQNHyperparams
from rloup.constants import num_actions_for
from rloup.engine import CoupEngine, obs_dim
from rloup.play import load_policy
from rloup.training.train_dqn import DoubleDQNConfig, DoubleDQNTrainer


def test_double_dqn_respects_action_mask_and_probs_sum_to_one():
    eng = CoupEngine(2, seed=0)
    eng.reset(seed=0)
    agent = DoubleDQNAgent(obs_dim(2), num_actions_for(2), hidden=64, num_layers=2, seed=1)

    for _ in range(30):
        if eng.done:
            eng.reset(seed=None)
        seat = eng.current_player
        obs = eng.get_obs(seat)
        mask = eng.legal_mask()

        act_greedy = agent.select_action(obs, mask, deterministic=True)
        assert mask[act_greedy]

        probs = agent.action_probs(obs, mask)
        legal = np.asarray(mask, dtype=bool)
        assert probs[~legal].sum() == pytest.approx(0.0, abs=1e-9)
        assert probs[legal].sum() == pytest.approx(1.0, abs=1e-5)

        # Batched epsilon-greedy exploration must never pick an illegal action.
        batch = agent.batched_actions(
            np.stack([obs] * 32), np.stack([mask] * 32), epsilon=1.0, deterministic=False
        )
        assert all(mask[int(a)] for a in batch)
        eng.step(act_greedy)


def test_double_dqn_update_and_checkpoint_roundtrip(tmp_path: Path):
    o_dim = obs_dim(2)
    n_act = num_actions_for(2)
    hyper = DoubleDQNHyperparams(batch_size=16, warmup=16, target_update=2)
    agent = DoubleDQNAgent(o_dim, n_act, hidden=32, num_layers=2, seed=4, hyper=hyper)

    rng = np.random.default_rng(4)
    for _ in range(32):
        obs = rng.standard_normal(o_dim).astype(np.float32)
        mask = np.zeros(n_act, dtype=bool)
        mask[[0, 1, 2]] = True
        nxt = rng.standard_normal(o_dim).astype(np.float32)
        agent.replay.push((obs, mask, int(rng.integers(0, 3)), 1.0, nxt, mask, False))

    assert agent.ready
    loss, mean_q = agent.update()
    assert np.isfinite(loss) and loss >= 0.0
    assert np.isfinite(mean_q)

    ckpt = agent.save(tmp_path / "ddqn.pt", {"step": 100})
    loaded = load_policy(ckpt)
    assert isinstance(loaded, DoubleDQNAgent)
    assert loaded.metadata["step"] == 100

    sample_obs = rng.standard_normal(o_dim).astype(np.float32)
    sample_mask = np.zeros(n_act, dtype=bool)
    sample_mask[[0, 2]] = True
    np.testing.assert_allclose(
        agent.q_values(sample_obs, sample_mask),
        loaded.q_values(sample_obs, sample_mask),
        atol=1e-6,
    )


@pytest.mark.parametrize("num_players,mode", [(2, "pfsp"), (2, "naive"), (4, "pfsp")])
def test_double_dqn_trainer_smoke(tmp_path: Path, num_players: int, mode: str):
    cfg = DoubleDQNConfig(
        num_players=num_players,
        mode=mode,
        total_steps=400,
        num_envs=4,
        iteration_steps=200,
        eval_interval=1,
        snapshot_interval=1,
        eval_games=6,
        save_dir=str(tmp_path / f"ddqn_{mode}_{num_players}p"),
        seed=7,
        device="cpu",
        hidden=32,
        num_layers=2,
        batch_size=16,
        warmup=32,
        replay_capacity=2000,
        updates_per_tick=2,
        verbose=False,
    )
    summary = DoubleDQNTrainer(cfg).train()
    assert summary["steps"] >= 400
    assert summary["updates"] > 0
    save_dir = Path(cfg.save_dir)
    assert (save_dir / "final.pt").is_file()
    assert (save_dir / "best.pt").is_file()
    assert (save_dir / "metrics.jsonl").is_file()
    loaded = DoubleDQNAgent.load(save_dir / "final.pt")
    assert loaded.n_actions == num_actions_for(num_players)
