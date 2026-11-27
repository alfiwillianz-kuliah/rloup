"""Tests for the Phase 2 training stack: masked PPO, PFSP pool and masked NFSP.

The suites are deliberately small (a few seconds in total) but they exercise the
same code paths as the long GPU runs: masked distributions, per-seat GAE across
an AEC episode boundary, PFSP sampling/EMA, a full PPO training loop in all
modes, an NFSP training loop, and checkpoint round-trips.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rloup import CoupEngine
from rloup.agents.nfsp import (
    NFSPAgent,
    NFSPHyperparams,
    PerSeatTransitionRecorder,
    ReplayBuffer,
    ReservoirBuffer,
    masked_double_dqn_loss,
)
from rloup.agents.ppo import (
    MASKED_LOGIT_VALUE,
    MaskedActorCritic,
    PPOAgent,
    PerAgentRolloutBuffer,
    SeatTrajectory,
)
from rloup.constants import num_actions_for
from rloup.engine import obs_dim
from rloup.training.metrics import METRIC_KEYS
from rloup.training.pool import OpponentPool, OpponentSpec, update_elo
from rloup.training.train_nfsp import NFSPConfig, NFSPTrainer
from rloup.training.train_ppo import PPOTrainer, TrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batched_masks(rows: int, n_actions: int, legal: list[list[int]]) -> torch.Tensor:
    mask = torch.zeros(rows, n_actions)
    for index, ids in enumerate(legal):
        mask[index, ids] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# Masking: logits, probabilities, entropy, gradients
# --------------------------------------------------------------------------- #
def test_masked_logits_probs_entropy_are_strictly_legal():
    n_actions, rows = 24, 5
    ac = MaskedActorCritic(173, n_actions).to(DEVICE).eval()
    legal = [[0, 1, 2, 3], [5, 9], [23], [0, 12, 18], [7, 8, 9, 10, 11]]
    obs = torch.rand(rows, 173, device=DEVICE)
    mask = _batched_masks(rows, n_actions, legal).to(DEVICE)

    logits = ac.masked_logits(obs, mask)
    for index, ids in enumerate(legal):
        allowed = set(ids)
        for action in range(n_actions):
            if action in allowed:
                assert float(logits[index, action].detach()) > MASKED_LOGIT_VALUE
            else:
                assert float(logits[index, action].detach()) == MASKED_LOGIT_VALUE

    probs = ac.action_probs(obs, mask)
    for index, ids in enumerate(legal):
        assert pytest.approx(1.0, abs=1e-6) == float(probs[index, ids].detach().sum())
        illegal = torch.ones(n_actions, dtype=torch.bool, device=DEVICE)
        illegal[ids] = False
        assert float(probs[index][illegal].detach().abs().sum()) == 0.0

    log_probs = torch.log(probs.clamp_min(1e-30))
    expected_entropy = -(probs * torch.where(probs > 0, log_probs, torch.zeros_like(probs))).sum(-1)
    _, _, entropy, _ = ac.get_action_and_value(obs, mask)
    assert torch.allclose(entropy, expected_entropy, atol=1e-5)

    # A row with a single legal action is deterministic and has zero entropy.
    single = _batched_masks(2, n_actions, [[11], [11]]).to(DEVICE)
    action, log_prob, entropy, _ = ac.get_action_and_value(obs[:2], single)
    assert torch.all(action == 11)
    assert torch.allclose(log_prob, torch.zeros_like(log_prob), atol=1e-6)
    assert torch.allclose(entropy, torch.zeros_like(entropy), atol=1e-6)


def test_masked_policy_gradients_are_finite_and_mask_respecting():
    n_actions = 24
    ac = MaskedActorCritic(173, n_actions).to(DEVICE)
    obs = torch.rand(6, 173, device=DEVICE)
    # Action 10 is illegal in every row, so it can never influence the loss.
    mask = _batched_masks(6, n_actions, [[0, 1], [2], [3, 4, 5], [6], [20, 21], [22, 23]]).to(DEVICE)
    action, log_prob, entropy, value = ac.get_action_and_value(obs, mask)
    loss = -(log_prob * 2.0).mean() + (value**2).mean() - 0.01 * entropy.mean()
    loss.backward()
    for name, parameter in ac.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    # The actor head must not see gradient through masked logits: perturbing the
    # illegal logits directly cannot change the loss.
    with torch.no_grad():
        before = loss_snapshot(ac, obs, mask, action)
        ac.actor.net[-1].bias[10] += 5.0
        after = loss_snapshot(ac, obs, mask, action)
    assert before == pytest.approx(after, abs=1e-5)


def loss_snapshot(ac: MaskedActorCritic, obs, mask, action) -> float:
    with torch.no_grad():
        _, log_prob, entropy, value = ac.get_action_and_value(obs, mask, action)
    return float(-log_prob.mean() + (value**2).mean() - 0.01 * entropy.mean())


def test_gumbel_sampling_matches_the_masked_categorical():
    n_actions = 24
    ac = MaskedActorCritic(173, n_actions).to(DEVICE).eval()
    obs = torch.rand(1, 173, device=DEVICE)
    mask = _batched_masks(1, n_actions, [[0, 3, 7]]).to(DEVICE)
    with torch.no_grad():
        probs = ac.action_probs(obs, mask)[0].cpu().numpy()
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(7)
    draws = np.zeros(n_actions, dtype=np.int64)
    samples = 6000
    obs_batch = obs.repeat(samples, 1)
    mask_batch = mask.repeat(samples, 1)
    with torch.no_grad():
        actions = ac.sample_action(obs_batch, mask_batch, generator=generator).cpu().numpy()
    for action in actions:
        draws[int(action)] += 1
    assert draws[[1, 2, 4, 5, 6]].sum() == 0  # never illegal
    empirical = draws / samples
    assert np.allclose(empirical, probs, atol=0.025)


def test_agent_never_selects_an_illegal_action():
    engine = CoupEngine(4, max_turns=120, seed=11)
    agent = PPOAgent(obs_dim(4), num_actions_for(4), device=DEVICE, seed=3)
    for game in range(4):
        engine.reset(seed=100 + game)
        while not engine.done:
            obs = engine.get_obs(engine.current_player)
            mask = engine.legal_mask()
            for _ in range(3):
                action = agent.select_action(obs, mask)
                assert mask[action] == 1
                probs = agent.action_probs(obs, mask)
                assert probs[action] > 0.0
                assert pytest.approx(1.0, abs=1e-5) == float(probs.sum())
            engine.step(int(np.flatnonzero(mask)[0]))


# --------------------------------------------------------------------------- #
# Per-seat GAE
# --------------------------------------------------------------------------- #
def test_gae_matches_the_closed_form():
    traj = SeatTrajectory(seat=0)
    traj.values = [0.5, 0.2]
    traj.rewards = [0.0, 1.0]
    traj.dones = [False, True]
    traj.actions = [0, 0]
    adv, ret = traj.advantages(gamma=0.99, gae_lambda=0.95)
    delta1 = 1.0 + 0.0 - 0.2
    delta0 = 0.0 + 0.99 * 0.2 - 0.5
    expected = np.array([delta0 + 0.99 * 0.95 * delta1, delta1])
    assert np.allclose(adv, expected)
    assert np.allclose(ret, expected + np.array([0.5, 0.2]))


def test_terminal_reward_lands_on_the_last_decision_of_each_seat():
    """An episode that ends on an opponent's turn must still pay every seat."""
    engine = CoupEngine(2, max_turns=200, seed=5)
    rng = np.random.default_rng(0)
    checked_opponent_finish = 0
    for game in range(12):
        buffer = PerAgentRolloutBuffer(2)
        engine.reset(seed=game)
        last_mover = None
        while not engine.done:
            seat = engine.current_player
            mask = engine.legal_mask()
            action = int(rng.choice(np.flatnonzero(mask)))
            buffer.add_step(seat, engine.get_obs(seat), mask, action, -0.5, 0.25)
            last_mover = seat
            engine.step(action)
        rewards = engine.rewards
        opponent_finished = engine.winner() != last_mover
        if opponent_finished:
            checked_opponent_finish += 1
        buffer.finish_episode(rewards)
        assert len(buffer.open) == 0
        for traj in buffer.completed:
            assert traj.dones[-1] is True
            assert traj.rewards[-1] == pytest.approx(float(rewards[traj.seat]))
            assert all(not done for done in traj.dones[:-1])
            assert all(reward == 0.0 for reward in traj.rewards[:-1])
            # GAEs are per seat: they only ever see this seat's own steps.
            adv, ret = traj.advantages()
            assert adv.shape == (len(traj),)
            assert np.isfinite(adv).all() and np.isfinite(ret).all()
        batch = buffer.flush()
        assert len(batch) == sum(len(t) for t in buffer.completed) + len(batch)
        assert set(batch.seats.tolist()) <= {0, 1}
        assert buffer.completed_steps == 0
    assert checked_opponent_finish > 0, "no game ended on the opponent's turn"


def test_concurrent_scopes_do_not_mix_trajectories():
    buffer = PerAgentRolloutBuffer(2)
    obs = np.zeros(4, dtype=np.float32)
    mask = np.array([1, 0], dtype=np.float32)
    buffer.add_step(0, obs, mask, 0, 0.0, 0.0, scope=0)
    buffer.add_step(0, obs, mask, 0, 0.0, 0.0, scope=1)
    buffer.add_step(0, obs, mask, 0, 0.0, 0.0, scope=1)
    assert buffer.open_steps == 3
    buffer.finish_episode([1.0, -1.0], scope=0)
    assert list(buffer.open) == [(1, 0)]
    assert buffer.completed_steps == 1
    buffer.finish_episode([-1.0, 1.0], scope=1)
    assert buffer.completed_steps == 3
    batch = buffer.flush()
    assert len(batch) == 3


# --------------------------------------------------------------------------- #
# Opponent pool: PFSP sampling, EMA, Elo
# --------------------------------------------------------------------------- #
def test_elo_update_is_symmetric_and_k_scaled():
    ra, rb = 1000.0, 1000.0
    new_a, new_b = update_elo(ra, rb, 1.0, k=32.0)
    assert new_a == pytest.approx(1016.0)
    assert new_b == pytest.approx(984.0)
    assert new_a + new_b == pytest.approx(ra + rb)
    draw_a, draw_b = update_elo(1200.0, 1000.0, 0.5, k=32.0)
    assert draw_a < 1200.0 and draw_b > 1000.0  # favourite loses rating on a draw
    assert draw_a + draw_b == pytest.approx(2200.0)


def test_pfsp_weights_prefer_opponents_the_learner_loses_to():
    pool = OpponentPool(obs_dim(2), num_actions_for(2), device=DEVICE, seed=0)
    ac = MaskedActorCritic(pool.obs_dim, pool.n_actions).to(DEVICE)
    for _ in range(3):
        pool.add_snapshot(ac, step=0)
    pool._win_rates[pool.snapshots[0].key] = 0.9  # learner dominates it
    pool._win_rates[pool.snapshots[1].key] = 0.1  # learner is losing
    pool._win_rates[pool.snapshots[2].key] = 0.5
    weights = pool.weights()
    expected = np.array([(1 - 0.9) ** 2, (1 - 0.1) ** 2, (1 - 0.5) ** 2]) + 0.05
    expected = expected / expected.sum()
    assert np.allclose(weights, expected)
    assert weights[1] > weights[2] > weights[0]


def test_pool_sampling_mixture_before_and_after_snapshots():
    pool = OpponentPool(obs_dim(2), num_actions_for(2), device=DEVICE, seed=1)
    draws = [pool.sample() for _ in range(4000)]
    kinds = {kind: sum(1 for d in draws if d.kind == kind) for kind in ("latest", "scripted", "snapshot")}
    # No snapshots yet: 85% latest, 15% scripted, never a snapshot.
    assert kinds["snapshot"] == 0
    assert 0.80 < kinds["latest"] / 4000 < 0.90
    assert 0.10 < kinds["scripted"] / 4000 < 0.20
    assert {d.name for d in draws if d.kind == "scripted"} <= {
        "random",
        "honest",
        "aggressive_bluffer",
        "cautious_skeptical",
        "card_counter",
    }

    ac = MaskedActorCritic(pool.obs_dim, pool.n_actions).to(DEVICE)
    for _ in range(4):
        pool.add_snapshot(ac, step=0)
    draws = [pool.sample() for _ in range(4000)]
    kinds = {kind: sum(1 for d in draws if d.kind == kind) for kind in ("latest", "scripted", "snapshot")}
    assert 0.20 < kinds["latest"] / 4000 < 0.30
    assert 0.52 < kinds["snapshot"] / 4000 < 0.68
    assert 0.09 < kinds["scripted"] / 4000 < 0.21


def test_pool_ema_tracks_win_rate_and_elo():
    pool = OpponentPool(obs_dim(2), num_actions_for(2), device=DEVICE, seed=2, ema_alpha=0.25)
    ac = MaskedActorCritic(pool.obs_dim, pool.n_actions).to(DEVICE)
    snapshot = pool.add_snapshot(ac, step=10)
    spec = OpponentSpec(kind="snapshot", key=snapshot.key, index=snapshot.index)
    assert pool.win_rate(spec.key) == pytest.approx(0.5)
    pool.record_result(spec, 1.0)
    assert pool.win_rate(spec.key) == pytest.approx(0.625)
    pool.record_result(spec, 1.0)
    assert pool.win_rate(spec.key) == pytest.approx(0.71875)
    assert pool.snapshots[0].win_rate == pytest.approx(0.71875)
    assert pool.learner_elo > 1000.0
    assert pool.snapshots[0].elo < 1000.0
    stats = pool.stats()
    assert stats["snapshot_count"] == 1
    assert pool.elo_table()[0]["key"] == snapshot.key

    scripted = pool.sample()
    while scripted.kind != "scripted":
        scripted = pool.sample()
    pool.record_result(scripted, 0.0)
    assert pool.win_rate(scripted.key) < 0.5


def test_pool_evicts_old_snapshots():
    pool = OpponentPool(obs_dim(2), num_actions_for(2), device=DEVICE, seed=0, max_snapshots=3)
    ac = MaskedActorCritic(pool.obs_dim, pool.n_actions).to(DEVICE)
    keys = [pool.add_snapshot(ac, step=i).key for i in range(5)]
    assert pool.snapshot_count == 3
    assert [s.key for s in pool.snapshots] == keys[-3:]
    assert pool.win_rate(keys[0]) == pytest.approx(0.5)  # falls back to the prior


# --------------------------------------------------------------------------- #
# Short end-to-end PPO runs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "num_players,mode", [(2, "pfsp"), (2, "naive"), (4, "pfsp"), (4, "naive")]
)
def test_short_ppo_run_and_checkpoint_roundtrip(tmp_path: Path, num_players: int, mode: str):
    config = TrainConfig(
        num_players=num_players,
        mode=mode,
        total_steps=256,
        num_envs=4,
        rollout_steps=128,
        eval_interval=10**6,
        eval_games=6,
        save_dir=str(tmp_path / f"ppo_{mode}_{num_players}p"),
        seed=4,
        device=str(DEVICE),
        minibatch_size=64,
        hidden=32,
        num_layers=2,
        num_snapshots=1,
        max_turns=120,
        verbose=False,
    )
    summary = PPOTrainer(config).train()
    save_dir = Path(config.save_dir)

    records = [json.loads(line) for line in (save_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(records) >= 2
    for record in records:
        assert set(METRIC_KEYS) <= set(record)
        assert record["num_players"] if "num_players" in record else num_players
        assert np.isfinite(record["avg_vs_baselines"])
        for name in ("random", "honest", "aggressive_bluffer", "cautious_skeptical", "card_counter"):
            assert 0.0 <= record[f"win_rate_{name}"] <= 1.0
    assert summary["steps"] >= config.total_steps
    assert (save_dir / "best.pt").exists()
    assert (save_dir / "final.pt").exists()
    assert (save_dir / "config.json").exists()
    assert list((save_dir / "snapshots").glob("*.pt"))

    agent = PPOAgent.load(save_dir / "final.pt", device="cpu")
    engine = CoupEngine(num_players, max_turns=120, seed=9)
    engine.reset(seed=3)
    obs, mask = engine.get_obs(engine.current_player), engine.legal_mask()
    action = agent.select_action(obs, mask, deterministic=True)
    assert mask[action] == 1
    probs = agent.action_probs(obs, mask)
    assert pytest.approx(1.0, abs=1e-5) == float(probs.sum())
    assert int(probs.argmax()) == action  # deterministic == argmax


def test_ppo_update_reduces_loss_on_a_fixed_batch(tmp_path: Path):
    config = TrainConfig(
        num_players=2,
        mode="naive",
        total_steps=192,
        num_envs=4,
        rollout_steps=96,
        eval_interval=10**6,
        eval_games=4,
        save_dir=str(tmp_path / "ppo_fit"),
        seed=1,
        device=str(DEVICE),
        hidden=32,
        num_layers=2,
        num_snapshots=0,
        epochs=4,
        minibatch_size=64,
        max_turns=120,
        verbose=False,
    )
    trainer = PPOTrainer(config)
    trainer.collect(96)
    first = trainer.update()
    trainer.buffer.clear()
    losses = [first["value_loss"]]
    for _ in range(3):
        trainer.collect(96)
        losses.append(trainer.update()["value_loss"])
    assert np.isfinite(losses).all()
    assert all(value >= 0.0 for value in losses)


# --------------------------------------------------------------------------- #
# NFSP pieces
# --------------------------------------------------------------------------- #
def test_nfsp_recorder_assigns_terminal_reward_to_last_decision():
    recorder = PerSeatTransitionRecorder()
    obs = np.arange(4, dtype=np.float32)
    mask = np.array([1, 1, 0, 0], dtype=np.float32)
    assert recorder.add_decision(0, obs, mask, 1) is None
    transition = recorder.add_decision(1, obs + 1, mask, 0)
    assert transition is None  # a different seat, not a successor
    successor = recorder.add_decision(0, obs + 2, mask, 1)
    assert successor is not None
    assert successor[3] == 0.0 and successor[6] is False
    assert np.allclose(successor[4], obs + 2)
    terminal = recorder.finish_episode([-1.0, 1.0])
    assert len(terminal) == 2
    by_reward = {float(t[3]): t for t in terminal}
    assert by_reward[-1.0][2] == 1 and by_reward[-1.0][6] is True
    assert by_reward[1.0][2] == 0 and by_reward[1.0][6] is True
    assert len(recorder) == 0


def test_reservoir_buffer_keeps_capacity_and_spreads_inserts():
    reservoir = ReservoirBuffer(10, 3, 4, seed=0)
    obs = np.zeros(3, dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)
    for i in range(200):
        reservoir.push(obs, mask, i % 4)
    assert len(reservoir) == 10
    assert reservoir.seen == 200
    sampled_actions = set(reservoir.actions[:10].tolist())
    assert len(sampled_actions) > 1  # later items replaced early ones


def test_masked_double_dqn_ignores_illegal_actions():
    torch.manual_seed(0)
    q = NFSPAgent(6, 5, hidden=16, num_layers=1, device=DEVICE, seed=0)
    obs = torch.rand(4, 6, device=DEVICE)
    mask = torch.zeros(4, 5, device=DEVICE)
    mask[:, 2] = 1.0
    mask[:, 4] = 1.0
    with torch.no_grad():
        values = q.q_network.masked(obs, mask)
        for row in range(4):
            assert float(values[row, 0]) == MASKED_LOGIT_VALUE
            assert float(values[row, 1]) == MASKED_LOGIT_VALUE
            assert float(values[row, 3]) == MASKED_LOGIT_VALUE
            assert float(values[row, 2]) > MASKED_LOGIT_VALUE
    loss = masked_double_dqn_loss(
        q.q_network,
        q.target_network,
        obs,
        mask,
        torch.full((4,), 2, device=DEVICE),
        torch.ones(4, device=DEVICE),
        obs,
        mask,
        torch.ones(4, device=DEVICE),
    )
    loss.backward()
    for name, parameter in q.q_network.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name


def test_nfsp_agent_save_load_and_average_policy():
    agent = NFSPAgent(obs_dim(2), num_actions_for(2), device=DEVICE, seed=5)
    engine = CoupEngine(2, max_turns=120, seed=1)
    engine.reset(seed=2)
    obs, mask = engine.get_obs(engine.current_player), engine.legal_mask()
    probs = agent.action_probs(obs, mask)
    assert pytest.approx(1.0, abs=1e-5) == float(probs.sum())
    assert probs[mask == 0].sum() == 0.0
    action = agent.select_action(obs, mask)
    assert mask[action] == 1


def test_short_nfsp_run_and_metrics_schema(tmp_path: Path):
    config = NFSPConfig(
        total_steps=512,
        num_envs=8,
        iteration_steps=256,
        eval_interval=2,
        eval_games=6,
        save_dir=str(tmp_path / "nfsp"),
        seed=6,
        device=str(DEVICE),
        hidden=32,
        num_layers=2,
        batch_size=32,
        replay_capacity=2000,
        reservoir_capacity=2000,
        target_update=10,
        q_updates_per_tick=2,
        sl_updates_per_tick=1,
        max_turns=120,
        verbose=False,
    )
    trainer = NFSPTrainer(config)
    trainer.hyper.warmup = 64
    summary = trainer.train()
    save_dir = Path(config.save_dir)
    records = [json.loads(line) for line in (save_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(records) >= 2
    for record in records:
        assert set(record) == set(METRIC_KEYS)
        for name in ("random", "honest", "aggressive_bluffer", "cautious_skeptical", "card_counter"):
            assert 0.0 <= record[f"win_rate_{name}"] <= 1.0
    assert summary["steps"] >= config.total_steps
    assert (save_dir / "best.pt").exists() and (save_dir / "final.pt").exists()
    assert (save_dir / "config.json").exists()

    reloaded = NFSPAgent.load(save_dir / "final.pt", device=DEVICE)
    engine = CoupEngine(2, max_turns=120, seed=4)
    engine.reset(seed=8)
    obs, mask = engine.get_obs(engine.current_player), engine.legal_mask()
    probs = reloaded.action_probs(obs, mask)
    assert np.allclose(probs, trainer.agent.action_probs(obs, mask), atol=1e-6)
    assert mask[int(probs.argmax())] == 1


def test_ppo_and_nfsp_metric_schemas_are_identical(tmp_path: Path):
    ppo_config = TrainConfig(
        num_players=2,
        mode="naive",
        total_steps=192,
        num_envs=4,
        rollout_steps=96,
        eval_interval=10**6,
        eval_games=4,
        save_dir=str(tmp_path / "ppo"),
        seed=0,
        device=str(DEVICE),
        hidden=16,
        num_layers=1,
        num_snapshots=0,
        max_turns=100,
        verbose=False,
    )
    PPOTrainer(ppo_config).train()
    nfsp_config = NFSPConfig(
        total_steps=192,
        num_envs=4,
        iteration_steps=96,
        eval_interval=10**6,
        eval_games=4,
        save_dir=str(tmp_path / "nfsp"),
        seed=0,
        device=str(DEVICE),
        hidden=16,
        num_layers=1,
        batch_size=16,
        replay_capacity=1000,
        reservoir_capacity=1000,
        q_updates_per_tick=1,
        sl_updates_per_tick=1,
        max_turns=100,
        verbose=False,
    )
    trainer = NFSPTrainer(nfsp_config)
    trainer.hyper.warmup = 32
    trainer.train()
    ppo_keys = set(json.loads((tmp_path / "ppo" / "metrics.jsonl").read_text().splitlines()[0]))
    nfsp_keys = set(json.loads((tmp_path / "nfsp" / "metrics.jsonl").read_text().splitlines()[0]))
    assert ppo_keys == nfsp_keys == set(METRIC_KEYS)


def test_replay_buffer_wraps_around():
    buffer = ReplayBuffer(4, 3, 2, seed=0)
    obs = np.zeros(3, dtype=np.float32)
    mask = np.ones(2, dtype=np.float32)
    for i in range(6):
        buffer.push((obs, mask, i % 2, float(i), obs, mask, i % 2 == 0))
    assert len(buffer) == 4
    assert buffer.actions.tolist()[:4] == [0, 1, 0, 1]
    batch = buffer.sample(8, DEVICE)
    assert batch["obs"].shape == (8, 3)
    assert torch.isfinite(batch["rewards"]).all()


def test_nfsp_hyperparameter_defaults_match_the_paper():
    hyper = NFSPHyperparams()
    assert hyper.eta == pytest.approx(0.1)
    assert (hyper.epsilon_start, hyper.epsilon_end) == (0.15, 0.02)
    assert (hyper.replay_capacity, hyper.reservoir_capacity) == (150_000, 300_000)
    assert NFSPAgent.epsilon_at(0.0) == pytest.approx(0.15)
    assert NFSPAgent.epsilon_at(1.0) == pytest.approx(0.02)
    assert NFSPAgent.epsilon_at(0.5) == pytest.approx(0.085)
