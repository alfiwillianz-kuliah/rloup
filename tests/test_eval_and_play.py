"""Tests for the Phase 3 evaluation suite, plotting and the play CLI.

The heavy numerical work of a full evaluation run is not repeated here.  What is
tested is everything that can silently go wrong: the Wilson interval algebra, the
legal-state construction of the impossible-claim probe (a bug there would quietly
invalidated every card-counting number), the round-robin / Elo book-keeping, the
4-player finish-rank mapping, behaviour telemetry, the figure pipeline and the
interactive CLI's non-interactive demo path.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rloup.constants import CLAIM_KINDS, Phase, Role
from rloup.eval.evaluate import (
    COIN_BRACKETS,
    Competitor,
    bradley_terry_elo,
    build_impossible_claim_state,
    coin_bracket,
    collect_behaviour,
    elo_from_win_rates,
    evaluate_tables,
    finish_rank,
    four_player_tables,
    impossible_claim_probe,
    iterative_elo,
    round_robin_2p,
    run_games,
    scripted_competitor,
    wilson_interval,
    wilson_record,
)
from rloup.eval.plots import (
    BLUFF_KIND_LABELS,
    generate_all,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Wilson score interval
# --------------------------------------------------------------------------- #
def test_wilson_interval_matches_published_values():
    # Textbook 95% Wilson intervals (z = 1.959964).
    low, high = wilson_interval(10, 20)
    assert low == pytest.approx(0.2993, abs=5e-4)
    assert high == pytest.approx(0.7007, abs=5e-4)

    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.27753, abs=5e-4)

    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.72247, abs=5e-4)
    assert high == pytest.approx(1.0, abs=1e-12)


def test_wilson_interval_is_symmetric_and_brackets_the_point_estimate():
    for trials in (5, 20, 200, 1000):
        for successes in range(0, trials + 1, max(1, trials // 7)):
            low, high = wilson_interval(successes, trials)
            p_hat = successes / trials
            assert 0.0 <= low <= p_hat + 1e-12
            assert p_hat - 1e-12 <= high <= 1.0
            mirror_low, mirror_high = wilson_interval(trials - successes, trials)
            assert low == pytest.approx(1.0 - mirror_high, abs=1e-9)
            assert high == pytest.approx(1.0 - mirror_low, abs=1e-9)


def test_wilson_interval_narrows_with_more_samples():
    widths = [wilson_interval(50, n)[1] - wilson_interval(50, n)[0] for n in (100, 1000, 10000)]
    assert widths == sorted(widths, reverse=True)
    # A degenerate (empty) sample must not divide by zero.
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_record_reports_counts_and_half_width():
    record = wilson_record(7, 10)
    assert record["successes"] == 7 and record["trials"] == 10
    assert record["rate"] == pytest.approx(0.7)
    assert record["half_width"] == pytest.approx(
        (record["ci_high"] - record["ci_low"]) / 2.0
    )
    assert 0.0 <= record["ci_low"] < 0.7 < record["ci_high"] <= 1.0


# --------------------------------------------------------------------------- #
# Coin brackets
# --------------------------------------------------------------------------- #
def test_coin_bracket_boundaries():
    assert COIN_BRACKETS == ("0-2", "3-6", "7-9", "10+")
    assert [coin_bracket(c) for c in (0, 2, 3, 6, 7, 9, 10, 14)] == [
        "0-2",
        "0-2",
        "3-6",
        "3-6",
        "7-9",
        "7-9",
        "10+",
        "10+",
    ]


# --------------------------------------------------------------------------- #
# Impossible-claim probe construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_players", (2, 4))
def test_build_impossible_claim_state_is_legal_and_provably_impossible(num_players):
    rng = np.random.default_rng(1234 + num_players)
    for index in range(120):
        state = build_impossible_claim_state(
            rng, num_players, block_phase=bool(index % 2)
        )
        engine = state.engine
        # A legal 15-card deal, two or more living players, a living viewer.
        assert engine.total_cards() == 15
        assert engine.num_alive >= 2
        assert engine.players[state.viewer].alive
        # The claim is impossible: the viewer knows all three copies.
        own = int(engine.hand_counts(state.viewer)[state.role])
        revealed = int(engine.revealed_counts()[state.role])
        assert own + revealed == 3
        assert engine._role_exhausted_for(state.viewer, Role(state.role))
        # ... and it really is a bluff from the claimant.
        assert state.role not in engine.players[state.claimant].hand
        # The observation exposes it and the decision is pass/challenge only.
        assert state.obs[engine.obs_layout.claimed_impossible] == 1.0
        assert engine.phase in (Phase.CHALLENGE_ACTION, Phase.CHALLENGE_BLOCK)
        assert engine.phase == (
            Phase.CHALLENGE_BLOCK if state.block_phase else Phase.CHALLENGE_ACTION
        )
        assert sorted(np.flatnonzero(state.mask).tolist()) == sorted(
            [state.pass_id, state.challenge_id]
        )


def test_challenging_an_impossible_claim_always_wins():
    rng = np.random.default_rng(7)
    for _ in range(40):
        state = build_impossible_claim_state(rng, 2)
        engine = state.engine
        engine.step(state.challenge_id)
        assert engine.stats[state.viewer]["challenges_won"] == 1
        assert engine.stats[state.claimant]["caught_bluffing"] == 1
        # The caught bluff cancels the action and costs the claimant influence.
        assert engine.players[state.claimant].influence <= 1


def test_impossible_claim_probe_separates_card_counters_from_noise():
    agents = [
        scripted_competitor("CardCounter"),
        scripted_competitor("Honest"),
        scripted_competitor("Random"),
    ]
    probe = impossible_claim_probe(agents, scenarios=200, seed=3, verbose=False)
    assert probe["scenarios"] == 200
    by_agent = probe["agents"]
    # Exact card counters never miss a provably impossible claim.
    assert by_agent["CardCounter"]["challenge_rate"] == pytest.approx(1.0)
    assert by_agent["Honest"]["challenge_rate"] == pytest.approx(1.0)
    # A uniform random policy sits at the 50% coin flip.
    assert 0.35 <= by_agent["Random"]["challenge_rate"] <= 0.65
    for record in by_agent.values():
        assert record["ci_low"] <= record["challenge_rate"] <= record["ci_high"]
        assert record["legal_action_count"] == pytest.approx(2.0)
        assert record["claimed_impossible_in_obs"] == 1.0
        assert record["mean_challenge_prob"] is None  # scripted agents


# --------------------------------------------------------------------------- #
# Elo / Bradley-Terry
# --------------------------------------------------------------------------- #
def test_bradley_terry_orders_a_transitive_field_and_anchors_random():
    names = ["A", "B", "Random"]
    pairs = {
        ("A", "B"): {"a_wins": 700, "b_wins": 300, "draws": 0},
        ("A", "Random"): {"a_wins": 900, "b_wins": 100, "draws": 0},
        ("B", "Random"): {"a_wins": 800, "b_wins": 200, "draws": 0},
    }
    elo = bradley_terry_elo(names, pairs, anchor="Random")
    assert elo["Random"] == pytest.approx(1000.0, abs=1e-9)
    assert elo["A"] > elo["B"] > elo["Random"]
    # 900 vs 100 is a 9:1 win ratio -> ~382 Elo above the anchor.
    assert elo["A"] - 1000.0 == pytest.approx(400.0 * math.log10(9.0), abs=40.0)

    alternative = iterative_elo(names, pairs, anchor="Random")
    assert alternative["A"] > alternative["B"] > alternative["Random"] - 1e-9
    assert alternative["Random"] == pytest.approx(1000.0, abs=1e-6)


def test_bradley_terry_handles_a_winless_anchor_and_empty_input():
    names = ["A", "Random"]
    pairs = {("A", "Random"): {"a_wins": 1000, "b_wins": 0, "draws": 0}}
    elo = bradley_terry_elo(names, pairs, anchor="Random")
    assert elo["A"] > elo["Random"]
    assert all(math.isfinite(value) for value in elo.values())
    assert bradley_terry_elo([], {}) == {}


def test_elo_from_win_rates_recovers_a_known_opponent_rating():
    opponents = {"Random": 1000.0, "Strong": 1200.0}
    # A player scoring exactly the Elo expectation against both must equal the
    # anchor rating.
    rates = {}
    for name, rating in opponents.items():
        rates[name] = 1.0 / (1.0 + 10.0 ** ((rating - 1000.0) / 400.0))
    elo = elo_from_win_rates(rates, opponents, games_per_opponent=200)
    assert elo == pytest.approx(1000.0, abs=1e-6)
    # Winning everything must be far above both anchors.
    assert elo_from_win_rates({"Random": 1.0, "Strong": 1.0}, opponents, games_per_opponent=200) > 1200.0


# --------------------------------------------------------------------------- #
# Finish ranks
# --------------------------------------------------------------------------- #
def test_finish_rank_maps_elimination_order_to_places():
    # 4 players: seat 2 out first, then seat 0, then seat 3 -> seat 1 wins.
    elim = [2, 0, 3]
    assert finish_rank(elim, 1, 1, 4) == 1
    assert finish_rank(elim, 1, 3, 4) == 2
    assert finish_rank(elim, 1, 0, 4) == 3
    assert finish_rank(elim, 1, 2, 4) == 4
    # A truncated game has no ranking at all.
    assert finish_rank(elim, None, 1, 4) is None


# --------------------------------------------------------------------------- #
# Batched game loop and the small tournaments
# --------------------------------------------------------------------------- #
def test_run_games_plays_every_scheduled_game_and_reports_winners():
    schedule = [
        [scripted_competitor("Random"), scripted_competitor("Honest")],
        [scripted_competitor("Honest"), scripted_competitor("Random")],
    ]
    results = run_games(schedule, num_players=2, num_envs=2, seed=0)
    assert len(results) == 2
    for index, result in enumerate(results):
        assert result["seats"] == [c.name for c in schedule[index]]
        assert result["turns"] >= 1
        if result["truncated"]:
            assert result["winner"] is None
        else:
            assert result["winner"] in (0, 1)
            assert len(result["elim_order"]) == 1
            assert sorted(result["rewards"]) == [-1.0, 1.0]


def test_run_games_rejects_malformed_schedules():
    with pytest.raises(ValueError):
        run_games([[scripted_competitor("Random")]], num_players=2)
    assert run_games([], num_players=2) == []


def test_small_round_robin_is_internally_consistent():
    competitors = [
        scripted_competitor("CardCounter"),
        scripted_competitor("Honest"),
        scripted_competitor("Random"),
    ]
    summary = round_robin_2p(competitors, games_per_pair=20, num_envs=8, seed=1, verbose=False)
    assert summary["total_games"] == 3 * 20
    assert summary["agents"] == ["CardCounter", "Honest", "Random"]
    matrix = np.asarray(summary["matrix"], dtype=float)
    assert matrix.shape == (3, 3)
    # Complementary win rates (no truncation expected at these lengths).
    if summary["draws"] == 0:
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert matrix[i, j] + matrix[j, i] == pytest.approx(1.0)
    for row in summary["head_to_head"]:
        assert row["games"] == 20
        assert row["ci_low"] <= row["rate"] <= row["ci_high"]
        assert row["a_seat0_games"] + row["a_seat1_games"] == 20
    # Elo anchored at Random, and the strongest scripted bot above it.
    assert summary["elo"]["Random"] == pytest.approx(1000.0, abs=1e-9)
    assert summary["elo"]["CardCounter"] > summary["elo"]["Random"]
    assert [row["agent"] for row in summary["ranking"]][-1] == "Random"


def test_small_four_player_tables_report_rates_and_ranks():
    learner = scripted_competitor("CardCounter")
    tables = four_player_tables()[:2] + [four_player_tables()[-1]]
    result = evaluate_tables(
        learner,
        tables,
        games_per_table=16,
        num_players=4,
        num_envs=8,
        seed=2,
        verbose=False,
    )
    assert result["chance"] == pytest.approx(0.25)
    assert set(result["tables"]) == {table["name"] for table in tables}
    for name, record in result["tables"].items():
        assert record["games"] == 16
        assert 0.0 <= record["rate"] <= 1.0
        assert record["ci_low"] <= record["rate"] <= record["ci_high"]
        # Each table rotates the learner through all four seats.
        assert sum(record["win_rate_by_seat"].values()) > 0.0 or record["wins"] == 0
        if record["ranked_games"]:
            assert 1.0 <= record["avg_finish_rank"] <= 4.0
            assert sum(record["rank_counts"].values()) == record["ranked_games"]
    assert result["overall"]["played"] + result["overall"]["truncated"] == 3 * 16


def test_collect_behaviour_produces_per_kind_bluff_and_challenge_telemetry():
    learner = scripted_competitor("Honest")
    behaviour = collect_behaviour(
        learner, num_players=2, games_per_table=30, num_envs=8, seed=3, verbose=False
    )
    assert behaviour["games"] == 120  # four tables x 30
    assert set(behaviour["per_table"]) == {"Honest", "Cautious", "CardCounter", "SelfPlay"}
    assert set(behaviour["bluff_rate_by_kind"]) == set(CLAIM_KINDS)
    # The scripted HonestAgent never claims a role it does not hold.
    assert behaviour["bluff_rate"] == pytest.approx(0.0)
    assert behaviour["claims"] > 0
    assert 0.0 <= behaviour["challenge_rate"] <= 1.0
    assert 0.0 <= behaviour["challenge_precision"] <= 1.0
    assert 0.0 <= behaviour["block_rate"] <= 1.0
    assert set(behaviour["coin_brackets"]) == set(COIN_BRACKETS)
    assert behaviour["vs_honest"]["games"] == 30


def test_competitor_requires_a_policy():
    bare = Competitor(name="nobody")
    assert bare.neural is False
    with pytest.raises(RuntimeError):
        bare.new_agent()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _synthetic_metrics(run: str, steps: int = 5) -> str:
    rows = []
    for i in range(steps):
        row = {
            "step": int(i * 500_000),
            "update": i,
            "episodes": i * 100,
            "wall_time": float(i),
            "steps_per_second": 10_000.0,
            "avg_vs_baselines": 0.2 + 0.1 * i,
            "strong_avg": 0.15 + 0.1 * i,
            "win_rate_random": 0.5 + 0.05 * i,
            "win_rate_honest": 0.3 + 0.08 * i,
            "win_rate_aggressive_bluffer": 0.4 + 0.06 * i,
            "win_rate_cautious_skeptical": 0.25 + 0.09 * i,
            "win_rate_card_counter": 0.2 + 0.1 * i,
            "bluff_rate": 0.2,
            "challenge_rate": 0.15,
            "challenge_precision": 0.3,
            "impossible_claim_challenge_rate": 1.0,
            "policy_loss": -0.01,
            "value_loss": 0.5 - 0.05 * i,
            "entropy": 0.7 - 0.1 * i,
            "approx_kl": 0.008,
            "clip_fraction": 0.1,
            "learner_elo": 1000.0 + 20 * i,
            "snapshot_count": i,
        }
        rows.append(json.dumps(row))
    return "\n".join(rows) + "\n"


def _synthetic_summary() -> dict:
    agents = ["PFSP-PPO", "Naive-PPO", "NFSP", "CardCounter", "Honest", "Cautious", "Bluffer", "Random"]
    size = len(agents)
    matrix = [[0.5 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i][j] = 0.5 + 0.04 * (size - i) - 0.04 * (size - j)
    elo = {name: 1000.0 + 30.0 * (size - index) for index, name in enumerate(agents)}
    behaviour = {}
    for name in ("PFSP-PPO", "Naive-PPO", "NFSP", "PFSP-PPO-4P"):
        behaviour[name] = {
            "games": 2000,
            "claims": 400,
            "bluffs": 80,
            "bluff_rate": 0.2,
            "bluff_rate_by_kind": {kind: 0.15 + 0.01 * k for k, kind in enumerate(CLAIM_KINDS)},
            "challenge_rate": 0.2,
            "challenge_precision": 0.35,
            "block_rate": 0.1,
            "coin_brackets": {bracket: {"TAX": 3} for bracket in COIN_BRACKETS},
            "vs_honest": {"games": 500},
        }
    probe_agents = {
        name: {
            "challenge_rate": 0.9,
            "ci_low": 0.88,
            "ci_high": 0.92,
            "mean_challenge_prob": 0.9,
        }
        for name in ("PFSP-PPO", "Naive-PPO", "NFSP", "CardCounter", "Honest", "Random")
    }
    models = {}
    for index, name in enumerate(("PFSP-PPO", "Naive-PPO", "NFSP")):
        rate = 0.52 + 0.05 * index
        models[name] = {
            "exploiter_win_rate": rate,
            "ci95_low": rate - 0.03,
            "ci95_high": rate + 0.03,
            "approx_exploitability": max(0.0, rate - 0.5),
            "games": 1000,
        }
    return {
        "round_robin_2p": {
            "agents": agents,
            "matrix": matrix,
            "elo": elo,
        },
        "four_player": {"tables": {"3xHonest": {"rate": 0.4}}, "overall": {"rate": 0.4}},
        "behaviour": behaviour,
        "impossible_claim_probe": {"two_player": {"agents": probe_agents}},
        "exploitability_2p": {"models": models},
    }


def test_plot_pipeline_writes_five_non_empty_pngs(tmp_path: Path):
    results = tmp_path / "results"
    for run in ("ppo_pfsp_2p", "ppo_naive_2p", "nfsp_2p", "ppo_pfsp_4p"):
        run_dir = results / run
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.jsonl").write_text(_synthetic_metrics(run), encoding="utf-8")
    summary_path = results / "eval_summary.json"
    summary_path.write_text(json.dumps(_synthetic_summary()), encoding="utf-8")

    paths = generate_all(results, tmp_path / "figures", summary_path=summary_path)
    names = [path.name for path in paths]
    assert names == [
        "learning_curves_2p.png",
        "learning_curve_4p.png",
        "round_robin_2p.png",
        "behavioral_diagnostics.png",
        "exploitability_2p.png",
    ]
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 10_000  # a real PNG, not an empty canvas
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_pipeline_requires_an_eval_summary(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        generate_all(tmp_path / "missing", tmp_path / "figures")


def test_committed_figures_are_rendered_when_present():
    summary = REPO_ROOT / "results" / "eval_summary.json"
    if not summary.exists():
        pytest.skip("no eval_summary.json yet")
    bars = [kind for kind, _ in BLUFF_KIND_LABELS]
    assert len(bars) == 8
    for name in (
        "learning_curves_2p.png",
        "learning_curve_4p.png",
        "round_robin_2p.png",
        "behavioral_diagnostics.png",
        "exploitability_2p.png",
    ):
        path = REPO_ROOT / "results" / "figures" / name
        if not path.exists():
            pytest.skip(f"{name} not generated yet")
        assert path.stat().st_size > 10_000


# --------------------------------------------------------------------------- #
# Play CLI
# --------------------------------------------------------------------------- #
def _run_play(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "rloup.play", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_play_demo_runs_a_full_2player_game():
    completed = _run_play("--demo", "--num-players", "2")
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "Winner:" in output or "truncated" in output
    assert "final:" in output
    assert "seat 0" in output and "seat 1" in output
    assert "influence" in output


def test_play_demo_runs_a_full_4player_game():
    completed = _run_play("--demo", "--num-players", "4")
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "final:" in output
    for seat in range(4):
        assert f"seat {seat}" in output
    assert "turns:" in output


def test_play_demo_with_scripted_card_counter_and_probabilities():
    completed = _run_play(
        "--demo", "--opponent", "card_counter", "--show-probs", "--seed", "5"
    )
    assert completed.returncode == 0, completed.stderr
    # The scripted bot has no distribution, so it is labelled as such.
    assert "scripted policy" in completed.stdout


def test_play_rejects_a_checkpoint_for_the_wrong_seat_count():
    checkpoint = REPO_ROOT / "results" / "ppo_pfsp_2p" / "final.pt"
    if not checkpoint.exists():
        pytest.skip("2-player checkpoint not trained yet")
    completed = _run_play("--checkpoint", str(checkpoint), "--demo", "--num-players", "4")
    assert completed.returncode != 0
    assert "players" in (completed.stderr + completed.stdout)
