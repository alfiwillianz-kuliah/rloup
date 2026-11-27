"""Tests for the scripted baseline agents."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rloup import CoupEngine
from rloup.agents import (
    BASELINES,
    AggressiveBlufferAgent,
    CardCounterAgent,
    CautiousSkepticalAgent,
    HonestAgent,
    RandomAgent,
    parse_obs,
)
from rloup.constants import NUM_ROLES, Phase, Role
from rloup.engine import obs_dim

AGENT_CASES = (
    RandomAgent,
    HonestAgent,
    AggressiveBlufferAgent,
    CautiousSkepticalAgent,
    CardCounterAgent,
)

A = int(Role.ASSASSIN)
C = int(Role.CAPTAIN)
D = int(Role.DUKE)
M = int(Role.AMBASSADOR)
T = int(Role.CONTESSA)


# --------------------------------------------------------------------------- #
# Observation parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", (2, 3, 4, 6))
def test_parse_obs_matches_engine(n):
    rng = np.random.default_rng(n)
    e = CoupEngine(n, max_turns=250)
    for game in range(30):
        e.reset(seed=game, randomize_start=True)
        while not e.done:
            seat = e.current_player
            obs = e.get_obs(seat)
            mask = e.legal_mask()
            view = parse_obs(obs, mask)
            assert view.num_players == n
            assert view.n_actions == e.n_actions
            assert view.own_hand == sorted(int(c) for c in e.players[seat].hand)
            assert view.seat.coins == e.players[seat].coins
            assert view.seat.influence == len(e.players[seat].hand)
            assert view.seat.alive is bool(e.players[seat].hand)
            assert view.phase == Phase(e.phase)
            assert view.num_alive == e.num_alive
            assert view.deck_size == len(e.deck)
            assert np.array_equal(
                view.legal_ids, np.flatnonzero(e.legal_mask())
            )
            for rel in range(n):
                squad = e.players[(seat + rel) % n]
                assert view.seats[rel].influence == len(squad.hand)
                assert view.seats[rel].coins == squad.coins
                assert view.seats[rel].alive is bool(squad.hand)
                assert np.array_equal(
                    view.seats[rel].revealed,
                    np.bincount(squad.revealed, minlength=NUM_ROLES),
                )
            if e.pending_action is not None:
                assert view.pending_action == e.pending_action
            assert view.in_exchange == (
                e.phase == Phase.EXCHANGE_SELECT and e.current_player == seat
            )
            e.step(int(rng.choice(view.legal_ids)))


def test_parse_obs_unseen_and_known_counts():
    e = CoupEngine(2)
    e.reset(seed=0, randomize_start=False)
    e.debug_set_hands({0: [D, C], 1: [T, M]}, coins={0: 2, 1: 2})
    view = parse_obs(e.get_obs(0), e.legal_mask())
    assert view.own_hand_counts.tolist() == [1, 0, 1, 0, 0]
    # 3 copies of each role minus what the viewer holds and what is revealed
    assert view.unseen.tolist() == [2, 3, 2, 3, 3]
    e.players[1].revealed = [T]
    view = parse_obs(e.get_obs(0), e.legal_mask())
    assert view.total_revealed.tolist() == [0, 0, 0, 0, 1]
    assert view.unseen.tolist() == [2, 3, 2, 3, 2]
    assert view.known.tolist() == [1, 0, 1, 0, 1]


def test_view_helpers():
    e = CoupEngine(2)
    e.reset(seed=0, randomize_start=False)
    e.debug_set_hands({0: [M, D], 1: [T, C]}, coins={0: 2, 1: 2})
    e.step(e.spec.exchange)
    e.step(e.spec.pass_)
    view = parse_obs(e.get_obs(0), e.legal_mask())
    assert view.in_exchange
    assert view.buffer == sorted(view.buffer)
    assert len(view.buffer) == 4
    options = view.keep_options()
    assert options and all(len(slots) == 2 for _, slots in options)
    assert view.has(Role.DUKE) and view.has(M)
    assert not view.has(Role.CAPTAIN)
    assert view.candidates("EXCHANGE_KEEP")
    assert view.claimed_role is None


# --------------------------------------------------------------------------- #
# Legal actions everywhere
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("agent_cls", AGENT_CASES)
@pytest.mark.parametrize("n", (2, 3, 4, 6))
def test_agents_only_choose_legal_actions(agent_cls, n):
    rng = np.random.default_rng(99 + n)
    e = CoupEngine(n, max_turns=250)
    for game in range(40):
        e.reset(seed=game * 13 + n, randomize_start=True)
        agents = [agent_cls(seed=game + i) for i in range(n)]
        guard = 0
        while not e.done:
            seat = e.current_player
            mask = e.legal_mask()
            action = agents[seat].select_action(e.get_obs(seat), mask, e, seat)
            assert 0 <= action < e.n_actions
            assert mask[action] == 1, (
                f"{agent_cls.__name__} chose illegal {e.spec.action_name(action)} "
                f"in {Phase(e.phase).name}"
            )
            e.step(action)
            guard += 1
            if guard > 3000:
                raise AssertionError("game did not terminate")


def test_agent_signature_accepts_engine_and_player_id():
    e = CoupEngine(3)
    e.reset(seed=0, randomize_start=False)
    seat = e.current_player
    obs, mask = e.get_obs(seat), e.legal_mask()
    for cls in AGENT_CASES:
        with_engine = cls(seed=1).select_action(obs, mask, e, seat)
        without = cls(seed=1).select_action(obs, mask)
        assert mask[with_engine] == 1
        assert mask[without] == 1
        assert with_engine == without, "the engine handle is never consulted"


def test_agents_do_not_look_at_hidden_cards():
    """Opponent hands must not change any agent's decision."""
    for cls in AGENT_CASES:
        e1 = CoupEngine(3)
        e1.reset(seed=7, randomize_start=False)
        e1.debug_set_hands({0: [D, C], 1: [T, M], 2: [A, M]}, coins={0: 2, 1: 2, 2: 2})
        e2 = CoupEngine(3)
        e2.reset(seed=7, randomize_start=False)
        e2.debug_set_hands({0: [D, C], 1: [A, A], 2: [T, T]}, coins={0: 2, 1: 2, 2: 2})
        a = cls(seed=5).select_action(e1.get_obs(0), e1.legal_mask())
        b = cls(seed=5).select_action(e2.get_obs(0), e2.legal_mask())
        assert a == b, f"{cls.__name__} peeked at a hidden hand"


@pytest.mark.parametrize("agent_cls", AGENT_CASES)
def test_agent_reset_reseeds(agent_cls):
    obs, mask = _opening_state()
    agent = agent_cls(seed=0)
    agent.reset(seed=11)
    a = agent.select_action(obs, mask)
    agent.reset(seed=11)
    b = agent.select_action(obs, mask)
    assert a == b


def _opening_state():
    e = CoupEngine(2)
    e.reset(seed=0, randomize_start=False)
    return e.get_obs(0), e.legal_mask()


# --------------------------------------------------------------------------- #
# Card counting maths
# --------------------------------------------------------------------------- #
def test_p_holds_is_the_exact_hypergeometric_probability():
    e = CoupEngine(2)
    e.reset(seed=0, randomize_start=False)
    e.debug_set_hands({0: [D, C], 1: [T, M]}, coins={0: 2, 1: 2})
    view = parse_obs(e.get_obs(0), e.legal_mask())
    unseen = view.unseen.sum()

    # P(seat 1's two cards contain at least one Assassin)
    expected = 1.0 - math.comb(int(unseen - view.unseen[A]), 2) / math.comb(int(unseen), 2)
    got = CardCounterAgent._p_holds(view, Role.ASSASSIN, 2)
    assert abs(got - expected) < 1e-12

    # a role we hold twice can only be one specific unseen card
    e.debug_set_hands({0: [D, D], 1: [T, M]}, coins={0: 2, 1: 2})
    view = parse_obs(e.get_obs(0), e.legal_mask())
    assert view.unseen[D] == 1
    total = int(view.unseen.sum())
    expected = 1.0 - math.comb(total - 1, 2) / math.comb(total, 2)
    assert CardCounterAgent._p_holds(view, Role.DUKE, 2) == pytest.approx(expected)

    # nothing left of a role -> probability zero
    e.debug_set_hands({0: [D, D], 1: [T, M]}, revealed={0: [D]}, coins={0: 2, 1: 2})
    view = parse_obs(e.get_obs(0), e.legal_mask())
    assert CardCounterAgent._p_holds(view, Role.DUKE, 1) == 0.0


def test_card_counter_bluff_posterior():
    e = CoupEngine(2)
    e.reset(seed=0, randomize_start=False)
    e.debug_set_hands({0: [D, C], 1: [T, M]}, coins={0: 2, 1: 2})
    agent = CardCounterAgent(seed=0)
    view = parse_obs(e.get_obs(0), e.legal_mask())
    beta = agent._bluff_rate(view, 1)
    assert beta == pytest.approx(0.25), "Beta(1, 3) prior"
    p_bluff = agent._p_bluff(view, Role.DUKE, 1)
    assert 0.0 <= p_bluff <= 1.0
    # a reputation for bluffing raises the posterior
    e.claim_counts[1, 5] = 4  # caught bluffing
    view = parse_obs(e.get_obs(0), e.legal_mask())
    assert agent._bluff_rate(view, 1) > beta
    # an impossible claim is essentially certain to be a bluff
    e.claim_counts[1, 5] = 0
    e.debug_set_hands({0: [D, D], 1: [T, M]}, revealed={0: [D]}, coins={0: 2, 1: 2})
    view = parse_obs(e.get_obs(0), e.legal_mask())
    assert agent._p_bluff(view, Role.DUKE, 1) > 0.9


def test_card_counter_always_challenges_a_provable_bluff():
    e = CoupEngine(2)
    e.reset(seed=0, randomize_start=False)
    e.debug_set_hands({0: [T, M], 1: [D, M]}, revealed={1: [D, D]}, coins={0: 2, 1: 2})
    e.step(e.spec.tax)  # provably impossible Duke claim
    agent = CardCounterAgent(seed=0)
    view = parse_obs(e.get_obs(1), e.legal_mask())
    assert view.claimed_impossible
    assert agent.select_action(e.get_obs(1), e.legal_mask()) == e.spec.challenge


# --------------------------------------------------------------------------- #
# Behavioural sanity of the baselines
# --------------------------------------------------------------------------- #
def _win_rate(a_cls, b_cls, games=200, base_seed=0) -> float:
    e = CoupEngine(2, max_turns=200)
    wins = 0
    for g in range(games):
        e.reset(seed=base_seed + g)
        agents = [a_cls(seed=g), b_cls(seed=g + 10_000)]
        steps = 0
        while not e.done:
            seat = e.current_player
            e.step(agents[seat].select_action(e.get_obs(seat), e.legal_mask()))
            steps += 1
            if steps > 3000:  # pragma: no cover - defensive
                raise AssertionError("game did not terminate")
        if e.winner() == 0:
            wins += 1
    return wins / games


@pytest.mark.parametrize("agent_cls", (HonestAgent, CardCounterAgent, CautiousSkepticalAgent))
def test_stronger_baselines_decisively_beat_random(agent_cls):
    rate = _win_rate(agent_cls, RandomAgent)
    assert rate >= 0.70, f"{agent_cls.__name__} only won {rate:.1%} vs RandomAgent"


def test_card_counter_beats_the_aggressive_bluffer():
    rate = _win_rate(CardCounterAgent, AggressiveBlufferAgent, games=100)
    assert rate >= 0.70, f"card counter only won {rate:.1%} vs the bluffer"


def test_bluffing_agent_actually_bluffs():
    e = CoupEngine(4, max_turns=250)
    e.reset(seed=3, randomize_start=True)
    agents = [AggressiveBlufferAgent(seed=i) for i in range(4)]
    while not e.done:
        seat = e.current_player
        e.step(agents[seat].select_action(e.get_obs(seat), e.legal_mask()))
    stats = e.get_episode_stats()
    claims = sum(sum(p["claims"].values()) for p in stats["players"].values())
    bluffs = sum(sum(p["bluffs"].values()) for p in stats["players"].values())
    assert claims > 0
    assert bluffs > 0, "the bluffer never bluffed"


def test_honest_agent_never_bluffs():
    e = CoupEngine(3, max_turns=250)
    e.reset(seed=8, randomize_start=True)
    agents = [HonestAgent(seed=i) for i in range(3)]
    while not e.done:
        seat = e.current_player
        e.step(agents[seat].select_action(e.get_obs(seat), e.legal_mask()))
    stats = e.get_episode_stats()
    for player in stats["players"].values():
        assert sum(player["bluffs"].values()) == 0
        assert player["caught_bluffing"] == 0
