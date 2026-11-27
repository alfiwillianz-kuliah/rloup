"""Rule-by-rule unit tests for :class:`rloup.engine.CoupEngine`."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from rloup.constants import (
    COPIES_PER_ROLE,
    DECK_SIZE,
    EXCHANGE_KEEP_OPTIONS,
    NUM_EXCHANGE_KEEP_SLOTS,
    ActionSpaceSpec,
    ActionType,
    Phase,
    Role,
    num_actions_for,
    players_from_num_actions,
)
from rloup.engine import CLAIM_FEATURES, CoupEngine, make_obs_layout, obs_dim

# plain ints, so asserts can compare against the engine's raw card ids
A = int(Role.ASSASSIN)
C = int(Role.CAPTAIN)
D = int(Role.DUKE)
M = int(Role.AMBASSADOR)
T = int(Role.CONTESSA)

PLAYER_COUNTS = (2, 3, 4, 6)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make(
    n: int = 2,
    seed: int = 0,
    hands: dict[int, list[int]] | None = None,
    coins: dict[int, int] | None = None,
    revealed: dict[int, list[int]] | None = None,
    deck: list[int] | None = None,
    max_turns: int = 200,
    **kwargs,
) -> CoupEngine:
    """Build an engine in a deterministic, optionally forced state."""
    e = CoupEngine(n, max_turns=max_turns, **kwargs)
    e.reset(seed=seed, randomize_start=False)
    if hands or coins or revealed or deck is not None:
        e.debug_set_hands(hands or {}, deck=deck, coins=coins, revealed=revealed)
    assert e.current_player == 0
    return e


def legal(e: CoupEngine) -> list[int]:
    return [int(a) for a in np.flatnonzero(e.legal_mask())]


def hand(e: CoupEngine, seat: int) -> list[int]:
    return sorted(int(c) for c in e.players[seat].hand)


def revealed(e: CoupEngine, seat: int) -> list[int]:
    return sorted(int(c) for c in e.players[seat].revealed)


def coins(e: CoupEngine) -> list[int]:
    return [p.coins for p in e.players]


def force_draw(e: CoupEngine, cards: list[int]) -> None:
    """Put ``cards`` on top of the deck so the next draw takes exactly them."""
    for card in cards:
        e.deck.remove(int(card))
    e.deck.extend(int(c) for c in cards)


# --------------------------------------------------------------------------- #
# Constants and action space
# --------------------------------------------------------------------------- #
def test_roles_and_deck_size():
    assert [int(r) for r in Role] == [0, 1, 2, 3, 4]
    assert COPIES_PER_ROLE == 3
    assert DECK_SIZE == 15
    assert [p.name for p in Phase] == [
        "CHOOSE_ACTION",
        "CHALLENGE_ACTION",
        "BLOCK_ACTION",
        "CHALLENGE_BLOCK",
        "LOSE_INFLUENCE",
        "EXCHANGE_SELECT",
        "GAME_OVER",
    ]
    assert [a.name for a in ActionType] == [
        "INCOME",
        "FOREIGN_AID",
        "COUP",
        "TAX",
        "ASSASSINATE",
        "STEAL",
        "EXCHANGE",
    ]


@pytest.mark.parametrize("n", (2, 3, 4, 5, 6))
def test_action_space_layout(n):
    spec = ActionSpaceSpec(n)
    assert spec.n_actions == 21 + 3 * (n - 1) == num_actions_for(n)
    assert players_from_num_actions(spec.n_actions) == n
    assert spec.n_actions == {
        2: 24,
        3: 27,
        4: 30,
        5: 33,
        6: 36,
    }[n]

    # fixed single-action slots
    assert spec.decode(0).kind == "INCOME"
    assert spec.decode(1).kind == "FOREIGN_AID"
    assert spec.decode(2).kind == "TAX"
    assert spec.decode(3).kind == "EXCHANGE"

    # targeted ranges are contiguous, seat-relative offsets 1..n-1
    for kind, start in (
        ("COUP", spec.coup_offset),
        ("ASSASSINATE", spec.assassinate_offset),
        ("STEAL", spec.steal_offset),
    ):
        ids = list(spec.target_actions(kind))
        assert len(ids) == n - 1
        assert ids[0] == start
        for offset, action_id in enumerate(ids, start=1):
            info = spec.decode(action_id)
            assert info.kind == kind
            assert info.rel_off == offset
            assert spec.decode(action_id) == info

    # every action id decodes and has a unique, non-empty name
    names = [spec.action_name(a) for a in range(spec.n_actions)]
    assert all(names)
    assert len(set(names)) == spec.n_actions

    # block slots map back to roles
    for role in (D, T, C, M):
        assert spec.decode(spec.block_action(role)).role == role
        assert spec.decode(spec.block_action(role)).kind == "BLOCK"
    assert spec.decode(spec.pass_).kind == "PASS"
    assert spec.decode(spec.challenge).kind == "CHALLENGE"
    assert spec.decode(spec.lose_slot(0)).index == 0
    assert spec.decode(spec.lose_slot(1)).index == 1
    for index in range(NUM_EXCHANGE_KEEP_SLOTS):
        assert spec.decode(spec.exchange_keep(index)).index == index

    # out-of-range handling
    with pytest.raises(ValueError):
        spec.decode(spec.n_actions)
    with pytest.raises(ValueError):
        spec.coup(0)
    with pytest.raises(ValueError):
        spec.coup(n)
    with pytest.raises(ValueError):
        ActionSpaceSpec(1)
    with pytest.raises(ValueError):
        ActionSpaceSpec(7)


def test_exchange_keep_options_shape():
    assert len(EXCHANGE_KEEP_OPTIONS) == 9
    assert EXCHANGE_KEEP_OPTIONS[:6] == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    assert EXCHANGE_KEEP_OPTIONS[6:] == ((0,), (1,), (2,))


# --------------------------------------------------------------------------- #
# Setup: deck, hands, coins
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_deck_composition_and_hands(n):
    e = make(n)
    assert e.total_cards() == DECK_SIZE
    assert len(e.deck) == DECK_SIZE - 2 * n
    flat = [int(c) for p in e.players for c in p.hand] + list(e.deck)
    assert sorted(flat) == sorted([int(r) for r in Role for _ in range(3)])
    for p in e.players:
        assert len(p.hand) == 2
        assert p.hand == sorted(p.hand), "hands are always sorted ascending"
        assert p.revealed == []
        assert p.alive


def test_two_player_starting_coins():
    """The first player gets 1 coin, the other 2 (official 2-player rule)."""
    for seed in range(40):
        e = CoupEngine(2)
        start = e.reset(seed=seed, randomize_start=True)
        assert coins(e)[start] == 1
        assert coins(e)[1 - start] == 2


def test_two_player_coin_rule_can_be_disabled():
    e = CoupEngine(2, two_player_coin_rule=False)
    e.reset(seed=0, randomize_start=False)
    assert coins(e) == [2, 2]


@pytest.mark.parametrize("n", (3, 4, 5, 6))
def test_multiplayer_starting_coins(n):
    e = CoupEngine(n)
    e.reset(seed=3, randomize_start=True)
    assert coins(e) == [2] * n


def test_randomize_start_covers_all_seats():
    e = CoupEngine(4)
    starts = {e.reset(seed=s, randomize_start=True) for s in range(60)}
    assert starts == {0, 1, 2, 3}
    for s in range(5):
        assert e.reset(seed=s, randomize_start=False) == 0


def test_reset_is_seed_deterministic():
    e = CoupEngine(3)
    e.reset(seed=7, randomize_start=True)
    first = ([p.hand[:] for p in e.players], e.deck[:], e.current_player)
    e.reset(seed=7, randomize_start=True)
    second = ([p.hand[:] for p in e.players], e.deck[:], e.current_player)
    assert first == second


def test_invalid_construction():
    with pytest.raises(ValueError):
        CoupEngine(1)
    with pytest.raises(ValueError):
        CoupEngine(7)
    with pytest.raises(ValueError):
        CoupEngine(2, max_turns=0)
    with pytest.raises(ValueError):
        CoupEngine(2, reward_mode="nope")


# --------------------------------------------------------------------------- #
# Legal masks
# --------------------------------------------------------------------------- #
def test_choose_action_mask_basics():
    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    ids = set(legal(e))
    spec = e.spec
    assert spec.income in ids and spec.foreign_aid in ids and spec.tax in ids
    assert spec.exchange in ids
    assert spec.coup(1) not in ids, "coup needs 7 coins"
    assert spec.assassinate(1) not in ids, "assassinate needs 3 coins"
    assert spec.steal(1) in ids, "steal is legal even against broke players"
    assert spec.pass_ not in ids

    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 7, 1: 2})
    assert e.spec.coup(1) in set(legal(e))

    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 3, 1: 2})
    assert e.spec.assassinate(1) in set(legal(e))


def test_mandatory_coup_at_ten_coins():
    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 10, 1: 2})
    ids = legal(e)
    assert ids == [e.spec.coup(1)], "only coup is legal at >= 10 coins"
    for coins_value in (10, 11, 15):
        e = make(hands={0: [D, C], 1: [A, M]}, coins={0: coins_value, 1: 2})
        assert legal(e) == [e.spec.coup(1)]


def test_steal_illegal_without_opponents_alive():
    e = make(3, hands={0: [D, C], 1: [A, M], 2: [T, T]}, revealed={2: [D, C]})
    e.players[2].hand = []
    ids = set(legal(e))
    assert e.spec.steal(1) in ids
    assert e.spec.steal(2) not in ids, "cannot target a dead seat"


def test_illegal_and_out_of_range_actions_raise():
    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    with pytest.raises(ValueError):
        e.step(e.spec.coup(1))
    with pytest.raises(ValueError):
        e.step(e.n_actions + 5)
    with pytest.raises(ValueError):
        e.step(-1)


# --------------------------------------------------------------------------- #
# Income / Foreign Aid
# --------------------------------------------------------------------------- #
def test_income():
    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.income)
    assert coins(e) == [3, 2]
    assert e.phase == Phase.CHOOSE_ACTION
    assert e.current_player == 1
    assert e.turn == 1


def test_foreign_aid_unblocked():
    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.foreign_aid)
    assert e.phase == Phase.BLOCK_ACTION
    assert e.current_player == 1
    assert set(legal(e)) == {e.spec.pass_, e.spec.block_duke}
    e.step(e.spec.pass_)
    assert coins(e) == [4, 2]
    assert e.current_player == 1
    assert e.phase == Phase.CHOOSE_ACTION


def test_foreign_aid_blocked_by_duke_unchallenged():
    e = make(hands={0: [D, C], 1: [D, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.foreign_aid)
    e.step(e.spec.block_duke)
    assert e.phase == Phase.CHALLENGE_BLOCK
    assert e.current_player == 0, "the actor may challenge the block"
    e.step(e.spec.pass_)
    assert coins(e) == [2, 2], "blocked foreign aid pays nothing"
    assert e.phase == Phase.CHOOSE_ACTION
    assert e.current_player == 1
    assert revealed(e, 1) == []


def test_foreign_aid_block_challenged_and_proven():
    e = make(hands={0: [D, C], 1: [D, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.foreign_aid)
    e.step(e.spec.block_duke)
    e.step(e.spec.challenge)
    # challenger (seat 0) loses an influence, block stands, no coins paid
    assert e.phase == Phase.LOSE_INFLUENCE
    assert e.current_player == 0
    e.step(e.spec.lose_slot(0))
    assert coins(e) == [2, 2]
    assert len(e.players[0].hand) == 1
    assert len(e.players[1].hand) == 2, "the blocker proved and redrew"
    assert e.phase == Phase.CHOOSE_ACTION
    assert e.current_player == 1


def test_foreign_aid_block_is_a_bluff_and_gets_caught():
    e = make(hands={0: [T, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.foreign_aid)
    e.step(e.spec.block_duke)  # seat 1 has no Duke
    e.step(e.spec.challenge)
    assert e.phase == Phase.LOSE_INFLUENCE
    assert e.current_player == 1
    e.step(e.spec.lose_slot(0))
    assert coins(e) == [4, 2], "the action resolves once the bluff is caught"
    assert len(e.players[1].hand) == 1


# --------------------------------------------------------------------------- #
# Tax (Duke)
# --------------------------------------------------------------------------- #
def test_tax_unchallenged():
    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.tax)
    assert e.phase == Phase.CHALLENGE_ACTION
    assert e.current_player == 1
    assert set(legal(e)) == {e.spec.pass_, e.spec.challenge}
    e.step(e.spec.pass_)
    assert coins(e) == [5, 2]
    assert e.current_player == 1


def test_tax_challenged_and_proven():
    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.tax)
    e.step(e.spec.challenge)
    assert e.current_player == 1 and e.phase == Phase.LOSE_INFLUENCE
    e.step(e.spec.lose_slot(0))
    assert coins(e) == [5, 2], "tax still pays after a failed challenge"
    assert len(e.players[0].hand) == 2, "proven Duke is replaced"
    assert D in e.deck
    assert len(e.players[1].hand) == 1


def test_tax_challenged_and_caught():
    e = make(hands={0: [T, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.tax)
    e.step(e.spec.challenge)
    assert e.current_player == 0 and e.phase == Phase.LOSE_INFLUENCE
    e.step(e.spec.lose_slot(0))
    assert coins(e) == [2, 2], "no tax income for a caught bluffer"
    assert len(e.players[0].hand) == 1


def test_proven_card_returns_to_deck_and_is_replaced():
    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.deck = [int(D)]  # single-card deck: the draw is deterministic
    before = sorted(hand(e, 0) + e.deck)
    e._prove_role(0, D)
    assert len(hand(e, 0)) == 2, "a replacement card is drawn"
    assert len(e.deck) == 1, "the proven card entered the deck and one left it"
    assert sorted(hand(e, 0) + e.deck) == before
    assert D in e.deck, "the proven Duke is back in the Court deck"


def test_challenge_keeps_tableau_sizes():
    e = make(hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.tax)
    e.step(e.spec.challenge)
    e.step(e.spec.lose_slot(0))
    assert len(e.players[0].hand) == 2, "the proven claim redraws"
    assert len(e.players[1].hand) == 1
    assert e.total_cards() == DECK_SIZE


# --------------------------------------------------------------------------- #
# Assassinate
# --------------------------------------------------------------------------- #
def test_assassinate_pays_up_front_and_kills():
    e = make(hands={0: [A, C], 1: [T, M]}, coins={0: 5, 1: 2})
    e.step(e.spec.assassinate(1))
    assert coins(e) == [2, 2], "3 coins are paid on declaration"
    assert e.phase == Phase.CHALLENGE_ACTION
    e.step(e.spec.pass_)
    assert e.phase == Phase.BLOCK_ACTION, "target gets the Contessa window"
    assert e.current_player == 1
    assert set(legal(e)) == {e.spec.pass_, e.spec.block_contessa}
    e.step(e.spec.pass_)
    assert e.phase == Phase.LOSE_INFLUENCE
    assert set(legal(e)) == {e.spec.lose_slot(0), e.spec.lose_slot(1)}
    e.step(e.spec.lose_slot(1))  # sorted hand [Ambassador, Contessa] -> lose slot 1
    assert revealed(e, 1) == [T]
    assert hand(e, 1) == [M]
    assert coins(e) == [2, 2]
    assert not e.terminated


def test_assassinate_blocked_by_contessa_no_refund():
    e = make(hands={0: [A, C], 1: [T, M]}, coins={0: 5, 1: 2})
    e.step(e.spec.assassinate(1))
    e.step(e.spec.pass_)
    e.step(e.spec.block_contessa)
    assert e.phase == Phase.CHALLENGE_BLOCK
    assert e.current_player == 0
    e.step(e.spec.pass_)
    assert coins(e) == [2, 2], "a standing block does not refund"
    assert revealed(e, 1) == []
    assert e.phase == Phase.CHOOSE_ACTION
    assert e.current_player == 1


def test_assassinate_challenge_succeeds_and_refunds():
    e = make(hands={0: [D, C], 1: [T, M]}, coins={0: 5, 1: 2})
    e.step(e.spec.assassinate(1))  # seat 0 has no Assassin
    assert coins(e) == [2, 2]
    e.step(e.spec.challenge)
    assert e.phase == Phase.LOSE_INFLUENCE
    assert e.current_player == 0
    e.step(e.spec.lose_slot(0))
    assert coins(e) == [5, 2], "a caught Assassinate is refunded"
    assert len(e.players[0].hand) == 1
    assert len(e.players[1].hand) == 2, "the action was cancelled"
    assert e.current_player == 1


def test_assassinate_double_loss_case_a_pass_block():
    """Target challenges the Assassin, loses, then declines the block window."""
    e = make(hands={0: [A, C], 1: [T, M]}, coins={0: 5, 1: 2})
    e.step(e.spec.assassinate(1))
    e.step(e.spec.challenge)  # truthful challenge -> challenger loses
    assert e.current_player == 1 and e.phase == Phase.LOSE_INFLUENCE
    e.step(e.spec.lose_slot(0))
    assert len(e.players[1].hand) == 1, "first influence gone"
    assert e.phase == Phase.BLOCK_ACTION, "the target still gets its block window"
    assert e.current_player == 1
    e.step(e.spec.pass_)
    assert e.terminated, "the assassinate then takes the last influence"
    assert revealed(e, 1) == sorted([T, M])
    assert e.rewards.tolist() == [1.0, -1.0]


def test_assassinate_double_loss_case_a_can_still_block():
    """After losing the challenge the target may still block with Contessa."""
    e = make(hands={0: [A, C], 1: [T, M]}, coins={0: 5, 1: 2})
    e.step(e.spec.assassinate(1))
    e.step(e.spec.challenge)
    e.step(e.spec.lose_slot(0))
    assert e.phase == Phase.BLOCK_ACTION
    e.step(e.spec.block_contessa)  # still alive, still has the Contessa
    assert e.phase == Phase.CHALLENGE_BLOCK
    e.step(e.spec.pass_)
    assert not e.terminated
    assert len(e.players[1].hand) == 1
    assert revealed(e, 1) == [M]
    assert coins(e) == [2, 2]
    assert e.phase == Phase.CHOOSE_ACTION


def test_assassinate_double_loss_case_b_bluff_caught():
    """Target bluffs Contessa, is challenged, then eats the assassinate."""
    e = make(hands={0: [A, C], 1: [T, M]}, coins={0: 5, 1: 2})
    e.step(e.spec.assassinate(1))
    e.step(e.spec.pass_)  # no challenge on the Assassin
    e.step(e.spec.block_contessa)  # seat 1 holds Contessa
    e.step(e.spec.pass_)  # seat 0 trusts the block
    assert e.current_player == 1 and not e.terminated
    assert len(e.players[1].hand) == 2

    # now the bluffing variant: seat 1 has no Contessa
    e = make(hands={0: [A, C], 1: [C, M]}, coins={0: 5, 1: 2})
    e.step(e.spec.assassinate(1))
    e.step(e.spec.pass_)
    e.step(e.spec.block_contessa)  # bluff
    assert e.phase == Phase.CHALLENGE_BLOCK
    e.step(e.spec.challenge)
    assert e.phase == Phase.LOSE_INFLUENCE and e.current_player == 1
    e.step(e.spec.lose_slot(0))
    assert e.terminated, "the assassinate then finishes the job"
    assert len(revealed(e, 1)) == 2, "both influences are gone"
    assert e.rewards.tolist() == [1.0, -1.0]


def test_assassinate_double_loss_guard_on_last_influence():
    """A final-influence bluff block dies from the challenge; no double pop."""
    e = make(
        hands={0: [A, C], 1: [C]},
        revealed={1: [M]},
        coins={0: 5, 1: 2},
    )
    e.step(e.spec.assassinate(1))
    e.step(e.spec.pass_)
    e.step(e.spec.block_contessa)  # last-influence bluff
    e.step(e.spec.challenge)
    # auto-resolved: the only card must be revealed, no exception, game over
    assert e.terminated
    assert len(e.players[1].hand) == 0
    assert len(revealed(e, 1)) == 2
    assert e.rewards.tolist() == [1.0, -1.0]


def test_third_party_challenge_then_target_block_window():
    e = make(
        4,
        hands={0: [A, C], 1: [D, M], 2: [T, M], 3: [C, D]},
        coins={0: 5, 1: 2, 2: 2, 3: 2},
    )
    e.step(e.spec.assassinate(2))
    assert e.phase == Phase.CHALLENGE_ACTION and e.current_player == 1
    e.step(e.spec.pass_)
    assert e.current_player == 2, "poll continues clockwise"
    e.step(e.spec.challenge)  # third party challenges the truthful Assassin
    assert e.current_player == 2 and e.phase == Phase.LOSE_INFLUENCE
    e.step(e.spec.lose_slot(0))
    assert e.phase == Phase.BLOCK_ACTION
    assert e.current_player == 2, "only the target may block an assassinate"
    assert set(legal(e)) == {e.spec.pass_, e.spec.block_contessa}


def test_third_party_challenger_dies_target_still_blocks():
    e = make(
        4,
        hands={0: [A, C], 1: [D], 2: [T, M], 3: [C, D]},
        revealed={1: [M]},
        coins={0: 5, 1: 2, 2: 2, 3: 2},
    )
    e.step(e.spec.assassinate(2))
    e.step(e.spec.challenge)  # seat 1 challenges and loses its only influence
    assert not e.terminated
    assert len(e.players[1].hand) == 0
    assert e.phase == Phase.BLOCK_ACTION
    assert e.current_player == 2, "the target keeps its Contessa window"


def test_target_challenger_dies_turn_ends_cleanly():
    e = make(
        3,
        hands={0: [A, C], 1: [D, M], 2: [C]},
        revealed={2: [M]},
        coins={0: 5, 1: 2, 2: 2},
    )
    e.step(e.spec.assassinate(2))
    e.step(e.spec.pass_)  # seat 1 declines
    e.step(e.spec.challenge)  # the target itself challenges and dies
    assert not e.terminated
    assert len(e.players[2].hand) == 0
    assert e.phase == Phase.CHOOSE_ACTION
    assert e.current_player == 1, "turn passes on; no block window for a dead target"


# --------------------------------------------------------------------------- #
# Coup
# --------------------------------------------------------------------------- #
def test_coup_costs_seven_and_cannot_be_blocked():
    e = make(hands={0: [A, C], 1: [T, M]}, coins={0: 7, 1: 2})
    e.step(e.spec.coup(1))
    assert coins(e) == [0, 2]
    assert e.phase == Phase.LOSE_INFLUENCE
    assert e.current_player == 1
    e.step(e.spec.lose_slot(0))
    assert len(e.players[1].hand) == 1
    assert e.current_player == 1
    assert e.phase == Phase.CHOOSE_ACTION


# --------------------------------------------------------------------------- #
# Steal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target_coins, stolen", [(0, 0), (1, 1), (2, 2), (5, 2)])
def test_steal_transfers_min_two(target_coins, stolen):
    e = make(hands={0: [C, D], 1: [T, M]}, coins={0: 2, 1: target_coins})
    e.step(e.spec.steal(1))
    assert e.phase == Phase.CHALLENGE_ACTION
    e.step(e.spec.pass_)
    assert e.phase == Phase.BLOCK_ACTION
    e.step(e.spec.pass_)
    assert coins(e)[0] == 2 + stolen
    assert coins(e)[1] == target_coins - stolen
    assert all(c >= 0 for c in coins(e))


@pytest.mark.parametrize("block_role", [C, M])
def test_steal_blocked_by_captain_or_ambassador(block_role):
    other = T if block_role == C else C
    e = make(hands={0: [C, D], 1: [block_role, other]}, coins={0: 2, 1: 3})
    e.step(e.spec.steal(1))
    e.step(e.spec.pass_)
    assert set(legal(e)) == {
        e.spec.pass_,
        e.spec.block_captain,
        e.spec.block_ambassador,
    }
    e.step(e.spec.block_action(block_role))
    assert e.phase == Phase.CHALLENGE_BLOCK
    e.step(e.spec.pass_)
    assert coins(e) == [2, 3], "standing block: nothing moves"
    assert e.phase == Phase.CHOOSE_ACTION
    assert e.current_player == 1


def test_steal_block_challenge_checks_the_right_role():
    # blocker really has the Captain -> challenger loses, nothing moves
    e = make(hands={0: [C, D], 1: [C, M]}, coins={0: 2, 1: 3})
    e.step(e.spec.steal(1))
    e.step(e.spec.pass_)
    e.step(e.spec.block_captain)
    e.step(e.spec.challenge)
    assert e.phase == Phase.LOSE_INFLUENCE and e.current_player == 0
    e.step(e.spec.lose_slot(0))
    assert coins(e) == [2, 3]

    # blocker bluffs the Captain -> caught, the steal still resolves
    e = make(hands={0: [C, D], 1: [T, M]}, coins={0: 2, 1: 3})
    e.step(e.spec.steal(1))
    e.step(e.spec.pass_)
    e.step(e.spec.block_captain)
    e.step(e.spec.challenge)
    e.step(e.spec.lose_slot(0))
    assert coins(e) == [4, 1], "the steal goes through after the failed block"
    assert len(e.players[1].hand) == 1


def test_steal_ambassador_block_bluff_caught():
    e = make(hands={0: [C, D], 1: [T, C]}, coins={0: 2, 1: 3})
    e.step(e.spec.steal(1))
    e.step(e.spec.pass_)
    e.step(e.spec.block_ambassador)  # seat 1 has no Ambassador
    e.step(e.spec.challenge)
    e.step(e.spec.lose_slot(0))
    assert coins(e) == [4, 1]


# --------------------------------------------------------------------------- #
# Exchange (Ambassador)
# --------------------------------------------------------------------------- #
def test_exchange_two_influences():
    e = make(hands={0: [M, D], 1: [T, C]}, coins={0: 2, 1: 2})
    force_draw(e, [C, T])
    deck_before = len(e.deck)
    e.step(e.spec.exchange)
    assert e.phase == Phase.CHALLENGE_ACTION
    e.step(e.spec.pass_)
    assert e.phase == Phase.EXCHANGE_SELECT
    assert e.current_player == 0
    assert len(e._exchange_buffer) == 4
    assert e._exchange_buffer == sorted([M, D, C, T])
    assert len(e.deck) == deck_before - 2
    assert e.total_cards() == DECK_SIZE

    options = [e.spec.decode(a).index for a in legal(e)]
    assert options, "at least one keep option must be legal"
    kept = [e._exchange_buffer[k] for k in EXCHANGE_KEEP_OPTIONS[options[0]]]
    e.step(e.spec.exchange_keep(options[0]))
    assert hand(e, 0) == sorted(kept)
    assert len(hand(e, 0)) == 2
    assert len(e.deck) == deck_before
    assert e.total_cards() == DECK_SIZE
    assert e.current_player == 1
    assert e.phase == Phase.CHOOSE_ACTION


def test_exchange_masks_duplicate_multisets():
    # buffer becomes [D, D, C, A]: pairs (0,1)=DD (0,2)=DC (0,3)=DA
    # (1,2)=DC dup (1,3)=DA dup (2,3)=CA -> legal indices 0, 1, 2, 5
    e = make(hands={0: [D, C], 1: [T, M]}, coins={0: 2, 1: 2})
    force_draw(e, [D, A])
    e.step(e.spec.exchange)
    e.step(e.spec.pass_)
    assert e._exchange_buffer == sorted([A, C, D, D])
    assert [e.spec.decode(a).index for a in legal(e)] == [0, 1, 2, 5]


def test_exchange_one_influence_offers_singles():
    e = make(
        hands={0: [M], 1: [T, C]},
        revealed={0: [D]},
        coins={0: 2, 1: 2},
    )
    force_draw(e, [A, C])
    e.step(e.spec.exchange)
    e.step(e.spec.pass_)
    assert e.phase == Phase.EXCHANGE_SELECT
    assert len(e._exchange_buffer) == 3
    options = [e.spec.decode(a).index for a in legal(e)]
    assert all(index >= 6 for index in options), "1-influence uses the 3 singles"
    e.step(e.spec.exchange_keep(options[0]))
    assert len(hand(e, 0)) == 1
    assert e.total_cards() == DECK_SIZE


def test_exchange_auto_resolves_when_no_choice_exists():
    e = make(
        hands={0: [D], 1: [T, C]},
        revealed={0: [M]},
        coins={0: 2, 1: 2},
    )
    force_draw(e, [D, D])
    deck_before = len(e.deck)
    e.step(e.spec.exchange)
    e.step(e.spec.pass_)
    # [D, D, D] -> every keep-1 subset is the same multiset -> auto-resolved
    assert e.phase == Phase.CHOOSE_ACTION
    assert e.current_player == 1
    assert hand(e, 0) == [D]
    assert len(e.deck) == deck_before
    assert e.total_cards() == DECK_SIZE


# --------------------------------------------------------------------------- #
# Influence loss
# --------------------------------------------------------------------------- #
def test_auto_resolve_single_card_left():
    e = make(hands={0: [A, C], 1: [T]}, revealed={1: [M]}, coins={0: 7, 1: 2})
    e.step(e.spec.coup(1))
    assert e.terminated, "single-card defender dies without a choice"
    assert len(revealed(e, 1)) == 2


def test_auto_resolve_identical_cards():
    e = make(hands={0: [A, C], 1: [C, C]}, coins={0: 7, 1: 2})
    e.step(e.spec.coup(1))
    # len(set(hand)) == 1 -> no real choice, so the engine resolves it itself
    assert e.phase == Phase.CHOOSE_ACTION
    assert len(e.players[1].hand) == 1
    assert revealed(e, 1) == [C]
    assert e.current_player == 1


def test_lose_influence_choice_lists_both_slots():
    e = make(hands={0: [A, C], 1: [T, M]}, coins={0: 7, 1: 2})
    e.step(e.spec.coup(1))
    assert set(legal(e)) == {e.spec.lose_slot(0), e.spec.lose_slot(1)}
    e.step(e.spec.lose_slot(1))
    assert revealed(e, 1) == [T], "slot 1 is the higher role id"
    assert hand(e, 1) == [M]


# --------------------------------------------------------------------------- #
# Rewards and termination
# --------------------------------------------------------------------------- #
def test_two_player_win_rewards():
    e = make(hands={0: [A, C], 1: [T]}, revealed={1: [M]}, coins={0: 7, 1: 2})
    assert e._reward_kind == "win"
    e.step(e.spec.coup(1))  # the last influence goes, game over
    assert e.terminated and e.done
    assert e.rewards.tolist() == [1.0, -1.0]
    assert e.winner() == 0


def _near_death_layout(n: int) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Seat 0 keeps two influences, every other seat is down to one card."""
    cards = [C, C, C, M, M, M, A, A, A, T, T, T, D]  # 13 cards left after [D, D]
    hands: dict[int, list[int]] = {0: [D, D]}
    revealed: dict[int, list[int]] = {}
    pool = iter(cards)
    for seat in range(1, n):
        hands[seat] = [next(pool)]
    for seat in range(1, n):
        revealed[seat] = [next(pool)]
    return hands, revealed


def _kill_everyone_from_seat_zero(e: CoupEngine, n: int) -> None:
    """Seat 0 coups the other seats in order (so deaths are 1, 2, ... n-1)."""
    for _ in range(n - 1):
        if e.done:
            return
        guard = 0
        while e.current_player != 0 and not e.done:
            for other in range(1, n):
                e.players[other].coins = 0  # keep them on the income action
            e.step(e.spec.income)
            guard += 1
            assert guard < 3 * n
        if e.done:
            return
        e.players[0].coins = 10  # mandatory-coup territory
        coups = [a for a in legal(e) if e.spec.decode(a).kind == "COUP"]
        assert coups
        e.step(coups[0])
        while e.phase == Phase.LOSE_INFLUENCE:
            e.step(e.spec.lose_slot(0))


@pytest.mark.parametrize(
    "n,expected",
    [
        # deaths happen 1 -> n-1, so places run 1 -> n-1 for seats 1..n-1
        (3, [1.0, -1.0, 0.0]),
        (4, [1.0, -1.0, -1 / 3, 1 / 3]),
        (6, [1.0, -1.0, -0.6, -0.2, 0.2, 0.6]),
    ],
)
def test_rank_rewards_are_zero_sum_and_spaced(n, expected):
    e = CoupEngine(n, reward_mode="rank")
    e.reset(seed=0, randomize_start=False)
    assert e._reward_kind == "rank"
    hands, revealed = _near_death_layout(n)
    e.debug_set_hands(hands, revealed=revealed, coins={0: 10})
    assert e.total_cards() == DECK_SIZE
    _kill_everyone_from_seat_zero(e, n)
    assert e.terminated
    assert np.allclose(e.rewards, expected, atol=1e-9)
    assert abs(float(e.rewards.sum())) < 1e-9


def test_explicit_win_mode_for_many_players():
    e = CoupEngine(4, reward_mode="win")
    e.reset(seed=0, randomize_start=False)
    hands, revealed = _near_death_layout(4)
    e.debug_set_hands(hands, revealed=revealed, coins={0: 10})
    _kill_everyone_from_seat_zero(e, 4)
    assert e.terminated
    assert np.allclose(e.rewards, [1.0, -1 / 3, -1 / 3, -1 / 3])
    assert abs(float(e.rewards.sum())) < 1e-9


def test_truncation_at_max_turns():
    e = CoupEngine(2, max_turns=2)
    e.reset(seed=0, randomize_start=False)
    e.step(e.spec.income)
    assert not e.done
    e.step(e.spec.income)
    assert e.truncated and not e.terminated
    assert e.phase == Phase.GAME_OVER
    assert e.rewards.tolist() == [0.0, 0.0]
    assert abs(float(e.rewards.sum())) < 1e-9
    with pytest.raises(RuntimeError):
        e.step(e.spec.income)


# --------------------------------------------------------------------------- #
# Fuzzing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_fuzz_thousand_games(n):
    """1000 random games per player count: no crash, invariants hold."""
    rng = np.random.default_rng(1000 + n)
    e = CoupEngine(n, max_turns=250)
    truncations = 0
    for game in range(1000):
        e.reset(seed=game * 7 + n, randomize_start=bool(game % 2))
        assert e.total_cards() == DECK_SIZE
        steps = 0
        while not e.done:
            mask = e.legal_mask()
            assert mask.any(), f"empty mask in {Phase(e.phase).name}"
            assert mask.sum() > 0
            action = int(rng.choice(np.flatnonzero(mask)))
            e.step(action)
            steps += 1
            assert steps < 4000, "game did not terminate"
            assert e.total_cards() == DECK_SIZE, "deck conservation violated"
            assert all(p.coins >= 0 for p in e.players), "negative coins"
            assert all(len(p.hand) <= 2 for p in e.players)
            assert all(len(p.revealed) <= 2 for p in e.players)
            assert len(e.alive_seats()) >= 1
        assert abs(float(e.rewards.sum())) < 1e-9, "terminal rewards must be zero-sum"
        if e.truncated:
            truncations += 1
            assert e.rewards.tolist() == [0.0] * n
        else:
            assert e.terminated
            assert e.winner() is not None
            assert e.phase == Phase.GAME_OVER
    assert truncations < 1000


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", PLAYER_COUNTS)
def test_obs_shape_and_range(n):
    e = make(n)
    layout = make_obs_layout(n)
    assert e.obs_dim == layout.dim == obs_dim(n) == 131 + 21 * n
    for seat in range(n):
        obs = e.get_obs(seat)
        assert obs.shape == (e.obs_dim,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert not np.isnan(obs).any()


def test_obs_seat_blocks_rotate_with_perspective():
    e = make(3, hands={0: [D, C], 1: [T, M], 2: [A, C]}, coins={0: 4, 1: 1, 2: 7})
    layout = make_obs_layout(3)
    obs0 = e.get_obs(0)
    obs1 = e.get_obs(1)
    for rel in range(3):
        seat = rel  # viewer 0
        other_rel = (seat - 1) % 3  # same seat seen from viewer 1
        a = obs0[layout.seat_offset(rel) : layout.seat_offset(rel) + layout.seat_stride]
        b = obs1[
            layout.seat_offset(other_rel) : layout.seat_offset(other_rel)
            + layout.seat_stride
        ]
        assert np.allclose(a, b), f"seat {seat} block differs between perspectives"
    # public globals are identical from every seat
    for key in ("total_revealed", "num_alive", "turn", "deck_size"):
        off = getattr(layout, key)
        assert obs0[off] == obs1[off], key
    assert np.allclose(
        obs0[layout.total_revealed : layout.total_revealed + 5],
        obs1[layout.total_revealed : layout.total_revealed + 5],
    )
    # `known_total` is private: own hand + revealed, per viewer
    own0 = obs0[layout.hand_counts : layout.hand_counts + 5]
    own1 = obs1[layout.hand_counts : layout.hand_counts + 5]
    revealed_cnt = np.rint(
        obs0[layout.total_revealed : layout.total_revealed + 5] * 3
    ).astype(int)
    for obs, own in ((obs0, own0), (obs1, own1)):
        expected = np.minimum(np.rint(own * 2).astype(int) + revealed_cnt, 3) / 3.0
        assert np.allclose(obs[layout.known_total : layout.known_total + 5], expected)


def test_obs_private_hand_is_only_visible_to_its_owner():
    e = make(2, hands={0: [D, C], 1: [T, M]}, coins={0: 2, 1: 2})
    layout = make_obs_layout(2)
    obs0, obs1 = e.get_obs(0), e.get_obs(1)
    own0 = obs0[layout.hand_counts : layout.hand_counts + 5]
    own1 = obs1[layout.hand_counts : layout.hand_counts + 5]
    assert np.allclose(own0, [1 / 2, 0, 1 / 2, 0, 0])  # Duke + Captain
    assert np.allclose(own1, [0, 0, 0, 1 / 2, 1 / 2])  # Ambassador + Contessa


def test_obs_does_not_leak_hidden_cards():
    e = make(3, hands={0: [D, C], 1: [T, M], 2: [A, C]}, coins={0: 3, 1: 4, 2: 2})
    before = e.get_obs(0)
    # swap every hidden opponent card for a different (legal) card
    e.debug_set_hands({0: [D, C], 1: [A, A], 2: [M, M]}, coins={0: 3, 1: 4, 2: 2})
    after = e.get_obs(0)
    assert np.array_equal(before, after), "opponent hands must not influence the obs"

    # shuffling the deck alone must not change anything either
    e.rng.shuffle(e.deck)
    assert np.array_equal(before, e.get_obs(0))

    # ...but revealing a card must be visible
    e.players[1].revealed = [A]
    e.players[1].hand = [A]
    changed = e.get_obs(0)
    assert not np.array_equal(before, changed)


def test_obs_marks_impossible_claims():
    layout = make_obs_layout(2)
    # seat 1 holds one Duke and has revealed two: the claim is provably a bluff
    e = make(2, hands={0: [T, M], 1: [D, M]}, revealed={1: [D, D]}, coins={0: 2, 1: 2})
    e.step(e.spec.tax)
    assert e._role_exhausted_for(1, D)
    assert e.get_obs(1)[layout.claimed_impossible] == 1.0

    # nothing is exhausted here, so the flag stays clear
    e = make(2, hands={0: [D, C], 1: [T, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.tax)
    assert e.get_obs(1)[layout.claimed_impossible] == 0.0


def test_obs_pending_features_follow_the_phase():
    e = make(2, hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    layout = make_obs_layout(2)
    e.step(e.spec.tax)
    obs = e.get_obs(1)
    assert obs[layout.phase + int(Phase.CHALLENGE_ACTION)] == 1.0
    assert obs[layout.pending_actor + 1] == 1.0, "actor is one seat clockwise"
    assert obs[layout.pending_action + int(ActionType.TAX)] == 1.0


def test_obs_exchange_buffer_is_private():
    e = make(2, hands={0: [M, D], 1: [T, C]}, coins={0: 2, 1: 2})
    e.step(e.spec.exchange)
    e.step(e.spec.pass_)
    layout = make_obs_layout(2)
    mine = e.get_obs(0)
    theirs = e.get_obs(1)
    assert mine[layout.in_exchange] == 1.0
    assert theirs[layout.in_exchange] == 0.0
    assert mine[layout.buffer_counts : layout.buffer_counts + 5].sum() > 0
    assert theirs[layout.buffer_counts : layout.buffer_counts + 5].sum() == 0


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
def test_episode_stats_structure():
    e = make(2, hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.income)
    stats = e.get_episode_stats()
    assert stats["num_players"] == 2
    assert set(stats["players"]) == {0, 1}
    for key in (
        "action_counts",
        "claims",
        "bluffs",
        "challenges_attempted",
        "challenges_won",
        "challenge_opportunities",
        "impossible_claim_opportunities",
        "challenges_on_impossible_claim",
        "blocks_attempted",
        "block_bluffs",
        "claims_proven",
        "caught_bluffing",
        "influence_lost",
        "coins_stolen",
        "bluff_rate",
    ):
        assert key in stats["players"][0], key
    assert stats["players"][0]["action_counts"]["INCOME"] == 1
    assert stats["players"][0]["bluff_rate"] == 0.0


def test_telemetry_tracks_truthful_claim_and_challenge():
    e = make(2, hands={0: [D, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.tax)
    e.step(e.spec.challenge)
    e.step(e.spec.lose_slot(0))
    stats = e.get_episode_stats()
    p0, p1 = stats["players"][0], stats["players"][1]
    assert p0["claims"]["tax_duke"] == 1
    assert p0["bluffs"]["tax_duke"] == 0
    assert p0["claims_proven"] == 1
    assert p1["challenges_attempted"] == 1
    assert p1["challenges_won"] == 0
    assert p1["challenge_opportunities"] == 1
    assert p0["bluff_rate"] == 0.0


def test_telemetry_tracks_bluff_and_caught_bluffing():
    e = make(2, hands={0: [T, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.tax)
    e.step(e.spec.challenge)
    e.step(e.spec.lose_slot(0))
    stats = e.get_episode_stats()
    p0, p1 = stats["players"][0], stats["players"][1]
    assert p0["claims"]["tax_duke"] == 1
    assert p0["bluffs"]["tax_duke"] == 1
    assert p0["caught_bluffing"] == 1
    assert p0["bluff_rate"] == 1.0
    assert p1["challenges_won"] == 1


def test_telemetry_tracks_blocks_and_impossible_claims():
    # seat 1 holds one Duke and has revealed two: a Duke claim is provably fake
    e = make(2, hands={0: [T, M], 1: [D, M]}, revealed={1: [D, D]}, coins={0: 2, 1: 2})
    assert e._role_exhausted_for(1, D)
    e.step(e.spec.tax)
    stats = e.get_episode_stats()
    assert stats["players"][1]["challenge_opportunities"] == 1
    assert stats["players"][1]["impossible_claim_opportunities"] == 1
    assert stats["players"][1]["challenges_attempted"] == 0
    e.step(e.spec.challenge)
    e.step(e.spec.lose_slot(0))
    stats = e.get_episode_stats()
    assert stats["players"][1]["challenges_on_impossible_claim"] == 1
    assert stats["players"][1]["challenges_won"] == 1
    assert stats["players"][0]["caught_bluffing"] == 1

    # blocks and block bluffs
    e = make(2, hands={0: [T, C], 1: [D, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.foreign_aid)
    e.step(e.spec.block_duke)
    stats = e.get_episode_stats()
    assert stats["players"][1]["blocks_attempted"] == 1
    assert stats["players"][1]["block_bluffs"] == 0
    assert stats["players"][1]["claims"]["block_duke"] == 1
    assert stats["players"][1]["bluffs"]["block_duke"] == 0

    e = make(2, hands={0: [T, C], 1: [A, M]}, coins={0: 2, 1: 2})
    e.step(e.spec.foreign_aid)
    e.step(e.spec.block_duke)  # seat 1 has no Duke
    stats = e.get_episode_stats()
    assert stats["players"][1]["block_bluffs"] == 1
    assert stats["players"][1]["bluffs"]["block_duke"] == 1


def test_telemetry_across_a_full_random_game():
    rng = np.random.default_rng(5)
    e = CoupEngine(4, max_turns=200)
    e.reset(seed=11, randomize_start=True)
    while not e.done:
        e.step(int(rng.choice(np.flatnonzero(e.legal_mask()))))
    stats = e.get_episode_stats()
    total_actions = sum(
        sum(p["action_counts"].values()) for p in stats["players"].values()
    )
    assert total_actions > 0
    assert stats["turns"] == e.turn
    assert set(stats["elim_order"]) == {
        s for s in range(4) if not e.players[s].alive
    }
    for p in stats["players"].values():
        assert 0.0 <= p["bluff_rate"] <= 1.0
        assert p["challenges_won"] <= p["challenges_attempted"]
        claimed = sum(p["claims"].values())
        assert sum(p["bluffs"].values()) <= claimed
