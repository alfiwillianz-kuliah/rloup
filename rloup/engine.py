"""Pure-Python Coup engine implementing the official base-game rules.

The engine owns all game state and exposes:

``reset`` / ``step`` / ``legal_mask`` / ``current_player``
    A minimal, fast decision interface.  ``step`` always applies to
    ``current_player`` and never returns anything; outcome information is read
    from ``terminated`` / ``truncated`` / ``rewards``.
``get_obs(player_id)``
    Ego-centric ``float32`` observation built only from information that
    ``player_id`` is allowed to see.
``get_episode_stats()``
    Per-episode behavioural telemetry used for analysis and evaluation.

Seats are integers ``0..N-1``.  "Clockwise" means increasing seat index modulo
``N``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .constants import (
    ACTION_CLAIM_KIND,
    ACTION_NAMES,
    ASSASSINATE_BLOCK_ROLES,
    ASSASSINATE_COST,
    BLOCK_ROLE_CLAIM_KIND,
    CLAIMED_ROLE,
    CLAIM_KINDS,
    COPIES_PER_ROLE,
    COUP_COST,
    DECK_SIZE,
    EXCHANGE_DRAW,
    EXCHANGE_KEEP_OPTIONS,
    EXCHANGE_KEEP_PAIRS,
    EXCHANGE_KEEP_SINGLES,
    FOREIGN_AID_BLOCK_ROLES,
    KIND_ASSASSINATE,
    KIND_BLOCK,
    KIND_CHALLENGE,
    KIND_COUP,
    KIND_EXCHANGE,
    KIND_EXCHANGE_KEEP,
    KIND_FOREIGN_AID,
    KIND_INCOME,
    KIND_LOSE_SLOT,
    KIND_PASS,
    KIND_STEAL,
    KIND_TAX,
    MANDATORY_COUP_COINS,
    NUM_DECISION_PHASES,
    NUM_ROLES,
    ActionSpaceSpec,
    ActionType,
    Phase,
    Role,
    STEAL_AMOUNT,
    STEAL_BLOCK_ROLES,
    TAX_AMOUNT,
)

# --------------------------------------------------------------------------- #
# Observation layout
# --------------------------------------------------------------------------- #

NUM_EVENTS = 8
EVENT_STRIDE = 8
NUM_EVENT_KINDS = 6
CLAIM_FEATURES = 7

# Event kinds stored in the sliding public-event window.
EV_ACTION = 0
EV_CHALLENGE_ACTION = 1
EV_BLOCK = 2
EV_CHALLENGE_BLOCK = 3
EV_PROVEN = 4
EV_INFLUENCE_LOST = 5

# Resume tags used by the influence-loss state machine.
RESUME_NEXT_TURN = "next_turn"
RESUME_AFTER_ACTION_CHALLENGER_LOSS = "after_action_challenger_loss"
RESUME_AFTER_BLOCK_CHALLENGER_LOSS = "after_block_challenger_loss"
RESUME_AFTER_BLOCKER_LOSS = "after_blocker_loss"


@dataclass(frozen=True)
class ObsLayout:
    """Offsets inside the flat ego-centric observation vector."""

    num_players: int
    hand_counts: int
    buffer_counts: int
    buffer_slots: int
    in_exchange: int
    seat_features: int
    seat_stride: int
    total_revealed: int
    known_total: int
    num_alive: int
    turn: int
    deck_size: int
    claimed_impossible: int
    phase: int
    pending_actor: int
    pending_target: int
    pending_action: int
    pending_blocker: int
    pending_block_role: int
    claims: int
    claims_stride: int
    events: int
    dim: int

    def seat_offset(self, relative_seat: int) -> int:
        return self.seat_features + self.seat_stride * relative_seat

    def claims_offset(self, relative_seat: int) -> int:
        return self.claims + self.claims_stride * relative_seat

    def event_offset(self, index: int) -> int:
        return self.events + EVENT_STRIDE * index


def make_obs_layout(num_players: int) -> ObsLayout:
    n = num_players
    hand_counts = 0
    buffer_counts = hand_counts + NUM_ROLES
    buffer_slots = buffer_counts + NUM_ROLES
    in_exchange = buffer_slots + 4 * NUM_ROLES
    seat_features = in_exchange + 1
    stride = 6 + NUM_ROLES  # alive, inf, coins, 3 flags, revealed x5
    total_revealed = seat_features + stride * n
    known_total = total_revealed + NUM_ROLES
    num_alive = known_total + NUM_ROLES
    turn = num_alive + 1
    deck_size = turn + 1
    claimed_impossible = deck_size + 1
    phase = claimed_impossible + 1
    pending_actor = phase + NUM_DECISION_PHASES
    pending_target = pending_actor + (n + 1)
    pending_action = pending_target + (n + 1)
    pending_blocker = pending_action + 8
    pending_block_role = pending_blocker + (n + 1)
    claims = pending_block_role + NUM_ROLES
    events = claims + CLAIM_FEATURES * n
    dim = events + NUM_EVENTS * EVENT_STRIDE
    return ObsLayout(
        num_players=n,
        hand_counts=hand_counts,
        buffer_counts=buffer_counts,
        buffer_slots=buffer_slots,
        in_exchange=in_exchange,
        seat_features=seat_features,
        seat_stride=stride,
        total_revealed=total_revealed,
        known_total=known_total,
        num_alive=num_alive,
        turn=turn,
        deck_size=deck_size,
        claimed_impossible=claimed_impossible,
        phase=phase,
        pending_actor=pending_actor,
        pending_target=pending_target,
        pending_action=pending_action,
        pending_blocker=pending_blocker,
        pending_block_role=pending_block_role,
        claims=claims,
        claims_stride=CLAIM_FEATURES,
        events=events,
        dim=dim,
    )


def obs_dim(num_players: int) -> int:
    """Flat observation size: ``131 + 21 * num_players``."""
    return make_obs_layout(num_players).dim


# --------------------------------------------------------------------------- #
# Player state
# --------------------------------------------------------------------------- #


@dataclass
class PlayerState:
    """Per-seat tableaus."""

    hand: list[int] = field(default_factory=list)
    coins: int = 2
    revealed: list[int] = field(default_factory=list)

    @property
    def influence(self) -> int:
        return len(self.hand)

    @property
    def alive(self) -> bool:
        return bool(self.hand)


def _counts(cards: Iterable[int], size: int = NUM_ROLES) -> np.ndarray:
    out = np.zeros(size, dtype=np.int64)
    for c in cards:
        out[int(c)] += 1
    return out


def _new_stats() -> dict[str, Any]:
    return {
        "action_counts": {name: 0 for name in ACTION_NAMES}
        | {"PASS": 0, "CHALLENGE": 0, "BLOCK": 0, "LOSE_SLOT": 0, "EXCHANGE_KEEP": 0},
        "claims": {k: 0 for k in CLAIM_KINDS},
        "bluffs": {k: 0 for k in CLAIM_KINDS},
        "challenges_attempted": 0,
        "challenges_won": 0,
        "challenge_opportunities": 0,
        "impossible_claim_opportunities": 0,
        "challenges_on_impossible_claim": 0,
        "blocks_attempted": 0,
        "block_bluffs": 0,
        "claims_proven": 0,
        "caught_bluffing": 0,
        "influence_lost": 0,
        "coins_stolen": 0,
        "tags": [],
    }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class CoupEngine:
    """Full Coup rules engine for 2 to 6 players."""

    MIN_PLAYERS = 2
    MAX_PLAYERS = 6

    def __init__(
        self,
        num_players: int = 2,
        max_turns: int = 200,
        two_player_coin_rule: bool = True,
        reward_mode: str = "auto",
        seed: int | None = None,
    ) -> None:
        if not self.MIN_PLAYERS <= num_players <= self.MAX_PLAYERS:
            raise ValueError(
                f"num_players must be in {self.MIN_PLAYERS}..{self.MAX_PLAYERS}, "
                f"got {num_players}"
            )
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if reward_mode not in ("auto", "win", "rank"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")

        self.num_players = int(num_players)
        self.max_turns = int(max_turns)
        self.two_player_coin_rule = bool(two_player_coin_rule)
        self.reward_mode = reward_mode
        self._reward_kind = self._resolve_reward_mode(reward_mode, num_players)

        self.spec = ActionSpaceSpec(self.num_players)
        self.n_actions = self.spec.n_actions
        self.obs_layout = make_obs_layout(self.num_players)
        self.obs_dim = self.obs_layout.dim

        self.players: list[PlayerState] = []
        self.deck: list[int] = []
        self.seed = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self.turn = 0
        self.phase = Phase.CHOOSE_ACTION
        self.current_player = 0
        self.pending_actor: int | None = None
        self.pending_action: ActionType | None = None
        self.pending_target: int | None = None
        self.pending_blocker: int | None = None
        self.pending_block_role: Role | None = None

        self.terminated = False
        self.truncated = False
        self.rewards = np.zeros(self.num_players, dtype=np.float64)
        self._elim_order: list[int] = []

        self._poll_order: list[int] = []
        self._poll_idx = 0

        self._lose_seat: int | None = None
        self._lose_resume: str | None = None
        self._exchange_buffer: list[int] | None = None

        self._events: deque[tuple[int, int | None, int | None]] = deque(maxlen=NUM_EVENTS)
        self.stats: list[dict[str, Any]] = []
        self.claim_counts = np.zeros((self.num_players, CLAIM_FEATURES), dtype=np.int64)
        self._last_action: int | None = None

        self.reset(seed=None, randomize_start=True)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_reward_mode(reward_mode: str, num_players: int) -> str:
        if reward_mode == "auto":
            return "win" if num_players == 2 else "rank"
        return reward_mode

    def reset(self, seed: int | None = None, randomize_start: bool = True) -> int:
        """Reset to a fresh game and return the seat that acts first.

        ``seed=None`` keeps drawing from the current RNG stream (so consecutive
        resets differ); an explicit ``seed`` makes the deal reproducible.
        """
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.players = [PlayerState() for _ in range(self.num_players)]
        full_deck = [int(r) for r in Role for _ in range(COPIES_PER_ROLE)]
        self.deck = list(full_deck)
        self.rng.shuffle(self.deck)

        for seat in range(self.num_players):
            hand = [self.deck.pop() for _ in range(2)]
            self.players[seat].hand = sorted(hand)

        start = int(self.rng.integers(self.num_players)) if randomize_start else 0
        for seat in range(self.num_players):
            self.players[seat].coins = 2
        if self.num_players == 2 and self.two_player_coin_rule:
            other = 1 - start
            self.players[start].coins = 1
            self.players[other].coins = 2

        self.turn = 0
        self.phase = Phase.CHOOSE_ACTION
        self.current_player = start
        self.pending_actor = None
        self.pending_action = None
        self.pending_target = None
        self.pending_blocker = None
        self.pending_block_role = None
        self.terminated = False
        self.truncated = False
        self.rewards = np.zeros(self.num_players, dtype=np.float64)
        self._elim_order = []
        self._poll_order = []
        self._poll_idx = 0
        self._lose_seat = None
        self._lose_resume = None
        self._exchange_buffer = None
        self._events.clear()
        self.stats = [_new_stats() for _ in range(self.num_players)]
        self.claim_counts = np.zeros((self.num_players, CLAIM_FEATURES), dtype=np.int64)
        self._last_action = None
        return start

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def num_alive(self) -> int:
        return sum(1 for p in self.players if p.alive)

    def is_alive(self, seat: int) -> bool:
        return self.players[seat].alive

    def alive_seats(self) -> list[int]:
        return [s for s in range(self.num_players) if self.players[s].alive]

    def winner(self) -> int | None:
        alive = self.alive_seats()
        return alive[0] if len(alive) == 1 else None

    def deck_size(self) -> int:
        return len(self.deck)

    def total_cards(self) -> int:
        """Number of cards in play; always 15 for a legal game.

        While a player is selecting a ``EXCHANGE`` keep-set their hand lives in
        the exchange buffer, so the buffer is counted instead of that hand.
        """
        total = len(self.deck)
        for p in self.players:
            total += len(p.revealed)
        buffer = self._exchange_buffer
        if buffer is None:
            for p in self.players:
                total += len(p.hand)
            return total
        total += len(buffer)
        for seat, p in enumerate(self.players):
            if seat != self.current_player:
                total += len(p.hand)
        return total

    def revealed_counts(self) -> np.ndarray:
        return _counts(c for p in self.players for c in p.revealed)

    def hand_counts(self, seat: int) -> np.ndarray:
        return _counts(self.players[seat].hand)

    def _rel(self, seat: int, viewer: int) -> int:
        return (seat - viewer) % self.num_players

    def _abs(self, viewer: int, rel: int) -> int:
        return (viewer + rel) % self.num_players

    def _poll_order_after(self, seat: int) -> list[int]:
        n = self.num_players
        return [self._abs(seat, k) for k in range(1, n) if self.is_alive(self._abs(seat, k))]

    def _next_alive_after(self, seat: int) -> int:
        order = self._poll_order_after(seat)
        return order[0] if order else seat

    def _target_rel_offsets(self, seat: int) -> list[int]:
        n = self.num_players
        return [k for k in range(1, n) if self.is_alive(self._abs(seat, k))]

    def _role_exhausted_for(self, viewer: int, role: Role) -> bool:
        """True when ``viewer`` can prove nobody holds ``role`` any more."""
        own = int(self.hand_counts(viewer)[int(role)])
        revealed = int(self.revealed_counts()[int(role)])
        return own + revealed >= COPIES_PER_ROLE

    def _push_event(self, kind: int, actor: int | None, target: int | None) -> None:
        self._events.append((int(kind), actor, target))

    def _prove_role(self, seat: int, role: Role) -> None:
        """Return the proven card to the deck, shuffle, draw a replacement."""
        hand = self.players[seat].hand
        idx = hand.index(int(role))
        hand.pop(idx)
        self.deck.append(int(role))
        self.rng.shuffle(self.deck)
        hand.append(self.deck.pop())
        hand.sort()

    def _record_claim(self, seat: int, claim_kind: str | None, role: Role | None) -> None:
        if claim_kind is None:
            return
        st = self.stats[seat]
        st["claims"][claim_kind] += 1
        held = role is not None and int(role) in self.players[seat].hand
        if not held:
            st["bluffs"][claim_kind] += 1
        if role is not None:
            self.claim_counts[seat, int(role)] += 1

    # ------------------------------------------------------------------ #
    # Legal actions
    # ------------------------------------------------------------------ #
    def legal_mask(self) -> np.ndarray:
        """``int8`` mask of legal action ids for ``current_player``."""
        mask = np.zeros(self.n_actions, dtype=np.int8)
        phase = self.phase
        if phase == Phase.CHOOSE_ACTION:
            self._fill_action_mask(mask, self.current_player)
        elif phase == Phase.CHALLENGE_ACTION:
            mask[self.spec.pass_] = 1
            mask[self.spec.challenge] = 1
        elif phase == Phase.BLOCK_ACTION:
            mask[self.spec.pass_] = 1
            kind = self.pending_action
            if kind == ActionType.FOREIGN_AID:
                roles: Sequence[Role] = FOREIGN_AID_BLOCK_ROLES
            elif kind == ActionType.ASSASSINATE:
                roles = ASSASSINATE_BLOCK_ROLES
            elif kind == ActionType.STEAL:
                roles = STEAL_BLOCK_ROLES
            else:  # pragma: no cover - defensive
                roles = ()
            for role in roles:
                mask[self.spec.block_action(role)] = 1
        elif phase == Phase.CHALLENGE_BLOCK:
            mask[self.spec.pass_] = 1
            mask[self.spec.challenge] = 1
        elif phase == Phase.LOSE_INFLUENCE:
            hand = self.players[self.current_player].hand
            for slot in range(len(hand)):
                mask[self.spec.lose_slot(slot)] = 1
            if not mask.any():  # pragma: no cover - defensive
                mask[self.spec.lose_slot(0)] = 1
        elif phase == Phase.EXCHANGE_SELECT:
            for index in self._exchange_canonical_indices():
                mask[self.spec.exchange_keep(index)] = 1
        else:  # GAME_OVER: keep the mask non-empty for safety
            mask[self.spec.pass_] = 1
        return mask

    def _fill_action_mask(self, mask: np.ndarray, seat: int) -> None:
        spec = self.spec
        coins = self.players[seat].coins
        rels = self._target_rel_offsets(seat)
        if coins >= MANDATORY_COUP_COINS:
            for rel in rels:
                mask[spec.coup(rel)] = 1
            return
        mask[spec.income] = 1
        mask[spec.foreign_aid] = 1
        mask[spec.tax] = 1
        if len(self.deck) >= EXCHANGE_DRAW:
            mask[spec.exchange] = 1
        if coins >= COUP_COST:
            for rel in rels:
                mask[spec.coup(rel)] = 1
        if coins >= ASSASSINATE_COST:
            for rel in rels:
                mask[spec.assassinate(rel)] = 1
        for rel in rels:
            mask[spec.steal(rel)] = 1

    def legal_actions(self) -> np.ndarray:
        """Indices of legal actions."""
        return np.flatnonzero(self.legal_mask())

    def action_name(self, action_id: int) -> str:
        return self.spec.action_name(action_id)

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> None:
        """Apply ``action`` for ``current_player``."""
        if self.done:
            raise RuntimeError("step() called on a finished game")
        a = int(action)
        if not 0 <= a < self.n_actions:
            raise ValueError(f"action id {a} out of range 0..{self.n_actions - 1}")
        if not self.legal_mask()[a]:
            raise ValueError(
                f"illegal action {a} ({self.spec.action_name(a)}) "
                f"in phase {Phase(self.phase).name}"
            )
        self._last_action = a
        info = self.spec.decode(a)
        phase = self.phase
        if phase == Phase.CHOOSE_ACTION:
            self._do_action(info)
        elif phase == Phase.CHALLENGE_ACTION:
            self._action_challenge_step(info)
        elif phase == Phase.BLOCK_ACTION:
            self._block_step(info)
        elif phase == Phase.CHALLENGE_BLOCK:
            self._block_challenge_step(info)
        elif phase == Phase.LOSE_INFLUENCE:
            self._lose_step(info)
        elif phase == Phase.EXCHANGE_SELECT:
            self._exchange_step(info)
        else:  # pragma: no cover - defensive
            raise RuntimeError("step() in GAME_OVER phase")

    # -- main action ---------------------------------------------------- #
    def _do_action(self, info) -> None:
        actor = self.current_player
        kind = info.kind
        target: int | None = None
        if info.rel_off is not None:
            target = self._abs(actor, info.rel_off)
        self.pending_actor = actor
        self.stats[actor]["action_counts"][kind] += 1
        self._push_event(EV_ACTION, actor, target)

        if kind == KIND_INCOME:
            self.players[actor].coins += 1
            self._next_turn()
            return
        if kind == KIND_FOREIGN_AID:
            self.pending_action = ActionType.FOREIGN_AID
            self.pending_target = None
            self._start_block_phase(self._poll_order_after(actor))
            return
        if kind == KIND_COUP:
            self.pending_action = ActionType.COUP
            self.pending_target = target
            self.players[actor].coins -= COUP_COST
            self._lose_influence(target, RESUME_NEXT_TURN)
            return

        if kind == KIND_TAX:
            action_type = ActionType.TAX
        elif kind == KIND_ASSASSINATE:
            action_type = ActionType.ASSASSINATE
        elif kind == KIND_STEAL:
            action_type = ActionType.STEAL
        elif kind == KIND_EXCHANGE:
            action_type = ActionType.EXCHANGE
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unexpected action kind {kind}")

        self.pending_action = action_type
        self.pending_target = target
        if action_type == ActionType.ASSASSINATE:
            self.players[actor].coins -= ASSASSINATE_COST
        claim_role = CLAIMED_ROLE[action_type]
        self._record_claim(actor, ACTION_CLAIM_KIND[action_type], claim_role)
        voters = self._poll_order_after(actor)
        self._start_challenge_poll(voters, Phase.CHALLENGE_ACTION)

    def _register_challenge_opportunity(self, voter: int, role: Role | None) -> None:
        if role is None:  # pragma: no cover - defensive
            return
        st = self.stats[voter]
        st["challenge_opportunities"] += 1
        if self._role_exhausted_for(voter, role):
            st["impossible_claim_opportunities"] += 1

    # -- polls ---------------------------------------------------------- #
    def _start_challenge_poll(self, voters: list[int], phase: Phase) -> None:
        if not voters:
            self._on_poll_exhausted(phase)
            return
        self._poll_order = voters
        self._poll_idx = 0
        self.phase = phase
        self.current_player = voters[0]
        self._register_challenge_opportunity(voters[0], self._claimed_role())

    def _start_block_phase(self, voters: list[int]) -> None:
        if not voters:
            self._resolve_action_now()
            return
        self._poll_order = voters
        self._poll_idx = 0
        self.phase = Phase.BLOCK_ACTION
        self.current_player = voters[0]

    def _advance_poll(self) -> bool:
        self._poll_idx += 1
        if self._poll_idx >= len(self._poll_order):
            return False
        self.current_player = self._poll_order[self._poll_idx]
        if self.phase in (Phase.CHALLENGE_ACTION, Phase.CHALLENGE_BLOCK):
            self._register_challenge_opportunity(self.current_player, self._claimed_role())
        return True

    def _on_poll_exhausted(self, phase: Phase) -> None:
        if phase == Phase.CHALLENGE_ACTION:
            self._on_action_unchallenged()
        elif phase == Phase.CHALLENGE_BLOCK:
            self._next_turn()  # block stands
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unexpected poll phase {phase}")

    def _on_action_unchallenged(self) -> None:
        kind = self.pending_action
        if kind == ActionType.TAX:
            self.players[self.pending_actor].coins += TAX_AMOUNT
            self._next_turn()
        elif kind == ActionType.EXCHANGE:
            self._start_exchange(self.pending_actor)
        elif kind in (ActionType.ASSASSINATE, ActionType.STEAL):
            target = self.pending_target
            if target is not None and self.is_alive(target):
                self._start_block_phase([target])
            else:  # pragma: no cover - defensive
                self._next_turn()
        else:  # pragma: no cover - defensive
            self._resolve_action_now()

    # -- challenge on the action ---------------------------------------- #
    def _action_challenge_step(self, info) -> None:
        voter = self.current_player
        if info.kind == KIND_PASS:
            self.stats[voter]["action_counts"]["PASS"] += 1
            if not self._advance_poll():
                self._on_poll_exhausted(Phase.CHALLENGE_ACTION)
        elif info.kind == KIND_CHALLENGE:
            self.stats[voter]["action_counts"]["CHALLENGE"] += 1
            self._resolve_action_challenge(voter)
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unexpected action {info.kind} in challenge poll")

    def _resolve_action_challenge(self, challenger: int) -> None:
        actor = self.pending_actor
        action_type = self.pending_action
        role = CLAIMED_ROLE[action_type]
        st = self.stats[challenger]
        st["challenges_attempted"] += 1
        if self._role_exhausted_for(challenger, role):
            st["challenges_on_impossible_claim"] += 1
        self._push_event(EV_CHALLENGE_ACTION, challenger, actor)

        if int(role) in self.players[actor].hand:
            # Challenge fails: the claimant proves the role.
            self.stats[actor]["claims_proven"] += 1
            self.claim_counts[actor, CLAIM_FEATURES - 1] += 1
            self._push_event(EV_PROVEN, actor, challenger)
            self._prove_role(actor, role)
            self._lose_influence(challenger, RESUME_AFTER_ACTION_CHALLENGER_LOSS)
        else:
            # Challenge succeeds: bluff caught, action cancelled.
            st["challenges_won"] += 1
            self.stats[actor]["caught_bluffing"] += 1
            self.claim_counts[actor, CLAIM_FEATURES - 2] += 1
            self._push_event(EV_PROVEN, challenger, actor)
            if action_type == ActionType.ASSASSINATE:
                self.players[actor].coins += ASSASSINATE_COST  # refund
            self._lose_influence(actor, RESUME_NEXT_TURN)

    # -- block ---------------------------------------------------------- #
    def _block_step(self, info) -> None:
        voter = self.current_player
        if info.kind == KIND_PASS:
            self.stats[voter]["action_counts"]["PASS"] += 1
            if not self._advance_poll():
                self._resolve_action_now()
        elif info.kind == KIND_BLOCK:
            self.stats[voter]["action_counts"]["BLOCK"] += 1
            self._declare_block(voter, info.role)
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unexpected action {info.kind} in block poll")

    def _declare_block(self, blocker: int, role: Role) -> None:
        self.pending_blocker = blocker
        self.pending_block_role = role
        self.stats[blocker]["blocks_attempted"] += 1
        if int(role) not in self.players[blocker].hand:
            self.stats[blocker]["block_bluffs"] += 1
        self._record_claim(blocker, BLOCK_ROLE_CLAIM_KIND[role], role)
        self._push_event(EV_BLOCK, blocker, self.pending_actor)
        voters = self._poll_order_after(blocker)
        self._start_challenge_poll(voters, Phase.CHALLENGE_BLOCK)

    def _block_challenge_step(self, info) -> None:
        voter = self.current_player
        if info.kind == KIND_PASS:
            self.stats[voter]["action_counts"]["PASS"] += 1
            if not self._advance_poll():
                self._on_poll_exhausted(Phase.CHALLENGE_BLOCK)
        elif info.kind == KIND_CHALLENGE:
            self.stats[voter]["action_counts"]["CHALLENGE"] += 1
            self._resolve_block_challenge(voter)
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unexpected action {info.kind} in block challenge poll")

    def _resolve_block_challenge(self, challenger: int) -> None:
        blocker = self.pending_blocker
        role = self.pending_block_role
        st = self.stats[challenger]
        st["challenges_attempted"] += 1
        if self._role_exhausted_for(challenger, role):
            st["challenges_on_impossible_claim"] += 1
        self._push_event(EV_CHALLENGE_BLOCK, challenger, blocker)

        if int(role) in self.players[blocker].hand:
            # Block was truthful: blocker re-draws, challenger loses influence,
            # the block stands and the action is cancelled.
            self.stats[blocker]["claims_proven"] += 1
            self.claim_counts[blocker, CLAIM_FEATURES - 1] += 1
            self._push_event(EV_PROVEN, blocker, challenger)
            self._prove_role(blocker, role)
            self._lose_influence(challenger, RESUME_AFTER_BLOCK_CHALLENGER_LOSS)
        else:
            # Bluff caught: blocker loses influence, then the action resolves.
            st["challenges_won"] += 1
            self.stats[blocker]["caught_bluffing"] += 1
            self.claim_counts[blocker, CLAIM_FEATURES - 2] += 1
            self._push_event(EV_PROVEN, challenger, blocker)
            self._lose_influence(blocker, RESUME_AFTER_BLOCKER_LOSS)

    # -- action resolution ---------------------------------------------- #
    def _resolve_action_now(self) -> None:
        """The pending action takes effect (no block, or block defeated)."""
        kind = self.pending_action
        actor = self.pending_actor
        if kind == ActionType.FOREIGN_AID:
            self.players[actor].coins += 2
            self._next_turn()
        elif kind == ActionType.TAX:  # pragma: no cover - routed earlier
            self.players[actor].coins += TAX_AMOUNT
            self._next_turn()
        elif kind == ActionType.ASSASSINATE:
            target = self.pending_target
            if target is not None and self.is_alive(target):
                self._lose_influence(target, RESUME_NEXT_TURN)
            else:
                self._next_turn()
        elif kind == ActionType.STEAL:
            target = self.pending_target
            if target is not None and self.is_alive(target):
                amount = min(STEAL_AMOUNT, self.players[target].coins)
                self.players[target].coins -= amount
                self.players[actor].coins += amount
                self.stats[actor]["coins_stolen"] += amount
            self._next_turn()
        elif kind == ActionType.EXCHANGE:  # pragma: no cover - routed earlier
            self._start_exchange(actor)
        else:  # pragma: no cover - defensive
            self._next_turn()

    # -- influence loss -------------------------------------------------- #
    def _lose_influence(self, seat: int, resume: str) -> None:
        if not self.is_alive(seat):  # pragma: no cover - defensive
            self._lose_resume = resume
            self._after_lose()
            return
        hand = self.players[seat].hand
        self._lose_seat = seat
        self._lose_resume = resume
        if len(set(hand)) == 1:
            # No real choice: one card left or two identical cards.
            self._apply_lose(seat, 0)
            self._after_lose()
        else:
            self.phase = Phase.LOSE_INFLUENCE
            self.current_player = seat

    def _lose_step(self, info) -> None:
        if info.kind != KIND_LOSE_SLOT:  # pragma: no cover - defensive
            raise RuntimeError(f"unexpected action {info.kind} in LOSE_INFLUENCE")
        seat = self.current_player
        if seat != self._lose_seat:  # pragma: no cover - defensive
            raise RuntimeError("influence loss seat mismatch")
        self.stats[seat]["action_counts"]["LOSE_SLOT"] += 1
        self._apply_lose(seat, int(info.index))
        self._after_lose()

    def _apply_lose(self, seat: int, slot: int) -> None:
        p = self.players[seat]
        card = p.hand.pop(slot)
        p.revealed.append(card)
        self.stats[seat]["influence_lost"] += 1
        self._push_event(EV_INFLUENCE_LOST, seat, None)
        if not p.hand:
            self._elim_order.append(seat)

    def _after_lose(self) -> None:
        resume = self._lose_resume
        self._lose_resume = None
        self._lose_seat = None
        if self._check_game_over():
            return
        if resume == RESUME_NEXT_TURN:
            self._next_turn()
        elif resume == RESUME_AFTER_ACTION_CHALLENGER_LOSS:
            self._continue_after_action_challenge_loss()
        elif resume == RESUME_AFTER_BLOCK_CHALLENGER_LOSS:
            self._next_turn()  # the block stood, action cancelled
        elif resume == RESUME_AFTER_BLOCKER_LOSS:
            self._resolve_action_now()  # the action now goes through
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unknown resume tag {resume!r}")

    def _continue_after_action_challenge_loss(self) -> None:
        """Claimant proved the role; the action still resolves."""
        action_type = self.pending_action
        if action_type == ActionType.TAX:
            self.players[self.pending_actor].coins += TAX_AMOUNT
            self._next_turn()
        elif action_type == ActionType.EXCHANGE:
            self._start_exchange(self.pending_actor)
        elif action_type in (ActionType.ASSASSINATE, ActionType.STEAL):
            target = self.pending_target
            if target is not None and self.is_alive(target):
                self._start_block_phase([target])
            else:
                self._next_turn()
        else:  # pragma: no cover - defensive
            self._resolve_action_now()

    # -- exchange ------------------------------------------------------- #
    def _exchange_canonical_indices(self) -> list[int]:
        """Indices of keep options with distinct resulting card multisets."""
        buffer = self._exchange_buffer or []
        if len(buffer) == 4:
            candidates = list(enumerate(EXCHANGE_KEEP_PAIRS))
        elif len(buffer) == 3:
            candidates = [
                (len(EXCHANGE_KEEP_PAIRS) + i, keep)
                for i, keep in enumerate(EXCHANGE_KEEP_SINGLES)
            ]
        else:  # pragma: no cover - defensive
            candidates = [(0, tuple(range(len(buffer))))]
        out: list[int] = []
        seen: set[tuple[int, ...]] = set()
        for index, keep in candidates:
            key = tuple(buffer[k] for k in keep)
            if key in seen:
                continue
            seen.add(key)
            out.append(index)
        return out

    def _start_exchange(self, seat: int) -> None:
        drawn = [self.deck.pop() for _ in range(EXCHANGE_DRAW)]
        buffer = sorted(self.players[seat].hand + drawn)
        self._exchange_buffer = buffer
        options = self._exchange_canonical_indices()
        if len(options) == 1:
            self._finish_exchange(seat, options[0])
            return
        self.phase = Phase.EXCHANGE_SELECT
        self.current_player = seat

    def _exchange_step(self, info) -> None:
        if info.kind != KIND_EXCHANGE_KEEP:  # pragma: no cover - defensive
            raise RuntimeError(f"unexpected action {info.kind} in EXCHANGE_SELECT")
        seat = self.current_player
        self.stats[seat]["action_counts"]["EXCHANGE_KEEP"] += 1
        self._finish_exchange(seat, int(info.index))

    def _finish_exchange(self, seat: int, index: int) -> None:
        buffer = self._exchange_buffer or []
        keep = EXCHANGE_KEEP_OPTIONS[index]
        kept = [buffer[k] for k in keep]
        returned = [card for j, card in enumerate(buffer) if j not in keep]
        self.deck.extend(returned)
        self.rng.shuffle(self.deck)
        self.players[seat].hand = sorted(int(c) for c in kept)
        self._exchange_buffer = None
        self._next_turn()

    # ------------------------------------------------------------------ #
    # Turn / termination
    # ------------------------------------------------------------------ #
    def _reset_pending(self) -> None:
        self.pending_actor = None
        self.pending_action = None
        self.pending_target = None
        self.pending_blocker = None
        self.pending_block_role = None
        self._poll_order = []
        self._poll_idx = 0

    def _next_turn(self) -> None:
        if self._check_game_over():
            return
        self.turn += 1
        if self.turn >= self.max_turns:
            self._truncate()
            return
        ref = self.pending_actor if self.pending_actor is not None else self.current_player
        nxt = self._next_alive_after(ref)
        self._reset_pending()
        self._exchange_buffer = None
        self.phase = Phase.CHOOSE_ACTION
        self.current_player = nxt

    def _check_game_over(self) -> bool:
        if self.num_alive <= 1:
            self._finish_game()
            return True
        return False

    def _finish_game(self) -> None:
        self.phase = Phase.GAME_OVER
        self.terminated = True
        self.truncated = False
        self._reset_pending()
        alive = self.alive_seats()
        if alive:
            self.current_player = alive[0]
        self._assign_terminal_rewards()

    def _truncate(self) -> None:
        self.phase = Phase.GAME_OVER
        self.terminated = False
        self.truncated = True
        self._reset_pending()
        self.rewards[:] = 0.0

    def _assign_terminal_rewards(self) -> None:
        n = self.num_players
        self.rewards[:] = 0.0
        if self._reward_kind == "win":
            alive = self.alive_seats()
            if not alive:  # pragma: no cover - defensive
                return
            winner = alive[0]
            if n == 2:
                for seat in range(n):
                    self.rewards[seat] = 1.0 if seat == winner else -1.0
            else:
                loser = -1.0 / (n - 1)
                for seat in range(n):
                    self.rewards[seat] = 1.0 if seat == winner else loser
        else:  # rank: linearly spaced +1 .. -1 by elimination order
            places = {seat: n - 1 - k for k, seat in enumerate(self._elim_order)}
            for seat in self.alive_seats():
                places[seat] = 0
            denom = n - 1
            for seat in range(n):
                place = places.get(seat, denom)
                self.rewards[seat] = 1.0 - 2.0 * place / denom

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _claimed_role(self) -> Role | None:
        if self.phase == Phase.CHALLENGE_ACTION and self.pending_action is not None:
            return CLAIMED_ROLE[self.pending_action]
        if self.phase == Phase.CHALLENGE_BLOCK:
            return self.pending_block_role
        return None

    def get_obs(self, player_id: int) -> np.ndarray:
        """Ego-centric ``float32`` observation for ``player_id``."""
        n = self.num_players
        L = self.obs_layout
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        viewer = int(player_id)
        me = self.players[viewer]

        own = self.hand_counts(viewer).astype(np.float32)
        obs[L.hand_counts : L.hand_counts + NUM_ROLES] = own / 2.0

        if (
            self.phase == Phase.EXCHANGE_SELECT
            and self.current_player == viewer
            and self._exchange_buffer is not None
        ):
            buffer = self._exchange_buffer
            bc = _counts(buffer).astype(np.float32)
            obs[L.buffer_counts : L.buffer_counts + NUM_ROLES] = bc / 4.0
            for slot in range(4):
                if slot < len(buffer):
                    obs[L.buffer_slots + slot * NUM_ROLES + int(buffer[slot])] = 1.0
            obs[L.in_exchange] = 1.0

        revealed = self.revealed_counts()
        for rel in range(n):
            seat = self._abs(viewer, rel)
            p = self.players[seat]
            o = L.seat_offset(rel)
            obs[o] = 1.0 if p.alive else 0.0
            obs[o + 1] = len(p.hand) / 2.0
            obs[o + 2] = min(p.coins, 12) / 12.0
            obs[o + 3] = 1.0 if p.coins >= ASSASSINATE_COST else 0.0
            obs[o + 4] = 1.0 if p.coins >= COUP_COST else 0.0
            obs[o + 5] = 1.0 if p.coins >= MANDATORY_COUP_COINS else 0.0
            obs[o + 6 : o + 6 + NUM_ROLES] = _counts(p.revealed) / 2.0

        obs[L.total_revealed : L.total_revealed + NUM_ROLES] = revealed.astype(np.float32) / 3.0
        known = np.minimum(own + revealed.astype(np.float32), COPIES_PER_ROLE).astype(
            np.float32
        )
        obs[L.known_total : L.known_total + NUM_ROLES] = known / 3.0
        obs[L.num_alive] = self.num_alive / n
        obs[L.turn] = min(self.turn, self.max_turns) / self.max_turns
        obs[L.deck_size] = len(self.deck) / DECK_SIZE

        claimed = self._claimed_role()
        if claimed is not None and self._role_exhausted_for(viewer, claimed):
            obs[L.claimed_impossible] = 1.0

        if self.phase != Phase.GAME_OVER:
            obs[L.phase + int(self.phase)] = 1.0
        obs[L.pending_actor + self._rel_slot(self.pending_actor, viewer)] = 1.0
        obs[L.pending_target + self._rel_slot(self.pending_target, viewer)] = 1.0
        action_slot = 7 if self.pending_action is None else int(self.pending_action)
        obs[L.pending_action + action_slot] = 1.0
        obs[L.pending_blocker + self._rel_slot(self.pending_blocker, viewer)] = 1.0
        if self.pending_block_role is not None:
            obs[L.pending_block_role + int(self.pending_block_role)] = 1.0

        claims = self.claim_counts.astype(np.float32)
        for rel in range(n):
            seat = self._abs(viewer, rel)
            o = L.claims_offset(rel)
            obs[o : o + NUM_ROLES] = np.minimum(claims[seat, :NUM_ROLES], 5) / 5.0
            obs[o + 5] = min(claims[seat, CLAIM_FEATURES - 2], 5) / 5.0
            obs[o + 6] = min(claims[seat, CLAIM_FEATURES - 1], 5) / 5.0

        events = list(self._events)
        pad = NUM_EVENTS - len(events)
        for index, (kind, actor, target) in enumerate(events):
            o = L.event_offset(pad + index)
            obs[o + kind] = 1.0
            obs[o + NUM_EVENT_KINDS] = self._rel_scalar(actor, viewer)
            obs[o + NUM_EVENT_KINDS + 1] = self._rel_scalar(target, viewer)

        return obs

    def _rel_slot(self, seat: int | None, viewer: int) -> int:
        if seat is None:
            return self.num_players
        return self._rel(int(seat), viewer)

    def _rel_scalar(self, seat: int | None, viewer: int) -> float:
        if seat is None:
            return 0.0
        return (self._rel(int(seat), viewer) + 1) / self.num_players

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #
    def get_episode_stats(self) -> dict[str, Any]:
        """Per-episode behavioural telemetry snapshot."""
        players = {}
        for seat in range(self.num_players):
            st = self.stats[seat]
            players[seat] = {
                "action_counts": dict(st["action_counts"]),
                "claims": dict(st["claims"]),
                "bluffs": dict(st["bluffs"]),
                "challenges_attempted": st["challenges_attempted"],
                "challenges_won": st["challenges_won"],
                "challenge_opportunities": st["challenge_opportunities"],
                "impossible_claim_opportunities": st["impossible_claim_opportunities"],
                "challenges_on_impossible_claim": st["challenges_on_impossible_claim"],
                "blocks_attempted": st["blocks_attempted"],
                "block_bluffs": st["block_bluffs"],
                "claims_proven": st["claims_proven"],
                "caught_bluffing": st["caught_bluffing"],
                "influence_lost": st["influence_lost"],
                "coins_stolen": st["coins_stolen"],
                "bluff_rate": self._bluff_rate(st),
            }
        return {
            "num_players": self.num_players,
            "turns": self.turn,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "winner": self.winner(),
            "elim_order": list(self._elim_order),
            "rewards": [float(r) for r in self.rewards],
            "players": players,
        }

    @staticmethod
    def _bluff_rate(st: dict[str, Any]) -> float:
        claims = sum(st["claims"].values())
        if claims == 0:
            return 0.0
        return sum(st["bluffs"].values()) / claims

    # ------------------------------------------------------------------ #
    # Debug / test helpers
    # ------------------------------------------------------------------ #
    def debug_set_hands(
        self,
        hands: dict[int, Sequence[int]],
        deck: Sequence[int] | None = None,
        coins: dict[int, int] | None = None,
        revealed: dict[int, Sequence[int]] | None = None,
    ) -> None:
        """Force hands (and optionally deck/coins/reveals), keeping the deck legal."""
        self._exchange_buffer = None
        for seat, cards in (revealed or {}).items():
            self.players[seat].revealed = [int(c) for c in cards]
        for seat, hand in hands.items():
            self.players[seat].hand = sorted(int(c) for c in hand)
        if coins:
            for seat, value in coins.items():
                self.players[seat].coins = int(value)
        if deck is not None:
            self.deck = [int(c) for c in deck]
            return
        remaining = [int(r) for r in Role for _ in range(COPIES_PER_ROLE)]
        for card in self.revealed_counts_flat():
            remaining.remove(card)
        for p in self.players:
            for card in p.hand:
                remaining.remove(int(card))
        self.deck = remaining

    def revealed_counts_flat(self) -> list[int]:
        return [int(c) for p in self.players for c in p.revealed]

    def debug_set_phase(self, phase: Phase, current_player: int | None = None) -> None:
        self.phase = Phase(phase)
        if current_player is not None:
            self.current_player = int(current_player)

    def render(self) -> str:
        """Compact textual state dump (debugging aid)."""
        lines = [f"turn={self.turn} phase={Phase(self.phase).name} actor={self.current_player}"]
        for seat, p in enumerate(self.players):
            coins = p.coins
            hand = ",".join(Role(c).name for c in p.hand)
            dead = ",".join(Role(c).name for c in p.revealed)
            lines.append(f"  seat {seat}: coins={coins:2d} hand=[{hand}] revealed=[{dead}]")
        lines.append(f"  deck={len(self.deck)}")
        return "\n".join(lines)


__all__ = [
    "CoupEngine",
    "ObsLayout",
    "make_obs_layout",
    "obs_dim",
    "NUM_EVENTS",
    "EVENT_STRIDE",
    "NUM_EVENT_KINDS",
    "CLAIM_FEATURES",
]
