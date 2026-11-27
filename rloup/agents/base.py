"""Scripted agents for the Coup environment.

Every agent draws its decisions exclusively from

* the flat ego-centric observation produced by ``CoupEngine.get_obs``, and
* the per-phase ``action_mask``.

Neither is allowed to contain hidden information, so no agent can cheat by
looking at another seat's hand.  ``engine`` / ``player_id`` are part of the
:class:`Agent` interface for symmetry with learning-based agents (which may
want a live engine handle) but the scripted baselines never read them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rloup.constants import (
    CLAIMED_ROLE,
    EXCHANGE_KEEP_OPTIONS,
    NUM_ROLES,
    ActionSpaceSpec,
    ActionType,
    Phase,
    Role,
    players_from_num_actions,
)
from rloup.engine import CLAIM_FEATURES, EVENT_STRIDE, ObsLayout, make_obs_layout

__all__ = [
    "Agent",
    "ObsView",
    "SeatView",
    "parse_obs",
    "RandomAgent",
    "HonestAgent",
    "AggressiveBlufferAgent",
    "CautiousSkepticalAgent",
    "CardCounterAgent",
    "BASELINES",
]


def _to_int(value: float, scale: int) -> int:
    return int(round(float(value) * scale))


@dataclass
class SeatView:
    """Relative-seat features decoded from the observation."""

    relative: int
    alive: bool
    influence: int
    coins: int
    can_assassinate: bool
    can_coup: bool
    must_coup: bool
    revealed: np.ndarray


@dataclass
class ObsView:
    """Structured view of one agent's observation."""

    obs: np.ndarray
    action_mask: np.ndarray
    num_players: int
    n_actions: int
    layout: ObsLayout
    spec: ActionSpaceSpec
    seat: SeatView
    seats: list[SeatView]
    own_hand_counts: np.ndarray
    own_hand: list[int]
    buffer_counts: np.ndarray
    buffer: list[int]
    buffer_slots: list[int]
    in_exchange: bool
    total_revealed: np.ndarray
    known: np.ndarray
    unseen: np.ndarray
    num_alive: int
    turn_ratio: float
    deck_size: int
    phase: Phase
    pending_actor_rel: int | None
    pending_target_rel: int | None
    pending_action: ActionType | None
    pending_blocker_rel: int | None
    pending_block_role: Role | None
    claimed_impossible: bool
    claims: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    events: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    legal_ids: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=int))

    # -- convenience --------------------------------------------------- #
    @property
    def must_coup(self) -> bool:
        return self.seat.must_coup

    @property
    def coins(self) -> int:
        return self.seat.coins

    def has(self, role: Role | int) -> bool:
        return int(self.own_hand_counts[int(role)]) > 0

    def influence_of(self, relative_seat: int) -> int:
        return self.seats[relative_seat].influence

    def coins_of(self, relative_seat: int) -> int:
        return self.seats[relative_seat].coins

    def target_rel(self, action_id: int) -> int | None:
        return self.spec.decode(int(action_id)).rel_off

    def candidates(self, kind: str) -> list[int]:
        """Legal action ids of ``kind`` (e.g. ``COUP``), in seat order."""
        out = []
        for action_id in self.legal_ids:
            info = self.spec.decode(int(action_id))
            if info.kind == kind:
                out.append(int(action_id))
        return out

    def ids_of_kind(self, kind: str) -> list[int]:
        return self.candidates(kind)

    @property
    def pass_id(self) -> int:
        return self.spec.pass_

    @property
    def challenge_id(self) -> int:
        return self.spec.challenge

    def can_pass(self) -> bool:
        return bool(self.action_mask[self.spec.pass_])

    def can_challenge(self) -> bool:
        return bool(self.action_mask[self.spec.challenge])

    def block_ids(self) -> list[int]:
        return self.candidates("BLOCK")

    @property
    def claimed_role(self) -> Role | None:
        if self.phase == Phase.CHALLENGE_ACTION and self.pending_action is not None:
            return CLAIMED_ROLE[self.pending_action]
        if self.phase == Phase.CHALLENGE_BLOCK:
            return self.pending_block_role
        return None

    def claimant_rel(self) -> int | None:
        """Relative seat of the player whose claim is currently pending."""
        if self.phase == Phase.CHALLENGE_ACTION:
            return self.pending_actor_rel
        if self.phase == Phase.CHALLENGE_BLOCK:
            return self.pending_blocker_rel
        return None

    def keep_options(self) -> list[tuple[int, tuple[int, ...]]]:
        """Legal exchange keep options as ``(action_id, slots)``."""
        out = []
        for action_id in self.legal_ids:
            info = self.spec.decode(int(action_id))
            if info.kind == "EXCHANGE_KEEP":
                out.append((int(action_id), EXCHANGE_KEEP_OPTIONS[int(info.index)]))
        return out


def parse_obs(obs: np.ndarray, action_mask: np.ndarray) -> ObsView:
    """Decode the flat ego-centric observation into :class:`ObsView`."""
    mask = np.asarray(action_mask)
    n_actions = int(mask.shape[0])
    num_players = players_from_num_actions(n_actions)
    layout = make_obs_layout(num_players)
    spec = ActionSpaceSpec(num_players)
    o = np.asarray(obs, dtype=np.float32)
    L = layout
    n = num_players

    own_counts = np.rint(o[L.hand_counts : L.hand_counts + NUM_ROLES] * 2).astype(int)
    own_hand: list[int] = []
    for role in range(NUM_ROLES):
        own_hand.extend([role] * int(own_counts[role]))

    buffer_counts = np.rint(o[L.buffer_counts : L.buffer_counts + NUM_ROLES] * 4).astype(int)
    buffer_slots: list[int] = []
    for slot in range(4):
        base = L.buffer_slots + slot * NUM_ROLES
        chunk = o[base : base + NUM_ROLES]
        if float(chunk.max()) > 0.5:
            buffer_slots.append(int(np.argmax(chunk)))
    buffer = sorted(buffer_slots)

    revealed = np.rint(o[L.total_revealed : L.total_revealed + NUM_ROLES] * 3).astype(int)
    known = np.rint(o[L.known_total : L.known_total + NUM_ROLES] * 3).astype(int)
    unseen = np.maximum(3 - own_counts - revealed, 0)

    seats: list[SeatView] = []
    for rel in range(n):
        base = L.seat_offset(rel)
        seats.append(
            SeatView(
                relative=rel,
                alive=bool(o[base] > 0.5),
                influence=_to_int(o[base + 1], 2),
                coins=_to_int(o[base + 2], 12),
                can_assassinate=bool(o[base + 3] > 0.5),
                can_coup=bool(o[base + 4] > 0.5),
                must_coup=bool(o[base + 5] > 0.5),
                revealed=np.rint(o[base + 6 : base + 6 + NUM_ROLES] * 2).astype(int),
            )
        )

    phase_idx = int(np.argmax(o[L.phase : L.phase + 6]))
    phase = Phase(phase_idx) if float(o[L.phase : L.phase + 6].max()) > 0.5 else Phase.GAME_OVER

    def rel_or_none(base: int) -> int | None:
        slot = int(np.argmax(o[base : base + n + 1]))
        return None if slot >= n else slot

    action_slot = int(np.argmax(o[L.pending_action : L.pending_action + 8]))
    pending_action = None if action_slot >= len(ActionType) else ActionType(action_slot)
    block_slot = o[L.pending_block_role : L.pending_block_role + NUM_ROLES]
    pending_block_role = (
        Role(int(np.argmax(block_slot))) if float(block_slot.max()) > 0.5 else None
    )

    claims = np.rint(
        o[L.claims : L.claims + CLAIM_FEATURES * n] * 5.0
    ).astype(int).reshape(n, CLAIM_FEATURES)
    events = np.asarray(
        o[L.events : L.events + EVENT_STRIDE * 8], dtype=np.float32
    ).reshape(8, EVENT_STRIDE)

    return ObsView(
        obs=o,
        action_mask=mask,
        num_players=n,
        n_actions=n_actions,
        layout=L,
        spec=spec,
        seat=seats[0],
        seats=seats,
        own_hand_counts=own_counts,
        own_hand=own_hand,
        buffer_counts=buffer_counts,
        buffer=buffer,
        buffer_slots=buffer_slots,
        in_exchange=bool(o[L.in_exchange] > 0.5),
        total_revealed=revealed,
        known=known,
        unseen=unseen,
        num_alive=_to_int(o[L.num_alive], n),
        turn_ratio=float(o[L.turn]),
        deck_size=_to_int(o[L.deck_size], 15),
        phase=phase,
        pending_actor_rel=rel_or_none(L.pending_actor),
        pending_target_rel=rel_or_none(L.pending_target),
        pending_action=pending_action,
        pending_blocker_rel=rel_or_none(L.pending_blocker),
        pending_block_role=pending_block_role,
        claimed_impossible=bool(o[L.claimed_impossible] > 0.5),
        claims=claims,
        events=events,
        legal_ids=np.flatnonzero(mask),
    )


class Agent:
    """Base class for scripted agents."""

    name = "agent"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    # -- interface ----------------------------------------------------- #
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        engine=None,
        player_id: int | None = None,
    ) -> int:
        """Choose a legal action id for the current decision."""
        del engine, player_id  # scripted agents never inspect the live engine
        view = parse_obs(obs, action_mask)
        legal = view.legal_ids
        if legal.size == 0:  # pragma: no cover - engine guarantees non-empty
            raise RuntimeError("no legal actions available")
        return int(self.act(view))

    # -- to implement -------------------------------------------------- #
    def act(self, view: ObsView) -> int:
        raise NotImplementedError

    # -- shared helpers ------------------------------------------------ #
    def _random_legal(self, view: ObsView) -> int:
        return int(self.rng.choice(view.legal_ids))

    @staticmethod
    def _expendable_slot(view: ObsView) -> int:
        """Index of the hand slot that is cheapest to reveal.

        Hands are always stored sorted ascending by role id, and the observation
        exposes exact counts, so the sorted hand (and therefore each slot) is
        fully reconstructable.
        """
        hand = view.own_hand
        if len(hand) <= 1:
            return 0
        value = {
            Role.DUKE: 3,
            Role.CONTESSA: 3,
            Role.ASSASSIN: 2,
            Role.CAPTAIN: 2,
            Role.AMBASSADOR: 1,
        }
        counts = view.own_hand_counts
        best_slot, best_score = 0, None
        for slot, card in enumerate(hand):
            # losing a duplicated role costs less, and low-value roles first
            score = value[Role(card)] - 4 * (int(counts[card]) - 1)
            if best_score is None or score < best_score:
                best_slot, best_score = slot, score
        return best_slot

    @staticmethod
    def _loss_action(view: ObsView, slot: int) -> int | None:
        action_id = view.spec.lose_slot(slot)
        return action_id if view.action_mask[action_id] else None

    def _default_loss(self, view: ObsView) -> int:
        slot = self._expendable_slot(view)
        chosen = self._loss_action(view, slot)
        if chosen is None:
            chosen = int(view.legal_ids[0])
        return chosen

    @staticmethod
    def _best_keep(view: ObsView) -> int:
        """Keep the most useful cards during an exchange."""
        value = {
            Role.DUKE: 5.0,
            Role.CONTESSA: 4.0,
            Role.ASSASSIN: 3.5,
            Role.CAPTAIN: 3.0,
            Role.AMBASSADOR: 2.0,
        }
        best_id, best_score = int(view.legal_ids[0]), -1e9
        for action_id, slots in view.keep_options():
            score = sum(value[Role(view.buffer[k])] for k in slots)
            # prefer keeping a role we already had (information advantage)
            for k in slots:
                if view.own_hand_counts[view.buffer[k]] > 0:
                    score += 0.25
            if score > best_score:
                best_id, best_score = action_id, score
        return best_id

    @staticmethod
    def _best_target(view: ObsView, ids: list[int]) -> int:
        """Target the richest/highest-influence opponent."""
        best_id, best_score = ids[0], -1e9
        for action_id in ids:
            rel = view.target_rel(action_id)
            if rel is None:  # pragma: no cover - defensive
                continue
            seat = view.seats[rel]
            score = seat.influence * 3.0 + seat.coins * 0.5
            if score > best_score:
                best_id, best_score = action_id, score
        return best_id

    @staticmethod
    def _richest_target(view: ObsView, ids: list[int]) -> int:
        """Target the opponent with the most coins and influence."""
        best_id, best_score = ids[0], -1e9
        for action_id in ids:
            rel = view.target_rel(action_id)
            if rel is None:  # pragma: no cover - defensive
                continue
            seat = view.seats[rel]
            score = seat.coins + 3.0 * seat.influence
            if score > best_score:
                best_id, best_score = action_id, score
        return int(best_id)

    def _main_action(self, view: ObsView) -> int:
        raise NotImplementedError

    def _challenge_action(self, view: ObsView) -> int:
        raise NotImplementedError

    def _block_action(self, view: ObsView) -> int:
        raise NotImplementedError

    def act(self, view: ObsView) -> int:
        phase = view.phase
        if phase == Phase.CHOOSE_ACTION:
            action = self._main_action(view)
            if view.action_mask[action]:
                return action
            return self._random_legal(view)
        if phase in (Phase.CHALLENGE_ACTION, Phase.CHALLENGE_BLOCK):
            action = self._challenge_action(view)
            if view.action_mask[action]:
                return action
            return int(view.pass_id) if view.can_pass() else self._random_legal(view)
        if phase == Phase.BLOCK_ACTION:
            action = self._block_action(view)
            if view.action_mask[action]:
                return action
            return int(view.pass_id) if view.can_pass() else self._random_legal(view)
        if phase == Phase.LOSE_INFLUENCE:
            return self._default_loss(view)
        if phase == Phase.EXCHANGE_SELECT:
            return self._best_keep(view)
        return self._random_legal(view)  # pragma: no cover - GAME_OVER

