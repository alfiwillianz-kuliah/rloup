"""Static definitions for the Coup game: roles, phases, actions and the flat
discrete action space shared by the engine, the PettingZoo wrapper and agents.

The action space is a single flat ``Discrete(n_actions)`` space.  Every action
id is decoded through :class:`ActionSpaceSpec`, which knows the seating order of
the engine and therefore which of the (legal) id ranges apply in a given phase.

Layout (``N = num_players``)::

    offset 0                      INCOME
    offset 1                      FOREIGN_AID
    offset 2                      TAX            (claims DUKE)
    offset 3                      EXCHANGE       (claims AMBASSADOR)
    4                .. 4+(N-1)-1            COUP(rel_off)
    4+(N-1)          .. 4+2*(N-1)-1          ASSASSINATE(rel_off)   (claims ASSASSIN)
    4+2*(N-1)        .. 4+3*(N-1)-1          STEAL(rel_off)         (claims CAPTAIN)
    4+3*(N-1)        .. +5                   PASS, CHALLENGE,
                                             BLOCK_DUKE, BLOCK_CONTESSA,
                                             BLOCK_CAPTAIN, BLOCK_AMBASSADOR
    +6, +7                                   LOSE_SLOT_0, LOSE_SLOT_1
    +8               .. +16                  6 keep-2 pairs (2 influence exchange)
                                             then 3 keep-1 singles (1 influence)

Total: ``21 + 3 * (N - 1)`` actions (24 for 2P, 30 for 4P, 36 for 6P).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #


class Role(IntEnum):
    """The five Court roles.  The integer value doubles as the deck/hand id."""

    DUKE = 0
    ASSASSIN = 1
    CAPTAIN = 2
    AMBASSADOR = 3
    CONTESSA = 4


NUM_ROLES = 5
COPIES_PER_ROLE = 3
DECK_SIZE = NUM_ROLES * COPIES_PER_ROLE  # 15

ROLE_NAMES: tuple[str, ...] = ("DUKE", "ASSASSIN", "CAPTAIN", "AMBASSADOR", "CONTESSA")


class Phase(IntEnum):
    """Which kind of decision is currently pending."""

    CHOOSE_ACTION = 0
    CHALLENGE_ACTION = 1
    BLOCK_ACTION = 2
    CHALLENGE_BLOCK = 3
    LOSE_INFLUENCE = 4
    EXCHANGE_SELECT = 5
    GAME_OVER = 6


NUM_DECISION_PHASES = 6  # phases observable through the phase one-hot


class ActionType(IntEnum):
    """Main turn actions."""

    INCOME = 0
    FOREIGN_AID = 1
    COUP = 2
    TAX = 3
    ASSASSINATE = 4
    STEAL = 5
    EXCHANGE = 6


NUM_ACTION_TYPES = 7

ACTION_NAMES: tuple[str, ...] = (
    "INCOME",
    "FOREIGN_AID",
    "COUP",
    "TAX",
    "ASSASSINATE",
    "STEAL",
    "EXCHANGE",
)
ACTION_TYPE_BY_NAME = {name: ActionType(i) for i, name in enumerate(ACTION_NAMES)}

#: Role a main action claims, or ``None`` when the action claims nothing.
CLAIMED_ROLE: dict[ActionType, Role | None] = {
    ActionType.INCOME: None,
    ActionType.FOREIGN_AID: None,
    ActionType.COUP: None,
    ActionType.TAX: Role.DUKE,
    ActionType.ASSASSINATE: Role.ASSASSIN,
    ActionType.STEAL: Role.CAPTAIN,
    ActionType.EXCHANGE: Role.AMBASSADOR,
}

COUP_COST = 7
ASSASSINATE_COST = 3
MANDATORY_COUP_COINS = 10
STEAL_AMOUNT = 2
TAX_AMOUNT = 3
EXCHANGE_DRAW = 2

# --------------------------------------------------------------------------- #
# Action-space vocabulary (semantic kinds, independent of the numeric layout)
# --------------------------------------------------------------------------- #

KIND_INCOME = "INCOME"
KIND_FOREIGN_AID = "FOREIGN_AID"
KIND_TAX = "TAX"
KIND_EXCHANGE = "EXCHANGE"
KIND_COUP = "COUP"
KIND_ASSASSINATE = "ASSASSINATE"
KIND_STEAL = "STEAL"
KIND_PASS = "PASS"
KIND_CHALLENGE = "CHALLENGE"
KIND_BLOCK = "BLOCK"
KIND_LOSE_SLOT = "LOSE_SLOT"
KIND_EXCHANGE_KEEP = "EXCHANGE_KEEP"

#: Claim (main action) kinds used by the behavioural telemetry.
CLAIM_KIND_TAX_DUKE = "tax_duke"
CLAIM_KIND_ASSASSINATE = "assassinate"
CLAIM_KIND_STEAL_CAPTAIN = "steal_captain"
CLAIM_KIND_EXCHANGE_AMBASSADOR = "exchange_ambassador"
CLAIM_KIND_BLOCK_DUKE = "block_duke"
CLAIM_KIND_BLOCK_CONTESSA = "block_contessa"
CLAIM_KIND_BLOCK_CAPTAIN = "block_captain"
CLAIM_KIND_BLOCK_AMBASSADOR = "block_ambassador"

CLAIM_KINDS: tuple[str, ...] = (
    CLAIM_KIND_TAX_DUKE,
    CLAIM_KIND_ASSASSINATE,
    CLAIM_KIND_STEAL_CAPTAIN,
    CLAIM_KIND_EXCHANGE_AMBASSADOR,
    CLAIM_KIND_BLOCK_DUKE,
    CLAIM_KIND_BLOCK_CONTESSA,
    CLAIM_KIND_BLOCK_CAPTAIN,
    CLAIM_KIND_BLOCK_AMBASSADOR,
)

#: Main action -> telemetry claim kind (only for actions that claim a role).
ACTION_CLAIM_KIND: dict[ActionType, str | None] = {
    ActionType.INCOME: None,
    ActionType.FOREIGN_AID: None,
    ActionType.COUP: None,
    ActionType.TAX: CLAIM_KIND_TAX_DUKE,
    ActionType.ASSASSINATE: CLAIM_KIND_ASSASSINATE,
    ActionType.STEAL: CLAIM_KIND_STEAL_CAPTAIN,
    ActionType.EXCHANGE: CLAIM_KIND_EXCHANGE_AMBASSADOR,
}

#: Block role -> telemetry claim kind.
BLOCK_ROLE_CLAIM_KIND: dict[Role, str] = {
    Role.DUKE: CLAIM_KIND_BLOCK_DUKE,
    Role.CONTESSA: CLAIM_KIND_BLOCK_CONTESSA,
    Role.CAPTAIN: CLAIM_KIND_BLOCK_CAPTAIN,
    Role.AMBASSADOR: CLAIM_KIND_BLOCK_AMBASSADOR,
}

#: Actions that may be blocked, and by whom / with which roles.
FOREIGN_AID_BLOCK_ROLES = (Role.DUKE,)
ASSASSINATE_BLOCK_ROLES = (Role.CONTESSA,)
STEAL_BLOCK_ROLES = (Role.CAPTAIN, Role.AMBASSADOR)

#: Utility-block sub-offsets.
UTIL_PASS = 0
UTIL_CHALLENGE = 1
UTIL_BLOCK_DUKE = 2
UTIL_BLOCK_CONTESSA = 3
UTIL_BLOCK_CAPTAIN = 4
UTIL_BLOCK_AMBASSADOR = 5
NUM_UTIL_SLOTS = 6

#: Utility sub-offset -> role it blocks with.
BLOCK_SLOT_TO_ROLE: dict[int, Role] = {
    UTIL_BLOCK_DUKE: Role.DUKE,
    UTIL_BLOCK_CONTESSA: Role.CONTESSA,
    UTIL_BLOCK_CAPTAIN: Role.CAPTAIN,
    UTIL_BLOCK_AMBASSADOR: Role.AMBASSADOR,
}
ROLE_TO_BLOCK_SLOT: dict[Role, int] = {r: s for s, r in BLOCK_SLOT_TO_ROLE.items()}

#: Keep-subsets offered when exchanging with 2 influences (choose 2 of 4).
EXCHANGE_KEEP_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)
#: Keep-subsets offered when exchanging with 1 influence (choose 1 of 3).
EXCHANGE_KEEP_SINGLES: tuple[tuple[int, ...], ...] = ((0,), (1,), (2,))
EXCHANGE_KEEP_OPTIONS: tuple[tuple[int, ...], ...] = EXCHANGE_KEEP_PAIRS + EXCHANGE_KEEP_SINGLES
NUM_EXCHANGE_KEEP_SLOTS = len(EXCHANGE_KEEP_OPTIONS)  # 9


# --------------------------------------------------------------------------- #
# Action decoding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ActionInfo:
    """Decoded meaning of a flat action id."""

    kind: str
    #: Seat-relative target offset in ``1 .. N-1`` for targeted actions.
    rel_off: int | None = None
    #: Index into ``EXCHANGE_KEEP_OPTIONS`` / number of the loss slot.
    index: int | None = None
    #: Blocking role for ``BLOCK`` kind.
    role: Role | None = None


class ActionSpaceSpec:
    """Flat discrete action-space layout for a given seat count."""

    __slots__ = (
        "num_players",
        "n_actions",
        "coup_offset",
        "assassinate_offset",
        "steal_offset",
        "util_offset",
        "lose_offset",
        "exchange_keep_offset",
    )

    def __init__(self, num_players: int) -> None:
        if not 2 <= num_players <= 6:
            raise ValueError(f"num_players must be in 2..6, got {num_players}")
        n = num_players
        self.num_players = n
        span = n - 1
        self.coup_offset = 4
        self.assassinate_offset = self.coup_offset + span
        self.steal_offset = self.assassinate_offset + span
        self.util_offset = self.steal_offset + span
        self.lose_offset = self.util_offset + NUM_UTIL_SLOTS
        self.exchange_keep_offset = self.lose_offset + 2
        self.n_actions = self.exchange_keep_offset + NUM_EXCHANGE_KEEP_SLOTS
        assert self.n_actions == 21 + 3 * span, (self.n_actions, n)

    # -- constructors ------------------------------------------------------ #
    def coup(self, rel_off: int) -> int:
        self._check_rel(rel_off)
        return self.coup_offset + rel_off - 1

    def assassinate(self, rel_off: int) -> int:
        self._check_rel(rel_off)
        return self.assassinate_offset + rel_off - 1

    def steal(self, rel_off: int) -> int:
        self._check_rel(rel_off)
        return self.steal_offset + rel_off - 1

    def _check_rel(self, rel_off: int) -> None:
        if not 1 <= rel_off <= self.num_players - 1:
            raise ValueError(
                f"rel_off must be in 1..{self.num_players - 1}, got {rel_off}"
            )

    # -- fixed slots ------------------------------------------------------- #
    @property
    def income(self) -> int:
        return 0

    @property
    def foreign_aid(self) -> int:
        return 1

    @property
    def tax(self) -> int:
        return 2

    @property
    def exchange(self) -> int:
        return 3

    @property
    def pass_(self) -> int:
        return self.util_offset + UTIL_PASS

    @property
    def challenge(self) -> int:
        return self.util_offset + UTIL_CHALLENGE

    @property
    def block_duke(self) -> int:
        return self.util_offset + UTIL_BLOCK_DUKE

    @property
    def block_contessa(self) -> int:
        return self.util_offset + UTIL_BLOCK_CONTESSA

    @property
    def block_captain(self) -> int:
        return self.util_offset + UTIL_BLOCK_CAPTAIN

    @property
    def block_ambassador(self) -> int:
        return self.util_offset + UTIL_BLOCK_AMBASSADOR

    def lose_slot(self, slot: int) -> int:
        if slot not in (0, 1):
            raise ValueError(f"loss slot must be 0 or 1, got {slot}")
        return self.lose_offset + slot

    def exchange_keep(self, index: int) -> int:
        if not 0 <= index < NUM_EXCHANGE_KEEP_SLOTS:
            raise ValueError(f"keep index must be in 0..{NUM_EXCHANGE_KEEP_SLOTS - 1}")
        return self.exchange_keep_offset + index

    def block_action(self, role: Role) -> int:
        return self.util_offset + ROLE_TO_BLOCK_SLOT[role]

    # -- decoding ---------------------------------------------------------- #
    def decode(self, action_id: int) -> ActionInfo:
        a = int(action_id)
        if not 0 <= a < self.n_actions:
            raise ValueError(f"action id {a} out of range 0..{self.n_actions - 1}")
        if a == 0:
            return ActionInfo(KIND_INCOME)
        if a == 1:
            return ActionInfo(KIND_FOREIGN_AID)
        if a == 2:
            return ActionInfo(KIND_TAX)
        if a == 3:
            return ActionInfo(KIND_EXCHANGE)
        rel = a - self.coup_offset + 1
        if a < self.assassinate_offset:
            return ActionInfo(KIND_COUP, rel_off=rel)
        rel = a - self.assassinate_offset + 1
        if a < self.steal_offset:
            return ActionInfo(KIND_ASSASSINATE, rel_off=rel)
        rel = a - self.steal_offset + 1
        if a < self.util_offset:
            return ActionInfo(KIND_STEAL, rel_off=rel)
        sub = a - self.util_offset
        if sub == UTIL_PASS:
            return ActionInfo(KIND_PASS)
        if sub == UTIL_CHALLENGE:
            return ActionInfo(KIND_CHALLENGE)
        if sub in BLOCK_SLOT_TO_ROLE:
            return ActionInfo(KIND_BLOCK, role=BLOCK_SLOT_TO_ROLE[sub])
        sub = a - self.lose_offset
        if 0 <= sub < 2:
            return ActionInfo(KIND_LOSE_SLOT, index=sub)
        sub = a - self.exchange_keep_offset
        return ActionInfo(KIND_EXCHANGE_KEEP, index=sub)

    def action_name(self, action_id: int) -> str:
        """Human readable name of a flat action id."""
        info = self.decode(action_id)
        if info.kind == KIND_COUP:
            return f"COUP(+{info.rel_off})"
        if info.kind == KIND_ASSASSINATE:
            return f"ASSASSINATE(+{info.rel_off})"
        if info.kind == KIND_STEAL:
            return f"STEAL(+{info.rel_off})"
        if info.kind == KIND_BLOCK:
            return f"BLOCK_{ROLE_NAMES[int(info.role)]}"
        if info.kind == KIND_LOSE_SLOT:
            return f"LOSE_SLOT_{info.index}"
        if info.kind == KIND_EXCHANGE_KEEP:
            keep = EXCHANGE_KEEP_OPTIONS[info.index]
            return "EXCHANGE_KEEP(" + ",".join(str(k) for k in keep) + ")"
        return info.kind

    # -- phase masks ------------------------------------------------------- #
    def target_actions(self, kind: str) -> range:
        """All action ids for a targeted action kind, regardless of legality."""
        if kind == KIND_COUP:
            return range(self.coup_offset, self.assassinate_offset)
        if kind == KIND_ASSASSINATE:
            return range(self.assassinate_offset, self.steal_offset)
        if kind == KIND_STEAL:
            return range(self.steal_offset, self.util_offset)
        raise ValueError(f"{kind} is not a targeted action")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ActionSpaceSpec(num_players={self.num_players}, "
            f"n_actions={self.n_actions})"
        )


def num_actions_for(num_players: int) -> int:
    """``21 + 3 * (num_players - 1)``."""
    return 21 + 3 * (num_players - 1)


def players_from_num_actions(n_actions: int) -> int:
    """Inverse of :func:`num_actions_for`."""
    rem = n_actions - 21
    if rem % 3 != 0 or rem < 0:
        raise ValueError(f"invalid action space size {n_actions}")
    return rem // 3 + 1
