"""Five scripted baseline agents (no hidden-information access).

``RandomAgent``
    Uniform over legal actions.
``HonestAgent``
    Never claims or blocks with a role it does not hold; challenges only when it
    can prove the claim impossible or the claimant is a known repeat bluff.
``AggressiveBlufferAgent``
    Claims and blocks freely, challenges often.
``CautiousSkepticalAgent``
    Truthful, risk-averse; challenges on evidence only, with a high bar.
``CardCounterAgent``
    Exact hypergeometric card counting over ``own_hand + total_revealed`` with
    expected-value based challenge, block and main-action choices.
"""

from __future__ import annotations

import math

from .base import Agent, ObsView
from rloup.constants import Role

__all__ = [
    "RandomAgent",
    "HonestAgent",
    "AggressiveBlufferAgent",
    "CautiousSkepticalAgent",
    "CardCounterAgent",
    "BASELINES",
]

#: How valuable it is to stop each pending action (used for challenge/block EV).
_ACTION_DENY_VALUE = {
    "TAX": 1.6,
    "ASSASSINATE": 2.2,
    "STEAL": 1.4,
    "EXCHANGE": 0.7,
    "FOREIGN_AID": 0.9,
}


class RandomAgent(Agent):
    """Uniform random legal actions."""

    name = "random"

    def _main_action(self, view: ObsView) -> int:
        return self._random_legal(view)

    def _challenge_action(self, view: ObsView) -> int:
        return self._random_legal(view)

    def _block_action(self, view: ObsView) -> int:
        return self._random_legal(view)


class HonestAgent(Agent):
    """Truthful play: only claims and blocks roles it actually holds."""

    name = "honest"

    # -- main action --------------------------------------------------- #
    def _main_action(self, view: ObsView) -> int:
        spec = view.spec
        if view.must_coup or view.coins >= 7:
            coups = view.ids_of_kind("COUP")
            if coups:
                return self._best_target(view, coups)
        if view.has(Role.DUKE):
            return spec.tax
        if view.has(Role.ASSASSIN) and view.coins >= 3:
            ids = view.ids_of_kind("ASSASSINATE")
            if ids:
                return self._best_target(view, ids)
        if view.has(Role.CAPTAIN):
            ids = view.ids_of_kind("STEAL")
            if ids:
                target = self._richest_target(view, ids)
                rel = view.target_rel(target)
                if rel is not None and view.coins_of(rel) >= 1:
                    return target
        if view.has(Role.AMBASSADOR) and view.coins <= 5:
            return spec.exchange
        if view.coins < 7:
            return spec.foreign_aid
        return spec.income

    # -- challenge ----------------------------------------------------- #
    def _challenge_action(self, view: ObsView) -> int:
        if view.claimed_impossible:
            return int(view.challenge_id)
        claimant = view.claimant_rel()
        if claimant is None:
            return int(view.pass_id)
        caught = int(view.claims[claimant][5])
        proven = int(view.claims[claimant][6])
        if caught >= 2 and proven == 0:
            return int(view.challenge_id)
        return int(view.pass_id)

    # -- block --------------------------------------------------------- #
    def _block_action(self, view: ObsView) -> int:
        for action_id in view.block_ids():
            role = view.spec.decode(action_id).role
            if role is not None and view.has(role):
                return int(action_id)
        return int(view.pass_id)


class AggressiveBlufferAgent(Agent):
    """Claims roles it does not hold, blocks often, challenges often."""

    name = "aggressive_bluffer"

    def _main_action(self, view: ObsView) -> int:
        spec = view.spec
        roll = float(self.rng.random())
        if view.must_coup or view.coins >= 7:
            coups = view.ids_of_kind("COUP")
            if coups:
                return self._best_target(view, coups)
        ids = view.ids_of_kind("ASSASSINATE")
        if ids and roll < 0.85:
            return self._best_target(view, ids)
        ids = view.ids_of_kind("STEAL")
        if ids and roll < 0.70:
            return self._richest_target(view, ids)
        if roll < 0.90:
            return spec.tax  # bluffs the Duke most of the time
        if roll < 0.95 and view.action_mask[spec.exchange]:
            return spec.exchange
        return spec.foreign_aid

    def _challenge_action(self, view: ObsView) -> int:
        if view.claimed_impossible:
            return int(view.challenge_id)
        roll = float(self.rng.random())
        threshold = 0.60 if view.seat.influence > 1 else 0.45
        if roll < threshold:
            return int(view.challenge_id)
        return int(view.pass_id)

    def _block_action(self, view: ObsView) -> int:
        blocks = view.block_ids()
        if not blocks:
            return int(view.pass_id)
        # prefer a truthful block when possible
        for action_id in blocks:
            role = view.spec.decode(action_id).role
            if role is not None and view.has(role):
                return int(action_id)
        roll = float(self.rng.random())
        threshold = 0.65 if view.seat.influence > 1 else 0.35
        if roll < threshold:
            return int(blocks[0])
        return int(view.pass_id)


class CautiousSkepticalAgent(Agent):
    """Truthful and risk-averse; challenges only with strong evidence."""

    name = "cautious_skeptical"

    def _main_action(self, view: ObsView) -> int:
        spec = view.spec
        if view.must_coup or view.coins >= 7:
            coups = view.ids_of_kind("COUP")
            if coups:
                return self._best_target(view, coups)
        if view.has(Role.DUKE):
            return spec.tax
        if view.has(Role.ASSASSIN) and view.coins >= 3:
            ids = view.ids_of_kind("ASSASSINATE")
            if ids:
                return self._best_target(view, ids)
        if view.has(Role.CAPTAIN):
            ids = view.ids_of_kind("STEAL")
            if ids:
                target = self._richest_target(view, ids)
                rel = view.target_rel(target)
                if rel is not None and view.coins_of(rel) >= 1:
                    return target
        if view.coins <= 3:
            return spec.income
        if view.has(Role.AMBASSADOR) and view.coins <= 6:
            return spec.exchange
        return spec.income

    def _challenge_action(self, view: ObsView) -> int:
        if view.claimed_impossible:
            return int(view.challenge_id)
        role = view.claimed_role
        claimant = view.claimant_rel()
        if role is None or claimant is None:
            return int(view.pass_id)
        influence = view.seats[claimant].influence
        p_true = self._p_holds(view, role, influence)
        caught = int(view.claims[claimant][5])
        proven = int(view.claims[claimant][6])
        suspicion = 0.0
        if caught > proven:
            suspicion += 0.15 * (caught - proven)
        threshold = 0.80 - min(suspicion, 0.3)
        if 1.0 - p_true >= threshold:
            return int(view.challenge_id)
        return int(view.pass_id)

    def _block_action(self, view: ObsView) -> int:
        for action_id in view.block_ids():
            role = view.spec.decode(action_id).role
            if role is not None and view.has(role):
                return int(action_id)
        return int(view.pass_id)

    # -- card counting -------------------------------------------------- #
    @staticmethod
    def _p_holds(view: ObsView, role: Role, hand_size: int) -> float:
        unseen = int(view.unseen[int(role)])
        total = int(view.unseen.sum())
        if hand_size <= 0 or total <= 0:
            return 0.0
        if unseen <= 0:
            return 0.0
        if total - unseen < hand_size:
            return 1.0
        return 1.0 - math.comb(total - unseen, hand_size) / math.comb(total, hand_size)


class CardCounterAgent(Agent):
    """Exact combinatorial card counting with expected-value decisions.

    The agent maintains a hypergeometric model of the unseen cards
    (``3 - own_hand - total_revealed`` copies of each role) and turns it into
    ``P(claimant holds the claimed role)``.  That card evidence is combined with
    a Beta(1, 3) bluff-rate prior - updated by the observed ``caught_bluffing``
    and ``proven_truthful`` counters - to obtain ``P(the claim is a bluff)``.
    Challenges are then taken when that posterior clears the risk-adjusted bar
    set by the influence at stake.
    """

    name = "card_counter"

    #: Pseudo-counts of the bluff-rate prior (truthful claims are the norm).
    PRIOR_CAUGHT = 1.0
    PRIOR_PROVEN = 3.0
    #: Prior probability that one opponent challenges an unproven claim.
    CHALLENGE_PRIOR = 0.50
    #: Prior probability that one opponent blocks an unblocked action.
    BLOCK_PRIOR = 0.30
    #: Coin-unit utilities of the main-turn options (1.0 == one coin).
    UTIL_INCOME = 1.0
    UTIL_FOREIGN_AID = 2.0
    UTIL_TAX = 3.0
    UTIL_COUP = 6.0
    UTIL_ASSASSINATE = 3.0
    UTIL_STEAL_BASE = 0.5
    UTIL_EXCHANGE_WEAK = 1.2
    UTIL_EXCHANGE = 0.6

    def __init__(self, seed: int | None = None, challenge_bar: float = 0.62) -> None:
        super().__init__(seed)
        self.challenge_bar = float(challenge_bar)

    # -- probability model ---------------------------------------------- #
    @staticmethod
    def _p_holds(view: ObsView, role: Role, hand_size: int) -> float:
        """P(a player holding ``hand_size`` unseen cards has >=1 ``role``)."""
        unseen = int(view.unseen[int(role)])
        total = int(view.unseen.sum())
        if hand_size <= 0 or total <= 0 or unseen <= 0:
            return 0.0
        if total - unseen < hand_size:
            return 1.0
        return 1.0 - math.comb(total - unseen, hand_size) / math.comb(total, hand_size)

    def _bluff_rate(self, view: ObsView, relative_seat: int) -> float:
        """Posterior bluff rate of a seat given its observed track record."""
        caught = float(view.claims[relative_seat][5])
        proven = float(view.claims[relative_seat][6])
        return (caught + self.PRIOR_CAUGHT) / (
            caught + proven + self.PRIOR_CAUGHT + self.PRIOR_PROVEN
        )

    def _p_bluff(self, view: ObsView, role: Role, relative_seat: int) -> float:
        """P(the pending claim is a bluff), from card evidence x reputation."""
        p_card = self._p_holds(view, role, view.seats[relative_seat].influence)
        p_lacks = max(0.0, 1.0 - p_card)
        if p_lacks <= 0.0:
            return 0.0
        beta = min(max(self._bluff_rate(view, relative_seat), 1e-3), 1.0 - 1e-3)
        odds = (beta / (1.0 - beta)) * (p_lacks / max(p_card, 1e-3))
        return odds / (1.0 + odds)

    def _influence_value(self, view: ObsView) -> float:
        """Marginal value of keeping one influence (threshold units)."""
        return 1.0 if view.seat.influence > 1 else 3.0

    def _bluff_cost(self, view: ObsView) -> float:
        """Cost of a caught bluff in coin units (1 coin == one income turn)."""
        return 5.0 if view.seat.influence > 1 else 12.0

    def _deny_value(self, view: ObsView) -> float:
        kind = None if view.pending_action is None else view.pending_action.name
        return _ACTION_DENY_VALUE.get(kind or "", 1.0)

    def _p_challenged(self, view: ObsView) -> float:
        opponents = max(view.num_alive - 1, 1)
        return 1.0 - (1.0 - self.CHALLENGE_PRIOR) ** opponents

    def _p_blocked(self, view: ObsView) -> float:
        opponents = max(view.num_alive - 1, 1)
        return 1.0 - (1.0 - self.BLOCK_PRIOR) ** opponents

    # -- decisions ------------------------------------------------------ #
    def _main_action(self, view: ObsView) -> int:
        spec = view.spec
        bluff_cost = self._bluff_cost(view)
        p_ch = self._p_challenged(view)
        p_block = self._p_blocked(view)
        candidates: list[tuple[float, int]] = []

        def add(utility: float, action_id: int, role: Role | None = None) -> None:
            if not view.action_mask[action_id]:
                return
            if role is not None and not view.has(role):
                utility -= p_ch * bluff_cost  # expected cost of a caught bluff
            candidates.append((utility, int(action_id)))

        # Utilities are in coin units (1.0 == one coin), except the influence
        # denial terms which are priced at ``COIN_PER_INFLUENCE``.
        add(self.UTIL_INCOME, spec.income)
        add(self.UTIL_FOREIGN_AID * (1.0 - p_block), spec.foreign_aid)
        add(self.UTIL_TAX, spec.tax, Role.DUKE)

        coups = view.ids_of_kind("COUP")
        if coups:
            add(self.UTIL_COUP, self._best_target(view, coups))
        assassinate = view.ids_of_kind("ASSASSINATE")
        if assassinate:
            add(
                self.UTIL_ASSASSINATE,
                self._best_target(view, assassinate),
                Role.ASSASSIN,
            )
        steals = view.ids_of_kind("STEAL")
        if steals:
            target = self._richest_target(view, steals)
            rel = view.target_rel(target)
            coins = view.coins_of(rel) if rel is not None else 0
            add(self.UTIL_STEAL_BASE + min(coins, 2), target, Role.CAPTAIN)
        weak_hand = not (view.has(Role.DUKE) or view.has(Role.CONTESSA))
        add(
            self.UTIL_EXCHANGE_WEAK if weak_hand else self.UTIL_EXCHANGE,
            spec.exchange,
            Role.AMBASSADOR,
        )

        if not candidates:  # pragma: no cover - defensive
            return self._random_legal(view)
        return max(candidates, key=lambda item: item[0])[1]

    def _challenge_action(self, view: ObsView) -> int:
        role = view.claimed_role
        claimant = view.claimant_rel()
        if role is None or claimant is None:  # pragma: no cover - defensive
            return int(view.pass_id)
        p_bluff = self._p_bluff(view, role, claimant)
        bar = self.challenge_bar
        if view.seats[claimant].influence <= 1:
            bar -= 0.10  # catching them removes them from the game
        if view.seat.influence <= 1:
            bar += 0.12  # our own last influence is precious
        if p_bluff >= bar:
            return int(view.challenge_id)
        return int(view.pass_id)

    def _block_action(self, view: ObsView) -> int:
        blocks = view.block_ids()
        if not blocks:  # pragma: no cover - defensive
            return int(view.pass_id)
        for action_id in blocks:
            role = view.spec.decode(action_id).role
            if role is not None and view.has(role):
                return int(action_id)
        # A bluff block only pays if a challenge is unlikely: being caught costs
        # an influence *and* lets the action through.
        saved = self._deny_value(view)
        loss = self._influence_value(view)
        p_ch = self._p_challenged(view)
        if (1.0 - p_ch) * saved - p_ch * (loss + saved) > 0.45:
            return int(blocks[0])
        return int(view.pass_id)


BASELINES = (
    RandomAgent,
    HonestAgent,
    AggressiveBlufferAgent,
    CautiousSkepticalAgent,
    CardCounterAgent,
)
