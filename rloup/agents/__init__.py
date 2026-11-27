"""Scripted baselines plus the two learning agents.

``base`` / ``baselines`` hold the five scripted opponents.  ``ppo`` adds the
action-masked actor-critic used by PPO (and by the PFSP opponent pool), and
``nfsp`` adds the action-masked Q / supervised networks of NFSP.
"""

from .base import Agent, ObsView, SeatView, parse_obs
from .baselines import (
    BASELINES,
    AggressiveBlufferAgent,
    CardCounterAgent,
    CautiousSkepticalAgent,
    HonestAgent,
    RandomAgent,
)
from .dqn import DoubleDQNAgent, DoubleDQNHyperparams
from .nfsp import NFSPAgent, QNetwork, SLNetwork
from .ppo import MaskedActorCritic, PPOAgent, PerAgentRolloutBuffer
from .slm import (
    LlamaServerManager,
    SLMAgent,
    SLMConfig,
    discover_local_gguf,
    list_local_ggufs,
)

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
    "MaskedActorCritic",
    "PPOAgent",
    "PerAgentRolloutBuffer",
    "NFSPAgent",
    "QNetwork",
    "SLNetwork",
    "DoubleDQNAgent",
    "DoubleDQNHyperparams",
    "SLMAgent",
    "SLMConfig",
    "LlamaServerManager",
    "discover_local_gguf",
    "list_local_ggufs",
]
