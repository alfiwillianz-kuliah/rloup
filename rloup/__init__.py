"""rloup - Coup reinforcement learning environment.

Public surface:

``rloup.CoupEngine``      pure rules engine (fast, dependency-light)
``rloup.CoupAECEnv``      PettingZoo AEC wrapper
``rloup.constants``       roles, phases, action space layout
``rloup.agents``          scripted baselines (Random .. CardCounter)
"""

from .constants import (
    ActionSpaceSpec,
    ActionType,
    Phase,
    Role,
    num_actions_for,
    players_from_num_actions,
)
from .engine import CoupEngine, ObsLayout, make_obs_layout, obs_dim
from .pet_env import CoupAECEnv, env, raw_env

__version__ = "0.1.0"

__all__ = [
    "ActionSpaceSpec",
    "ActionType",
    "CoupAECEnv",
    "CoupEngine",
    "ObsLayout",
    "Phase",
    "Role",
    "env",
    "make_obs_layout",
    "num_actions_for",
    "obs_dim",
    "players_from_num_actions",
    "raw_env",
]
