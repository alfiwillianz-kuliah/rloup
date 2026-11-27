"""Training stack for rloup: masked PPO, action-masked NFSP and PFSP opponents.

Public surface::

    TrainConfig, PPOTrainer          vectorised batched-inference PPO
    NFSPTrainer, NFSPConfig          action-masked NFSP for 2-player Coup
    OpponentPool, update_elo         PFSP opponent pool and Elo book-keeping
    evaluate_vs_baselines            the shared seat-balanced benchmark

Run a trainer from the command line with::

    python -m rloup.training.train_ppo --num-players 2 --mode pfsp --save-dir results/ppo_pfsp_2p
    python -m rloup.training.train_nfsp --num-players 2 --save-dir results/nfsp_2p
"""

from __future__ import annotations

from .metrics import METRIC_KEYS, MetricsWriter
from .pool import OpponentPool, OpponentSpec, update_elo
from .vec import (
    STRONG_BASELINES,
    BehaviourStats,
    EvaluationResult,
    evaluate_vs_baselines,
)

__all__ = [
    "METRIC_KEYS",
    "MetricsWriter",
    "OpponentPool",
    "OpponentSpec",
    "STRONG_BASELINES",
    "BehaviourStats",
    "EvaluationResult",
    "evaluate_vs_baselines",
    "update_elo",
]


def __getattr__(name: str):
    """Lazy re-export of the trainers (they pull in torch)."""
    if name in ("PPOTrainer", "TrainConfig", "EnvSlot"):
        from . import train_ppo

        return getattr(train_ppo, name)
    if name in ("NFSPTrainer", "NFSPConfig"):
        from . import train_nfsp

        return getattr(train_nfsp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
