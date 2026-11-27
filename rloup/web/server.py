"""Zero-dependency HTTP server powering the rloup Web Arena & Tournament Dashboard.

Run locally::

    python -m rloup.web --host 127.0.0.1 --port 8765

Features
--------
* Auto-discovers trained RL checkpoints (PPO / Double-DQN / NFSP / exploiters)
  and scripted baselines — safe to expose publicly (no local LLM subprocesses
  or outbound HTTP calls).
* Interactive 2-player and 4-player Coup Arena: play against any mix of bots or
  spectate AI-vs-AI with live RL action probabilities and value / Q estimates.
* Live 2-player round-robin and 4-player table tournaments running in background
  worker threads with streaming progress, Bradley-Terry Elo, exact 95% Wilson
  intervals, head-to-head heatmaps, and cancellation.
* Lab Benchmarks viewer serving ``results/eval_summary.json``,
  ``results/slm_eval.json`` (static benchmark comparison), and
  ``results/figures/*.png``.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
import json
import mimetypes
from pathlib import Path
import threading
import time
from typing import Any, Sequence
import urllib.parse
import uuid

import numpy as np

from rloup.agents import (
    Agent,
    AggressiveBlufferAgent,
    CardCounterAgent,
    CautiousSkepticalAgent,
    DoubleDQNAgent,
    HonestAgent,
    NFSPAgent,
    PPOAgent,
    RandomAgent,
)
from rloup.constants import (
    ACTION_NAMES,
    ROLE_NAMES,
    Phase,
    Role,
    players_from_num_actions,
)
from rloup.engine import CoupEngine
from rloup.eval.evaluate import (
    bradley_terry_elo,
    finish_rank,
    wilson_interval,
)
from rloup.play import describe_action, load_policy

__all__ = ["WebApp", "create_server", "main"]

STATIC_DIR = Path(__file__).resolve().parent / "static"

_SCRIPTED_REGISTRY: tuple[tuple[str, str, str, type[Agent]], ...] = (
    (
        "baseline:card_counter",
        "CardCounter",
        "Exact hypergeometric card counter + EV heuristic",
        CardCounterAgent,
    ),
    (
        "baseline:honest",
        "Honest",
        "Never bluffs; challenges impossible claims & repeat liars",
        HonestAgent,
    ),
    (
        "baseline:cautious",
        "Cautious",
        "Truthful & risk-averse; high evidence bar for challenges",
        CautiousSkepticalAgent,
    ),
    (
        "baseline:bluffer",
        "Bluffer",
        "Aggressive bluffer; claims Duke and blocks/challenges freely",
        AggressiveBlufferAgent,
    ),
    (
        "baseline:random",
        "Random",
        "Uniform random over legal actions (Elo 1000 anchor)",
        RandomAgent,
    ),
)

_KNOWN_RL_META: dict[str, tuple[str, str]] = {
    "results/ppo_pfsp_2p/final.pt": (
        "PFSP-PPO (2P)",
        "2M steps masked PPO with Prioritized Fictitious Self-Play",
    ),
    "results/ddqn_pfsp_2p/final.pt": (
        "Double-DQN (2P)",
        "1M steps Action-Masked Double DQN with PFSP opponent pool",
    ),
    "results/ppo_naive_2p/final.pt": (
        "Naive-PPO (2P)",
        "2M steps pure self-play masked PPO ablation",
    ),
    "results/nfsp_2p/final.pt": (
        "NFSP (2P)",
        "1M steps Neural Fictitious Self-Play average policy",
    ),
    "results/ppo_pfsp_4p/final.pt": (
        "PFSP-PPO (4P)",
        "2M steps 4-player masked PPO + PFSP (93.4% impossible-claim detection)",
    ),
    "results/ddqn_pfsp_4p/final.pt": (
        "Double-DQN (4P)",
        "1M steps 4-player Action-Masked Double DQN with PFSP opponent pool",
    ),
    "results/exploiters/vs_pfsp_2p/final.pt": (
        "Exploiter vs PFSP (2P)",
        "500k steps PPO best-response exploiter trained vs PFSP-PPO",
    ),
    "results/exploiters/vs_ddqn_2p/final.pt": (
        "Exploiter vs Double-DQN (2P)",
        "500k steps PPO best-response exploiter trained vs Double-DQN",
    ),
    "results/exploiters/vs_naive_2p/final.pt": (
        "Exploiter vs Naive (2P)",
        "500k steps PPO best-response exploiter trained vs Naive-PPO",
    ),
    "results/exploiters/vs_nfsp_2p/final.pt": (
        "Exploiter vs NFSP (2P)",
        "500k steps PPO best-response exploiter trained vs NFSP",
    ),
}


# --------------------------------------------------------------------------- #
# Game & tournament state
# --------------------------------------------------------------------------- #


@dataclass
class GameSession:
    game_id: str
    engine: CoupEngine
    seat_specs: list[dict[str, Any]]
    agents: dict[int, Agent]
    human_seats: set[int]
    reveal_ai_hands: bool = False
    show_inspector: bool = True
    seed: int = 0
    created_at: float = field(default_factory=time.time)
    log: list[dict[str, Any]] = field(default_factory=list)
    last_inspector: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class TournamentJob:
    tournament_id: str
    mode: str  # "round_robin_2p" | "table_4p"
    bot_ids: list[str]
    bot_names: dict[str, str]
    bot_categories: dict[str, str]
    games_per_pair: int
    seed: int
    total_games: int
    status: str = "running"  # running | completed | cancelled | failed
    error: str | None = None
    completed_games: int = 0
    current_matchup: str = "initializing"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pairs: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)
    four_p_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    behaviour: dict[str, dict[str, int]] = field(default_factory=dict)
    recent_matches: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# WebApp core
# --------------------------------------------------------------------------- #


class WebApp:
    """Stateful controller for RL/baseline models, games, tournaments, and benchmarks."""

    def __init__(self, root: str | Path = ".", *, device: str = "auto") -> None:
        self.root = Path(root).resolve()
        self.device = self._resolve_device(device)
        self._policy_cache: dict[str, Agent] = {}
        self._policy_lock = threading.Lock()
        self._games: dict[str, GameSession] = {}
        self._tournaments: dict[str, TournamentJob] = {}
        self._catalog_cache: dict[str, dict[str, Any]] = {}
        self._elo_hints: dict[str, float] = {}
        self._load_elo_hints()
        self.refresh_catalog()
        atexit.register(self.close)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return str(device)
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def close(self) -> None:
        for job in list(self._tournaments.values()):
            job.cancel_event.set()

    def _load_elo_hints(self) -> None:
        summary_path = self.root / "results" / "eval_summary.json"
        if not summary_path.exists():
            return
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            elo = data.get("round_robin_2p", {}).get("elo", {})
            mapping = {
                "PFSP-PPO": "rl:ppo_pfsp_2p",
                "Double-DQN": "rl:ddqn_pfsp_2p",
                "Naive-PPO": "rl:ppo_naive_2p",
                "NFSP": "rl:nfsp_2p",
                "CardCounter": "baseline:card_counter",
                "Honest": "baseline:honest",
                "Cautious": "baseline:cautious",
                "Bluffer": "baseline:bluffer",
                "Random": "baseline:random",
            }
            for disp, bid in mapping.items():
                if disp in elo:
                    self._elo_hints[bid] = round(float(elo[disp]), 1)
        except Exception:
            pass

    # -- model catalog (RL checkpoints + scripted baselines only) -------- #
    def refresh_catalog(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = [
            {
                "id": "human",
                "name": "Human (You)",
                "category": "human",
                "players": [2, 3, 4, 5, 6],
                "description": "Play interactively from your browser",
                "elo": None,
                "badge": "YOU",
            }
        ]

        # 1. Trained RL checkpoints
        rl_paths: list[Path] = []
        results_dir = self.root / "results"
        if results_dir.exists():
            for rel in _KNOWN_RL_META:
                p = self.root / rel
                if p.is_file():
                    rl_paths.append(p)
            for p in sorted(results_dir.glob("*/final.pt")):
                if p not in rl_paths:
                    rl_paths.append(p)
            for p in sorted(results_dir.glob("exploiters/*/final.pt")):
                if p not in rl_paths:
                    rl_paths.append(p)

        import torch

        for path in rl_paths:
            rel_str = path.relative_to(self.root).as_posix()
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                fmt = str(payload.get("format", ""))
                n_actions = int(payload.get("n_actions", 24))
                n_players = players_from_num_actions(n_actions)
                if "ddqn" in fmt:
                    algo = "DDQN"
                elif "nfsp" in fmt:
                    algo = "NFSP"
                else:
                    algo = "PPO"
            except Exception:
                continue
            parent_key = path.parent.name
            if "exploiters" in path.parts:
                bot_id = f"rl:exploiter_{parent_key}"
            else:
                bot_id = f"rl:{parent_key}"
            name, desc = _KNOWN_RL_META.get(
                rel_str,
                (f"{parent_key} ({n_players}P)", f"Trained {algo} checkpoint ({rel_str})"),
            )
            entries.append(
                {
                    "id": bot_id,
                    "name": name,
                    "category": "rl",
                    "algo": algo,
                    "players": [n_players],
                    "checkpoint": rel_str,
                    "description": desc,
                    "elo": self._elo_hints.get(bot_id),
                    "badge": f"{algo} {n_players}P",
                }
            )

        # 2. Scripted baselines
        for bot_id, name, desc, _ in _SCRIPTED_REGISTRY:
            entries.append(
                {
                    "id": bot_id,
                    "name": name,
                    "category": "baseline",
                    "players": [2, 3, 4, 5, 6],
                    "description": desc,
                    "elo": self._elo_hints.get(bot_id),
                    "badge": "BOT",
                }
            )

        self._catalog_cache = {entry["id"]: entry for entry in entries}
        return entries

    def list_models(self) -> dict[str, Any]:
        models = self.refresh_catalog()
        return {
            "device": self.device,
            "models": models,
        }

    def _get_spec(self, bot_id: str) -> dict[str, Any]:
        spec = self._catalog_cache.get(bot_id)
        if spec is None:
            self.refresh_catalog()
            spec = self._catalog_cache.get(bot_id)
        if spec is None:
            raise ValueError(f"Unknown model id: {bot_id!r}")
        return spec

    def _instantiate_agent(
        self, bot_id: str, *, num_players: int, seed: int = 0
    ) -> Agent:
        spec = self._get_spec(bot_id)
        if num_players not in spec["players"]:
            raise ValueError(
                f"{spec['name']} supports {spec['players']} players, not {num_players}"
            )
        cat = spec["category"]
        if cat == "baseline":
            for bid, _, _, cls in _SCRIPTED_REGISTRY:
                if bid == bot_id:
                    return cls(seed=seed)
            raise ValueError(f"Unknown baseline {bot_id!r}")
        if cat == "rl":
            ckpt = str(self.root / spec["checkpoint"])
            with self._policy_lock:
                agent = self._policy_cache.get(ckpt)
                if agent is None:
                    agent = load_policy(ckpt, device=self.device)
                    self._policy_cache[ckpt] = agent
            return agent
        raise ValueError(f"Cannot instantiate agent for category {cat!r}")

    # -- interactive game arena ------------------------------------------ #
    def create_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        seats_in = payload.get("seats")
        if not isinstance(seats_in, list) or len(seats_in) not in (2, 3, 4, 5, 6):
            raise ValueError("seats must be a list of 2..6 bot ids")
        num_players = len(seats_in)
        seed = int(payload.get("seed", int(time.time() * 1000) % 100_000))
        reveal_ai_hands = bool(payload.get("reveal_ai_hands", False))
        show_inspector = bool(payload.get("show_inspector", True))

        seat_specs: list[dict[str, Any]] = []
        human_seats: set[int] = set()
        agents: dict[int, Agent] = {}

        for idx, bot_id in enumerate(seats_in):
            spec = self._get_spec(str(bot_id))
            if num_players not in spec["players"]:
                raise ValueError(
                    f"Seat {idx} ({spec['name']}) does not support {num_players}-player games"
                )
            seat_specs.append(spec)
            if spec["category"] == "human":
                human_seats.add(idx)
            else:
                agents[idx] = self._instantiate_agent(
                    str(bot_id), num_players=num_players, seed=seed + idx * 97
                )

        engine = CoupEngine(num_players, max_turns=200, seed=seed)
        engine.reset(seed=seed)

        game_id = uuid.uuid4().hex[:12]
        session = GameSession(
            game_id=game_id,
            engine=engine,
            seat_specs=seat_specs,
            agents=agents,
            human_seats=human_seats,
            reveal_ai_hands=reveal_ai_hands and not human_seats,
            show_inspector=show_inspector,
            seed=seed,
        )
        self._games[game_id] = session
        if len(self._games) > 64:
            oldest = sorted(self._games.values(), key=lambda g: g.created_at)[:-32]
            for old in oldest:
                self._games.pop(old.game_id, None)

        if payload.get("auto_advance", False) and human_seats:
            self._advance_until_human_or_done(session)

        return self.serialize_game(session)

    def get_game(self, game_id: str) -> dict[str, Any]:
        session = self._games.get(game_id)
        if session is None:
            raise KeyError(f"Unknown game_id {game_id!r}")
        return self.serialize_game(session)

    def submit_human_action(
        self, game_id: str, action_id: int, *, auto_advance: bool = False
    ) -> dict[str, Any]:
        session = self._games.get(game_id)
        if session is None:
            raise KeyError(f"Unknown game_id {game_id!r}")
        with session.lock:
            engine = session.engine
            if engine.done:
                raise ValueError("Game is already finished")
            seat = engine.current_player
            if seat not in session.human_seats:
                raise ValueError(f"Seat {seat} is controlled by AI, not human")
            mask = engine.legal_mask()
            aid = int(action_id)
            if not (0 <= aid < len(mask) and mask[aid]):
                raise ValueError(f"Action {aid} is not legal in current state")
            self._apply_and_record(session, seat, aid, inspector=None)
            if auto_advance:
                self._advance_until_human_or_done(session)
            return self.serialize_game(session)

    def step_ai(self, game_id: str, *, reveal_ai_hands: bool | None = None) -> dict[str, Any]:
        session = self._games.get(game_id)
        if session is None:
            raise KeyError(f"Unknown game_id {game_id!r}")
        with session.lock:
            if reveal_ai_hands is not None and not session.human_seats:
                session.reveal_ai_hands = bool(reveal_ai_hands)
            engine = session.engine
            if engine.done:
                return self.serialize_game(session)
            seat = engine.current_player
            if seat in session.human_seats:
                return self.serialize_game(session)
            self._step_single_ai(session)
            return self.serialize_game(session)

    def _advance_until_human_or_done(self, session: GameSession) -> None:
        guard = 0
        while not session.engine.done and session.engine.current_player not in session.human_seats:
            self._step_single_ai(session)
            guard += 1
            if guard > 400:
                break

    def _inspect_agent(
        self, engine: CoupEngine, agent: Agent, seat: int, obs: np.ndarray, mask: np.ndarray
    ) -> dict[str, Any] | None:
        probs: np.ndarray | None = None
        value: float | None = None
        value_label = "Critic Value V(s)"
        if isinstance(agent, PPOAgent):
            import torch

            probs = np.asarray(agent.action_probs(obs, mask), dtype=np.float64)
            with torch.inference_mode():
                t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                value = float(agent.actor_critic.value(t).item())
        elif isinstance(agent, NFSPAgent):
            probs = np.asarray(agent.action_probs(obs, mask), dtype=np.float64)
        elif isinstance(agent, DoubleDQNAgent):
            probs = np.asarray(agent.action_probs(obs, mask), dtype=np.float64)
            value = agent.max_q(obs, mask)
            value_label = "Max Q(s, a)"
        if probs is None:
            return None
        order = np.argsort(probs)[::-1][:5]
        top = [
            {
                "action_id": int(a),
                "label": describe_action(engine, int(a)),
                "prob": round(float(probs[a]), 4),
            }
            for a in order
            if probs[a] > 1e-4
        ]
        return {
            "kind": "neural",
            "seat": seat,
            "name": f"Seat {seat}",
            "value": round(value, 3) if value is not None else None,
            "value_label": value_label,
            "top_actions": top,
        }

    def _step_single_ai(self, session: GameSession) -> None:
        engine = session.engine
        seat = engine.current_player
        agent = session.agents[seat]
        obs = engine.get_obs(seat)
        mask = engine.legal_mask()

        inspector: dict[str, Any] | None = None
        if session.show_inspector and isinstance(agent, (PPOAgent, NFSPAgent, DoubleDQNAgent)):
            inspector = self._inspect_agent(engine, agent, seat, obs, mask)
            if inspector is not None:
                inspector["name"] = session.seat_specs[seat]["name"]

        t0 = time.monotonic()
        action = int(agent.select_action(obs, mask))
        latency_ms = (time.monotonic() - t0) * 1000.0

        if session.show_inspector and inspector is None:
            inspector = {
                "kind": "scripted",
                "seat": seat,
                "name": session.seat_specs[seat]["name"],
                "note": session.seat_specs[seat].get("description", "Scripted policy"),
                "latency_ms": round(latency_ms, 2),
            }

        self._apply_and_record(session, seat, action, inspector=inspector)

    def _apply_and_record(
        self,
        session: GameSession,
        seat: int,
        action_id: int,
        *,
        inspector: dict[str, Any] | None,
    ) -> None:
        engine = session.engine
        phase_before = Phase(engine.phase).name
        turn_before = int(engine.turn)
        info = engine.spec.decode(action_id)
        desc = describe_action(engine, action_id)

        claimed_role: Role | None = None
        if info.kind == "TAX":
            claimed_role = Role.DUKE
        elif info.kind == "ASSASSINATE":
            claimed_role = Role.ASSASSIN
        elif info.kind == "STEAL":
            claimed_role = Role.CAPTAIN
        elif info.kind == "EXCHANGE":
            claimed_role = Role.AMBASSADOR
        elif info.kind == "BLOCK":
            claimed_role = info.role
        is_bluff = (
            claimed_role is not None
            and int(claimed_role) not in engine.players[seat].hand
        )

        public_desc = desc
        if info.kind == "EXCHANGE_KEEP" and seat not in session.human_seats:
            public_desc = "completed EXCHANGE (kept cards hidden)"

        entry = {
            "index": len(session.log) + 1,
            "turn": turn_before,
            "phase": phase_before,
            "seat": seat,
            "actor_name": session.seat_specs[seat]["name"],
            "action_id": int(action_id),
            "kind": info.kind,
            "text": public_desc,
            "full_text": desc,
            "is_bluff": bool(is_bluff),
            "inspector": inspector,
        }
        session.log.append(entry)
        if inspector is not None:
            session.last_inspector = inspector
        engine.step(action_id)

    def serialize_game(self, session: GameSession) -> dict[str, Any]:
        engine = session.engine
        done = bool(engine.done)
        can_reveal_all = done or (not session.human_seats and session.reveal_ai_hands)
        cur = int(engine.current_player)
        human_turn = (not done) and (cur in session.human_seats)

        seats_out: list[dict[str, Any]] = []
        ep_stats = engine.get_episode_stats()["players"]
        for s in range(engine.num_players):
            p = engine.players[s]
            spec = session.seat_specs[s]
            show_hand = can_reveal_all or (s in session.human_seats)
            hand = (
                [str(ROLE_NAMES[int(c)]) for c in p.hand]
                if show_hand
                else ["HIDDEN"] * len(p.hand)
            )
            revealed = [str(ROLE_NAMES[int(c)]) for c in p.revealed]
            seat_obj: dict[str, Any] = {
                "seat": s,
                "bot_id": spec["id"],
                "name": spec["name"],
                "category": spec["category"],
                "badge": spec.get("badge", ""),
                "alive": bool(p.alive),
                "coins": int(p.coins),
                "influence": int(p.influence),
                "hand": hand,
                "revealed": revealed,
                "is_current": (not done) and (s == cur),
                "is_human": s in session.human_seats,
                "caught_bluffing": int(ep_stats[s]["caught_bluffing"]),
                "claims_proven": int(ep_stats[s]["claims_proven"]),
            }
            if done:
                seat_obj["bluff_rate"] = round(float(ep_stats[s]["bluff_rate"]), 3)
                seat_obj["challenges_attempted"] = int(ep_stats[s]["challenges_attempted"])
                seat_obj["challenges_won"] = int(ep_stats[s]["challenges_won"])
            seats_out.append(seat_obj)

        legal_actions: list[dict[str, Any]] = []
        if human_turn:
            mask = engine.legal_mask()
            hand_set = set(engine.players[cur].hand)
            for aid in np.flatnonzero(mask):
                aid_int = int(aid)
                info = engine.spec.decode(aid_int)
                target_seat = (
                    (cur + int(info.rel_off)) % engine.num_players
                    if info.rel_off is not None
                    else None
                )
                role_needed: Role | None = None
                if info.kind == "TAX":
                    role_needed = Role.DUKE
                elif info.kind == "ASSASSINATE":
                    role_needed = Role.ASSASSIN
                elif info.kind == "STEAL":
                    role_needed = Role.CAPTAIN
                elif info.kind == "EXCHANGE":
                    role_needed = Role.AMBASSADOR
                elif info.kind == "BLOCK":
                    role_needed = info.role
                truthful = None if role_needed is None else (int(role_needed) in hand_set)
                legal_actions.append(
                    {
                        "action_id": aid_int,
                        "kind": info.kind,
                        "label": describe_action(engine, aid_int),
                        "role": str(ROLE_NAMES[int(role_needed)])
                        if role_needed is not None
                        else None,
                        "truthful": truthful,
                        "target_seat": target_seat,
                    }
                )

        viewer = cur if human_turn else (next(iter(session.human_seats), cur))
        claimed_role = engine._claimed_role()
        claimed_impossible = bool(
            claimed_role is not None
            and not done
            and engine._role_exhausted_for(viewer, claimed_role)
        )
        pending = {
            "actor": engine.pending_actor,
            "target": engine.pending_target,
            "action": ACTION_NAMES[int(engine.pending_action)]
            if engine.pending_action is not None
            else None,
            "blocker": engine.pending_blocker,
            "block_role": str(ROLE_NAMES[int(engine.pending_block_role)])
            if engine.pending_block_role is not None
            else None,
            "claimed_role": str(ROLE_NAMES[int(claimed_role)])
            if claimed_role is not None
            else None,
            "claimed_impossible": claimed_impossible,
        }

        log_out: list[dict[str, Any]] = []
        for entry in session.log[-80:]:
            log_out.append(
                {
                    "index": entry["index"],
                    "turn": entry["turn"],
                    "phase": entry["phase"],
                    "seat": entry["seat"],
                    "actor_name": entry["actor_name"],
                    "kind": entry["kind"],
                    "text": entry["full_text"] if can_reveal_all else entry["text"],
                    "is_bluff": entry["is_bluff"] if can_reveal_all else None,
                    "inspector": entry["inspector"] if session.show_inspector else None,
                }
            )

        winner = None if engine.truncated else engine.winner()
        return {
            "game_id": session.game_id,
            "num_players": engine.num_players,
            "turn": int(engine.turn),
            "phase": Phase(engine.phase).name,
            "done": done,
            "truncated": bool(engine.truncated),
            "winner": winner,
            "winner_name": session.seat_specs[winner]["name"]
            if winner is not None
            else None,
            "current_player": cur,
            "human_turn": human_turn,
            "has_human": bool(session.human_seats),
            "reveal_ai_hands": bool(session.reveal_ai_hands),
            "show_inspector": bool(session.show_inspector),
            "deck_size": int(engine.deck_size()),
            "pending": pending,
            "seats": seats_out,
            "legal_actions": legal_actions,
            "log": log_out,
            "last_inspector": session.last_inspector if session.show_inspector else None,
        }

    # -- live tournaments ------------------------------------------------ #
    def start_tournament(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "round_robin_2p"))
        if mode not in ("round_robin_2p", "table_4p"):
            raise ValueError(f"Unsupported tournament mode: {mode!r}")

        bot_ids = [str(b) for b in (payload.get("bots") or [])]
        if any(b == "human" for b in bot_ids):
            raise ValueError("Human cannot be entered into an automated tournament")

        seed = int(payload.get("seed", 42))
        bot_names: dict[str, str] = {}
        bot_categories: dict[str, str] = {}
        req_players = 2 if mode == "round_robin_2p" else 4

        if mode == "round_robin_2p":
            bot_ids = list(dict.fromkeys(bot_ids))
            if len(bot_ids) < 2:
                raise ValueError("Select at least 2 bots for a 2P round-robin tournament")
        else:
            if len(bot_ids) != 4:
                raise ValueError("Select exactly 4 seats for a 4-player table championship")

        for idx, bid in enumerate(bot_ids):
            spec = self._get_spec(bid)
            if req_players not in spec["players"]:
                raise ValueError(
                    f"{spec['name']} does not support {req_players}-player tournaments"
                )
            key = bid if mode == "round_robin_2p" else f"{bid}#{idx}"
            bot_names[key] = (
                spec["name"]
                if mode == "round_robin_2p"
                else f"Seat {idx + 1}: {spec['name']}"
            )
            bot_categories[key] = spec["category"]

        games_per_pair = max(2, min(int(payload.get("games_per_pair", 40)), 400))
        if mode == "round_robin_2p":
            if games_per_pair % 2 == 1:
                games_per_pair += 1
            num_pairs = len(bot_ids) * (len(bot_ids) - 1) // 2
            total_games = num_pairs * games_per_pair
            keys = list(bot_ids)
        else:
            games_per_pair = max(4, ((games_per_pair + 3) // 4) * 4)
            total_games = games_per_pair
            keys = [f"{bid}#{idx}" for idx, bid in enumerate(bot_ids)]

        tid = uuid.uuid4().hex[:10]
        job = TournamentJob(
            tournament_id=tid,
            mode=mode,
            bot_ids=bot_ids,
            bot_names=bot_names,
            bot_categories=bot_categories,
            games_per_pair=games_per_pair,
            seed=seed,
            total_games=total_games,
            behaviour={
                k: {"claims": 0, "bluffs": 0, "ch_att": 0, "ch_won": 0} for k in keys
            },
            four_p_stats={
                k: {"games": 0, "wins": 0, "rank_sum": 0, "ranks": {1: 0, 2: 0, 3: 0, 4: 0}}
                for k in keys
            },
        )
        self._tournaments[tid] = job

        worker = threading.Thread(
            target=self._run_tournament_worker,
            args=(job,),
            daemon=True,
        )
        worker.start()
        return self.serialize_tournament(job)

    def get_tournament(self, tournament_id: str) -> dict[str, Any]:
        job = self._tournaments.get(tournament_id)
        if job is None:
            raise KeyError(f"Unknown tournament_id {tournament_id!r}")
        return self.serialize_tournament(job)

    def cancel_tournament(self, tournament_id: str) -> dict[str, Any]:
        job = self._tournaments.get(tournament_id)
        if job is None:
            raise KeyError(f"Unknown tournament_id {tournament_id!r}")
        job.cancel_event.set()
        return self.serialize_tournament(job)

    def list_tournaments(self) -> list[dict[str, Any]]:
        jobs = sorted(
            self._tournaments.values(), key=lambda j: j.started_at, reverse=True
        )[:10]
        return [self.serialize_tournament(j) for j in jobs]

    def _run_tournament_worker(self, job: TournamentJob) -> None:
        try:
            if job.mode == "round_robin_2p":
                self._run_2p_round_robin(job)
            else:
                self._run_4p_table(job)
            with job.lock:
                job.status = "cancelled" if job.cancel_event.is_set() else "completed"
                job.finished_at = time.time()
                job.current_matchup = "finished" if job.status == "completed" else "cancelled"
        except Exception as exc:
            with job.lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = time.time()

    def _record_behaviour(
        self, job: TournamentJob, key: str, player_stats: dict[str, Any]
    ) -> None:
        b = job.behaviour[key]
        b["claims"] += sum(int(v) for v in player_stats["claims"].values())
        b["bluffs"] += sum(int(v) for v in player_stats["bluffs"].values())
        b["ch_att"] += int(player_stats["challenges_attempted"])
        b["ch_won"] += int(player_stats["challenges_won"])

    def _run_2p_round_robin(self, job: TournamentJob) -> None:
        shared_cache: dict[str, Agent] = {}

        def get_agent(bid: str, seed: int) -> Agent:
            if job.bot_categories[bid] == "baseline":
                return self._instantiate_agent(bid, num_players=2, seed=seed)
            if bid not in shared_cache:
                shared_cache[bid] = self._instantiate_agent(bid, num_players=2, seed=seed)
            return shared_cache[bid]

        pair_list = list(combinations(job.bot_ids, 2))
        game_counter = 0

        for pair_idx, (a_id, b_id) in enumerate(pair_list):
            if job.cancel_event.is_set():
                return
            with job.lock:
                job.current_matchup = f"{job.bot_names[a_id]} vs {job.bot_names[b_id]}"

            half = job.games_per_pair // 2
            chunk_size = 24
            a_fixed = (
                get_agent(a_id, job.seed + pair_idx * 101)
                if job.bot_categories[a_id] != "baseline"
                else None
            )
            b_fixed = (
                get_agent(b_id, job.seed + pair_idx * 101 + 1)
                if job.bot_categories[b_id] != "baseline"
                else None
            )

            key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            with job.lock:
                job.pairs.setdefault(
                    key, {"games": 0, "a_wins": 0, "b_wins": 0, "draws": 0}
                )

            schedule = [(a_id, b_id)] * half + [(b_id, a_id)] * (
                job.games_per_pair - half
            )
            for start in range(0, len(schedule), chunk_size):
                if job.cancel_event.is_set():
                    return
                batch = schedule[start : start + chunk_size]
                engines: list[CoupEngine] = []
                seat_agents: list[list[Agent]] = []
                for k, (s0, s1) in enumerate(batch):
                    gseed = job.seed * 7_919 + game_counter + k
                    eng = CoupEngine(2, max_turns=200, seed=gseed)
                    eng.reset(seed=gseed)
                    engines.append(eng)
                    ag0 = (
                        a_fixed
                        if s0 == a_id and a_fixed is not None
                        else (
                            b_fixed
                            if s0 == b_id and b_fixed is not None
                            else get_agent(s0, gseed + 1)
                        )
                    )
                    ag1 = (
                        a_fixed
                        if s1 == a_id and a_fixed is not None
                        else (
                            b_fixed
                            if s1 == b_id and b_fixed is not None
                            else get_agent(s1, gseed + 2)
                        )
                    )
                    seat_agents.append([ag0, ag1])

                while any(not e.done for e in engines):
                    if job.cancel_event.is_set():
                        return
                    neural_buckets: dict[int, tuple[Agent, list[tuple[int, np.ndarray, np.ndarray]]]] = {}
                    actions = [0] * len(engines)

                    for i, eng in enumerate(engines):
                        if eng.done:
                            continue
                        seat = eng.current_player
                        ag = seat_agents[i][seat]
                        obs = eng.get_obs(seat)
                        mask = eng.legal_mask()
                        if isinstance(ag, (PPOAgent, NFSPAgent, DoubleDQNAgent)):
                            neural_buckets.setdefault(id(ag), (ag, []))[1].append(
                                (i, obs, mask)
                            )
                        else:
                            actions[i] = int(ag.select_action(obs, mask))

                    for ag, rows in neural_buckets.values():
                        obs_b = np.stack([r[1] for r in rows])
                        mask_b = np.stack([r[2] for r in rows])
                        if isinstance(ag, PPOAgent):
                            chosen = [int(a) for a in ag.batched_sample(obs_b, mask_b)]
                        else:
                            chosen = [int(a) for a in ag.batched_actions(obs_b, mask_b)]
                        for (i, _, _), act in zip(rows, chosen):
                            actions[i] = act

                    for i, eng in enumerate(engines):
                        if not eng.done:
                            eng.step(actions[i])

                with job.lock:
                    rec = job.pairs[key]
                    for k, eng in enumerate(engines):
                        s0, s1 = batch[k]
                        winner = None if eng.truncated else eng.winner()
                        rec["games"] += 1
                        if winner is None:
                            rec["draws"] += 1
                            win_name = "Draw"
                        else:
                            win_id = s0 if winner == 0 else s1
                            win_name = job.bot_names[win_id]
                            if win_id == key[0]:
                                rec["a_wins"] += 1
                            else:
                                rec["b_wins"] += 1
                        ep = eng.get_episode_stats()["players"]
                        self._record_behaviour(job, s0, ep[0])
                        self._record_behaviour(job, s1, ep[1])
                        job.completed_games += 1
                        job.recent_matches.append(
                            {
                                "game": job.completed_games,
                                "matchup": f"{job.bot_names[s0]} vs {job.bot_names[s1]}",
                                "winner": win_name,
                                "turns": int(eng.turn),
                            }
                        )
                    job.recent_matches = job.recent_matches[-25:]
                game_counter += len(batch)

    def _run_4p_table(self, job: TournamentJob) -> None:
        keys = [f"{bid}#{idx}" for idx, bid in enumerate(job.bot_ids)]
        fixed_agents: dict[str, Agent] = {}
        for idx, bid in enumerate(job.bot_ids):
            k = keys[idx]
            if job.bot_categories[k] != "baseline":
                fixed_agents[k] = self._instantiate_agent(
                    bid, num_players=4, seed=job.seed + idx * 31
                )

        with job.lock:
            job.current_matchup = "4P Seat-Rotated Table"

        chunk_size = 24
        for start in range(0, job.total_games, chunk_size):
            if job.cancel_event.is_set():
                return
            count = min(chunk_size, job.total_games - start)
            engines: list[CoupEngine] = []
            seat_keys: list[list[str]] = []
            seat_agents: list[list[Agent]] = []
            for k in range(count):
                gidx = start + k
                rot = gidx % 4
                rotated_keys = keys[rot:] + keys[:rot]
                rotated_bids = job.bot_ids[rot:] + job.bot_ids[:rot]
                gseed = job.seed * 7_919 + gidx
                eng = CoupEngine(4, max_turns=200, seed=gseed)
                eng.reset(seed=gseed)
                engines.append(eng)
                seat_keys.append(rotated_keys)
                ags: list[Agent] = []
                for s, (rk, rbid) in enumerate(zip(rotated_keys, rotated_bids)):
                    if rk in fixed_agents:
                        ags.append(fixed_agents[rk])
                    else:
                        ags.append(
                            self._instantiate_agent(rbid, num_players=4, seed=gseed + s)
                        )
                seat_agents.append(ags)

            while any(not e.done for e in engines):
                if job.cancel_event.is_set():
                    return
                neural_buckets: dict[int, tuple[Agent, list[tuple[int, np.ndarray, np.ndarray]]]] = {}
                actions = [0] * len(engines)
                for i, eng in enumerate(engines):
                    if eng.done:
                        continue
                    seat = eng.current_player
                    ag = seat_agents[i][seat]
                    obs = eng.get_obs(seat)
                    mask = eng.legal_mask()
                    if isinstance(ag, (PPOAgent, NFSPAgent, DoubleDQNAgent)):
                        neural_buckets.setdefault(id(ag), (ag, []))[1].append(
                            (i, obs, mask)
                        )
                    else:
                        actions[i] = int(ag.select_action(obs, mask))
                for ag, rows in neural_buckets.values():
                    obs_b = np.stack([r[1] for r in rows])
                    mask_b = np.stack([r[2] for r in rows])
                    if isinstance(ag, PPOAgent):
                        chosen = [int(a) for a in ag.batched_sample(obs_b, mask_b)]
                    else:
                        chosen = [int(a) for a in ag.batched_actions(obs_b, mask_b)]
                    for (i, _, _), act in zip(rows, chosen):
                        actions[i] = act
                for i, eng in enumerate(engines):
                    if not eng.done:
                        eng.step(actions[i])

            with job.lock:
                for k, eng in enumerate(engines):
                    rkeys = seat_keys[k]
                    winner = None if eng.truncated else eng.winner()
                    elim = [int(s) for s in eng._elim_order]
                    ep = eng.get_episode_stats()["players"]
                    for s, rk in enumerate(rkeys):
                        st = job.four_p_stats[rk]
                        st["games"] += 1
                        if winner == s:
                            st["wins"] += 1
                        place = finish_rank(elim, winner, s, 4)
                        if place is not None:
                            st["rank_sum"] += place
                            st["ranks"][place] += 1
                        self._record_behaviour(job, rk, ep[s])
                    job.completed_games += 1
                    job.recent_matches.append(
                        {
                            "game": job.completed_games,
                            "matchup": "4P Table",
                            "winner": job.bot_names[rkeys[winner]]
                            if winner is not None
                            else "Truncated",
                            "turns": int(eng.turn),
                        }
                    )
                job.recent_matches = job.recent_matches[-25:]

    def serialize_tournament(self, job: TournamentJob) -> dict[str, Any]:
        with job.lock:
            now = job.finished_at or time.time()
            elapsed = max(now - job.started_at, 1e-6)
            rate = job.completed_games / elapsed
            remaining = max(0, job.total_games - job.completed_games)
            eta = (remaining / rate) if (rate > 0 and job.status == "running") else 0.0

            out: dict[str, Any] = {
                "tournament_id": job.tournament_id,
                "mode": job.mode,
                "status": job.status,
                "error": job.error,
                "games_per_pair": job.games_per_pair,
                "completed_games": job.completed_games,
                "total_games": job.total_games,
                "progress": (job.completed_games / job.total_games)
                if job.total_games
                else 1.0,
                "current_matchup": job.current_matchup,
                "elapsed_s": round(elapsed, 2),
                "games_per_s": round(rate, 2),
                "eta_s": round(eta, 1),
                "recent_matches": list(job.recent_matches),
            }

            if job.mode == "round_robin_2p":
                ids = list(job.bot_ids)
                anchor = "baseline:random" if "baseline:random" in ids else ids[-1]
                elo_map = bradley_terry_elo(ids, job.pairs, anchor=anchor)
                size = len(ids)
                matrix: list[list[dict[str, Any] | None]] = [
                    [None for _ in range(size)] for _ in range(size)
                ]
                totals: dict[str, dict[str, int]] = {
                    bid: {"wins": 0, "losses": 0, "draws": 0, "games": 0}
                    for bid in ids
                }
                for (a, b), rec in job.pairs.items():
                    i, j = ids.index(a), ids.index(b)
                    g = rec["games"]
                    if g > 0:
                        la, ha = wilson_interval(rec["a_wins"], g)
                        lb, hb = wilson_interval(rec["b_wins"], g)
                        matrix[i][j] = {
                            "rate": round(rec["a_wins"] / g, 4),
                            "wins": rec["a_wins"],
                            "losses": rec["b_wins"],
                            "draws": rec["draws"],
                            "games": g,
                            "ci_low": round(la, 4),
                            "ci_high": round(ha, 4),
                        }
                        matrix[j][i] = {
                            "rate": round(rec["b_wins"] / g, 4),
                            "wins": rec["b_wins"],
                            "losses": rec["a_wins"],
                            "draws": rec["draws"],
                            "games": g,
                            "ci_low": round(lb, 4),
                            "ci_high": round(hb, 4),
                        }
                        totals[a]["wins"] += rec["a_wins"]
                        totals[a]["losses"] += rec["b_wins"]
                        totals[a]["draws"] += rec["draws"]
                        totals[a]["games"] += g
                        totals[b]["wins"] += rec["b_wins"]
                        totals[b]["losses"] += rec["a_wins"]
                        totals[b]["draws"] += rec["draws"]
                        totals[b]["games"] += g

                standings = []
                for bid in ids:
                    t = totals[bid]
                    b = job.behaviour[bid]
                    ci_l, ci_h = wilson_interval(t["wins"], t["games"])
                    standings.append(
                        {
                            "id": bid,
                            "name": job.bot_names[bid],
                            "category": job.bot_categories[bid],
                            "elo": round(float(elo_map.get(bid, 1000.0)), 1),
                            "wins": t["wins"],
                            "losses": t["losses"],
                            "draws": t["draws"],
                            "games": t["games"],
                            "win_rate": round(t["wins"] / t["games"], 4)
                            if t["games"]
                            else 0.0,
                            "ci_low": round(ci_l, 4),
                            "ci_high": round(ci_h, 4),
                            "bluff_rate": round(b["bluffs"] / b["claims"], 4)
                            if b["claims"]
                            else 0.0,
                            "challenge_precision": round(b["ch_won"] / b["ch_att"], 4)
                            if b["ch_att"]
                            else 0.0,
                        }
                    )
                standings.sort(key=lambda r: (r["elo"], r["win_rate"]), reverse=True)
                out["bots"] = [
                    {"id": bid, "name": job.bot_names[bid], "category": job.bot_categories[bid]}
                    for bid in ids
                ]
                out["standings"] = standings
                out["matrix"] = matrix
                out["anchor"] = job.bot_names[anchor]
            else:
                keys = [f"{bid}#{idx}" for idx, bid in enumerate(job.bot_ids)]
                standings = []
                for k in keys:
                    st = job.four_p_stats[k]
                    b = job.behaviour[k]
                    g = st["games"]
                    ranked = sum(st["ranks"].values())
                    ci_l, ci_h = wilson_interval(st["wins"], g)
                    standings.append(
                        {
                            "id": k,
                            "name": job.bot_names[k],
                            "category": job.bot_categories[k],
                            "wins": st["wins"],
                            "games": g,
                            "win_rate": round(st["wins"] / g, 4) if g else 0.0,
                            "ci_low": round(ci_l, 4),
                            "ci_high": round(ci_h, 4),
                            "avg_rank": round(st["rank_sum"] / ranked, 2)
                            if ranked
                            else 0.0,
                            "ranks": dict(st["ranks"]),
                            "bluff_rate": round(b["bluffs"] / b["claims"], 4)
                            if b["claims"]
                            else 0.0,
                            "challenge_precision": round(b["ch_won"] / b["ch_att"], 4)
                            if b["ch_att"]
                            else 0.0,
                        }
                    )
                standings.sort(key=lambda r: (r["win_rate"], -r["avg_rank"]), reverse=True)
                out["standings"] = standings
            return out

    # -- saved assignment benchmarks (static read-only JSON & figures) --- #
    def get_benchmarks(self) -> dict[str, Any]:
        eval_path = self.root / "results" / "eval_summary.json"
        slm_path = self.root / "results" / "slm_eval.json"
        fig_dir = self.root / "results" / "figures"
        figures = (
            sorted(p.name for p in fig_dir.glob("*.png"))
            if fig_dir.exists()
            else []
        )
        return {
            "eval_summary": json.loads(eval_path.read_text(encoding="utf-8"))
            if eval_path.exists()
            else None,
            "slm_eval": json.loads(slm_path.read_text(encoding="utf-8"))
            if slm_path.exists()
            else None,
            "figures": figures,
        }


# --------------------------------------------------------------------------- #
# HTTP request handler
# --------------------------------------------------------------------------- #


def _make_handler(app: WebApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(data).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            mime, _ = mimetypes.guess_type(str(path))
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            obj = json.loads(raw) if raw.strip() else {}
            if not isinstance(obj, dict):
                raise ValueError("JSON body must be an object")
            return obj

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            try:
                if path == "/" or path == "/index.html":
                    self._send_file(STATIC_DIR / "index.html")
                    return
                if path.startswith("/static/"):
                    rel = path[len("/static/") :]
                    target = (STATIC_DIR / rel).resolve()
                    if not target.is_relative_to(STATIC_DIR.resolve()):
                        self._send_json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
                        return
                    self._send_file(target)
                    return
                if path.startswith("/figures/"):
                    rel = path[len("/figures/") :]
                    fig_root = (app.root / "results" / "figures").resolve()
                    target = (fig_root / rel).resolve()
                    if not target.is_relative_to(fig_root):
                        self._send_json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
                        return
                    self._send_file(target)
                    return
                if path == "/api/models":
                    self._send_json(app.list_models())
                    return
                if path == "/api/benchmarks":
                    self._send_json(app.get_benchmarks())
                    return
                if path == "/api/tournaments":
                    self._send_json({"tournaments": app.list_tournaments()})
                    return
                if path.startswith("/api/game/"):
                    game_id = path[len("/api/game/") :].strip("/")
                    self._send_json(app.get_game(game_id))
                    return
                if path.startswith("/api/tournament/"):
                    tid = path[len("/api/tournament/") :].strip("/")
                    self._send_json(app.get_tournament(tid))
                    return
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except KeyError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json_body()
                if path == "/api/game/new":
                    self._send_json(app.create_game(body))
                    return
                if path.startswith("/api/game/") and path.endswith("/action"):
                    game_id = path[len("/api/game/") : -len("/action")].strip("/")
                    if "action_id" not in body:
                        raise ValueError("missing action_id")
                    self._send_json(
                        app.submit_human_action(
                            game_id,
                            int(body["action_id"]),
                            auto_advance=bool(body.get("auto_advance", False)),
                        )
                    )
                    return
                if path.startswith("/api/game/") and path.endswith("/step"):
                    game_id = path[len("/api/game/") : -len("/step")].strip("/")
                    reveal = body.get("reveal_ai_hands")
                    self._send_json(
                        app.step_ai(
                            game_id,
                            reveal_ai_hands=None if reveal is None else bool(reveal),
                        )
                    )
                    return
                if path == "/api/tournament/start":
                    self._send_json(app.start_tournament(body))
                    return
                if path.startswith("/api/tournament/") and path.endswith("/cancel"):
                    tid = path[len("/api/tournament/") : -len("/cancel")].strip("/")
                    self._send_json(app.cancel_tournament(tid))
                    return
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except KeyError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return Handler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    root: str | Path = ".",
    device: str = "auto",
    app: WebApp | None = None,
) -> tuple[ThreadingHTTPServer, WebApp]:
    web_app = app or WebApp(root=root, device=device)
    server = ThreadingHTTPServer((host, int(port)), _make_handler(web_app))
    server.daemon_threads = True
    return server, web_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rloup.web",
        description="Launch the rloup Coup Web Arena & Tournament Dashboard.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server, app = create_server(args.host, args.port, device=args.device)
    actual_port = server.server_address[1]
    print(
        f"rloup Web Arena running at http://{args.host}:{actual_port}  "
        f"(device={app.device})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down rloup Web Arena...", flush=True)
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
