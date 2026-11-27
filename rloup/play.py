"""Interactive command-line Coup against the trained rloup agents.

Examples
--------
Play a 2-player game against the final PFSP-PPO policy::

    python -m rloup.play

Inspect the policy's action probabilities and critic value while playing::

    python -m rloup.play --show-probs

Face the NFSP average policy instead, or a scripted card counter::

    python -m rloup.play --opponent nfsp
    python -m rloup.play --opponent card_counter

Spectate or run a one-game narrated transcript (no input needed)::

    python -m rloup.play --ai-vs-ai
    python -m rloup.play --demo --num-players 4

You always sit in seat 0.  Only your own hand is ever displayed while the game
is running; every opponent's revealed (dead) cards, coin counts and influence
are public information and are always shown.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from rloup.agents import Agent
from rloup.agents.baselines import CardCounterAgent
from rloup.agents.dqn import DoubleDQNAgent
from rloup.agents.nfsp import NFSPAgent
from rloup.agents.ppo import PPOAgent
from rloup.agents.slm import SLMAgent, SLMConfig
from rloup.constants import (
    EXCHANGE_KEEP_OPTIONS,
    ROLE_NAMES,
    Phase,
    num_actions_for,
    players_from_num_actions,
)
from rloup.engine import CoupEngine

__all__ = [
    "DEFAULT_CHECKPOINTS",
    "OPPONENT_CHECKPOINTS",
    "main",
    "describe_action",
    "load_policy",
    "numbered_actions",
]

#: Default model file per seat count.
DEFAULT_CHECKPOINTS: dict[int, str] = {
    2: "results/ppo_pfsp_2p/final.pt",
    4: "results/ppo_pfsp_4p/final.pt",
}

#: ``--opponent`` family -> seat count -> checkpoint.
OPPONENT_CHECKPOINTS: dict[str, dict[int, str]] = {
    "pfsp": {2: "results/ppo_pfsp_2p/final.pt", 4: "results/ppo_pfsp_4p/final.pt"},
    "naive": {2: "results/ppo_naive_2p/final.pt", 4: "results/ppo_naive_2p/final.pt"},
    "nfsp": {2: "results/nfsp_2p/final.pt", 4: "results/nfsp_2p/final.pt"},
    "ddqn": {2: "results/ddqn_pfsp_2p/final.pt", 4: "results/ddqn_pfsp_4p/final.pt"},
}

OPPONENT_NAMES: dict[str, str] = {
    "pfsp": "PFSP-PPO",
    "naive": "Naive-PPO",
    "nfsp": "NFSP",
    "ddqn": "Double-DQN",
    "card_counter": "CardCounter",
    "slm": "SLM",
}

RULE = "=" * 72
THIN = "-" * 72


def load_policy(path: str | Path, device: str = "cpu") -> Agent:
    """Load a PPO, NFSP, or Double DQN checkpoint, dispatching on format."""
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    fmt = payload.get("format")
    if fmt == "rloup.ppo.v1":
        return PPOAgent.load(path, device=device)
    if fmt == "rloup.nfsp.v1":
        return NFSPAgent.load(path, device=device)
    if fmt == "rloup.ddqn.v1":
        return DoubleDQNAgent.load(path, device=device)
    raise ValueError(f"{path} is not a rloup checkpoint (format={fmt!r})")


def _role_name(card: int) -> str:
    return str(ROLE_NAMES[int(card)])


def describe_action(engine: CoupEngine, action_id: int) -> str:
    """Human readable description of a flat action id in the current state."""
    spec = engine.spec
    info = spec.decode(int(action_id))
    kind = info.kind
    if kind == "INCOME":
        return "INCOME (+1 coin)"
    if kind == "FOREIGN_AID":
        return "FOREIGN_AID (+2 coins, blockable by DUKE)"
    if kind == "TAX":
        return "TAX (+3 coins, claims DUKE)"
    if kind == "EXCHANGE":
        return "EXCHANGE (draw 2, claims AMBASSADOR)"
    if kind in ("COUP", "ASSASSINATE", "STEAL"):
        target = (engine.current_player + int(info.rel_off)) % engine.num_players
        label = {
            "COUP": "COUP (costs 7, unavoidable)",
            "ASSASSINATE": "ASSASSINATE (costs 3, claims ASSASSIN)",
            "STEAL": "STEAL 2 coins (claims CAPTAIN)",
        }[kind]
        return f"{label} -> seat {target}"
    if kind == "BLOCK":
        return f"BLOCK with {_role_name(info.role)}"
    if kind == "LOSE_SLOT":
        slot = int(info.index)
        hand = engine.players[engine.current_player].hand
        card = _role_name(hand[slot]) if slot < len(hand) else "?"
        return f"reveal hand slot {slot + 1} ({card})"
    if kind == "EXCHANGE_KEEP":
        buffer = engine._exchange_buffer or []
        keep = EXCHANGE_KEEP_OPTIONS[int(info.index)]
        cards = ", ".join(_role_name(buffer[k]) for k in keep if k < len(buffer))
        return f"keep [{cards}]"
    if kind == "PASS":
        return "PASS"
    if kind == "CHALLENGE":
        return "CHALLENGE"
    return kind  # pragma: no cover - defensive


def numbered_actions(engine: CoupEngine) -> list[tuple[int, str]]:
    """``(action_id, description)`` for every legal action, in id order."""
    mask = engine.legal_mask()
    return [
        (int(action_id), describe_action(engine, int(action_id)))
        for action_id in np.flatnonzero(mask)
    ]


def _action_probs(agent: Agent, obs: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    getter = getattr(agent, "action_probs", None)
    if getter is None:
        return None
    try:
        return np.asarray(getter(obs, mask), dtype=np.float64)
    except Exception:  # pragma: no cover - scripted agents simply have no probs
        return None


def _value(agent: Agent, obs: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    if isinstance(agent, DoubleDQNAgent) and mask is not None:
        return agent.max_q(obs, mask)
    if not isinstance(agent, PPOAgent):
        return None
    import torch

    with torch.inference_mode():
        tensor = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), device=agent.device
        ).unsqueeze(0)
        return float(agent.actor_critic.value(tensor).item())


class CoupCLI:
    """One interactive or narrated Coup game."""

    def __init__(
        self,
        engine: CoupEngine,
        agents: dict[int, Agent],
        names: list[str],
        *,
        human_seat: int | None,
        show_probs: bool = False,
        out: Any = print,
    ) -> None:
        self.engine = engine
        self.agents = agents
        self.names = names
        self.human_seat = human_seat
        self.show_probs = show_probs
        self.out = out
        self.history: list[str] = []

    # -- display -------------------------------------------------------- #
    def _seat_label(self, seat: int) -> str:
        return self.names[seat]

    def render_state(self, heading: str | None = None, compact: bool = False) -> None:
        engine = self.engine
        seat = engine.current_player
        self.out("")
        self.out(RULE)
        first = heading or f"Turn {engine.turn} | phase {Phase(engine.phase).name}"
        self.out(f" {first} | to act: {self._seat_label(seat)}")
        self.out(RULE)
        for s in range(engine.num_players):
            player = engine.players[s]
            marker = "->" if s == seat else "  "
            if self.human_seat is not None and s == self.human_seat:
                hand = ", ".join(_role_name(c) for c in player.hand) or "-"
            elif not player.alive:
                hand = "(out)"
            else:
                hand = "??"
            revealed = ", ".join(_role_name(c) for c in player.revealed) or "-"
            self.out(
                f" {marker} seat {s} {self._seat_label(s):<22s} "
                f"{player.coins:2d} coins | influence {player.influence} | "
                f"hand [{hand}] | revealed [{revealed}]"
            )
        self.out(f"      deck: {engine.deck_size()} cards")
        if self.history and not compact:
            self.out(THIN)
            self.out(" recent:")
            for line in self.history[-5:]:
                self.out(f"   {line}")

    def render_legal(self) -> list[tuple[int, str]]:
        options = numbered_actions(self.engine)
        self.out(THIN)
        for index, (action_id, description) in enumerate(options, start=1):
            self.out(f"  {index:2d}. {description}")
        return options

    def _narration(self, seat: int, action_id: int) -> str:
        return (
            f"t{self.engine.turn:<3d} seat {seat} "
            f"{self._seat_label(seat):<22s} {describe_action(self.engine, action_id)}"
        )

    def show_policy(self, seat: int, obs: np.ndarray, mask: np.ndarray) -> None:
        agent = self.agents[seat]
        probs = _action_probs(agent, obs, mask)
        value = _value(agent, obs, mask)
        if probs is None:
            self.out(f"      [{self.names[seat]}] scripted policy (no distribution)")
            return
        order = np.argsort(probs)[::-1][:3]
        parts = [
            f"{describe_action(self.engine, int(a))}={probs[a]:.2f}"
            for a in order
            if probs[a] > 1e-4
        ]
        value_text = f" | value {value:+.2f}" if value is not None else ""
        self.out(f"      [{self.names[seat]}] " + "  ".join(parts) + value_text)

    # -- interaction ----------------------------------------------------- #
    def ask_human(self, options: Sequence[tuple[int, str]]) -> int:
        while True:
            try:
                raw = input("  choice> ").strip()
            except EOFError:  # pragma: no cover - non-interactive stdin
                self.out("  (stdin closed - taking the first legal action)")
                return options[0][0]
            if raw in ("q", "quit", "exit"):
                raise KeyboardInterrupt
            if not raw:
                continue
            try:
                index = int(raw)
            except ValueError:
                self.out("  please type the number of an action")
                continue
            if 1 <= index <= len(options):
                return options[index - 1][0]
            self.out(f"  pick a number between 1 and {len(options)}")

    # -- main loop -------------------------------------------------------- #
    def play(self, interactive: bool, narrate_turns: bool = False) -> dict[str, Any]:
        engine = self.engine
        turn_marker = -1
        while not engine.done:
            seat = engine.current_player
            human_turn = (
                interactive and self.human_seat is not None and seat == self.human_seat
            )
            if narrate_turns and engine.turn != turn_marker:
                turn_marker = engine.turn
                self.render_state(compact=True)
            obs = engine.get_obs(seat)
            mask = engine.legal_mask()
            if human_turn:
                self.render_state()
                options = self.render_legal()
                action = self.ask_human(options)
            else:
                agent = self.agents[seat]
                if self.show_probs and not isinstance(agent, SLMAgent):
                    self.show_policy(seat, obs, mask)
                action = int(
                    agent.select_action(obs, mask, engine=None, player_id=seat)
                )
                if self.show_probs and isinstance(agent, SLMAgent) and agent.last_reason:
                    self.out(f"      [{self.names[seat]}] {agent.last_reason}")
            self.history.append(self._narration(seat, action))
            if narrate_turns:
                self.out(f"  -> {self.history[-1]}")
            engine.step(action)
        return self.report()

    # -- result ---------------------------------------------------------- #
    def report(self) -> dict[str, Any]:
        engine = self.engine
        winner = engine.winner()
        stats = engine.get_episode_stats()
        self.out("")
        self.out(RULE)
        if engine.truncated:
            self.out(" Game truncated (turn limit reached)")
        elif winner is not None:
            self.out(f" Winner: {self._seat_label(winner)}")
        self.out(RULE)
        for seat in range(engine.num_players):
            player = engine.players[seat]
            hand = ", ".join(_role_name(c) for c in player.hand) or "-"
            revealed = ", ".join(_role_name(c) for c in player.revealed) or "-"
            telemetry = stats["players"][seat]
            self.out(
                f" seat {seat} {self._seat_label(seat):<22s} "
                f"coins {player.coins:2d} | hand [{hand}] | revealed [{revealed}] | "
                f"bluff {telemetry['bluff_rate']:.2f} | "
                f"challenges {telemetry['challenges_attempted']} "
                f"({telemetry['challenges_won']} won)"
            )
        self.out(f" turns: {engine.turn} | deck left: {engine.deck_size()}")
        return {
            "winner": winner,
            "truncated": bool(engine.truncated),
            "turns": int(engine.turn),
            "rewards": [float(r) for r in engine.rewards],
            "stats": stats,
        }


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #
def _resolve_opponent_source(
    args: argparse.Namespace,
) -> tuple[Callable[[], Agent], str, Callable[[], None]]:
    """Return ``(agent_factory, display_name, cleanup_fn)`` for non-human seats."""
    noop: Callable[[], None] = lambda: None
    family = str(args.opponent)
    if args.checkpoint:
        path = Path(args.checkpoint)
        name = OPPONENT_NAMES.get(family, family)
    elif family == "card_counter":
        return (lambda: CardCounterAgent(seed=args.seed), "CardCounter", noop)
    elif family == "slm":
        cfg = SLMConfig(
            base_url=args.slm_url,
            model=args.slm_model,
            temperature=float(args.slm_temperature),
            include_reason=bool(args.show_probs or args.slm_reasoning),
            max_tokens=128 if (args.show_probs or args.slm_reasoning) else 24,
            parallel_slots=2,
        )
        shared = SLMAgent(config=cfg, seed=args.seed)
        shared.ensure_ready()
        label = f"SLM({shared.model_label})" if shared.model_label != "SLM" else "SLM"
        return (lambda: shared, label, shared.close)
    else:
        path = Path(OPPONENT_CHECKPOINTS[family][args.num_players])
        name = OPPONENT_NAMES[family]
    if not path.exists():
        if args.demo:
            print(
                f"warning: {path} not found, using the scripted card counter for the demo",
                file=sys.stderr,
                flush=True,
            )
            return (lambda: CardCounterAgent(seed=args.seed), "CardCounter", noop)
        raise SystemExit(
            f"checkpoint {path} not found - train it first (see README) or pass --opponent card_counter"
        )
    template = load_policy(path, device=args.device)
    expected_actions = num_actions_for(args.num_players)
    if getattr(template, "n_actions", expected_actions) != expected_actions:
        raise SystemExit(
            f"{path} was trained for {players_from_num_actions(template.n_actions)} "
            f"players but --num-players {args.num_players} was requested"
        )
    return (lambda: template, name, noop)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rloup.play",
        description="Play Coup against the trained rloup agents.",
    )
    parser.add_argument("--checkpoint", default=None, help="explicit policy checkpoint")
    parser.add_argument(
        "--num-players", type=int, choices=(2, 4), default=2, help="seat count"
    )
    parser.add_argument(
        "--opponent",
        choices=("pfsp", "naive", "nfsp", "ddqn", "card_counter", "slm"),
        default="pfsp",
        help="which trained policy (or scripted / local SLM bot) fills the other seats",
    )
    parser.add_argument(
        "--slm-model",
        default=None,
        help="path to a local .gguf model for --opponent slm (auto-discovers if omitted)",
    )
    parser.add_argument(
        "--slm-url",
        default=None,
        help="existing OpenAI-compatible /v1 endpoint for --opponent slm (llama-server/Ollama/vLLM)",
    )
    parser.add_argument(
        "--slm-reasoning",
        action="store_true",
        help="ask the SLM for a brief rationale before picking its action",
    )
    parser.add_argument(
        "--slm-temperature",
        type=float,
        default=0.2,
        help="sampling temperature for --opponent slm",
    )
    parser.add_argument(
        "--show-probs",
        action="store_true",
        help="print the AI action distribution and critic value (or SLM rationale)",
    )
    parser.add_argument("--ai-vs-ai", action="store_true", help="spectate: no human seat")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="non-interactive one-game narrated transcript for quick testing",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent_factory, opponent_name, cleanup = _resolve_opponent_source(args)
    try:
        engine = CoupEngine(int(args.num_players), max_turns=int(args.max_turns), seed=args.seed)
        engine.reset(seed=args.seed)

        spectate = bool(args.ai_vs_ai or args.demo)
        human_seat = None if spectate else 0
        agents: dict[int, Agent] = {}
        names: list[str] = []
        for seat in range(engine.num_players):
            if human_seat is not None and seat == human_seat:
                names.append("You")
                continue
            agent = agent_factory()
            agents[seat] = agent
            names.append(opponent_name if engine.num_players == 2 else f"{opponent_name}-{seat}")

        cli = CoupCLI(
            engine,
            agents,
            names,
            human_seat=human_seat,
            show_probs=bool(args.show_probs),
        )
        try:
            result = cli.play(interactive=not spectate, narrate_turns=spectate)
        except KeyboardInterrupt:
            print("\n  quit - no result", flush=True)
            return 130

        if spectate:
            winner = result["winner"]
            label = "truncated" if winner is None else cli._seat_label(winner)
            print("", flush=True)
            print(f"  final: {label} | turns {result['turns']}", flush=True)
        return 0
    finally:
        cleanup()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
