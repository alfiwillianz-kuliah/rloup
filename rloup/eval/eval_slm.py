"""Evaluate a local Small Language Model (SLM) bot against rloup agents.

Runs seat-balanced 2-player matches against the trained RL policies and
scripted baselines plus the controlled impossible-claim card-counting probe,
then reports win rates (with exact 95% Wilson CIs), anchored Elo, behavioural
telemetry, and SLM inference latency / parse rate.

Example
-------
Using the auto-discovered local GGUF model on the RTX 5060 Ti::

    python -m rloup.eval.eval_slm --games 40 --probe-scenarios 50

Or pointing at a specific GGUF model or running server::

    python -m rloup.eval.eval_slm --slm-model ~/Gemma/models/edge/gemma-4-E4B_q4_0-it.gguf
    python -m rloup.eval.eval_slm --slm-url http://127.0.0.1:8080/v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from rloup.agents.slm import SLMAgent, SLMConfig
from rloup.constants import CLAIM_KINDS, Phase
from rloup.engine import CoupEngine
from rloup.eval.evaluate import (
    Competitor,
    build_impossible_claim_state,
    elo_from_win_rates,
    load_competitors,
    wilson_record,
)

DEFAULT_OPPONENTS: tuple[str, ...] = (
    "PFSP-PPO",
    "Naive-PPO",
    "NFSP",
    "CardCounter",
    "Honest",
    "Random",
)


def _play_slm_matchup(
    slm: SLMAgent,
    opponent: Competitor,
    *,
    games: int,
    num_envs: int,
    seed: int,
    max_turns: int = 200,
) -> dict[str, Any]:
    """Seat-balanced 2P matchup with concurrent SLM batching."""
    first = games // 2
    # slm_seat[g] is 0 for the first half, 1 for the second half.
    slm_seats = [0] * first + [1] * (games - first)
    num_envs = max(1, min(int(num_envs), games))

    engines: list[CoupEngine] = []
    opp_agents = []
    game_idx: list[int] = []
    for i in range(num_envs):
        eng = CoupEngine(2, max_turns=max_turns, seed=seed * 1_000_003 + i)
        eng.reset(seed=seed * 7_919 + 11 + i)
        engines.append(eng)
        opp_agents.append(opponent.new_agent(seed=seed * 65_537 + i))
        game_idx.append(i)

    next_game = num_envs
    finished = 0
    wins = 0
    draws = 0
    total_turns = 0
    claims = 0
    bluffs = 0
    ch_att = 0
    ch_won = 0
    bluff_by_kind = {k: 0 for k in CLAIM_KINDS}
    claim_by_kind = {k: 0 for k in CLAIM_KINDS}

    while finished < games:
        slm_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        opp_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        actions = [0] * num_envs

        for i in range(num_envs):
            eng = engines[i]
            if eng.done:
                continue
            g = game_idx[i]
            seat = eng.current_player
            obs = eng.get_obs(seat)
            mask = eng.legal_mask()
            if seat == slm_seats[g]:
                slm_rows.append((i, obs, mask))
            elif opponent.neural:
                opp_rows.append((i, obs, mask))
            else:
                actions[i] = int(opp_agents[i].select_action(obs, mask))

        if slm_rows:
            chosen = slm.batched_actions(
                np.stack([r[1] for r in slm_rows]),
                np.stack([r[2] for r in slm_rows]),
            )
            for (i, _, _), act in zip(slm_rows, chosen):
                actions[i] = int(act)

        if opp_rows:
            chosen = opponent.batched_actions(
                np.stack([r[1] for r in opp_rows]),
                np.stack([r[2] for r in opp_rows]),
            )
            for (i, _, _), act in zip(opp_rows, chosen):
                actions[i] = int(act)

        for i in range(num_envs):
            eng = engines[i]
            if eng.done:
                continue
            eng.step(actions[i])
            if not eng.done:
                continue

            g = game_idx[i]
            slm_seat = slm_seats[g]
            total_turns += int(eng.turn)
            if eng.truncated:
                draws += 1
            elif eng.winner() == slm_seat:
                wins += 1

            st = eng.get_episode_stats()["players"][slm_seat]
            for kind in CLAIM_KINDS:
                claim_by_kind[kind] += int(st["claims"][kind])
                bluff_by_kind[kind] += int(st["bluffs"][kind])
            claims += sum(int(v) for v in st["claims"].values())
            bluffs += sum(int(v) for v in st["bluffs"].values())
            ch_att += int(st["challenges_attempted"])
            ch_won += int(st["challenges_won"])

            finished += 1
            if next_game < games:
                eng.reset(seed=seed * 7_919 + 11 + next_game)
                opp_agents[i] = opponent.new_agent(seed=seed * 65_537 + next_game)
                game_idx[i] = next_game
                next_game += 1

    rec = wilson_record(wins, games)
    rec.update(
        {
            "opponent": opponent.name,
            "games": games,
            "wins": wins,
            "losses": games - wins - draws,
            "draws": draws,
            "avg_turns": total_turns / max(games, 1),
            "claims": claims,
            "bluffs": bluffs,
            "bluff_rate": (bluffs / claims) if claims else 0.0,
            "challenges_attempted": ch_att,
            "challenges_won": ch_won,
            "challenge_precision": (ch_won / ch_att) if ch_att else 0.0,
        }
    )
    return rec


def run_impossible_probe(
    slm: SLMAgent, *, scenarios: int = 50, num_players: int = 2, seed: int = 101
) -> dict[str, Any]:
    """Controlled card-counting deduction probe for the SLM."""
    rng = np.random.default_rng(seed)
    states = [
        build_impossible_claim_state(rng, num_players, block_phase=bool(i % 2))
        for i in range(scenarios)
    ]
    obs_batch = np.stack([s.obs for s in states])
    mask_batch = np.stack([s.mask for s in states])
    actions = slm.batched_actions(obs_batch, mask_batch)
    challenged = sum(
        1 for st, act in zip(states, actions) if int(act) == int(st.challenge_id)
    )
    rec = wilson_record(challenged, scenarios)
    rec["challenge_rate"] = rec["rate"]
    rec["scenarios"] = scenarios
    return rec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rloup.eval.eval_slm",
        description="Benchmark a local SLM agent against rloup RL models and baselines.",
    )
    parser.add_argument("--slm-model", default=None, help="Path to local .gguf model")
    parser.add_argument(
        "--slm-url", default=None, help="Existing OpenAI-compatible /v1 server URL"
    )
    parser.add_argument("--games", type=int, default=40, help="Games per opponent (even)")
    parser.add_argument(
        "--probe-scenarios",
        type=int,
        default=50,
        help="Controlled impossible-claim probe scenarios",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Concurrent games / llama-server parallel slots",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Disable JSON reason field for faster throughput",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=list(DEFAULT_OPPONENTS),
        help="Opponent names to face",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out", default="results/slm_eval.json", help="Output JSON summary path"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SLMConfig(
        base_url=args.slm_url,
        model=args.slm_model,
        temperature=float(args.temperature),
        include_reason=not bool(args.fast),
        max_tokens=32 if args.fast else 96,
        parallel_slots=max(1, int(args.parallel)),
    )

    competitors = {
        c.name: c for c in load_competitors(device=args.device, verbose=False)
    }
    selected = [competitors[name] for name in args.opponents if name in competitors]
    if not selected:
        raise SystemExit(f"No valid opponents in {args.opponents!r}")

    # Load anchor Elos from eval_summary.json when available.
    anchor_elos: dict[str, float] = {}
    summary_path = Path("results/eval_summary.json")
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        anchor_elos = {
            k: float(v)
            for k, v in data.get("round_robin_2p", {}).get("elo", {}).items()
        }

    t_start = time.time()
    with SLMAgent(config=config, seed=args.seed) as slm:
        print(
            f"Benchmarking local SLM ({slm.model_label}) | "
            f"games/opp={args.games} | parallel={args.parallel} | "
            f"reason={'off' if args.fast else 'on'}",
            flush=True,
        )
        matchups: dict[str, dict[str, Any]] = {}
        for idx, opp in enumerate(selected):
            t0 = time.time()
            rec = _play_slm_matchup(
                slm,
                opp,
                games=int(args.games),
                num_envs=int(args.parallel),
                seed=int(args.seed) + idx * 1000,
            )
            matchups[opp.name] = rec
            print(
                f"  vs {opp.name:<12s}: {rec['wins']:3d}/{rec['games']} "
                f"({rec['rate'] * 100:5.1f}%, 95% CI [{rec['ci_low'] * 100:4.1f}%, {rec['ci_high'] * 100:4.1f}%]) "
                f"| bluff {rec['bluff_rate'] * 100:4.1f}% | {time.time() - t0:.1f}s",
                flush=True,
            )

        probe: dict[str, Any] | None = None
        if int(args.probe_scenarios) > 0:
            t0 = time.time()
            probe = run_impossible_probe(
                slm, scenarios=int(args.probe_scenarios), seed=int(args.seed) + 9999
            )
            print(
                f"  impossible-claim probe: {probe['successes']}/{probe['scenarios']} "
                f"({probe['challenge_rate'] * 100:5.1f}%, CI [{probe['ci_low'] * 100:4.1f}%, {probe['ci_high'] * 100:4.1f}%]) "
                f"| {time.time() - t0:.1f}s",
                flush=True,
            )

        slm_stats = slm.stats()

    win_rates = {name: rec["rate"] for name, rec in matchups.items()}
    anchored_elo = (
        elo_from_win_rates(
            win_rates, anchor_elos, games_per_opponent=int(args.games)
        )
        if anchor_elos
        else None
    )
    total_claims = sum(rec["claims"] for rec in matchups.values())
    total_bluffs = sum(rec["bluffs"] for rec in matchups.values())
    total_ch_att = sum(rec["challenges_attempted"] for rec in matchups.values())
    total_ch_won = sum(rec["challenges_won"] for rec in matchups.values())

    report = {
        "model": slm_stats["model"],
        "games_per_opponent": int(args.games),
        "include_reason": not bool(args.fast),
        "temperature": float(args.temperature),
        "wall_time_s": time.time() - t_start,
        "anchored_elo": anchored_elo,
        "avg_win_rate": float(np.mean(list(win_rates.values()))),
        "matchups": matchups,
        "impossible_claim_probe": probe,
        "behaviour": {
            "claims": total_claims,
            "bluffs": total_bluffs,
            "bluff_rate": (total_bluffs / total_claims) if total_claims else 0.0,
            "challenges_attempted": total_ch_att,
            "challenges_won": total_ch_won,
            "challenge_precision": (total_ch_won / total_ch_att) if total_ch_att else 0.0,
        },
        "slm_stats": slm_stats,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("", flush=True)
    if anchored_elo is not None:
        print(f"Estimated Elo (anchored to round robin): {anchored_elo:.1f}", flush=True)
    print(
        f"SLM calls: {slm_stats['calls']} | parse rate: {slm_stats['parse_rate'] * 100:.1f}% | "
        f"mean latency: {slm_stats['mean_latency_ms']:.1f} ms | saved -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
