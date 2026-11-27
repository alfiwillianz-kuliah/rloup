"""Tests for the local SLM agent, prompt formatting, parsing, and play CLI wiring."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rloup.agents.base import parse_obs
from rloup.agents.slm import (
    SLMAgent,
    SLMConfig,
    describe_legal_action,
    discover_local_gguf,
    format_obs_prompt,
    parse_choice_response,
)
from rloup.constants import Phase, Role
from rloup.engine import CoupEngine

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
def test_parse_choice_response_handles_json_wrapped_and_plaintext():
    assert parse_choice_response('{"choice": 2, "reason": "Tax with Duke"}', 4) == (
        2,
        "Tax with Duke",
    )
    assert parse_choice_response('```json\n{"reason":"safe","choice":3}\n```', 4) == (
        3,
        "safe",
    )
    assert parse_choice_response("I pick option 2 because it is best.", 3) == (2, None)
    # Truncated JSON mentioning "Opp+1" must NOT falsely parse choice=1.
    assert parse_choice_response('{"reason": "Opp+1 declared STEAL', 3) == (None, None)
    # Out-of-range choice must return None so caller falls back cleanly.
    assert parse_choice_response('{"choice": 9}', 3) == (None, None)
    assert parse_choice_response("", 3) == (None, None)


# --------------------------------------------------------------------------- #
# Prompt formatting & strict no-hidden-info guarantee
# --------------------------------------------------------------------------- #
def test_format_obs_prompt_contains_no_hidden_opponent_hand():
    eng_a = CoupEngine(2, seed=7)
    eng_a.reset(seed=7)
    eng_a.debug_set_hands(
        hands={0: [Role.DUKE, Role.CONTESSA], 1: [Role.ASSASSIN, Role.CAPTAIN]},
        coins={0: 4, 1: 3},
    )
    eng_a.current_player = 0
    eng_a.phase = Phase.CHOOSE_ACTION

    eng_b = CoupEngine(2, seed=7)
    eng_b.reset(seed=7)
    eng_b.debug_set_hands(
        hands={0: [Role.DUKE, Role.CONTESSA], 1: [Role.AMBASSADOR, Role.DUKE]},
        coins={0: 4, 1: 3},
    )
    eng_b.current_player = 0
    eng_b.phase = Phase.CHOOSE_ACTION

    prompt_a, legal_a = format_obs_prompt(parse_obs(eng_a.get_obs(0), eng_a.legal_mask()))
    prompt_b, legal_b = format_obs_prompt(parse_obs(eng_b.get_obs(0), eng_b.legal_mask()))
    assert prompt_a == prompt_b
    assert legal_a == legal_b
    assert "hand=[DUKE, CONTESSA]" in prompt_a
    assert "TAX (+3 coins, claims DUKE [truthful])" in prompt_a


def test_format_obs_prompt_flags_impossible_claim():
    eng = CoupEngine(2, seed=11)
    eng.reset(seed=11)
    eng.debug_set_hands(
        hands={0: [Role.DUKE, Role.DUKE], 1: [Role.CAPTAIN, Role.CONTESSA]},
        revealed={1: [Role.DUKE]},
        coins={0: 2, 1: 2},
    )
    eng.current_player = 1
    eng.phase = Phase.CHOOSE_ACTION
    eng.step(eng.spec.tax)  # seat 1 bluffs DUKE -> seat 0 gets CHALLENGE_ACTION
    assert eng.current_player == 0
    assert eng.phase == Phase.CHALLENGE_ACTION

    view = parse_obs(eng.get_obs(0), eng.legal_mask())
    prompt, legal = format_obs_prompt(view)
    assert len(legal) == 2
    assert "IMPOSSIBLE" in prompt
    assert "CHALLENGE wins 100%" in prompt


# --------------------------------------------------------------------------- #
# SLMAgent behavior with mock completion backend
# --------------------------------------------------------------------------- #
def test_slm_agent_plays_full_2p_and_4p_games_with_mock_backend():
    def mock_completion(payload: dict[str, Any]) -> dict[str, Any]:
        schema = payload["response_format"]["json_schema"]["schema"]
        choices = schema["properties"]["choice"]["enum"]
        # Pick the last legal choice (exercises non-trivial index mapping).
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"reason": "mock pick", "choice": choices[-1]}
                        )
                    }
                }
            ]
        }

    for num_players in (2, 4):
        eng = CoupEngine(num_players, seed=42)
        eng.reset(seed=42)
        agent = SLMAgent(
            config=SLMConfig(include_reason=True),
            seed=0,
            completion_fn=mock_completion,
        )
        steps = 0
        while not eng.done and steps < 300:
            seat = eng.current_player
            obs = eng.get_obs(seat)
            mask = eng.legal_mask()
            action = agent.select_action(obs, mask)
            assert mask[action], f"illegal action {action}"
            eng.step(action)
            steps += 1
        assert eng.done
        assert agent.calls > 0
        assert agent.fallbacks == 0
        assert agent.valid_parses == agent.calls


def test_slm_agent_single_legal_action_fast_path_skips_backend():
    called = 0

    def fail_if_called(_: dict[str, Any]) -> dict[str, Any]:
        nonlocal called
        called += 1
        raise AssertionError("backend must not be called when only 1 action is legal")

    eng = CoupEngine(2, seed=3)
    eng.reset(seed=3)
    eng.debug_set_hands(
        hands={0: [Role.DUKE, Role.CONTESSA], 1: [Role.ASSASSIN, Role.CAPTAIN]},
        coins={0: 10, 1: 2},
    )
    eng.current_player = 0
    eng.phase = Phase.CHOOSE_ACTION
    mask = eng.legal_mask()
    assert int(mask.sum()) == 1  # mandatory coup at 10 coins in 2P

    agent = SLMAgent(completion_fn=fail_if_called, seed=0)
    action = agent.select_action(eng.get_obs(0), mask)
    assert mask[action]
    assert called == 0
    assert agent.fast_path_skips == 1


def test_slm_agent_falls_back_gracefully_on_backend_error():
    def broken(_: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("server offline")

    eng = CoupEngine(2, seed=5)
    eng.reset(seed=5)
    agent = SLMAgent(completion_fn=broken, seed=1)
    mask = eng.legal_mask()
    action = agent.select_action(eng.get_obs(0), mask)
    assert mask[action]
    assert agent.fallbacks == 1
    assert agent.valid_parses == 0
    assert "fallback" in (agent.last_reason or "")


def test_slm_agent_batched_actions_concurrent():
    def mock_completion(payload: dict[str, Any]) -> dict[str, Any]:
        return {"choices": [{"message": {"content": '{"choice": 1}'}}]}

    eng = CoupEngine(2, seed=9)
    eng.reset(seed=9)
    obs = eng.get_obs(eng.current_player)
    mask = eng.legal_mask()
    agent = SLMAgent(
        config=SLMConfig(include_reason=False, parallel_slots=4),
        completion_fn=mock_completion,
        seed=0,
    )
    actions = agent.batched_actions(np.stack([obs] * 6), np.stack([mask] * 6))
    assert len(actions) == 6
    assert all(mask[a] for a in actions)
    assert agent.valid_parses == 6


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #
def test_discover_local_gguf_prefers_edge_models_and_skips_vocabs(tmp_path: Path):
    vocab = tmp_path / "ggml-vocab-gemma-4.gguf"
    vocab.write_bytes(b"\x00" * (12 * 1024 * 1024))
    tiny = tmp_path / "gemma-3-270m-it-Q8_0.gguf"
    tiny.write_bytes(b"\x00" * (12 * 1024 * 1024))
    edge = tmp_path / "gemma-4-E4B_q4_0-it.gguf"
    edge.write_bytes(b"\x00" * (12 * 1024 * 1024))

    picked = discover_local_gguf(search_dirs=[tmp_path])
    assert picked is not None
    assert picked.name == "gemma-4-E4B_q4_0-it.gguf"


# --------------------------------------------------------------------------- #
# End-to-end HTTP + play.py CLI integration with local mock OpenAI server
# --------------------------------------------------------------------------- #
def test_play_cli_with_slm_opponent_against_mock_http_server():
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
            choices = schema.get("properties", {}).get("choice", {}).get("enum", [1])
            reply = {
                "id": "mock",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"reason": "play solid", "choice": choices[0]}
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            encoded = json.dumps(reply).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "rloup.play",
                "--demo",
                "--opponent",
                "slm",
                "--slm-url",
                f"http://127.0.0.1:{port}/v1",
                "--show-probs",
                "--seed",
                "4",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert completed.returncode == 0, completed.stderr
    assert "final:" in completed.stdout
    assert "SLM" in completed.stdout
    assert "play solid" in completed.stdout
