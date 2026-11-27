"""Tests for the rloup Web Arena & Tournament HTTP server."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from rloup.web.server import WebApp, create_server

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def web_server():
    app = WebApp(root=REPO_ROOT, device="cpu")
    server, _ = create_server("127.0.0.1", 0, app=app)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        yield base, app
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def _get(url: str) -> tuple[int, Any]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
        ctype = resp.headers.get("Content-Type", "")
        return resp.status, (json.loads(raw) if "json" in ctype else raw)


def _post(url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_static_assets_and_models_catalog_exclude_live_slms(web_server):
    base, _ = web_server
    st, html = _get(f"{base}/")
    assert st == 200 and "RLOUP" in html

    st, css = _get(f"{base}/static/style.css")
    assert st == 200 and "--brass" in css

    st, js = _get(f"{base}/static/app.js")
    assert st == 200 and "startTournament" in js

    st, catalog = _get(f"{base}/api/models")
    assert st == 200
    ids = {m["id"] for m in catalog["models"]}
    categories = {m["category"] for m in catalog["models"]}
    assert "human" in ids
    assert "baseline:card_counter" in ids
    assert "baseline:random" in ids
    # Public web server must never expose live local or external SLM endpoints.
    assert "slm" not in categories
    assert not any(bid.startswith("slm") for bid in ids)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post(
            f"{base}/api/game/new",
            {"seats": ["human", "slm_url:http://127.0.0.1:8080/v1"]},
        )
    assert exc_info.value.code == 400


def test_human_vs_bot_game_hides_opponent_hand_until_done(web_server):
    base, _ = web_server
    st, game = _post(
        f"{base}/api/game/new",
        {
            "seats": ["human", "baseline:card_counter"],
            "seed": 12,
            "auto_advance": True,
            "reveal_ai_hands": True,  # must be ignored when a human is seated!
        },
    )
    assert st == 200
    assert game["has_human"] is True
    assert game["reveal_ai_hands"] is False
    if not game["done"]:
        opp_hand = game["seats"][1]["hand"]
        assert all(card == "HIDDEN" for card in opp_hand)
        human_hand = game["seats"][0]["hand"]
        assert all(card != "HIDDEN" for card in human_hand)

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post(f"{base}/api/game/{game['game_id']}/action", {"action_id": 999})
        assert exc_info.value.code == 400

    guard = 0
    while not game["done"] and guard < 200:
        if game["human_turn"]:
            aid = game["legal_actions"][0]["action_id"]
            _, game = _post(
                f"{base}/api/game/{game['game_id']}/action",
                {"action_id": aid, "auto_advance": True},
            )
        else:
            _, game = _post(f"{base}/api/game/{game['game_id']}/step", {})
        guard += 1

    assert game["done"] is True
    assert "bluff_rate" in game["seats"][0]
    assert all(card != "HIDDEN" for card in game["seats"][1]["hand"])


def test_ai_vs_ai_inspector_supports_ppo_and_double_dqn(web_server):
    base, app = web_server
    ids = {m["id"] for m in app.list_models()["models"]}
    seat0 = "rl:ppo_pfsp_2p" if "rl:ppo_pfsp_2p" in ids else "baseline:honest"
    seat1 = "rl:ddqn_pfsp_2p" if "rl:ddqn_pfsp_2p" in ids else "baseline:card_counter"

    st, game = _post(
        f"{base}/api/game/new",
        {
            "seats": [seat0, seat1],
            "seed": 5,
            "show_inspector": True,
            "reveal_ai_hands": True,
        },
    )
    assert st == 200
    assert game["has_human"] is False

    seen_kinds = set()
    seen_labels = set()
    for _ in range(12):
        if game["done"]:
            break
        _, game = _post(f"{base}/api/game/{game['game_id']}/step", {})
        if game["last_inspector"]:
            seen_kinds.add(game["last_inspector"]["kind"])
            if game["last_inspector"].get("value_label"):
                seen_labels.add(game["last_inspector"]["value_label"])
    if seat0.startswith("rl:") or seat1.startswith("rl:"):
        assert "neural" in seen_kinds
    if seat1 == "rl:ddqn_pfsp_2p":
        assert "Max Q(s, a)" in seen_labels


def test_live_2p_round_robin_tournament(web_server):
    base, _ = web_server
    st, job = _post(
        f"{base}/api/tournament/start",
        {
            "mode": "round_robin_2p",
            "bots": ["baseline:card_counter", "baseline:honest", "baseline:random"],
            "games_per_pair": 6,
            "seed": 7,
        },
    )
    assert st == 200
    tid = job["tournament_id"]
    deadline = time.time() + 20.0
    while job["status"] == "running" and time.time() < deadline:
        time.sleep(0.1)
        _, job = _get(f"{base}/api/tournament/{tid}")

    assert job["status"] == "completed", job.get("error")
    assert job["completed_games"] == 18
    assert len(job["standings"]) == 3
    assert len(job["matrix"]) == 3
    assert job["standings"][0]["elo"] >= job["standings"][-1]["elo"]


def test_live_4p_table_tournament(web_server):
    base, _ = web_server
    st, job = _post(
        f"{base}/api/tournament/start",
        {
            "mode": "table_4p",
            "bots": [
                "baseline:card_counter",
                "baseline:honest",
                "baseline:cautious",
                "baseline:random",
            ],
            "games_per_pair": 8,
            "seed": 9,
        },
    )
    assert st == 200
    tid = job["tournament_id"]
    deadline = time.time() + 20.0
    while job["status"] == "running" and time.time() < deadline:
        time.sleep(0.1)
        _, job = _get(f"{base}/api/tournament/{tid}")

    assert job["status"] == "completed", job.get("error")
    assert job["completed_games"] == 8
    assert len(job["standings"]) == 4
    assert sum(row["wins"] for row in job["standings"]) <= 8


def test_benchmarks_endpoint_returns_saved_eval_and_slm_benchmark(web_server):
    base, _ = web_server
    st, bench = _get(f"{base}/api/benchmarks")
    assert st == 200
    assert "figures" in bench
    assert "slm_eval" in bench
    if bench["figures"]:
        with urllib.request.urlopen(f"{base}/figures/{bench['figures'][0]}", timeout=5) as r:
            assert r.status == 200
            assert r.read(8) == b"\x89PNG\r\n\x1a\n"
