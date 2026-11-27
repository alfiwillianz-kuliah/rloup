"""Local Small Language Model (SLM) agent for Coup.

``SLMAgent`` is a blind :class:`~rloup.agents.base.Agent` that sees **only**
the ego-centric observation vector and legal action mask (never hidden opponent
hands).  It converts :class:`~rloup.agents.base.ObsView` into a concise tactical
prompt, queries any OpenAI-compatible local inference server (``llama-server``,
Ollama, vLLM, LM Studio, ...), and uses structured JSON Schema decoding so the
chosen action is guaranteed to be a legal index.

Zero-config local GPU mode (RTX 5060 Ti / CUDA)
-----------------------------------------------
When constructed without ``base_url``, :class:`SLMAgent` auto-discovers a local
``.gguf`` model (preferring Gemma 4 E4B / E2B edge models) and manages a local
``llama-server`` subprocess with full GPU offload (``-ngl 99``) on an ephemeral
localhost port, shutting it down cleanly via :meth:`close`, context manager
exit, or ``atexit``.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from typing import Any, Callable, Sequence
import urllib.error
import urllib.request

import numpy as np

from rloup.constants import (
    ACTION_NAMES,
    EXCHANGE_KEEP_OPTIONS,
    NUM_ROLES,
    ROLE_NAMES,
    ActionType,
    Phase,
    Role,
)
from rloup.engine import (
    EV_ACTION,
    EV_BLOCK,
    EV_CHALLENGE_ACTION,
    EV_CHALLENGE_BLOCK,
    EV_INFLUENCE_LOST,
    EV_PROVEN,
    NUM_EVENT_KINDS,
)
from .base import Agent, ObsView, parse_obs

__all__ = [
    "SLMConfig",
    "LlamaServerManager",
    "SLMAgent",
    "discover_local_gguf",
    "list_local_ggufs",
    "describe_legal_action",
    "format_obs_prompt",
    "parse_choice_response",
]

# --------------------------------------------------------------------------- #
# System prompt & rules summary (kept compact for fast KV prefix caching)
# --------------------------------------------------------------------------- #

COUP_SYSTEM_PROMPT = """\
You are an expert player of the card game Coup (base game, 15-card deck: 3 copies each of DUKE, ASSASSIN, CAPTAIN, AMBASSADOR, CONTESSA).
Rules summary:
- Main actions: INCOME (+1 coin, unblockable), FOREIGN_AID (+2, blockable by DUKE), COUP (cost 7, unavoidable, mandatory at 10+ coins), TAX (+3, claims DUKE), ASSASSINATE (cost 3, claims ASSASSIN, blockable by CONTESSA), STEAL (+2 from target, claims CAPTAIN, blockable by CAPTAIN/AMBASSADOR), EXCHANGE (draw 2 & swap, claims AMBASSADOR).
- Any role claim or block may be CHALLENGED. Caught bluff = claimant loses 1 influence & action fails. Proven claim = challenger loses 1 influence & claimant replaces card.
- Tactical guidelines:
  1. If you hold DUKE, TAX (+3) is strictly better than INCOME (+1).
  2. At 7+ coins, COUP eliminates an opponent's influence unavoidably.
  3. If a claim is marked IMPOSSIBLE (all 3 copies are in your hand + revealed), ALWAYS CHALLENGE (100% win).
  4. Block attacks targeting you when you hold the blocking role; if targeted by ASSASSINATE with only 1 influence left and no CONTESSA, bluffing BLOCK_CONTESSA is often better than dying immediately.
  5. Selective bluffing (especially TAX or STEAL) can win, but avoid reckless challenges on 1 influence unless the claim is impossible or very suspicious.
Respond ONLY with valid JSON matching the schema. Keep any reason concise (under 15 words). Pick the single best legal action number."""

_EVENT_LABELS: dict[int, str] = {
    EV_ACTION: "action",
    EV_CHALLENGE_ACTION: "challenged action",
    EV_BLOCK: "blocked",
    EV_CHALLENGE_BLOCK: "challenged block",
    EV_PROVEN: "proved claim",
    EV_INFLUENCE_LOST: "lost influence",
}

#: Preferred GGUF filename stems (highest priority first) when auto-discovering.
_PREFERRED_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "gemma-4-e4b",
    "gemma-4-e2b",
    "gemma-3-4b",
    "qwen3-4b",
    "qwen2.5-3b",
    "llama-3.2-3b",
    "gemma-3-1b",
    "gemma-3-270m",
    "gemma4-26b",
)


# --------------------------------------------------------------------------- #
# Model discovery & local llama-server lifecycle
# --------------------------------------------------------------------------- #


def _default_model_search_dirs() -> list[Path]:
    home = Path.home()
    return [
        Path.cwd() / "models",
        home / "Gemma" / "models" / "edge",
        home / "Gemma" / "models" / "tiny",
        home / "Gemma" / "models",
        home / "models",
        home / ".cache" / "llama.cpp",
        home / ".cache" / "lm-studio" / "models",
    ]


def _model_rank(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    for idx, token in enumerate(_PREFERRED_MODEL_SUBSTRINGS):
        if token in name:
            return (0, idx, name)
    # Prefer smaller-to-medium SLMs (< 6 GB) over giant quantised files.
    try:
        size_mb = path.stat().st_size // (1024 * 1024)
    except OSError:
        size_mb = 999_999
    bucket = 1 if 150 <= size_mb <= 6_500 else 2
    return (bucket, size_mb, name)


def list_local_ggufs(
    search_dirs: Sequence[str | Path] | None = None,
) -> list[Path]:
    """Return all discovered local ``.gguf`` SLM files, ranked best-first."""
    dirs = (
        [Path(d).expanduser() for d in search_dirs]
        if search_dirs is not None
        else _default_model_search_dirs()
    )
    found: dict[Path, Path] = {}
    env_model = os.environ.get("RLOUP_SLM_MODEL")
    if env_model:
        candidate = Path(env_model).expanduser()
        if candidate.is_file():
            found[candidate.resolve()] = candidate

    for root in dirs:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".gguf":
            found[root.resolve()] = root
            continue
        for path in root.glob("*.gguf"):
            lower = path.name.lower()
            if lower.startswith("ggml-vocab-") or "mmproj" in lower:
                continue
            try:
                if path.stat().st_size < 10 * 1024 * 1024:
                    continue
            except OSError:
                continue
            found[path.resolve()] = path

    return sorted(found.values(), key=_model_rank)


def discover_local_gguf(
    search_dirs: Sequence[str | Path] | None = None,
) -> Path | None:
    """Find the best local ``.gguf`` SLM file for single-GPU inference.

    Respects ``RLOUP_SLM_MODEL`` if set.  Skips llama.cpp vocabulary-only files
    (``ggml-vocab-*.gguf``) and tiny non-model shards (< 10 MB).
    """
    env_model = os.environ.get("RLOUP_SLM_MODEL")
    if env_model:
        candidate = Path(env_model).expanduser()
        if candidate.is_file():
            return candidate

    models = list_local_ggufs(search_dirs=search_dirs)
    return models[0] if models else None


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass
class SLMConfig:
    """Configuration for :class:`SLMAgent` and its optional local server."""

    base_url: str | None = None
    model: str | None = None
    api_key: str = "local"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 128
    include_reason: bool = True
    use_json_schema: bool = True
    disable_thinking: bool = True
    timeout: float = 30.0
    # Managed llama-server settings (used only when base_url is None)
    server_binary: str = "llama-server"
    host: str = "127.0.0.1"
    port: int | None = None
    gpu_layers: int = 99
    ctx_size: int = 4096
    parallel_slots: int = 4
    flash_attn: str = "on"
    startup_timeout: float = 60.0
    extra_server_args: tuple[str, ...] = field(default_factory=tuple)


class LlamaServerManager:
    """Manages a local ``llama-server`` subprocess bound to localhost."""

    def __init__(self, config: SLMConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self.port: int | None = None
        self.model_path: Path | None = None
        self._base_url: str | None = None
        self._registered_atexit = False

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("LlamaServerManager has not been started")
        return self._base_url

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> str:
        """Start ``llama-server`` if not already running and return its ``/v1`` URL."""
        if self.running and self._base_url is not None:
            return self._base_url

        binary = shutil.which(self.config.server_binary)
        if binary is None:
            raise RuntimeError(
                f"{self.config.server_binary!r} not found in PATH. "
                "Install llama.cpp or pass --slm-url pointing to an OpenAI-compatible server."
            )

        model_path = (
            Path(self.config.model).expanduser()
            if self.config.model
            else discover_local_gguf()
        )
        if model_path is None or not model_path.is_file():
            raise FileNotFoundError(
                f"No local GGUF model found (requested={self.config.model!r}). "
                "Pass --slm-model /path/to/model.gguf or set RLOUP_SLM_MODEL."
            )
        self.model_path = model_path
        self.port = self.config.port or _find_free_port(self.config.host)

        total_ctx = max(int(self.config.ctx_size), 1536 * max(1, int(self.config.parallel_slots)))
        cmd = [
            binary,
            "-m",
            str(model_path),
            "--host",
            self.config.host,
            "--port",
            str(self.port),
            "-ngl",
            str(self.config.gpu_layers),
            "-c",
            str(total_ctx),
            "-np",
            str(max(1, int(self.config.parallel_slots))),
            "-fa",
            str(self.config.flash_attn),
            "--log-disable",
        ]
        if self.config.disable_thinking:
            cmd.extend(["--reasoning-budget", "0"])
        cmd.extend(self.config.extra_server_args)

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not self._registered_atexit:
            atexit.register(self.stop)
            self._registered_atexit = True

        health_url = f"http://{self.config.host}:{self.port}/health"
        deadline = time.monotonic() + float(self.config.startup_timeout)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                err = ""
                if self.process.stderr is not None:
                    err = self.process.stderr.read()[-1500:]
                raise RuntimeError(
                    f"llama-server exited early (code {self.process.returncode}): {err}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        if data.get("status") == "ok":
                            self._base_url = f"http://{self.config.host}:{self.port}/v1"
                            return self._base_url
            except Exception:
                pass
            time.sleep(0.2)

        self.stop()
        raise TimeoutError(
            f"llama-server did not become ready within {self.config.startup_timeout:.1f}s"
        )

    def stop(self) -> None:
        """Terminate the managed ``llama-server`` subprocess if running."""
        proc = self.process
        if proc is None:
            return
        self.process = None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except Exception:
                pass

    def __enter__(self) -> LlamaServerManager:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
# Observation -> text prompt (strictly from ObsView, zero hidden info)
# --------------------------------------------------------------------------- #


def _role_str(role: int) -> str:
    return str(ROLE_NAMES[int(role)])


def _seat_tag(rel: int | None) -> str:
    if rel is None:
        return "none"
    return "You" if rel == 0 else f"Opp+{rel}"


def _format_role_multiset(counts: np.ndarray) -> str:
    items: list[str] = []
    for role in range(NUM_ROLES):
        c = int(counts[role])
        if c > 0:
            items.extend([_role_str(role)] * c)
    return ", ".join(items) if items else "none"


def describe_legal_action(view: ObsView, action_id: int) -> str:
    """Ego-centric human-readable description of a legal flat action id."""
    info = view.spec.decode(int(action_id))
    kind = info.kind
    if kind == "INCOME":
        return "INCOME (+1 coin, safe)"
    if kind == "FOREIGN_AID":
        return "FOREIGN_AID (+2 coins, blockable by DUKE)"
    if kind == "TAX":
        holds = "truthful" if view.has(Role.DUKE) else "BLUFF"
        return f"TAX (+3 coins, claims DUKE [{holds}])"
    if kind == "EXCHANGE":
        holds = "truthful" if view.has(Role.AMBASSADOR) else "BLUFF"
        return f"EXCHANGE (draw 2 & swap, claims AMBASSADOR [{holds}])"
    if kind in ("COUP", "ASSASSINATE", "STEAL"):
        rel = int(info.rel_off)
        opp = view.seats[rel]
        target = f"Opp+{rel} ({opp.coins}c, {opp.influence}inf)"
        if kind == "COUP":
            return f"COUP {target} (cost 7, unavoidable)"
        if kind == "ASSASSINATE":
            holds = "truthful" if view.has(Role.ASSASSIN) else "BLUFF"
            return f"ASSASSINATE {target} (cost 3, claims ASSASSIN [{holds}])"
        holds = "truthful" if view.has(Role.CAPTAIN) else "BLUFF"
        return f"STEAL up to 2 coins from {target} (claims CAPTAIN [{holds}])"
    if kind == "PASS":
        return "PASS (do not challenge / do not block)"
    if kind == "CHALLENGE":
        return "CHALLENGE the pending claim"
    if kind == "BLOCK":
        role = info.role
        holds = "truthful" if (role is not None and view.has(role)) else "BLUFF"
        return f"BLOCK claiming {_role_str(int(role))} [{holds}]"
    if kind == "LOSE_SLOT":
        slot = int(info.index)
        card = _role_str(view.own_hand[slot]) if slot < len(view.own_hand) else "?"
        return f"LOSE {card} (slot {slot + 1})"
    if kind == "EXCHANGE_KEEP":
        keep = EXCHANGE_KEEP_OPTIONS[int(info.index)]
        cards = ", ".join(
            _role_str(view.buffer[k]) for k in keep if k < len(view.buffer)
        )
        return f"KEEP [{cards}]"
    return view.spec.action_name(action_id)


def _decode_recent_events(view: ObsView) -> list[str]:
    lines: list[str] = []
    n = view.num_players
    for row in view.events:
        kind_slice = row[:NUM_EVENT_KINDS]
        if float(kind_slice.max()) < 0.5:
            continue
        kind = int(np.argmax(kind_slice))
        actor_s = float(row[NUM_EVENT_KINDS])
        target_s = float(row[NUM_EVENT_KINDS + 1])
        actor_rel = int(round(actor_s * n)) - 1 if actor_s > 0.0 else None
        target_rel = int(round(target_s * n)) - 1 if target_s > 0.0 else None
        label = _EVENT_LABELS.get(kind, f"event_{kind}")
        if target_rel is not None and 0 <= target_rel < n:
            lines.append(f"{_seat_tag(actor_rel)} {label} -> {_seat_tag(target_rel)}")
        else:
            lines.append(f"{_seat_tag(actor_rel)} {label}")
    return lines[-5:]


def format_obs_prompt(view: ObsView) -> tuple[str, list[int]]:
    """Render ``view`` into a compact user prompt and return ``(prompt, legal_ids)``."""
    legal_ids = [int(a) for a in view.legal_ids]
    lines: list[str] = []

    hand_str = ", ".join(_role_str(c) for c in view.own_hand) or "none"
    lines.append(
        f"Phase: {Phase(view.phase).name} | Alive: {view.num_alive}/{view.num_players} | Deck: {view.deck_size}"
    )
    lines.append(
        f"You: hand=[{hand_str}], coins={view.coins}, influence={view.seat.influence}"
        + (" [MUST COUP]" if view.must_coup else "")
    )
    if view.in_exchange and view.buffer:
        buf_str = ", ".join(_role_str(c) for c in view.buffer)
        lines.append(f"Exchange pool: [{buf_str}]")

    lines.append("Opponents:")
    for rel in range(1, view.num_players):
        seat = view.seats[rel]
        status = "ALIVE" if seat.alive else "OUT"
        rev = _format_role_multiset(seat.revealed)
        claims_row = view.claims[rel]
        claimed_roles = [
            f"{_role_str(r)}x{int(claims_row[r])}"
            for r in range(NUM_ROLES)
            if int(claims_row[r]) > 0
        ]
        caught = int(claims_row[5])
        proven = int(claims_row[6])
        hist_parts: list[str] = []
        if claimed_roles:
            hist_parts.append("claimed " + ",".join(claimed_roles))
        if caught or proven:
            hist_parts.append(f"caught={caught} proven={proven}")
        hist = f" | {'; '.join(hist_parts)}" if hist_parts else ""
        lines.append(
            f"- Opp+{rel}: {status}, coins={seat.coins}, inf={seat.influence}, revealed=[{rev}]{hist}"
        )

    unseen_str = ", ".join(
        f"{_role_str(r)}={int(view.unseen[r])}" for r in range(NUM_ROLES)
    )
    lines.append(f"Unseen copies left (3 - hand - revealed): {unseen_str}")

    if view.pending_action is not None:
        act_name = ACTION_NAMES[int(view.pending_action)]
        actor = _seat_tag(view.pending_actor_rel)
        target = (
            f" -> {_seat_tag(view.pending_target_rel)}"
            if view.pending_target_rel is not None
            else ""
        )
        pending = f"Pending: {actor} declared {act_name}{target}"
        if view.pending_blocker_rel is not None and view.pending_block_role is not None:
            pending += (
                f"; {_seat_tag(view.pending_blocker_rel)} BLOCKED with "
                f"{_role_str(int(view.pending_block_role))}"
            )
        lines.append(pending)

    if view.claimed_role is not None:
        claimant = _seat_tag(view.claimant_rel())
        role_name = _role_str(int(view.claimed_role))
        unseen_cnt = int(view.unseen[int(view.claimed_role)])
        if view.claimed_impossible:
            lines.append(
                f"ALERT: {claimant} claims {role_name}, which is IMPOSSIBLE "
                f"(0 unseen copies left — all 3 are in your hand/revealed). CHALLENGE wins 100%!"
            )
        else:
            lines.append(
                f"Claim under review: {claimant} claims {role_name} ({unseen_cnt} unseen copies exist)."
            )

    recent = _decode_recent_events(view)
    if recent:
        lines.append("Recent events: " + " | ".join(recent))

    lines.append("Legal actions:")
    for idx, aid in enumerate(legal_ids, start=1):
        lines.append(f"{idx}. {describe_legal_action(view, aid)}")
    lines.append(f"Choose 1..{len(legal_ids)}.")
    return "\n".join(lines), legal_ids


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


def _build_response_format(num_choices: int, include_reason: bool) -> dict[str, Any]:
    enum_vals = list(range(1, num_choices + 1))
    if include_reason:
        # Putting `reason` before `choice` lets the SLM think in 1 short sentence
        # before committing to the action index; maxLength prevents truncation.
        properties: dict[str, Any] = {
            "reason": {"type": "string", "maxLength": 120},
            "choice": {"type": "integer", "enum": enum_vals},
        }
        required = ["reason", "choice"]
    else:
        properties = {
            "choice": {"type": "integer", "enum": enum_vals},
        }
        required = ["choice"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "coup_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def parse_choice_response(
    text: str, num_choices: int
) -> tuple[int | None, str | None]:
    """Extract ``(1-based choice, optional reason)`` from JSON or plain text."""
    raw = (text or "").strip()
    if not raw:
        return None, None

    # 1. Try direct JSON or embedded JSON object.
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "choice" in obj:
            try:
                val = int(obj["choice"])
            except (TypeError, ValueError):
                val = -1
            reason = obj.get("reason")
            reason_str = str(reason).strip() if isinstance(reason, str) else None
            if 1 <= val <= num_choices:
                return val, reason_str

    # 2. Fallback: extract `"choice": <int>`.
    m = re.search(r'"choice"\s*:\s*(\d+)', raw)
    if m:
        val = int(m.group(1))
        if 1 <= val <= num_choices:
            return val, None

    # Only scan plain-text digits when the output isn't a truncated JSON object
    # (otherwise numbers like "Opp+1" inside a truncated reason would mis-parse).
    if not raw.lstrip().startswith("{"):
        for m in re.finditer(r"(?<![+A-Za-z_])(\d+)(?![A-Za-z_])", raw):
            val = int(m.group(1))
            if 1 <= val <= num_choices:
                return val, None

    return None, None


# --------------------------------------------------------------------------- #
# SLMAgent
# --------------------------------------------------------------------------- #


class SLMAgent(Agent):
    """Coup agent powered by a local Small Language Model via OpenAI-compatible API.

    Parameters
    ----------
    config:
        :class:`SLMConfig` controlling endpoint, model path, temperature, and
        optional local ``llama-server`` startup.
    seed:
        RNG seed used only if the SLM fails and a random legal fallback is needed.
    completion_fn:
        Optional ``(payload: dict) -> dict`` callable replacing HTTP requests
        (useful for deterministic unit tests without starting a server).
    server_manager:
        Optional pre-started :class:`LlamaServerManager` to share across
        multiple seats without spawning duplicate server instances.
    """

    name = "slm"

    def __init__(
        self,
        config: SLMConfig | None = None,
        *,
        seed: int | None = None,
        completion_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        server_manager: LlamaServerManager | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.config = config or SLMConfig()
        self._completion_fn = completion_fn
        self._owns_server = False
        self._server: LlamaServerManager | None = server_manager
        self._resolved_url: str | None = (
            self.config.base_url or os.environ.get("RLOUP_SLM_URL")
        )
        if self._resolved_url:
            self._resolved_url = self._resolved_url.rstrip("/")

        # Runtime telemetry
        self.calls: int = 0
        self.fast_path_skips: int = 0
        self.valid_parses: int = 0
        self.fallbacks: int = 0
        self.total_latency_s: float = 0.0
        self.last_reason: str | None = None
        self.last_prompt: str | None = None
        self.last_error: str | None = None

    @property
    def model_label(self) -> str:
        """Short display label for the active model."""
        if self._server is not None and self._server.model_path is not None:
            return self._server.model_path.stem
        if self.config.model:
            return Path(self.config.model).stem
        return "SLM"

    @property
    def mean_latency_ms(self) -> float:
        return (self.total_latency_s / self.calls * 1000.0) if self.calls else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model_label,
            "calls": self.calls,
            "fast_path_skips": self.fast_path_skips,
            "valid_parses": self.valid_parses,
            "fallbacks": self.fallbacks,
            "parse_rate": (self.valid_parses / self.calls) if self.calls else 1.0,
            "mean_latency_ms": self.mean_latency_ms,
            "last_error": self.last_error,
        }

    def ensure_ready(self) -> str | None:
        """Ensure the inference backend is reachable (starts local server if needed)."""
        if self._completion_fn is not None:
            return None
        if self._resolved_url:
            return self._resolved_url
        if self._server is None:
            self._server = LlamaServerManager(self.config)
            self._owns_server = True
        self._resolved_url = self._server.start().rstrip("/")
        return self._resolved_url

    def close(self) -> None:
        """Stop the local ``llama-server`` if this agent spawned it."""
        if self._owns_server and self._server is not None:
            self._server.stop()
            self._owns_server = False

    def __enter__(self) -> SLMAgent:
        self.ensure_ready()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- inference --------------------------------------------------------- #
    def _build_payload(self, prompt: str, num_choices: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model or "local",
            "messages": [
                {"role": "system", "content": COUP_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.config.temperature),
            "top_p": float(self.config.top_p),
            "max_tokens": int(self.config.max_tokens),
        }
        if self.config.use_json_schema:
            payload["response_format"] = _build_response_format(
                num_choices, self.config.include_reason
            )
        if self.config.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _call_backend(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._completion_fn is not None:
            return self._completion_fn(payload)
        base_url = self.ensure_ready()
        assert base_url is not None
        url = (
            base_url
            if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions"
        )
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=float(self.config.timeout)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if content:
            return str(content)
        # Fallback if a reasoning model put text in reasoning_content.
        return str(msg.get("reasoning_content") or "")

    def act(self, view: ObsView) -> int:
        legal = [int(a) for a in view.legal_ids]
        if len(legal) == 1:
            self.fast_path_skips += 1
            self.last_reason = "only legal action"
            return legal[0]

        prompt, legal_ids = format_obs_prompt(view)
        self.last_prompt = prompt
        payload = self._build_payload(prompt, len(legal_ids))

        t0 = time.monotonic()
        self.calls += 1
        choice_idx: int | None = None
        reason: str | None = None
        try:
            response = self._call_backend(payload)
            text = self._extract_text(response)
            choice_idx, reason = parse_choice_response(text, len(legal_ids))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            reason = f"fallback ({type(exc).__name__})"
        finally:
            self.total_latency_s += time.monotonic() - t0

        if choice_idx is not None:
            self.valid_parses += 1
            self.last_reason = reason
            return legal_ids[choice_idx - 1]

        self.fallbacks += 1
        self.last_reason = reason or "fallback (unparseable response)"
        return self._random_legal(view)

    def batched_actions(
        self,
        obs_batch: np.ndarray,
        mask_batch: np.ndarray,
        max_workers: int | None = None,
    ) -> list[int]:
        """Select actions concurrently across a batch of observations."""
        obs_arr = np.asarray(obs_batch)
        mask_arr = np.asarray(mask_batch)
        n = int(obs_arr.shape[0])
        if n <= 1:
            return [self.select_action(obs_arr[i], mask_arr[i]) for i in range(n)]

        self.ensure_ready()
        workers = max(1, int(max_workers or self.config.parallel_slots))
        views = [parse_obs(obs_arr[i], mask_arr[i]) for i in range(n)]
        results: list[int] = [0] * n
        pending: list[tuple[int, ObsView]] = []
        for idx, view in enumerate(views):
            if view.legal_ids.size == 1:
                self.fast_path_skips += 1
                results[idx] = int(view.legal_ids[0])
            else:
                pending.append((idx, view))

        if not pending:
            return results

        def _query(item: tuple[int, ObsView]) -> tuple[int, int, bool, float, str | None]:
            idx, view = item
            prompt, legal_ids = format_obs_prompt(view)
            payload = self._build_payload(prompt, len(legal_ids))
            t0 = time.monotonic()
            try:
                response = self._call_backend(payload)
                text = self._extract_text(response)
                choice_idx, reason = parse_choice_response(text, len(legal_ids))
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                choice_idx, reason = None, f"fallback ({type(exc).__name__})"
            dt = time.monotonic() - t0
            if choice_idx is not None:
                return idx, legal_ids[choice_idx - 1], True, dt, reason
            return idx, int(legal_ids[0]), False, dt, reason

        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            for idx, action, ok, dt, reason in pool.map(_query, pending):
                self.calls += 1
                self.total_latency_s += dt
                if ok:
                    self.valid_parses += 1
                    results[idx] = action
                    self.last_reason = reason
                else:
                    self.fallbacks += 1
                    results[idx] = self._random_legal(views[idx])
                    self.last_reason = reason or "fallback"

        return results
