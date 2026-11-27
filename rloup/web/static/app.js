/* RLOUP — Coup RL Arena & Tournament Dashboard */

const ROLE_META = {
  DUKE: {
    pip: "DK",
    sub: "Tax +3",
    svg: `<svg viewBox="0 0 32 32" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M5 24h22M6 20l2.5-11 7.5 5 7.5-5L26 20H6z"/><circle cx="16" cy="7" r="1.4" fill="currentColor"/></svg>`,
  },
  ASSASSIN: {
    pip: "AS",
    sub: "Kill −3",
    svg: `<svg viewBox="0 0 32 32" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M24 6L12 18l-2 6 6-2L28 10 24 6z"/><path d="M9 15l8 8M6 26l4-4"/></svg>`,
  },
  CAPTAIN: {
    pip: "CP",
    sub: "Steal +2",
    svg: `<svg viewBox="0 0 32 32" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="16" cy="8" r="2.5"/><path d="M16 10.5V26M11 14h10M7 19a9 9 0 0 0 18 0"/><path d="M5 19h4M23 19h4"/></svg>`,
  },
  AMBASSADOR: {
    pip: "AM",
    sub: "Exchange",
    svg: `<svg viewBox="0 0 32 32" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M16 6v20M10 26h12M7 11h18"/><path d="M7 11l-3 7h6l-3-7zM25 11l-3 7h6l-3-7z"/></svg>`,
  },
  CONTESSA: {
    pip: "CT",
    sub: "Block Kill",
    svg: `<svg viewBox="0 0 32 32" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M16 5l9 3.5v7.5c0 5.5-3.8 9.2-9 11-5.2-1.8-9-5.5-9-11V8.5L16 5z"/><path d="M16 11v10M12 15h8"/></svg>`,
  },
  HIDDEN: {
    pip: "··",
    sub: "Court",
    svg: `<svg viewBox="0 0 32 32" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.2"><polygon points="16,4 27,16 16,28 5,16"/><polygon points="16,9 22,16 16,23 10,16" opacity="0.65"/><circle cx="16" cy="16" r="1.5" fill="currentColor"/></svg>`,
  },
};

const COIN_SVG = `<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="8" cy="8" r="6"/><circle cx="8" cy="8" r="3" opacity="0.55"/></svg>`;

const FIGURE_LABELS = {
  "round_robin_2p.png": "9×9 2-Player Round-Robin Heatmap & Bradley-Terry Elo",
  "learning_curves_2p.png": "2-Player Training Curves (PFSP-PPO, Double-DQN, Naive-PPO, NFSP)",
  "learning_curve_4p.png": "4-Player PFSP-PPO Scaling Across Baseline Tables",
  "behavioral_diagnostics.png": "Bluffing Structure, Challenge Precision & Card-Counting Probe",
  "exploitability_2p.png": "Approximate Exploitability vs 500k-Step PPO Best Response",
};

const state = {
  models: [],
  seatCount: 2,
  selectedSeats: ["human", "rl:ppo_pfsp_2p"],
  game: null,
  stepping: false,
  autoPlayTimer: null,
  tMode: "round_robin_2p",
  tSelected: new Set([
    "rl:ppo_pfsp_2p",
    "rl:ddqn_pfsp_2p",
    "rl:ppo_naive_2p",
    "rl:nfsp_2p",
    "baseline:card_counter",
    "baseline:honest",
    "baseline:random",
  ]),
  tFourSeats: ["rl:ppo_pfsp_4p", "rl:ddqn_pfsp_4p", "baseline:card_counter", "baseline:honest"],
  tId: null,
  tPollTimer: null,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
async function apiGet(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function apiPost(url, body = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 3200);
}

function esc(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function firstRlId(players, preferredId = null) {
  if (preferredId && state.models.some((x) => x.id === preferredId && x.players.includes(players))) {
    return preferredId;
  }
  const m = state.models.find((x) => x.category === "rl" && x.players.includes(players));
  return m ? m.id : "baseline:card_counter";
}

// ---------------------------------------------------------------------------
// Catalog & Navigation
// ---------------------------------------------------------------------------
async function refreshModels() {
  const data = await apiGet("/api/models");
  state.models = data.models || [];

  const chip = document.getElementById("device-pill");
  chip.classList.toggle("cuda", data.device === "cuda");
  chip.querySelector(".meta-chip-text").textContent =
    data.device === "cuda" ? "CUDA · Ready" : `${String(data.device).toUpperCase()} · Ready`;

  const ids = new Set(state.models.map((m) => m.id));
  state.selectedSeats = state.selectedSeats.map((id, i) =>
    ids.has(id) ? id : i === 0 ? "human" : firstRlId(state.seatCount)
  );
  state.tSelected = new Set([...state.tSelected].filter((id) => ids.has(id)));
  if (state.tSelected.size < 2) {
    state.models
      .filter((m) => m.category !== "human" && m.players.includes(2))
      .slice(0, 5)
      .forEach((m) => state.tSelected.add(m.id));
  }

  renderSeatSelectors();
  renderTournamentBotList();
}

function switchTab(tabName) {
  document.querySelectorAll(".nav-tab").forEach((b) => {
    const active = b.dataset.tab === tabName;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${tabName}`);
  });
  if (tabName === "benchmarks") loadBenchmarks();
}

function initTabs() {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  document.getElementById("wordmark-link").addEventListener("click", (e) => {
    e.preventDefault();
    switchTab("arena");
  });
}

// ---------------------------------------------------------------------------
// Arena Setup
// ---------------------------------------------------------------------------
function renderSeatSelectors() {
  const container = document.getElementById("seat-selectors");
  const compatible = state.models.filter((m) => m.players.includes(state.seatCount));

  while (state.selectedSeats.length < state.seatCount) {
    state.selectedSeats.push(
      state.selectedSeats.length === 1 ? firstRlId(state.seatCount) : "baseline:card_counter"
    );
  }
  state.selectedSeats = state.selectedSeats.slice(0, state.seatCount);

  const compatIds = new Set(compatible.map((m) => m.id));
  state.selectedSeats = state.selectedSeats.map((id, idx) =>
    compatIds.has(id) ? id : idx === 0 ? "human" : firstRlId(state.seatCount)
  );

  const groups = [
    ["human", "Interactive"],
    ["rl", "Trained RL Policies"],
    ["baseline", "Scripted Baselines"],
  ];

  container.innerHTML = state.selectedSeats
    .map((curId, seatIdx) => {
      const opts = groups
        .map(([cat, label]) => {
          const items = compatible.filter((m) => m.category === cat);
          if (!items.length) return "";
          return `<optgroup label="${esc(label)}">${items
            .map((m) => {
              const eloStr = m.elo ? ` (Elo ${m.elo})` : "";
              return `<option value="${esc(m.id)}" ${m.id === curId ? "selected" : ""}>${esc(m.name)}${eloStr}</option>`;
            })
            .join("")}</optgroup>`;
        })
        .join("");
      return `
        <div class="seat-field">
          <label><span>Seat ${seatIdx}</span><span class="mono">${seatIdx === 0 ? "South" : `Opp ${seatIdx}`}</span></label>
          <select data-seat="${seatIdx}">${opts}</select>
        </div>
      `;
    })
    .join("");

  container.querySelectorAll("select").forEach((sel) => {
    sel.addEventListener("change", () => {
      state.selectedSeats[Number(sel.dataset.seat)] = sel.value;
    });
  });
}

function applyPreset(preset) {
  if (preset === "human-vs-pfsp") {
    state.seatCount = 2;
    state.selectedSeats = ["human", firstRlId(2, "rl:ppo_pfsp_2p")];
  } else if (preset === "human-vs-ddqn") {
    state.seatCount = 2;
    state.selectedSeats = ["human", firstRlId(2, "rl:ddqn_pfsp_2p")];
  } else if (preset === "pfsp-vs-ddqn") {
    state.seatCount = 2;
    state.selectedSeats = [firstRlId(2, "rl:ppo_pfsp_2p"), firstRlId(2, "rl:ddqn_pfsp_2p")];
    document.getElementById("opt-xray").checked = true;
  } else if (preset === "4p-mixed") {
    state.seatCount = 4;
    state.selectedSeats = [
      "human",
      firstRlId(4, "rl:ppo_pfsp_4p"),
      firstRlId(4, "rl:ddqn_pfsp_4p"),
      "baseline:card_counter",
    ];
  }
  document.querySelectorAll("#seat-count-toggle .toggle-btn").forEach((b) => {
    b.classList.toggle("active", Number(b.dataset.seats) === state.seatCount);
  });
  renderSeatSelectors();
}

// ---------------------------------------------------------------------------
// Game Loop & Tabletop Rendering
// ---------------------------------------------------------------------------
function stopAutoPlay() {
  if (state.autoPlayTimer) {
    clearTimeout(state.autoPlayTimer);
    state.autoPlayTimer = null;
  }
  const btn = document.getElementById("btn-autoplay");
  if (btn) btn.textContent = "Play";
}

async function startNewGame() {
  stopAutoPlay();
  const btn = document.getElementById("start-game-btn");
  btn.disabled = true;
  btn.textContent = "Dealing…";
  try {
    const payload = {
      seats: state.selectedSeats,
      seed: Number(document.getElementById("game-seed").value || 0),
      show_inspector: document.getElementById("opt-inspector").checked,
      reveal_ai_hands: document.getElementById("opt-xray").checked,
      auto_advance: false,
    };
    const game = await apiPost("/api/game/new", payload);
    state.game = game;
    renderGame();
    if (game.has_human && !game.human_turn && !game.done) {
      await advanceAiUntilHuman();
    }
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Deal Hand";
  }
}

async function stepSingleAi() {
  if (!state.game || state.game.done || state.game.human_turn || state.stepping) return;
  state.stepping = true;
  try {
    const game = await apiPost(`/api/game/${state.game.game_id}/step`, {
      reveal_ai_hands: document.getElementById("opt-xray").checked,
    });
    state.game = game;
    renderGame();
  } catch (err) {
    stopAutoPlay();
    toast(err.message);
  } finally {
    state.stepping = false;
  }
}

async function advanceAiUntilHuman() {
  while (state.game && !state.game.done && !state.game.human_turn) {
    await stepSingleAi();
    await new Promise((r) => setTimeout(r, 170));
  }
}

async function submitHumanAction(actionId) {
  if (!state.game || !state.game.human_turn || state.stepping) return;
  state.stepping = true;
  document.querySelectorAll(".act-btn").forEach((b) => (b.disabled = true));
  try {
    const game = await apiPost(`/api/game/${state.game.game_id}/action`, {
      action_id: actionId,
      auto_advance: false,
    });
    state.game = game;
    renderGame();
  } catch (err) {
    toast(err.message);
  } finally {
    state.stepping = false;
  }
  if (state.game && !state.game.done && !state.game.human_turn) {
    await advanceAiUntilHuman();
  }
}

function toggleAutoPlay() {
  if (state.autoPlayTimer) {
    stopAutoPlay();
    return;
  }
  const btn = document.getElementById("btn-autoplay");
  btn.textContent = "Pause";
  const tick = async () => {
    if (!state.game || state.game.done || state.game.human_turn) {
      stopAutoPlay();
      return;
    }
    await stepSingleAi();
    if (state.game && !state.game.done && !state.game.human_turn && btn.textContent === "Pause") {
      const delay = Number(document.getElementById("speed-slider").value || 650);
      state.autoPlayTimer = setTimeout(tick, delay);
    } else {
      stopAutoPlay();
    }
  };
  tick();
}

function renderCard(role, dead = false) {
  const meta = ROLE_META[role] || ROLE_META.HIDDEN;
  const title = role === "HIDDEN" ? "Hidden" : role[0] + role.slice(1).toLowerCase();
  return `
    <div class="card role-${esc(role)} ${dead ? "is-dead" : ""}">
      ${dead ? '<span class="dead-stamp">LOST</span>' : ""}
      <div class="card-inner">
        <span class="card-top-pip">${meta.pip}</span>
        <span class="card-emblem">${meta.svg}</span>
        <div>
          <div class="card-name">${esc(title)}</div>
          <div class="card-sub">${esc(meta.sub)}</div>
        </div>
      </div>
    </div>
  `;
}

function renderSeatBlock(s, done) {
  const liveCards = s.hand.map((c) => renderCard(c, false)).join("");
  const deadCards = s.revealed.map((c) => renderCard(c, true)).join("");
  const foot = done
    ? `<span>bluff ${(s.bluff_rate * 100).toFixed(0)}%</span><span>challenges ${s.challenges_won}/${s.challenges_attempted}</span>`
    : `<span>influence ${s.influence}/2</span><span>proven ${s.claims_proven}</span><span>caught ${s.caught_bluffing}</span>`;
  return `
    <div class="seat-block ${s.is_current ? "is-turn" : ""} ${s.alive ? "" : "is-out"}">
      <div class="seat-head">
        <div class="seat-ident">
          ${s.is_current ? '<span class="turn-pip"></span>' : ""}
          <span class="seat-name">${esc(s.name)}</span>
          <span class="seat-idx">S${s.seat}</span>
        </div>
        <div class="coin-counter" title="${s.coins} coins">${COIN_SVG}<span>${s.coins}</span></div>
      </div>
      <div class="seat-cards">${liveCards}${deadCards}</div>
      <div class="seat-foot">${foot}</div>
    </div>
  `;
}

function renderGame() {
  const g = state.game;
  if (!g) return;

  document.getElementById("hud-turn").textContent = `Turn ${g.turn}`;
  document.getElementById("hud-phase").textContent = g.phase.replaceAll("_", " ");
  document.getElementById("hud-deck").textContent = `${g.deck_size} in court`;

  const specCtrl = document.getElementById("spectate-controls");
  specCtrl.style.display = !g.has_human && !g.done ? "flex" : "none";

  // Arrange seats spatially: Seat 0 at South, remaining seats along North
  const southSeat = g.seats[0];
  const oppSeats = g.seats.slice(1);

  // Center status banner
  const p = g.pending;
  let centerHtml = "";
  if (g.done) {
    centerHtml = g.truncated
      ? `<div class="court-winner">Turn Limit Reached (${g.turn} turns)</div>`
      : `<div class="court-winner">${esc(g.winner_name)} Wins the Court</div>
         <div class="deck-visual">Completed in ${g.turn} turns · All hands revealed</div>`;
  } else if (p && p.action) {
    const actor = p.actor !== null ? g.seats[p.actor]?.name : "?";
    const target = p.target !== null ? ` on ${g.seats[p.target]?.name}` : "";
    let line = `<strong>${esc(actor)}</strong> declares ${esc(p.action.replaceAll("_", " "))}${esc(target)}`;
    if (p.blocker !== null && p.block_role) {
      line += ` · <strong>${esc(g.seats[p.blocker]?.name)}</strong> blocks as ${esc(p.block_role)}`;
    }
    const alert = p.claimed_impossible
      ? `<div class="court-alert">IMPOSSIBLE CLAIM — ALL 3 COPIES ACCOUNTED FOR</div>`
      : "";
    centerHtml = `<div class="court-status">${line}</div>${alert}`;
  } else {
    const activeName = g.seats[g.current_player]?.name || "";
    centerHtml = `
      <div class="court-status">Awaiting <strong>${esc(activeName)}</strong></div>
      <div class="deck-visual">${g.deck_size} cards remaining in Court deck</div>
    `;
  }

  const board = document.getElementById("table-board");
  board.innerHTML = `
    <div class="opponents-row count-${oppSeats.length}">
      ${oppSeats.map((s) => renderSeatBlock(s, g.done)).join("")}
    </div>
    <div class="court-center">${centerHtml}</div>
    <div class="south-row">${renderSeatBlock(southSeat, g.done)}</div>
  `;

  // Action dock
  const dock = document.getElementById("action-dock");
  const btns = document.getElementById("action-buttons");
  if (g.human_turn && g.legal_actions.length) {
    dock.classList.remove("hidden");
    document.getElementById("action-prompt").textContent = `Your Move — ${g.phase.replaceAll("_", " ")}`;
    document.getElementById("action-hint").textContent =
      g.legal_actions.length <= 9 ? `Press 1–${g.legal_actions.length} or click` : `${g.legal_actions.length} legal options`;
    btns.innerHTML = g.legal_actions
      .map((a, idx) => {
        const tag =
          a.truthful === true
            ? '<span class="tag-truth">Truth</span>'
            : a.truthful === false
            ? '<span class="tag-bluff">Bluff</span>'
            : "";
        const kbd = idx < 9 ? `<span class="kbd">${idx + 1}</span>` : "";
        return `<button class="act-btn kind-${esc(a.kind)}" data-aid="${a.action_id}" type="button">${kbd}<span>${esc(a.label)}</span>${tag}</button>`;
      })
      .join("");
    btns.querySelectorAll(".act-btn").forEach((b) => {
      b.addEventListener("click", () => submitHumanAction(Number(b.dataset.aid)));
    });
  } else {
    dock.classList.add("hidden");
  }

  renderInspector(g.last_inspector);

  const logEl = document.getElementById("game-log");
  document.getElementById("log-count").textContent = String(g.log.length);
  if (!g.log.length) {
    logEl.innerHTML = `<p class="muted">Game dealt — awaiting first move.</p>`;
  } else {
    logEl.innerHTML = [...g.log]
      .reverse()
      .map((entry) => {
        const bluffTag = entry.is_bluff ? `<span class="ledger-bluff">BLUFF</span>` : "";
        return `
          <div class="ledger-row">
            <span class="ledger-turn">T${entry.turn}</span>
            <div>
              <span class="ledger-actor">${esc(entry.actor_name)}</span>${bluffTag}
              <div class="ledger-text">${esc(entry.text)}</div>
            </div>
          </div>
        `;
      })
      .join("");
  }
}

function renderInspector(ins) {
  const badge = document.getElementById("inspector-badge");
  const body = document.getElementById("inspector-content");
  if (!ins) {
    badge.textContent = "Idle";
    body.innerHTML = `<p class="muted">Action distribution and critic / Q-value estimate appear here after each AI decision.</p>`;
    return;
  }
  if (ins.kind === "neural") {
    badge.textContent = ins.name;
    const bars = (ins.top_actions || [])
      .map(
        (row) => `
        <div class="prob-row">
          <div class="prob-meta">
            <span class="prob-label">${esc(row.label)}</span>
            <span class="mono">${(row.prob * 100).toFixed(1)}%</span>
          </div>
          <div class="prob-track"><div class="prob-fill" style="width:${Math.min(100, row.prob * 100).toFixed(1)}%"></div></div>
        </div>
      `
      )
      .join("");
    const valRow =
      ins.value !== null && ins.value !== undefined
        ? `<div class="val-row"><span class="muted">${esc(ins.value_label || "Critic V(s)")}</span><span>${ins.value >= 0 ? "+" : ""}${ins.value.toFixed(3)}</span></div>`
        : "";
    body.innerHTML = bars + valRow;
  } else {
    badge.textContent = "Scripted";
    body.innerHTML = `<p><strong>${esc(ins.name)}</strong> — <span class="muted">${esc(ins.note || "")}</span></p>`;
  }
}

// ---------------------------------------------------------------------------
// Tournament Tab
// ---------------------------------------------------------------------------
function renderTournamentBotList() {
  const container = document.getElementById("t-bot-list");
  const reqPlayers = state.tMode === "round_robin_2p" ? 2 : 4;
  const eligible = state.models.filter(
    (m) => m.category !== "human" && m.players.includes(reqPlayers)
  );

  if (state.tMode === "round_robin_2p") {
    document.getElementById("t-mode-hint").textContent =
      "Seat-balanced round-robin with live Bradley-Terry Elo and 95% Wilson intervals.";
    document.getElementById("t-games-label").textContent = "Games / pair";
    const groups = [
      ["rl", "Trained RL Policies"],
      ["baseline", "Scripted Baselines"],
    ];
    container.innerHTML = groups
      .map(([cat, title]) => {
        const items = eligible.filter((m) => m.category === cat);
        if (!items.length) return "";
        return (
          `<div class="t-group-head">${esc(title)}</div>` +
          items
            .map(
              (m) => `
            <label class="t-check">
              <span class="t-check-left">
                <input type="checkbox" value="${esc(m.id)}" ${state.tSelected.has(m.id) ? "checked" : ""} />
                <span class="t-check-name">${esc(m.name)}</span>
              </span>
              <span class="t-check-meta">${m.elo ? m.elo : esc(m.badge)}</span>
            </label>
          `
            )
            .join("")
        );
      })
      .join("");

    container.querySelectorAll("input[type=checkbox]").forEach((cb) => {
      cb.addEventListener("change", () => {
        if (cb.checked) state.tSelected.add(cb.value);
        else state.tSelected.delete(cb.value);
      });
    });
  } else {
    document.getElementById("t-mode-hint").textContent =
      "Seat-rotated 4-player table championship tracking win rate and average finish rank.";
    document.getElementById("t-games-label").textContent = "Total games (×4)";
    const compatIds = new Set(eligible.map((m) => m.id));
    state.tFourSeats = state.tFourSeats.map((id, i) =>
      compatIds.has(id) ? id : i === 0 ? firstRlId(4) : "baseline:card_counter"
    );
    container.innerHTML = state.tFourSeats
      .map(
        (curId, idx) => `
        <div class="seat-field" style="margin-bottom:8px;">
          <label><span>Seat ${idx + 1}</span><span class="mono">Rotated</span></label>
          <select data-tseat="${idx}">
            ${eligible
              .map(
                (m) =>
                  `<option value="${esc(m.id)}" ${m.id === curId ? "selected" : ""}>${esc(m.name)}</option>`
              )
              .join("")}
          </select>
        </div>
      `
      )
      .join("");
    container.querySelectorAll("select").forEach((sel) => {
      sel.addEventListener("change", () => {
        state.tFourSeats[Number(sel.dataset.tseat)] = sel.value;
      });
    });
  }
}

async function startTournament() {
  const errBox = document.getElementById("t-error");
  errBox.classList.add("hidden");
  const bots =
    state.tMode === "round_robin_2p" ? [...state.tSelected] : [...state.tFourSeats];
  try {
    const job = await apiPost("/api/tournament/start", {
      mode: state.tMode,
      bots,
      games_per_pair: Number(document.getElementById("t-games").value || 40),
      seed: Number(document.getElementById("t-seed").value || 42),
    });
    state.tId = job.tournament_id;
    renderTournament(job);
    pollTournament();
  } catch (err) {
    errBox.textContent = err.message;
    errBox.classList.remove("hidden");
  }
}

async function cancelTournament() {
  if (!state.tId) return;
  try {
    const job = await apiPost(`/api/tournament/${state.tId}/cancel`, {});
    renderTournament(job);
  } catch (err) {
    toast(err.message);
  }
}

function pollTournament() {
  clearTimeout(state.tPollTimer);
  if (!state.tId) return;
  state.tPollTimer = setTimeout(async () => {
    try {
      const job = await apiGet(`/api/tournament/${state.tId}`);
      renderTournament(job);
      if (job.status === "running") pollTournament();
    } catch (err) {
      toast(err.message);
    }
  }, 400);
}

function heatBg(rate) {
  const diff = (rate - 0.5) * 2;
  if (diff >= 0) {
    return `rgba(92, 148, 110, ${(0.08 + diff * 0.34).toFixed(2)})`;
  }
  return `rgba(201, 68, 50, ${(0.08 + Math.abs(diff) * 0.34).toFixed(2)})`;
}

function renderTournament(job) {
  const isRunning = job.status === "running";
  document.getElementById("t-start-btn").disabled = isRunning;
  document.getElementById("t-cancel-btn").classList.toggle("hidden", !isRunning);

  const errBox = document.getElementById("t-error");
  if (job.error) {
    errBox.textContent = job.error;
    errBox.classList.remove("hidden");
  } else {
    errBox.classList.add("hidden");
  }

  document.getElementById("t-title").textContent =
    job.mode === "round_robin_2p" ? "2-Player Round-Robin" : "4-Player Table Championship";
  document.getElementById("t-subtitle").textContent = isRunning
    ? `Playing ${job.current_matchup}`
    : `${job.status[0].toUpperCase() + job.status.slice(1)} · ${job.completed_games} of ${job.total_games} games in ${job.elapsed_s}s`;

  document.getElementById("t-progress-text").textContent = `${job.completed_games} / ${job.total_games}`;
  document.getElementById("t-speed-text").textContent = `${job.games_per_s} g/s`;
  document.getElementById("t-eta-text").textContent = isRunning ? `${job.eta_s}s left` : `${job.elapsed_s}s`;
  document.getElementById("t-progress-fill").style.width = `${(job.progress * 100).toFixed(1)}%`;

  const wrap = document.getElementById("t-standings-wrap");
  if (job.mode === "round_robin_2p") {
    const rows = (job.standings || [])
      .map(
        (r, i) => `
        <tr>
          <td class="num muted">${i + 1}</td>
          <td><strong>${esc(r.name)}</strong><span class="cat-tag">${esc(r.category)}</span></td>
          <td class="num elo-num">${r.elo.toFixed(1)}</td>
          <td class="num">${(r.win_rate * 100).toFixed(1)}%</td>
          <td class="num muted">[${(r.ci_low * 100).toFixed(0)}%, ${(r.ci_high * 100).toFixed(0)}%]</td>
          <td class="num">${r.wins}–${r.losses}${r.draws ? `–${r.draws}` : ""}</td>
          <td class="num">${(r.bluff_rate * 100).toFixed(1)}%</td>
          <td class="num">${(r.challenge_precision * 100).toFixed(1)}%</td>
        </tr>
      `
      )
      .join("");
    wrap.innerHTML = `
      <table class="ledger-table">
        <thead>
          <tr>
            <th class="num">#</th><th>Competitor</th><th class="num">BT Elo</th>
            <th class="num">Win %</th><th class="num">95% CI</th><th class="num">Record</th>
            <th class="num">Bluff %</th><th class="num">Chall. Prec.</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;

    const matSec = document.getElementById("t-matrix-section");
    matSec.classList.remove("hidden");
    const bots = job.bots || [];
    const header = bots.map((b) => `<th class="num">${esc(b.name)}</th>`).join("");
    const mRows = bots
      .map((rb, i) => {
        const cells = bots
          .map((cb, j) => {
            if (i === j) return `<td class="heat-cell heat-diag">—</td>`;
            const c = job.matrix[i][j];
            if (!c) return `<td class="heat-cell muted">·</td>`;
            const tip = `${rb.name} vs ${cb.name}: ${c.wins}–${c.losses} (95% CI ${(c.ci_low * 100).toFixed(1)}%–${(c.ci_high * 100).toFixed(1)}%)`;
            return `<td class="heat-cell" style="background:${heatBg(c.rate)}" title="${esc(tip)}">${(c.rate * 100).toFixed(0)}%</td>`;
          })
          .join("");
        return `<tr><td><strong>${esc(rb.name)}</strong></td>${cells}</tr>`;
      })
      .join("");
    document.getElementById("t-matrix-wrap").innerHTML = `
      <table class="ledger-table">
        <thead><tr><th>Row vs Column</th>${header}</tr></thead>
        <tbody>${mRows}</tbody>
      </table>
    `;
  } else {
    document.getElementById("t-matrix-section").classList.add("hidden");
    const rows = (job.standings || [])
      .map(
        (r, i) => `
        <tr>
          <td class="num muted">${i + 1}</td>
          <td><strong>${esc(r.name)}</strong></td>
          <td class="num elo-num">${(r.win_rate * 100).toFixed(1)}%</td>
          <td class="num muted">[${(r.ci_low * 100).toFixed(0)}%, ${(r.ci_high * 100).toFixed(0)}%]</td>
          <td class="num">${r.avg_rank.toFixed(2)}</td>
          <td class="num">${r.ranks["1"] || 0} / ${r.ranks["2"] || 0} / ${r.ranks["3"] || 0} / ${r.ranks["4"] || 0}</td>
          <td class="num">${(r.bluff_rate * 100).toFixed(1)}%</td>
        </tr>
      `
      )
      .join("");
    wrap.innerHTML = `
      <table class="ledger-table">
        <thead>
          <tr>
            <th class="num">#</th><th>Seat</th><th class="num">Win %</th>
            <th class="num">95% CI</th><th class="num">Avg Rank</th>
            <th class="num">1st / 2nd / 3rd / 4th</th><th class="num">Bluff %</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  const recentSec = document.getElementById("t-recent-section");
  if (job.recent_matches && job.recent_matches.length) {
    recentSec.classList.remove("hidden");
    document.getElementById("t-recent-list").innerHTML = [...job.recent_matches]
      .reverse()
      .slice(0, 12)
      .map(
        (m) =>
          `<span class="recent-chip">#${m.game} ${esc(m.matchup)} → <strong>${esc(m.winner)}</strong> (${m.turns}t)</span>`
      )
      .join("");
  } else {
    recentSec.classList.add("hidden");
  }
}

// ---------------------------------------------------------------------------
// Lab Benchmarks Tab (Static Eval + Offline SLM Reference)
// ---------------------------------------------------------------------------
async function loadBenchmarks() {
  const sumEl = document.getElementById("bench-summary");
  const figEl = document.getElementById("bench-figures");
  try {
    const data = await apiGet("/api/benchmarks");
    const ev = data.eval_summary;
    const slm = data.slm_eval;

    let html = "";
    if (ev && ev.round_robin_2p) {
      const rr = ev.round_robin_2p;
      const top = rr.ranking[0];
      const topRate = top.win_rate ?? top.avg_win_rate ?? 0;
      const fourP = ev.four_player?.overall;
      const probe2 = ev.impossible_claim_probe?.two_player?.agents || {};
      const probe4 = ev.impossible_claim_probe?.four_player?.agents?.["PFSP-PPO-4P"];
      const expModels = ev.exploitability_2p?.models || {};
      const ddqnRank = rr.ranking.find((r) => r.agent === "Double-DQN");
      const totalGames = rr.total_games || 0;

      document.getElementById("bench-heading").textContent =
        `${rr.agents.length}×${rr.agents.length} Round-Robin (${totalGames.toLocaleString()} Games), Exploitability & Offline SLM Reference`;

      html += `<div class="kpi-strip">
        <div class="kpi-cell">
          <span class="kicker">Top 2P Policy</span>
          <div class="kpi-val">${esc(top.agent)}</div>
          <div class="kpi-sub">Elo ${top.elo.toFixed(1)} · ${(topRate * 100).toFixed(1)}% avg win</div>
        </div>
        ${
          ddqnRank
            ? `<div class="kpi-cell">
          <span class="kicker">Double-DQN (2P)</span>
          <div class="kpi-val">${ddqnRank.elo.toFixed(1)}</div>
          <div class="kpi-sub">${((ddqnRank.win_rate ?? 0) * 100).toFixed(1)}% avg · Probe ${probe2["Double-DQN"] ? (probe2["Double-DQN"].challenge_rate * 100).toFixed(1) + "%" : "—"}</div>
        </div>`
            : ""
        }
        <div class="kpi-cell">
          <span class="kicker">4P Scaling (Chance 25%)</span>
          <div class="kpi-val">${fourP ? (fourP.rate * 100).toFixed(1) + "%" : "—"}</div>
          <div class="kpi-sub">Avg rank ${fourP?.avg_finish_rank?.toFixed(2) || "—"} · Probe ${probe4 ? (probe4.challenge_rate * 100).toFixed(1) + "%" : "—"}</div>
        </div>
        ${
          slm
            ? `<div class="kpi-cell">
          <span class="kicker">Offline SLM Reference</span>
          <div class="kpi-val">${slm.anchored_elo.toFixed(1)}</div>
          <div class="kpi-sub">${esc(slm.model)} · ${(slm.impossible_claim_probe.challenge_rate * 100).toFixed(0)}% probe</div>
        </div>`
            : ""
        }
      </div>`;

      const rows = rr.ranking
        .map((r, i) => {
          const wr = r.win_rate ?? r.avg_win_rate ?? 0;
          const pr = probe2[r.agent];
          const ex = expModels[r.agent];
          return `
            <tr>
              <td class="num muted">${i + 1}</td>
              <td><strong>${esc(r.agent)}</strong></td>
              <td class="num elo-num">${r.elo.toFixed(1)}</td>
              <td class="num">${(wr * 100).toFixed(1)}%</td>
              <td class="num">${pr ? (pr.challenge_rate * 100).toFixed(1) + "%" : "—"}</td>
              <td class="num">${ex ? ex.approx_exploitability.toFixed(3) : "—"}</td>
            </tr>
          `;
        })
        .join("");

      const slmRow = slm
        ? `<tr class="slm-ref-row">
            <td class="num muted">Ref</td>
            <td><strong>${esc(slm.model)}</strong><span class="ref-badge">Offline SLM Benchmark</span></td>
            <td class="num elo-num">${slm.anchored_elo.toFixed(1)}</td>
            <td class="num">${(slm.avg_win_rate * 100).toFixed(1)}% <span class="muted">(vs 6)</span></td>
            <td class="num">${(slm.impossible_claim_probe.challenge_rate * 100).toFixed(1)}%</td>
            <td class="num muted">—</td>
          </tr>`
        : "";

      html += `
        <div class="table-wrap">
          <table class="ledger-table">
            <thead>
              <tr>
                <th class="num">#</th><th>Agent</th><th class="num">Bradley-Terry Elo</th>
                <th class="num">Avg Win %</th><th class="num">Impossible Probe</th>
                <th class="num">Approx. Exploitability</th>
              </tr>
            </thead>
            <tbody>${rows}${slmRow}</tbody>
          </table>
        </div>
      `;
    } else {
      html = `<p class="muted">No saved eval_summary.json found.</p>`;
    }
    sumEl.innerHTML = html;

    const bust = Date.now();
    figEl.innerHTML = (data.figures || [])
      .map((name) => {
        const cap = FIGURE_LABELS[name] || name;
        const src = `/figures/${encodeURIComponent(name)}?v=${bust}`;
        return `
          <figure class="fig-frame" data-src="${esc(src)}" data-cap="${esc(cap)}" style="margin:0;">
            <img src="${esc(src)}" alt="${esc(cap)}" loading="lazy" />
            <figcaption class="fig-caption"><span>${esc(cap)}</span><span>${esc(name)}</span></figcaption>
          </figure>
        `;
      })
      .join("");

    figEl.querySelectorAll(".fig-frame").forEach((el) => {
      el.addEventListener("click", () => openLightbox(el.dataset.src, el.dataset.cap));
    });
  } catch (err) {
    sumEl.innerHTML = `<div class="error-note">${esc(err.message)}</div>`;
  }
}

function openLightbox(src, cap) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-cap").textContent = cap;
  document.getElementById("lightbox").classList.remove("hidden");
}

function closeLightbox() {
  document.getElementById("lightbox").classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Events & Keyboard Shortcuts
// ---------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  initTabs();

  document.querySelectorAll("#seat-count-toggle .toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#seat-count-toggle .toggle-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.seatCount = Number(btn.dataset.seats);
      renderSeatSelectors();
    });
  });

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });

  document.getElementById("rand-seed-btn").addEventListener("click", () => {
    document.getElementById("game-seed").value = String(Math.floor(Math.random() * 100000));
  });

  document.getElementById("start-game-btn").addEventListener("click", startNewGame);
  document.getElementById("empty-deal-btn").addEventListener("click", () => {
    applyPreset("human-vs-pfsp");
    startNewGame();
  });
  document.getElementById("btn-step").addEventListener("click", stepSingleAi);
  document.getElementById("btn-autoplay").addEventListener("click", toggleAutoPlay);

  document.querySelectorAll("#t-mode-toggle .toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#t-mode-toggle .toggle-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.tMode = btn.dataset.tmode;
      renderTournamentBotList();
    });
  });

  document.getElementById("t-start-btn").addEventListener("click", startTournament);
  document.getElementById("t-cancel-btn").addEventListener("click", cancelTournament);
  document.getElementById("reload-bench-btn").addEventListener("click", loadBenchmarks);

  document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
  document.getElementById("lightbox").addEventListener("click", (e) => {
    if (e.target.id === "lightbox") closeLightbox();
  });

  // Number keys 1..9 trigger legal actions during human turn; Escape closes lightbox
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeLightbox();
      return;
    }
    if (e.target && ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (state.game && state.game.human_turn && !state.stepping) {
      const n = Number(e.key);
      if (Number.isInteger(n) && n >= 1 && n <= state.game.legal_actions.length) {
        submitHumanAction(state.game.legal_actions[n - 1].action_id);
      }
    }
  });

  refreshModels().catch((err) => toast(`Failed to load models: ${err.message}`));
});
