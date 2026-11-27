"""Publication-grade figures for the rloup Phase 3 results.

Reads ``results/<run>/metrics.jsonl`` (the shared training-metric schema) and
``results/eval_summary.json`` (produced by :mod:`rloup.eval.evaluate`) and writes
five 180-DPI PNGs::

    python -m rloup.eval.plots
    python -m rloup.eval.plots --results-root results --out-dir /tmp/figures

Figures
-------
``learning_curves_2p``      average/strong win rate plus entropy and value loss
``learning_curve_4p``      4-player PFSP win rate per baseline
``round_robin_2p``         8x8 win-rate heatmap and Elo bar chart
``behavioral_diagnostics`` bluff rate per claim kind, challenge precision and
                           controlled impossible-claim detection
``exploitability_2p``      exploiter win rate and approximate exploitability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "BLUFF_KIND_LABELS",
    "RUN_DIRS",
    "generate_all",
    "load_metrics",
    "load_summary",
    "plot_behavioral_diagnostics",
    "plot_exploitability_2p",
    "plot_learning_curve_4p",
    "plot_learning_curves_2p",
    "plot_round_robin_2p",
]

#: Display name -> results directory name.
RUN_DIRS: dict[str, str] = {
    "PFSP-PPO": "ppo_pfsp_2p",
    "Naive-PPO": "ppo_naive_2p",
    "NFSP": "nfsp_2p",
    "Double-DQN": "ddqn_pfsp_2p",
    "PFSP-PPO-4P": "ppo_pfsp_4p",
    "Double-DQN-4P": "ddqn_pfsp_4p",
}

COLORS: dict[str, str] = {
    "PFSP-PPO": "#1f4e79",
    "Naive-PPO": "#c0504d",
    "NFSP": "#4f8a3d",
    "Double-DQN": "#d97706",
    "PFSP-PPO-4P": "#7030a0",
    "Double-DQN-4P": "#b45309",
    "CardCounter": "#7f7f7f",
}

#: The five scripted baselines, in plotting order.
BASELINE_NAMES: tuple[str, ...] = ("Random", "Bluffer", "Honest", "Cautious", "CardCounter")

#: Claim kinds shown in the behavioural figure.
BLUFF_KIND_LABELS: tuple[tuple[str, str], ...] = (
    ("tax_duke", "Tax\n(DUKE)"),
    ("assassinate", "Assassinate\n(ASSASSIN)"),
    ("steal_captain", "Steal\n(CAPTAIN)"),
    ("exchange_ambassador", "Exchange\n(AMB.)"),
    ("block_duke", "Block\n(DUKE)"),
    ("block_contessa", "Block\n(CONTESSA)"),
    ("block_captain", "Block\n(CAPTAIN)"),
    ("block_ambassador", "Block\n(AMB.)"),
)

STYLE: dict[str, Any] = {
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
    "lines.linewidth": 1.6,
    "lines.markersize": 3.0,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``metrics.jsonl`` file into a list of records (empty if absent)."""
    target = Path(path)
    if target.is_dir():
        target = target / "metrics.jsonl"
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def load_summary(path: str | Path) -> dict[str, Any]:
    """Read ``eval_summary.json``."""
    target = Path(path)
    if target.is_dir():
        target = target / "eval_summary.json"
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found - run `python -m rloup.eval.evaluate` first"
        )
    return json.loads(target.read_text(encoding="utf-8"))


def _steps_in_millions(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / 1e6


def _annotate_bars(
    ax: plt.Axes,
    bars,
    values: Sequence[float],
    fmt: str = "{:.2f}",
    horizontal: bool = False,
) -> None:
    """Value labels on top of (or at the end of) a bar series."""
    for bar, value in zip(bars, values):
        if horizontal:
            ax.annotate(
                fmt.format(value),
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2.0),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
            )
        else:
            ax.annotate(
                fmt.format(value),
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )


# --------------------------------------------------------------------------- #
# 1. 2-player learning curves
# --------------------------------------------------------------------------- #
def plot_learning_curves_2p(
    results_dir: str | Path,
    out_path: str | Path,
    models: Sequence[str] = ("PFSP-PPO", "Naive-PPO", "NFSP", "Double-DQN"),
) -> Path:
    """Average and strong-baseline win rate plus the entropy / value-loss panel."""
    results_dir = Path(results_dir)
    out_path = Path(out_path)
    with plt.rc_context(STYLE):
        fig, (ax_rate, ax_loss) = plt.subplots(1, 2, figsize=(11.0, 4.0))

        for name in models:
            records = load_metrics(results_dir / RUN_DIRS[name])
            if not records:
                continue
            steps = _steps_in_millions([r["step"] for r in records])
            color = COLORS.get(name, None)
            ax_rate.plot(
                steps,
                [r["avg_vs_baselines"] for r in records],
                marker="o",
                color=color,
                label=f"{name} (all baselines)",
            )
            ax_rate.plot(
                steps,
                [r["strong_avg"] for r in records],
                marker="s",
                linestyle="--",
                color=color,
                alpha=0.75,
                label=f"{name} (strong baselines)",
            )
        ax_rate.axhline(0.5, color="#444444", linestyle=":", linewidth=1.0)
        ax_rate.set_xlabel("learner steps (millions)")
        ax_rate.set_ylabel("seat-balanced win rate")
        ax_rate.set_title("2-player win rate during training")
        ax_rate.set_ylim(0.0, 1.0)
        ax_rate.legend(loc="lower right", ncol=1)

        loss_ax = ax_loss
        value_ax = ax_loss.twinx()
        value_ax.grid(False)
        for name in ("PFSP-PPO", "Naive-PPO"):
            records = load_metrics(results_dir / RUN_DIRS[name])
            if not records:
                continue
            steps = _steps_in_millions([r["step"] for r in records])
            color = COLORS.get(name)
            loss_ax.plot(steps, [r["entropy"] for r in records], color=color, marker="o")
            value_ax.plot(
                steps,
                [r["value_loss"] for r in records],
                color=color,
                marker="x",
                linestyle="--",
                alpha=0.8,
            )
        from matplotlib.lines import Line2D

        loss_ax.set_xlabel("learner steps (millions)")
        loss_ax.set_ylabel("policy entropy")
        value_ax.set_ylabel("value loss")
        loss_ax.set_yscale("log")
        value_ax.set_yscale("log")
        loss_ax.set_title("Optimisation diagnostics")
        handles = [
            Line2D([], [], color=COLORS["PFSP-PPO"], marker="o", label="PFSP-PPO entropy"),
            Line2D([], [], color=COLORS["Naive-PPO"], marker="o", label="Naive-PPO entropy"),
            Line2D(
                [],
                [],
                color=COLORS["PFSP-PPO"],
                marker="x",
                linestyle="--",
                label="PFSP-PPO value loss",
            ),
            Line2D(
                [],
                [],
                color=COLORS["Naive-PPO"],
                marker="x",
                linestyle="--",
                label="Naive-PPO value loss",
            ),
        ]
        loss_ax.legend(handles=handles, loc="upper right")
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 2. 4-player learning curve
# --------------------------------------------------------------------------- #
def plot_learning_curve_4p(results_dir: str | Path, out_path: str | Path) -> Path:
    """4-player PFSP-PPO win rate against every baseline, with the 25% line."""
    results_dir = Path(results_dir)
    out_path = Path(out_path)
    records = load_metrics(Path(results_dir) / RUN_DIRS["PFSP-PPO-4P"])
    if not records:
        raise FileNotFoundError("no 4-player metrics found")
    steps = _steps_in_millions([r["step"] for r in records])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        palette = {
            "Random": "#9e9e9e",
            "Bluffer": "#e07b39",
            "Honest": "#2e7d32",
            "Cautious": "#1565c0",
            "CardCounter": "#6a1b9a",
        }
        for name in BASELINE_NAMES:
            key = f"win_rate_{name.lower()}"
            if key not in records[0]:
                key = {
                    "Cautious": "win_rate_cautious_skeptical",
                    "Bluffer": "win_rate_aggressive_bluffer",
                    "CardCounter": "win_rate_card_counter",
                }.get(name, key)
            values = [r[key] for r in records]
            ax.plot(steps, values, marker="o", color=palette[name], label=f"vs {name}")
        ax.plot(
            steps,
            [r["avg_vs_baselines"] for r in records],
            color="#111111",
            linewidth=2.2,
            marker="s",
            label="average",
        )
        ax.axhline(0.25, color="#c62828", linestyle="--", linewidth=1.1)
        ax.annotate(
            "chance = 25%",
            xy=(steps[0], 0.25),
            xytext=(4, -11),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color="#c62828",
        )
        ax.set_xlabel("learner steps (millions)")
        ax.set_ylabel("win rate (seat-rotated, 4 players)")
        ax.set_title("4-player PFSP-PPO: win rate per baseline table")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper left", ncol=2)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 3. Round robin heatmap + Elo
# --------------------------------------------------------------------------- #
def plot_round_robin_2p(
    summary: dict[str, Any], out_path: str | Path
) -> Path:
    """8x8 annotated win-rate heatmap plus the Elo bar chart."""
    out_path = Path(out_path)
    section = summary.get("round_robin_2p") or {}
    names = section.get("agents") or []
    matrix = np.asarray(section.get("matrix") or [], dtype=np.float64)
    elo = section.get("elo") or {}
    if not names or matrix.size == 0:
        raise ValueError("eval_summary.json has no round_robin_2p section")
    with plt.rc_context(STYLE):
        fig, (ax_map, ax_elo) = plt.subplots(
            1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]}
        )
        size = len(names)
        masked = np.ma.masked_where(np.arange(size)[:, None] == np.arange(size)[None, :], matrix)
        image = ax_map.imshow(
            masked, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto"
        )
        ax_map.set_xticks(range(size), names, rotation=35, ha="right")
        ax_map.set_yticks(range(size), names)
        ax_map.set_xlabel("opponent")
        ax_map.set_ylabel("agent")
        ax_map.set_title("Head-to-head win rate (row vs column), 1,000 games/pair")
        ax_map.grid(False)
        for row in range(size):
            for col in range(size):
                if row == col:
                    ax_map.text(col, row, "-", ha="center", va="center", fontsize=7, color="#555555")
                    continue
                value = matrix[row, col]
                ax_map.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#111111" if 0.25 < value < 0.75 else "#ffffff",
                )
        colorbar = fig.colorbar(image, ax=ax_map, fraction=0.046, pad=0.02)
        colorbar.set_label("win rate")

        order = sorted(names, key=lambda name: elo.get(name, 0.0))
        values = [elo.get(name, 0.0) for name in order]
        colors = [
            COLORS.get(name, "#1f4e79")
            if name in ("PFSP-PPO", "Naive-PPO", "NFSP", "Double-DQN")
            else "#8fa8c0"
            for name in order
        ]
        bars = ax_elo.barh(order, values, color=colors, height=0.68)
        _annotate_bars(ax_elo, bars, values, fmt="{:.0f}", horizontal=True)
        ax_elo.axvline(1000.0, color="#c62828", linestyle="--", linewidth=1.1)
        ax_elo.annotate(
            "Random = 1000",
            xy=(1000.0, len(order) - 0.35),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color="#c62828",
            va="center",
        )
        low = min(values) - 40.0
        high = max(values) + 40.0
        ax_elo.set_xlim(low, high)
        ax_elo.set_xlabel("Bradley-Terry Elo")
        ax_elo.set_title("Elo ratings (anchored at Random = 1000)")
        ax_elo.grid(axis="y", visible=False)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 4. Behavioural diagnostics
# --------------------------------------------------------------------------- #
def plot_behavioral_diagnostics(
    summary: dict[str, Any], out_path: str | Path
) -> Path:
    """Bluff rate per claim kind, challenge precision and probe detection."""
    out_path = Path(out_path)
    behaviour = summary.get("behaviour") or {}
    probe_2p = (summary.get("impossible_claim_probe") or {}).get("two_player", {}).get("agents", {})
    probe_4p = (summary.get("impossible_claim_probe") or {}).get("four_player", {}).get("agents", {})
    probe = {**probe_4p, **probe_2p}
    agents = [
        name
        for name in ("PFSP-PPO", "Double-DQN", "Naive-PPO", "NFSP", "PFSP-PPO-4P")
        if name in behaviour
    ]
    if not agents:
        raise ValueError("eval_summary.json has no behaviour section")
    with plt.rc_context(STYLE):
        fig, (ax_kind, ax_quality) = plt.subplots(
            1, 2, figsize=(12.2, 4.4), gridspec_kw={"width_ratios": [1.45, 1.0]}
        )

        labels = [label for _, label in BLUFF_KIND_LABELS]
        keys = [key for key, _ in BLUFF_KIND_LABELS]
        positions = np.arange(len(keys), dtype=np.float64)
        width = 0.8 / max(len(agents), 1)
        for index, agent in enumerate(agents):
            rates = behaviour[agent].get("bluff_rate_by_kind", {})
            values = [float(rates.get(key, 0.0)) for key in keys]
            ax_kind.bar(
                positions + (index - (len(agents) - 1) / 2.0) * width,
                values,
                width=width,
                label=agent,
                color=COLORS.get(agent),
            )
        overall = [float(behaviour[agent].get("bluff_rate", 0.0)) for agent in agents]
        for index, agent in enumerate(agents):
            ax_kind.axhline(
                overall[index],
                color=COLORS.get(agent),
                linestyle=":",
                linewidth=0.9,
                alpha=0.7,
            )
        ax_kind.set_xticks(positions, labels, fontsize=7)
        ax_kind.set_ylabel("bluff rate (claims without the role)")
        ax_kind.set_title("Bluff rate by claim kind (dotted = pooled rate)")
        ax_kind.set_ylim(0.0, 1.05)
        ax_kind.legend(loc="upper right", ncol=2)

        positions = np.arange(len(agents), dtype=np.float64)
        width = 0.36
        precision = [
            float(behaviour[name].get("challenge_precision", 0.0)) for name in agents
        ]
        impossible = [
            float(probe.get(name, {}).get("challenge_rate", 0.0)) for name in agents
        ]
        ax_quality.bar(
            positions - width / 2.0,
            precision,
            width=width,
            color="#1f4e79",
            label="challenge precision",
        )
        ax_quality.bar(
            positions + width / 2.0,
            impossible,
            width=width,
            color="#c0504d",
            label="impossible-claim detection",
        )
        # The scripted card counter is the reference: it deduces every
        # provably impossible claim, so its rate is exactly 1.0.
        reference = float(probe.get("CardCounter", {}).get("challenge_rate", 1.0))
        ax_quality.axhline(
            reference,
            color="#c0504d",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax_quality.annotate(
            f"CardCounter (exact) = {reference:.2f}",
            xy=(positions[-1] + 0.45, reference),
            xytext=(0, -12),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color="#c0504d",
        )
        random_rate = probe.get("Random", {}).get("challenge_rate")
        if random_rate is not None:
            ax_quality.axhline(
                0.5,
                color="#777777",
                linestyle=":",
                linewidth=1.0,
            )
            ax_quality.annotate(
                "random = 0.50",
                xy=(positions[-1] + 0.45, 0.5),
                xytext=(0, -11),
                textcoords="offset points",
                ha="right",
                fontsize=7,
                color="#777777",
            )
        ax_quality.set_xticks(positions, agents, rotation=20, ha="right")
        ax_quality.set_ylim(0.0, 1.3)
        ax_quality.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_quality.set_ylabel("rate")
        ax_quality.set_title("Challenge quality and card-counting deduction")
        ax_quality.legend(loc="upper left", ncol=2)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 5. Exploitability
# --------------------------------------------------------------------------- #
def plot_exploitability_2p(summary: dict[str, Any], out_path: str | Path) -> Path:
    """Exploiter win rate with 95% error bars and the approximate exploitability."""
    out_path = Path(out_path)
    section = summary.get("exploitability_2p") or {}
    models = section.get("models") or {}
    if not models:
        raise ValueError("eval_summary.json has no exploitability_2p section")
    names = list(models.keys())
    win_rates = [float(models[name]["exploiter_win_rate"]) for name in names]
    lows = [float(models[name]["ci95_low"]) for name in names]
    highs = [float(models[name]["ci95_high"]) for name in names]
    err_low = [rate - low for rate, low in zip(win_rates, lows)]
    err_high = [high - rate for rate, high in zip(win_rates, highs)]
    exploitability = [float(models[name]["approx_exploitability"]) for name in names]

    with plt.rc_context(STYLE):
        fig, (ax_rate, ax_gap) = plt.subplots(1, 2, figsize=(10.4, 4.0))
        positions = np.arange(len(names), dtype=np.float64)
        colors = [COLORS.get(name, "#555555") for name in names]
        bars = ax_rate.bar(
            positions,
            win_rates,
            width=0.6,
            color=colors,
            yerr=[err_low, err_high],
            capsize=4,
            error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
        )
        _annotate_bars(ax_rate, bars, win_rates, fmt="{:.3f}")
        ax_rate.axhline(0.5, color="#c62828", linestyle="--", linewidth=1.2)
        ax_rate.annotate(
            "Nash equilibrium = 50%",
            xy=(positions[-1] + 0.32, 0.5),
            xytext=(0, 7),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="#c62828",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
        )
        ax_rate.set_xticks(positions, [f"vs {name}" for name in names])
        ax_rate.set_ylim(0.0, max(1.0, max(highs) + 0.08))
        ax_rate.set_ylabel("exploiter win rate (95% Wilson CI)")
        ax_rate.set_title("Trained exploiter vs frozen target")

        bars = ax_gap.bar(positions, exploitability, width=0.6, color=colors)
        _annotate_bars(ax_gap, bars, exploitability, fmt="{:.3f}")
        ax_gap.set_xticks(positions, [f"vs {name}" for name in names])
        ax_gap.set_ylabel("approx exploitability = max(0, win rate - 0.5)")
        ax_gap.set_title("Approximate exploitability (lower is better)")
        ax_gap.set_ylim(0.0, max(0.2, max(exploitability) * 1.35 + 1e-3))
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def generate_all(
    results_dir: str | Path = "results",
    out_dir: str | Path = "results/figures",
    summary_path: str | Path | None = None,
) -> list[Path]:
    """Write all five figures and return their paths."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(summary_path if summary_path is not None else results_dir)
    paths = [
        plot_learning_curves_2p(results_dir, out_dir / "learning_curves_2p.png"),
        plot_learning_curve_4p(results_dir, out_dir / "learning_curve_4p.png"),
        plot_round_robin_2p(summary, out_dir / "round_robin_2p.png"),
        plot_behavioral_diagnostics(summary, out_dir / "behavioral_diagnostics.png"),
        plot_exploitability_2p(summary, out_dir / "exploitability_2p.png"),
    ]
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rloup.eval.plots",
        description="Render the Phase 3 figures from metrics.jsonl and eval_summary.json.",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--summary", default=None, help="override eval_summary.json path")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = build_parser().parse_args(argv)
    paths = generate_all(
        results_dir=args.results_root,
        out_dir=args.out_dir,
        summary_path=args.summary,
    )
    if not args.quiet:
        for path in paths:
            size = path.stat().st_size if path.exists() else 0
            print(f"wrote {path} ({size / 1024:.0f} KiB)", flush=True)
    return paths


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
