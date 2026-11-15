"""Training report: one PNG with every curve that matters, from a run's metrics.jsonl."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl.metrics import read_metrics
from viz.theme import VizConfig


def smooth(values, window: int) -> np.ndarray:
    """Trailing moving average that ignores missing (NaN) points."""
    y = np.asarray(values, dtype=float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        if np.isnan(y[i]):
            continue  # keep gaps as gaps (e.g. an opponent type that stopped appearing)
        chunk = y[max(0, i - window + 1) : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            out[i] = chunk.mean()
    return out


def load_run(run_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """(update records, phase-advance events) of a run."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: is this a training run folder?")
    rows = read_metrics(path)
    return [r for r in rows if "event" not in r], [r for r in rows if r.get("event") == "phase_advance"]


def final_summary(updates: list[dict], tail: int) -> dict[str, float]:
    """Mean outcome rates over the last ``tail`` updates that finished episodes."""
    recent = [r for r in updates if r.get("episode/win_rate") is not None][-tail:]
    if not recent:
        return {}
    mean = lambda key: float(np.mean([r[key] for r in recent if r.get(key) is not None]))  # noqa: E731
    return {"win": mean("episode/win_rate"), "draw": mean("episode/draw_rate"), "loss": mean("episode/loss_rate"),
            "length": mean("episode/episode_length"), "step": float(recent[-1]["step"])}


def build_report(run_dir: str | Path, cfg: VizConfig, out_path: str | Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run_dir)
    updates, phases = load_run(run)
    if not updates:
        raise ValueError(f"{run}: metrics.jsonl has no update records yet")
    steps = np.array([r["step"] for r in updates], dtype=float)
    w = cfg.report.smoothing_window

    def series(key: str, raw: bool = False) -> np.ndarray:
        values = [np.nan if r.get(key) is None else r[key] for r in updates]
        return np.asarray(values, dtype=float) if raw else smooth(values, w)

    keys = sorted({k for r in updates for k in r})
    fig, axes = plt.subplots(4, 2, figsize=(cfg.report.figure_width, cfg.report.figure_height), sharex=True)
    fig.patch.set_facecolor("#14171c")
    text_color = "#e6eaef"

    def style(ax, title: str, ylabel: str = "") -> None:
        ax.set_facecolor("#1c2128")
        ax.set_title(title, color=text_color, fontsize=12, loc="left")
        ax.set_ylabel(ylabel, color="#98a4b1")
        ax.tick_params(colors="#98a4b1")
        ax.grid(alpha=0.18)
        for spine in ax.spines.values():
            spine.set_color("#303945")
        for ev in phases:
            ax.axvline(ev["step"], color="#ffcc4d", ls="--", lw=0.9, alpha=0.7)

    def lines(ax, items: list[tuple[str, str, str]]) -> None:
        for key, label, color in items:
            if key in keys:
                ax.plot(steps, series(key), label=label, color=color, lw=1.8)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(facecolor="#1c2128", edgecolor="#303945", labelcolor=text_color, fontsize=9, loc="best")

    ax = axes[0, 0]; style(ax, "Outcomes (rolling mean)", "fraction of episodes")
    lines(ax, [("episode/win_rate", "win", "#4cd37b"), ("episode/draw_rate", "draw", "#a3abb7"), ("episode/loss_rate", "loss", "#ff5d55")])
    ax.set_ylim(-0.02, 1.02)
    for ev in phases:
        ax.text(ev["step"], 1.0, f" {ev['phase']}", color="#ffcc4d", fontsize=8, va="top", rotation=90)

    ax = axes[0, 1]; style(ax, "Win rate by opponent", "win rate")
    palette = ["#6ea8fe", "#ff9f1c", "#2ec4b6", "#c77dff", "#ff6b9d", "#b8e986", "#f4d35e"]
    opp = [k for k in keys if k.startswith("episode/win_rate/opp_")]
    lines(ax, [(k, k.split("opp_")[1], palette[i % len(palette)]) for i, k in enumerate(opp)])
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1, 0]; style(ax, "Return per role (per episode)", "return")
    roles = [k for k in keys if k.startswith("episode/return/")]
    lines(ax, [(k, k.split("/")[-1].replace("slot", "slot "), palette[i % len(palette)]) for i, k in enumerate(roles)])

    ax = axes[1, 1]; style(ax, "Reward components (per episode, summed over players)", "reward")
    comps = [k for k in keys if k.startswith("episode/reward/")]
    lines(ax, [(k, k.split("/")[-1], palette[i % len(palette)]) for i, k in enumerate(comps)])

    ax = axes[2, 0]; style(ax, "Policy: entropy and KL", "entropy")
    lines(ax, [("train/entropy", "entropy", "#6ea8fe")])
    twin = ax.twinx()
    if "train/approx_kl" in keys:
        twin.plot(steps, series("train/approx_kl"), color="#ff9f1c", lw=1.4, label="approx KL")
    twin.tick_params(colors="#98a4b1"); twin.set_ylabel("KL", color="#98a4b1")

    ax = axes[2, 1]; style(ax, "Critic: explained variance and value loss", "explained variance")
    lines(ax, [("train/explained_variance", "explained variance", "#4cd37b")])
    twin = ax.twinx()
    if "train/value_loss" in keys:
        twin.plot(steps, series("train/value_loss"), color="#ff6b9d", lw=1.4)
    twin.tick_params(colors="#98a4b1"); twin.set_ylabel("value loss", color="#98a4b1")

    ax = axes[3, 0]; style(ax, "Episode statistics (per episode)", "count / steps")
    ax.set_ylabel("events per episode", color="#98a4b1")
    lines(ax, [("episode/pickups", "flag pickups", "#ffcc4d"), ("episode/tags_inflicted", "tags made", "#4cd37b"),
               ("episode/tags_suffered", "tags suffered", "#ff5d55")])
    twin = ax.twinx()
    if "episode/episode_length" in keys:
        twin.plot(steps, series("episode/episode_length"), color="#a3abb7", lw=1.4, ls=":", label="episode length (right axis)")
    twin.tick_params(colors="#98a4b1"); twin.set_ylabel("steps", color="#98a4b1")
    twin.legend(facecolor="#1c2128", edgecolor="#303945", labelcolor=text_color, fontsize=9, loc="upper right")
    ax.set_xlabel("environment steps", color="#98a4b1")

    ax = axes[3, 1]; style(ax, "Speed and schedules", "steps / second")
    lines(ax, [("perf/steps_per_sec", "steps/s", "#6ea8fe")])
    twin = ax.twinx()
    for key, label, color in (("schedule/scout_scale", "scout bonus scale", "#c77dff"), ("curriculum/phase", "phase", "#ffcc4d")):
        if key in keys:
            twin.plot(steps, series(key, raw=key == "curriculum/phase"), color=color, lw=1.2, label=label)
    twin.legend(facecolor="#1c2128", edgecolor="#303945", labelcolor=text_color, fontsize=9, loc="center right")
    twin.tick_params(colors="#98a4b1")
    ax.set_xlabel("environment steps", color="#98a4b1")

    summary = final_summary(updates, cfg.report.tail_updates)
    headline = f"{run.name}   ·   {int(steps[-1]):,} steps   ·   {len(updates)} updates"
    if summary:
        headline += (f"   ·   last {cfg.report.tail_updates} updates: win {summary['win']:.0%}  draw {summary['draw']:.0%}"
                     f"  loss {summary['loss']:.0%}  (episodes ≈ {summary['length']:.0f} steps)")
    fig.suptitle(headline, color=text_color, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out_path) if out_path else Path(cfg.paths.output_dir) / f"{run.name}_report.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=cfg.report.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out
