#!/usr/bin/env python3
"""Graphs for the report from files that already exist (no training, no evaluation is run here).

  uv run python scripts/make_rl_plots.py --run "Run 1=runs/v2_run1" --run "Run 2=runs/v2_run2_multimap" \
      --eval "Run 1 (6.0M)=outputs/eval/v2_run1_6.0M.json" --eval "Run 2 (9.0M)=outputs/eval/v2_run2_9M.json" \
      --out outputs/plots

Writes (only the ones whose inputs were given):
  training_curves.png     rolling win rate and policy entropy of every --run against the environment steps
  training_per_map.png    rolling training win rate per map of every --run that trained on a map pool
  eval_by_opponent.png    evaluation win rate (with Wilson 95% interval) per opponent, one bar per --eval
  eval_held_out_maps.png  evaluation win rate per held-out map; hatched = the policy never trained on that layout
Settings (sizes, smoothing, colours) are in configs/plots/rl_plots.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from rl.schema import from_dict, read_yaml  # noqa: E402


@dataclass(frozen=True)
class FigureCfg:
    dpi: int
    bar_width: float
    curve_width: float
    row_height: float
    curve_height: float


@dataclass(frozen=True)
class PlotCfg:
    figure: FigureCfg
    smoothing_window: int
    colors: list[str]
    unseen_hatch: str


def wilson(count: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = count / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def split_arg(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise SystemExit(f"expected LABEL=PATH, got {text!r}")
    label, path = text.split("=", 1)
    return label.strip(), path.strip()


def read_metrics(run: str) -> list[dict]:
    rows = []
    with open(Path(run) / "metrics.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rolling(xs: list[float], window: int) -> list[float]:
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= window:
            acc -= xs[i - window]
        out.append(acc / min(i + 1, window))
    return out


def series(rows: list[dict], key: str, window: int) -> tuple[list[float], list[float]]:
    pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None and "step" in r]
    pts.sort(key=lambda p: p[0])  # resume points repeat or step back by a few updates
    return [p[0] for p in pts], rolling([p[1] for p in pts], window)


def plot_training_curves(runs, cfg: PlotCfg, out: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(cfg.figure.curve_width, 2 * cfg.figure.curve_height), sharex=True)
    for i, (label, path) in enumerate(runs):
        rows = read_metrics(path)
        color = cfg.colors[i % len(cfg.colors)]
        for ax, key in ((axes[0], "episode/win_rate"), (axes[1], "train/entropy")):
            xs, ys = series(rows, key, cfg.smoothing_window)
            ax.plot([x / 1e6 for x in xs], ys, color=color, label=label, linewidth=1.6)
    axes[0].set_ylabel("training win rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("policy entropy")
    axes[1].set_xlabel("environment steps (millions)")
    axes[0].set_title(f"Training curves (rolling mean over {cfg.smoothing_window} updates)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    path = out / "training_curves.png"
    fig.savefig(path, dpi=cfg.figure.dpi)
    plt.close(fig)
    return path


def plot_training_per_map(runs, cfg: PlotCfg, out: Path) -> Path | None:
    prefix = "episode/win_rate/map_"
    panels = []
    for label, path in runs:
        rows = read_metrics(path)
        keys = sorted({k for r in rows for k in r if k.startswith(prefix)})
        if keys:
            panels.append((label, rows, keys))
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), 1, figsize=(cfg.figure.curve_width, cfg.figure.curve_height * len(panels)),
                             squeeze=False, sharex=True)
    for ax, (label, rows, keys) in zip(axes[:, 0], panels):
        for i, key in enumerate(keys):
            xs, ys = series(rows, key, cfg.smoothing_window)
            ax.plot([x / 1e6 for x in xs], ys, color=cfg.colors[i % len(cfg.colors)], label=key[len(prefix):], linewidth=1.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("training win rate")
        ax.set_title(f"{label}: win rate per training map")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", ncol=3)
    axes[-1, 0].set_xlabel("environment steps (millions)")
    fig.tight_layout()
    path = out / "training_per_map.png"
    fig.savefig(path, dpi=cfg.figure.dpi)
    plt.close(fig)
    return path


def plot_eval_by_opponent(evals, cfg: PlotCfg, out: Path) -> Path:
    data = [(label, json.load(open(path))) for label, path in evals]
    opponents: list[str] = []
    for _, d in data:
        for name in d["results"]:
            if name not in opponents:
                opponents.append(name)
    height = max(3.0, cfg.figure.row_height * len(opponents) * (len(data) * 0.45 + 0.6))
    fig, ax = plt.subplots(figsize=(cfg.figure.bar_width, height))
    slot = 0.8 / len(data)
    for j, (label, d) in enumerate(data):
        ys, vals, lows, highs = [], [], [], []
        for i, name in enumerate(opponents):
            cell = d["results"].get(name)
            if cell is None:
                continue
            w = cell["all"]["win"]
            ys.append(i + (j - (len(data) - 1) / 2) * slot)
            vals.append(100 * w["rate"])
            lows.append(100 * (w["rate"] - w["lo"]))
            highs.append(100 * (w["hi"] - w["rate"]))
        ax.barh(ys, vals, height=slot * 0.9, xerr=[lows, highs], color=cfg.colors[j % len(cfg.colors)], label=label,
                error_kw={"elinewidth": 1, "capsize": 3, "ecolor": "black"})
    ax.set_yticks(range(len(opponents)))
    ax.set_yticklabels(opponents)
    ax.invert_yaxis()
    ax.set_xlim(0, 102)
    ax.set_xlabel("win rate % (bars: 95% Wilson interval; the rest are draws and losses)")
    ax.set_title("Evaluation win rate per opponent")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(data))
    fig.tight_layout()
    path = out / "eval_by_opponent.png"
    fig.savefig(path, dpi=cfg.figure.dpi)
    plt.close(fig)
    return path


def held_out_rates(d: dict) -> dict[str, tuple[int, int, bool]]:
    """map -> (wins, games, seen). Old files have no held_out_seen: only the default layout counts as seen then."""
    seen_flags = d.get("held_out_seen") or {}
    out = {}
    for name, per_opp in d.get("held_out", {}).items():
        wins = sum(c["all"]["win"]["count"] for c in per_opp.values())
        games = sum(c["all"]["win"]["n"] for c in per_opp.values())
        out[name] = (wins, games, bool(seen_flags.get(name, name == "default")))
    return out


def plot_eval_held_out(evals, cfg: PlotCfg, out: Path) -> Path | None:
    data = [(label, held_out_rates(json.load(open(path)))) for label, path in evals]
    data = [(label, rates) for label, rates in data if rates]
    if not data:
        return None
    maps = sorted({m for _, rates in data for m in rates})
    fig, ax = plt.subplots(figsize=(cfg.figure.bar_width, max(3.0, cfg.figure.row_height * len(maps) * (len(data) * 0.45 + 0.6))))
    slot = 0.8 / len(data)
    for j, (label, rates) in enumerate(data):
        for i, name in enumerate(maps):
            if name not in rates:
                continue
            wins, games, seen = rates[name]
            lo, hi = wilson(wins, games)
            rate = wins / games
            y = i + (j - (len(data) - 1) / 2) * slot
            ax.barh(y, 100 * rate, height=slot * 0.9, xerr=[[100 * (rate - lo)], [100 * (hi - rate)]],
                    color=cfg.colors[j % len(cfg.colors)], hatch="" if seen else cfg.unseen_hatch, edgecolor="white",
                    error_kw={"elinewidth": 1, "capsize": 3, "ecolor": "black"})
    ax.set_yticks(range(len(maps)))
    ax.set_yticklabels(maps)
    ax.invert_yaxis()
    ax.set_xlim(0, 102)
    ax.set_xlabel("win rate % over the scripted opponents (bars: 95% Wilson interval)")
    ax.set_title("Evaluation maps (hatched = never trained on that layout)")
    ax.grid(axis="x", alpha=0.3)
    handles = [Patch(facecolor=cfg.colors[j % len(cfg.colors)], label=label) for j, (label, _) in enumerate(data)]
    handles.append(Patch(facecolor="white", edgecolor="gray", hatch=cfg.unseen_hatch, label="map never trained on"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(handles))
    fig.tight_layout()
    path = out / "eval_held_out_maps.png"
    fig.savefig(path, dpi=cfg.figure.dpi)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "plots" / "rl_plots.yaml"))
    p.add_argument("--run", action="append", default=[], metavar="LABEL=RUN_DIR", help="a training run folder (metrics.jsonl)")
    p.add_argument("--eval", action="append", default=[], metavar="LABEL=FILE.json", help="an rl.evaluate JSON")
    p.add_argument("--out", default="outputs/plots")
    args = p.parse_args(argv)
    cfg = from_dict(PlotCfg, read_yaml(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [split_arg(r) for r in args.run]
    evals = [split_arg(e) for e in args.eval]
    made = []
    if runs:
        made.append(plot_training_curves(runs, cfg, out))
        made.append(plot_training_per_map(runs, cfg, out))
    if evals:
        made.append(plot_eval_by_opponent(evals, cfg, out))
        made.append(plot_eval_held_out(evals, cfg, out))
    for path in made:
        if path is not None:
            print("saved ->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
