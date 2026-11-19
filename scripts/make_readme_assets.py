#!/usr/bin/env python3
"""Two README figures that are not made by scripts/make_rl_plots.py (no training, no evaluation is run here).

  uv run python scripts/make_readme_assets.py --out docs/img

Writes:
  maps_gallery.png   every layout of configs/maps with its walls, bases and spawn cells; marks which recipe trained on it
  experiments.png    wins / draws / losses of Experiment A and B of the research report (numbers from the YAML)
Settings and the report numbers are in configs/plots/readme_assets.yaml.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from rl.config import load_rl_config  # noqa: E402
from rl.schema import ConfigError, from_dict, read_yaml  # noqa: E402


@dataclass(frozen=True)
class FigureCfg:
    dpi: int
    map_cell: float
    map_columns: int
    bar_width: float


@dataclass(frozen=True)
class Colors:
    win: str
    draw: str
    loss: str
    wall: str
    red: str
    blue: str
    spawn_red: str
    spawn_blue: str


@dataclass(frozen=True)
class Recipe:
    label: str
    config: str


@dataclass(frozen=True)
class MapsCfg:
    directory: str
    recipes: list[Recipe]


@dataclass(frozen=True)
class Bar:
    label: str
    win: int
    draw: int
    loss: int


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    games: int
    bars: list[Bar]


@dataclass(frozen=True)
class AssetCfg:
    figure: FigureCfg
    colors: Colors
    maps: MapsCfg
    experiments: list[Experiment]

    def __post_init__(self) -> None:
        for exp in self.experiments:
            for bar in exp.bars:
                if bar.win + bar.draw + bar.loss != exp.games:
                    raise ConfigError(f"experiment {exp.id}, {bar.label!r}: win+draw+loss must equal games ({exp.games})")


def load_cfg(path: Path) -> AssetCfg:
    return from_dict(AssetCfg, read_yaml(path))


def _pool_labels(config_path: Path) -> set[str]:
    cfg = load_rl_config(config_path)
    return {Path(m.path).stem for ph in cfg.curriculum.phases for m in ph.maps}


def maps_gallery(cfg: AssetCfg, out: Path) -> Path:
    files = sorted((ROOT / cfg.maps.directory).glob("*.yaml"))
    pools = [(r.label, _pool_labels(ROOT / r.config)) for r in cfg.maps.recipes]
    cols = cfg.figure.map_columns
    rows = (len(files) + cols - 1) // cols
    first = read_yaml(files[0])
    w, h = first["width"], first["height"]
    cell = cfg.figure.map_cell
    fig, axes = plt.subplots(rows, cols, figsize=(cols * (w * cell + 0.3), rows * (h * cell + 0.75)), squeeze=False)
    c = cfg.colors
    for ax in axes.ravel():
        ax.axis("off")
    for ax, f in zip(axes.ravel(), files):
        d = read_yaml(f)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.add_patch(Rectangle((0, 0), w, h, facecolor="#f7fafc", edgecolor="#a0aec0", lw=1))
        for x in range(1, w):
            ax.plot([x, x], [0, h], color="#e2e8f0", lw=0.5)
        for y in range(1, h):
            ax.plot([0, w], [y, y], color="#e2e8f0", lw=0.5)
        for x, y in d["red_spawns"]:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=c.spawn_red, lw=0))
        for x, y in d["blue_spawns"]:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=c.spawn_blue, lw=0))
        for x, y in d["obstacles"]:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=c.wall, edgecolor="white", lw=0.6))
        for (x, y), color, name in ((d["red_base"], c.red, "R"), (d["blue_base"], c.blue, "B")):
            ax.add_patch(Rectangle((x + 0.1, y + 0.1), 0.8, 0.8, facecolor=color, edgecolor="none"))
            ax.text(x + 0.5, y + 0.52, name, color="white", ha="center", va="center", fontsize=8, weight="bold")
        tags = [f"{label.split(' (')[0]}: {'seen' if f.stem in pool else 'UNSEEN'}" for label, pool in pools]
        ax.set_title(f"{f.stem}\n" + "  |  ".join(tags), fontsize=8)
    fig.suptitle("Map layouts (11x7, mirror-symmetric): walls dark, bases R/B, light squares = fixed spawn cells", fontsize=10)
    fig.tight_layout(h_pad=2.2, rect=(0, 0, 1, 0.96))
    path = out / "maps_gallery.png"
    fig.savefig(path, dpi=cfg.figure.dpi)
    plt.close(fig)
    return path


def experiments_figure(cfg: AssetCfg, out: Path) -> Path:
    c, bw = cfg.colors, cfg.figure.bar_width
    fig, axes = plt.subplots(1, len(cfg.experiments), figsize=(5.4 * len(cfg.experiments), 4.2), squeeze=False)
    for ax, exp in zip(axes.ravel(), cfg.experiments):
        for i, (key, color) in enumerate((("win", c.win), ("draw", c.draw), ("loss", c.loss))):
            xs = [j + (i - 1) * bw for j in range(len(exp.bars))]
            vals = [getattr(b, key) for b in exp.bars]
            ax.bar(xs, vals, width=bw, color=color, label={"win": "Wins", "draw": "Draws", "loss": "Losses"}[key])
            for x, v in zip(xs, vals):
                ax.text(x, v + exp.games * 0.01, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(exp.bars)))
        ax.set_xticklabels([b.label for b in exp.bars], fontsize=8)
        ax.set_ylim(0, exp.games * 1.1)
        ax.set_ylabel(f"games (of {exp.games} per bar)")
        ax.set_title(exp.title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    path = out / "experiments.png"
    fig.savefig(path, dpi=cfg.figure.dpi)
    plt.close(fig)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "plots" / "readme_assets.yaml"))
    p.add_argument("--out", default=str(ROOT / "docs" / "img"))
    args = p.parse_args()
    cfg = load_cfg(Path(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in (maps_gallery(cfg, out), experiments_figure(cfg, out)):
        print("saved ->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
