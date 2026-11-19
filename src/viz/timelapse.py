"""'Watch it learn': the same scenario replayed by checkpoints from early to late training, side by side."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from ctf.config import CTFConfig  # noqa: E402
from viz.controllers import ControllerFactory, ControllerSpec  # noqa: E402
from viz.draw import Fonts, Geometry, ViewState, draw_board_panel, text, with_alpha  # noqa: E402
from viz.export import write_gif, write_mp4  # noqa: E402
from viz.frames import Recording  # noqa: E402
from viz.session import record_episode  # noqa: E402
from viz.theme import VizConfig, rgb  # noqa: E402


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, str, Path]]:
    """(training step, phase name, path) for every saved checkpoint of a run, oldest first."""
    from rl.checkpoint import load_checkpoint, migrate_config
    from rl.config import RLConfig
    from rl.schema import from_dict

    found = []
    for path in sorted((Path(run_dir) / "checkpoints").glob("*.pt")):
        if path.name == "latest.pt":
            continue
        payload = load_checkpoint(path, exact_layout=False)
        phases = from_dict(RLConfig, migrate_config(payload["config"])).curriculum.phases
        found.append((int(payload["counters"]["total_steps"]), phases[int(payload["curriculum"]["index"])].name, path))
    return sorted(found)


def pick_evenly(items: list, count: int) -> list:
    """``count`` items spread evenly across the list, always including the first and last."""
    if count >= len(items) or len(items) <= 2:
        return list(items)
    idx = sorted({round(i * (len(items) - 1) / (count - 1)) for i in range(count)})
    return [items[i] for i in idx]


def build_timelapse(
    checkpoints: list[tuple[int, str, Path]],
    opponent: ControllerSpec,
    env_config: CTFConfig,
    cfg: VizConfig,
    out_path: str | Path,
    *,
    seed: int,
    view: ViewState | None = None,
) -> Path:
    """Record every checkpoint (as RED) against ``opponent`` on the same seed and tile the replays."""
    pygame.font.init()
    factory = ControllerFactory()
    records: list[tuple[str, Recording]] = []
    for step, phase, path in checkpoints:
        red = factory.build(ControllerSpec(path.stem, "rl", str(path), True, {}), env_config, seed)
        blue = factory.build(opponent, env_config, seed + 1)
        records.append((f"{step / 1e6:.2f}M steps  ·  {phase}", record_episode(env_config, red, blue, seed=seed)))

    tl = cfg.timelapse
    cols = min(tl.columns, len(records))
    rows = -(-len(records) // cols)
    cell_h = tl.panel_height + tl.title_height
    w, h = cols * tl.panel_width, rows * cell_h
    w, h = w - w % 2, h - h % 2
    surface = pygame.Surface((w, h))
    fonts = Fonts(cfg)
    base = view or ViewState(fog=False, belief=False, vision=False, trails=True, probabilities=False, arrows=True,
                             labels=True, coordinates=False, spawns=False, view_team=cfg.overlays.view_team)
    base = replace(base, mode="replay", selected=None, show_help=False, flash_t=cfg.board.event_flash_seconds * 0.35)
    geo = Geometry.from_meta(records[0][1].meta)
    longest = max(len(r.frames) for _, r in records)
    arrays: list[np.ndarray] = []
    for t in range(longest):
        surface.fill(rgb(cfg.palette.background))
        for i, (title, rec) in enumerate(records):
            x, y = (i % cols) * tl.panel_width, (i // cols) * cell_h
            idx = min(t, len(rec.frames) - 1)
            head = pygame.Rect(x, y, tl.panel_width, tl.title_height)
            pygame.draw.rect(surface, rgb(cfg.palette.panel), head)
            text(surface, fonts.label_bold, title, (x + 12, head.centery - 9), rgb(cfg.palette.text), "midleft")
            last = rec.frames[-1]
            if idx == len(rec.frames) - 1 and last.done:
                verdict = {"RED": ("WIN", cfg.palette.win), "BLUE": ("LOSS", cfg.palette.loss)}.get(last.winner, ("DRAW", cfg.palette.draw))
                text(surface, fonts.label_bold, verdict[0], (head.right - 12, head.centery - 9), rgb(verdict[1]), "midright")
            else:
                text(surface, fonts.small, f"t = {rec.frames[idx].t}", (head.right - 12, head.centery - 9), rgb(cfg.palette.text_dim), "midright")
            text(surface, fonts.tiny, f"RED = this checkpoint   ·   BLUE = {opponent.label}   ·   seed {seed}",
                 (x + 12, head.bottom - 8), rgb(cfg.palette.text_dim), "midleft")
            area = pygame.Rect(x, y + tl.title_height, tl.panel_width, tl.panel_height)
            draw_board_panel(surface, cfg, fonts, geo, rec.frames, idx, base, area.inflate(-8, -8))
            pygame.draw.rect(surface, rgb(cfg.palette.border), pygame.Rect(x, y, tl.panel_width, cell_h), 1)
        arrays.append(np.ascontiguousarray(pygame.surfarray.array3d(surface).swapaxes(0, 1)))
    out = Path(out_path)
    writer = write_mp4 if out.suffix.lower() == ".mp4" else write_gif
    return writer(arrays, out, tl.fps, tl.hold_final_frames)
