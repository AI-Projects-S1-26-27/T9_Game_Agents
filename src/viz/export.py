"""Render recordings to images, GIF and MP4 (works headless, e.g. on Colab)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no window needed for exporting
import pygame  # noqa: E402

from viz.draw import Fonts, Geometry, ViewState, render_scene  # noqa: E402
from viz.frames import Recording  # noqa: E402
from viz.theme import VizConfig  # noqa: E402


def _to_array(surface: pygame.Surface) -> np.ndarray:
    return np.ascontiguousarray(pygame.surfarray.array3d(surface).swapaxes(0, 1))


def render_frames(rec: Recording, cfg: VizConfig, view: ViewState, size: tuple[int, int] | None = None,
                  indices: list[int] | None = None) -> list[np.ndarray]:
    """Every frame of the recording rendered exactly as the viewer shows it (RGB arrays)."""
    pygame.font.init()
    w, h = size or (cfg.export.width, cfg.export.height)
    w, h = w - w % 2, h - h % 2  # video encoders need even dimensions
    surface = pygame.Surface((w, h))
    fonts = Fonts(cfg)
    geo = Geometry.from_meta(rec.meta)
    v = replace(view, mode="replay", playing=False, show_help=False, message="", flash_t=cfg.board.event_flash_seconds * 0.35)
    out = []
    for i in indices if indices is not None else range(len(rec.frames)):
        render_scene(surface, cfg, fonts, geo, rec.meta, rec.frames, i, v)
        out.append(_to_array(surface))
    return out


def save_png(rec: Recording, path: str | Path, cfg: VizConfig, view: ViewState, index: int) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_frames(rec, cfg, view, indices=[index])[0]).save(path)
    return path


def write_gif(arrays: list[np.ndarray], path: str | Path, fps: int, hold_last: int = 0) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(a) for a in arrays]
    images += [images[-1]] * hold_last
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=False)
    return path


def write_mp4(arrays: list[np.ndarray], path: str | Path, fps: int, hold_last: int = 0) -> Path:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("MP4 export needs imageio and imageio-ffmpeg (uv sync --extra viz)") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(arrays) + [arrays[-1]] * hold_last
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1, quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    return path


def export_gif(rec: Recording, path: str | Path, cfg: VizConfig, view: ViewState, hold_last: int = 8) -> Path:
    return write_gif(render_frames(rec, cfg, view), path, cfg.export.fps, hold_last)


def export_mp4(rec: Recording, path: str | Path, cfg: VizConfig, view: ViewState, hold_last: int = 8) -> Path:
    return write_mp4(render_frames(rec, cfg, view), path, cfg.export.fps, hold_last)
