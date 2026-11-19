"""Command line entry point:  python -m viz <command> ...

  play       open the live viewer (watch a checkpoint or scripted team play; edit maps)
  replay     open a saved episode recording
  mapmaker   open the viewer directly in the map maker
  record     play one episode headlessly and save it (optionally straight to GIF / MP4)
  export     turn a saved recording into GIF / MP4 / PNG
  timelapse  the same scenario played by early, middle and late checkpoints, side by side
  report     one-page training report (PNG) from a run folder
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from ctf.config import load_config
from rl.schema import ConfigError, parse_override
from viz.controllers import ControllerSpec, load_controller_specs
from viz.theme import ViewTeam, VizConfig, load_viz_config

DEFAULT_CONFIG = "configs/viz/default.yaml"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=DEFAULT_CONFIG, help="viewer configuration (theme, keys, paths)")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                   help="override a viewer setting by dotted path, e.g. --set window.width=1600")


def _load_cfg(args) -> VizConfig:
    return load_viz_config(args.config, dict(parse_override(t) for t in args.overrides))


def _specs(cfg: VizConfig, path: str | None, checkpoint: str | None) -> list[ControllerSpec]:
    specs = load_controller_specs(path or cfg.paths.controllers_file)
    if checkpoint:
        specs = [ControllerSpec(f"RL ({Path(checkpoint).parent.parent.name})", "rl", checkpoint, True, {}), *specs]
    return specs


def _pick(specs: list[ControllerSpec], token: str | None, default: int) -> int:
    if token is None:
        return default % len(specs)
    if token.isdigit() and int(token) < len(specs):
        return int(token)
    for i, s in enumerate(specs):
        if token.lower() in s.label.lower():
            return i
    raise ConfigError(f"no controller matches {token!r}; available: {[s.label for s in specs]}")


def _add_controller_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--map", default=None, help="map/environment YAML (default: paths.default_map)")
    p.add_argument("--controllers", default=None, help="controllers YAML (default: paths.controllers_file)")
    p.add_argument("--checkpoint", default=None, help="add this checkpoint as the first (RL) controller")
    p.add_argument("--red", default=None, help="RED controller: index or part of its label")
    p.add_argument("--blue", default=None, help="BLUE controller: index or part of its label")
    p.add_argument("--seed", type=int, default=0)


def _viewer(args, mode: str):
    from viz.viewer import Viewer

    cfg = _load_cfg(args)
    specs = _specs(cfg, args.controllers, args.checkpoint)
    map_path = Path(args.map or cfg.paths.default_map)
    return Viewer(cfg, env_config=load_config(map_path), specs=specs, mode=mode, seed=args.seed, map_path=map_path,
                  red_index=_pick(specs, args.red, 0), blue_index=_pick(specs, args.blue, 1))


def cmd_play(args) -> int:
    _viewer(args, "live").run()
    return 0


def cmd_mapmaker(args) -> int:
    _viewer(args, "edit").run()
    return 0


def cmd_replay(args) -> int:
    from viz.frames import Recording
    from viz.viewer import Viewer

    cfg = _load_cfg(args)
    rec = Recording.load(args.recording)
    m = rec.meta
    env = replace(load_config(cfg.paths.default_map), width=m.width, height=m.height,
                  obstacles=[tuple(c) for c in m.obstacles], red_base=tuple(m.red_base), blue_base=tuple(m.blue_base),
                  red_spawns=[tuple(c) for c in m.red_spawns], blue_spawns=[tuple(c) for c in m.blue_spawns],
                  visibility_radius=m.visibility_radius, max_episode_steps=m.max_steps)
    Viewer(cfg, env_config=env, specs=_specs(cfg, args.controllers, None), recording=rec, mode="replay").run()
    return 0


def cmd_record(args) -> int:
    from viz.controllers import ControllerFactory
    from viz.export import export_gif, export_mp4
    from viz.draw import ViewState
    from viz.session import record_episode

    cfg = _load_cfg(args)
    specs = _specs(cfg, args.controllers, args.checkpoint)
    env = load_config(args.map or cfg.paths.default_map)
    factory = ControllerFactory()
    red = factory.build(specs[_pick(specs, args.red, 0)], env, args.seed)
    blue = factory.build(specs[_pick(specs, args.blue, 1)], env, args.seed + 1)
    rec = record_episode(env, red, blue, seed=args.seed)
    out = Path(args.out or Path(cfg.paths.output_dir) / f"episode_seed{args.seed}.json.gz")
    rec.save(out)
    print(f"{rec.outcome} after {rec.frames[-1].t} steps -> {out}")
    o = cfg.overlays
    view = ViewState(fog=o.fog, belief=o.belief, vision=o.vision, trails=o.trails, probabilities=o.probabilities, arrows=o.arrows,
                     labels=o.labels, coordinates=o.coordinates, spawns=o.spawns, view_team=o.view_team,
                     selected=args.select)
    if args.gif:
        print("gif ->", export_gif(rec, args.gif, cfg, view))
    if args.mp4:
        print("mp4 ->", export_mp4(rec, args.mp4, cfg, view))
    return 0


def cmd_export(args) -> int:
    from viz.draw import ViewState
    from viz.export import export_gif, export_mp4, save_png
    from viz.frames import Recording

    cfg = _load_cfg(args)
    rec = Recording.load(args.recording)
    o = cfg.overlays
    view = ViewState(fog=o.fog, belief=o.belief, vision=o.vision, trails=o.trails, probabilities=o.probabilities, arrows=o.arrows,
                     labels=o.labels, coordinates=o.coordinates, spawns=o.spawns, view_team=ViewTeam(args.view_team or o.view_team.value),
                     selected=args.select)
    if args.gif:
        print("gif ->", export_gif(rec, args.gif, cfg, view))
    if args.mp4:
        print("mp4 ->", export_mp4(rec, args.mp4, cfg, view))
    if args.png:
        print("png ->", save_png(rec, args.png, cfg, view, args.frame if args.frame >= 0 else len(rec.frames) - 1))
    return 0


def cmd_timelapse(args) -> int:
    from viz.timelapse import build_timelapse, list_checkpoints, pick_evenly

    cfg = _load_cfg(args)
    found = list_checkpoints(args.run)
    if not found:
        print(f"no checkpoints in {args.run}/checkpoints", file=sys.stderr)
        return 2
    chosen = pick_evenly(found, args.count)
    specs = _specs(cfg, args.controllers, None)
    opponent = specs[_pick(specs, args.opponent, 1)]
    env = load_config(args.map or cfg.paths.default_map)
    out = Path(args.out or Path(cfg.paths.output_dir) / f"{Path(args.run).name}_timelapse_seed{args.seed}.mp4")
    print("checkpoints:", ", ".join(f"{s / 1e6:.2f}M" for s, _, _ in chosen), "| opponent:", opponent.label)
    print("saved ->", build_timelapse(chosen, opponent, env, cfg, out, seed=args.seed))
    return 0


def cmd_report(args) -> int:
    from viz.report import build_report, final_summary, load_run

    cfg = _load_cfg(args)
    out = build_report(args.run, cfg, args.out)
    updates, _ = load_run(args.run)
    s = final_summary(updates, cfg.report.tail_updates)
    if s:
        print(f"last {cfg.report.tail_updates} updates: win {s['win']:.1%}  draw {s['draw']:.1%}  loss {s['loss']:.1%}")
    print("saved ->", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m viz", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("play", help="live viewer"); _common(p); _add_controller_args(p); p.set_defaults(func=cmd_play)
    p = sub.add_parser("mapmaker", help="map maker"); _common(p); _add_controller_args(p); p.set_defaults(func=cmd_mapmaker)
    p = sub.add_parser("replay", help="replay a recording"); _common(p); p.add_argument("recording")
    p.add_argument("--controllers", default=None); p.set_defaults(func=cmd_replay)
    p = sub.add_parser("record", help="record one episode headlessly"); _common(p); _add_controller_args(p)
    p.add_argument("--out", default=None); p.add_argument("--gif", default=None); p.add_argument("--mp4", default=None)
    p.add_argument("--select", default=None, help="agent id to highlight in exported videos, e.g. R1")
    p.set_defaults(func=cmd_record)
    p = sub.add_parser("export", help="recording -> gif / mp4 / png"); _common(p); p.add_argument("recording")
    p.add_argument("--gif"); p.add_argument("--mp4"); p.add_argument("--png"); p.add_argument("--frame", type=int, default=-1)
    p.add_argument("--select", default=None); p.add_argument("--view-team", choices=["red", "blue", "none"], default=None)
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("timelapse", help="checkpoints side by side"); _common(p)
    p.add_argument("--run", required=True, help="training run folder (contains checkpoints/)")
    p.add_argument("--count", type=int, default=4, help="how many checkpoints to show, spread over training")
    p.add_argument("--opponent", default=None, help="BLUE controller: index or part of its label")
    p.add_argument("--controllers", default=None); p.add_argument("--map", default=None)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_timelapse)
    p = sub.add_parser("report", help="training report PNG"); _common(p)
    p.add_argument("--run", required=True); p.add_argument("--out", default=None); p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    if args.command in ("record", "export", "timelapse", "report"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless commands never open a window
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
