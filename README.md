# Multi-Agent CTF (Prototype 1)

Capture-the-Flag grid environment for multi-agent coordination research, with optional fog of war and seeded spawn randomization. Core package: game rules and API only (no built-in AI). `step()` is deterministic; randomness enters only through seeded spawn modes. Optional Pygame demo for watching a scripted episode.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11+).

```bash
uv sync --extra dev          # environment + tests
uv sync --extra dev --extra rl   # also the RL dependencies (torch, tensorboard, ...)
```

`ctf`, `coordination` and `strategies` are installed as packages, so no `PYTHONPATH` tweaking is needed.

## Tests

```bash
uv run pytest
```

## Text demo

Scripted episode (RED captures BLUE flag and wins):

```bash
uv run python scripts/run_demo.py
```

## Visualization

```bash
uv run python visualization/demo.py
```

SPACE = pause/resume, ESC = quit, R = restart.

## RL agent (work in progress)

`src/rl/` holds the MAPPO agent (CNN actor-critic, role-conditioned heads, centralized critic).
Everything tunable lives in `configs/rl/default.yaml`; every key is required and unknown keys are
errors. Override any value from the command line with a dotted path.

```bash
uv sync --extra dev --extra rl                      # install (adds torch, tensorboard, matplotlib)
uv run pytest -q                                    # run all tests

# 1. pick rollout.num_workers / torch_threads / model size for your machine
uv run python -m rl.benchmark --workers 0 2 3 --updates 2

# 2. train (a run lives in runs/<run.name>/: metrics.jsonl, config.yaml, checkpoints/, tensorboard/)
uv run python -m rl.train --set run.name=first_run --set rollout.num_workers=3
uv run python -m rl.train --set run.name=first_run --set rollout.num_workers=3 --resume latest
uv run python -m rl.train --print-config            # show the resolved configuration

# 3. watch learning curves
uv run tensorboard --logdir runs
```

Ctrl+C saves a checkpoint before exiting, so a run can always be continued with `--resume latest`.

The shipped default trains on the v2 rules with **4 agents per team** (two attackers, a defender, a scout):
`env.config_path: configs/ctf_v2.yaml`, `roles.slots: [attacker, attacker, defender, scout]`. Metrics are split
per opponent label (`win_rate/opp_<label>`, `draw_rate/...`, `loss_rate/...`) and per spawn style
(`win_rate/spawns_fixed|random`).

Checkpoints from before the v2 rules (for example the 2M-step Colab run, 3 agents) still load for playing and
for the viewer: features added later are simply not shown to the older network. They cannot be resumed or
played on a 4-agent map. See `src/rl/checkpoint.py` (`migrate_config`, `ObservationLayout`).

## Opponents

Opponent kinds (`random`, `stay`, `scripted`, `minimax`) are built from YAML entries; see `src/rl/opponents.py`.
`minimax` is a depth-limited alpha-beta searcher over joint team moves (`src/rl/minimax.py`, explained in
`docs/minimax.md`). It is meant as a strong fixed evaluation opponent; it is also available in the viewer
(`uv run python -m viz play --map configs/ctf_v2.yaml --red scripted --blue minimax`).

## Environment rules

`configs/default.yaml` keeps the original rules (3 agents per team). `configs/ctf_v2.yaml` (same map, 4 agents per
team) turns on the v2 flag rules: only slots 1 and 2 (the attackers) may carry the flag, a carrier is tagged only when enemies on its cell outnumber its
non-carrier teammates there, and a tagged carrier's flag stays where it was tagged until its own team touches it
(sends it home) or an eligible enemy picks it up. Every rule is a YAML key; see `docs/environment.md` section 3.6b.

```bash
uv run python -m viz play --map configs/ctf_v2.yaml --red scripted --blue random
```

## Visualisation (`src/viz`)

A viewer for replays and live play (every player labelled with its id and role), a map maker, GIF/MP4
export, a "watch it learn" time-lapse over checkpoints, and a one-page training report. Press **H** in the
viewer for every key. Looks, layout and key bindings live in `configs/viz/default.yaml`; the controllers you
can put on each side are in `configs/viz/controllers.yaml`.

```bash
uv sync --extra dev --extra rl --extra viz

uv run python -m viz play --checkpoint runs/first_run/checkpoints/latest.pt        # live viewer
uv run python -m viz play --map configs/maps/center_gate.yaml --red scripted --blue random
uv run python -m viz mapmaker --map configs/maps/default.yaml                      # paint walls, bases, spawns
uv run python -m viz replay outputs/viz/episode_seed3.json.gz                      # replay a saved episode

# headless (no window; works on Colab): record, export, time-lapse, report
uv run python -m viz record --checkpoint runs/first_run/checkpoints/latest.pt --blue leash --seed 3 --mp4 ep.mp4
uv run python -m viz timelapse --run runs/first_run --count 4 --opponent leash --seed 3
uv run python -m viz report --run runs/first_run
```

Maps are ordinary environment configs, so a map saved in the map maker can be used for training or
evaluation with `--set env.config_path=configs/maps/<name>.yaml`. A trained policy only plays maps of the
size it was trained on (11x7) and assumes a left-right symmetric map; the map maker's mirror mode keeps maps symmetric.

## Configuration

All parameters have a single default in `src/ctf/config.py`; YAML files override them and are validated strictly (unknown keys and wrong types are errors). See `configs/default.yaml` for every option with comments.

## Documentation

Full Prototype 1 spec: [docs/environment.md](docs/environment.md)

## Layout

- `src/ctf/` — environment
- `src/coordination/` — shared team map (scout line-of-sight sharing)
- `src/strategies/` — strategy interface (`BaseStrategy`), `Role`, dummy strategy
- `configs/default.yaml` — default map and rules
- `scripts/demo_scenario.py` — shared action script for demos
- `visualization/` — Pygame layer
- `tests/`
