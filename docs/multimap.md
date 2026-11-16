# Multi-map training

Run 1 trained on one layout (`configs/ctf_v2.yaml`) and did not transfer (held-out maps: 0% to 93%, see the report).
A phase can now train on a POOL of maps: every episode draws one map by weight.

## Config

```yaml
curriculum:
  phases:
    - name: real_setting
      maps:                                   # empty list (`maps: []`) = every episode uses the base map
        - {path: configs/maps/default.yaml,       label: default,       weight: 2.0}
        - {path: configs/maps/center_pillar.yaml, label: center_pillar, weight: 1.0}
```

* `path`: a map YAML. Only `obstacles`, `red_base`, `blue_base`, `red_spawns`, `blue_spawns` are used; rules,
  visibility, spawn mode and episode length come from the base config plus the phase's `env_overrides`.
* `label`: `[a-z0-9_]+`, unique in the phase. It names the metric series `episode/win_rate/map_<label>` in `metrics.jsonl`
  (next to the per-opponent `episode/win_rate/opp_<label>`); a phase without a pool logs no per-map series.
* `weight`: > 0, relative draw probability.
* A phase with a pool may not set `obstacles`, `red_base`, `blue_base`, `red_spawns` or `blue_spawns` in
  `env_overrides` (they would silently replace the map). Phases 1 and 2 keep their spawn overrides and `maps: []`.
* Older checkpoints and configs without `maps` load as `maps: []` (`migrate_config`), so Run 1 is unchanged.

## What a pool map must satisfy (checked at startup, `rl.maps`)

* same width, height and `agents_per_team` as the base environment (the network shape contains the board size);
* left-right mirror symmetric walls and mirrored bases (BLUE's view is flipped, so one policy plays both sides);
* a walkable path between the two bases.

The trainer also refuses a pool map that is one of the held-out maps of `noise_free_eval` (same file, or the same
walls and bases under another name), naming both files. `configs/rl/eval_train.yaml` therefore lists only the UNSEEN
maps (`center_gate`, `long_wall_south`), so the training curve `eval_held_out/...` tracks transfer.

## Split used for Run 2

Train pool (phase 3): `default` (weight 2), `center_pillar`, `open_field`, `pillars`, `two_corridors`.
Held out, never trained on: `center_gate` (wide central gate) and `long_wall_south` (one bottom-only gap). They were
the worst maps for the Run 1 policy and are structurally different, so they measure real transfer. Expect unseen-map
results to stay mediocre and report them honestly. Adding them to the pool for a final model removes the clean unseen
set: decide that before the report is written and keep the old split's numbers as the unseen measurement.

## Evaluation

`python -m rl.evaluate` prints the held-out maps in two groups. A map is SEEN when the checkpoint's own config trained on
that layout (base map or any phase pool; compared by walls and bases, so `default.yaml` counts as seen for a policy
trained on `ctf_v2.yaml`), otherwise UNSEEN. The JSON gets `held_out_seen: {map: bool}` next to `held_out`. Checkpoints
from before this change have empty pools, so only the base layout is seen for them.

## Implementation notes

* `CTFTrainEnv` keeps one runtime per map (environments, feature builders, reward calculators with THAT map's path
  distances, env config). The map is drawn in `reset()`; the opponent is built from the drawn map's config.
* With zero or one map there is NO extra random draw per episode, so seeds of old configs reproduce.
* `info["map"]` is in the episode summary ("" without a pool).
