"""configs/rl/all_maps.yaml (Run 3 recipe) next to configs/rl/default.yaml (Run 2 recipe)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rl.checkpoint import config_to_plain
from rl.config import load_base_env_config, load_rl_config, phase_map_bases
from rl.eval_config import load_eval_config
from rl.schema import ConfigError
from rl.train_eval import validate_map_pools
from rl_helpers import ROOT

CFG = ROOT / "configs" / "rl"
LAST = 2  # the real_setting phase holds the pool


def _labels(cfg) -> list[str]:
    return [m.label for m in cfg.curriculum.phases[LAST].maps]


def test_default_is_the_run2_recipe_and_all_maps_adds_exactly_the_two_held_out_maps() -> None:
    default, allmaps = load_rl_config(CFG / "default.yaml"), load_rl_config(CFG / "all_maps.yaml")
    assert _labels(default) == ["default", "center_pillar", "open_field", "pillars", "two_corridors"]
    assert _labels(allmaps) == _labels(default) + ["center_gate", "long_wall_south"]
    assert [m.weight for m in allmaps.curriculum.phases[LAST].maps][-2:] == [1.0, 2.0]
    # nothing else may drift between the two recipes except the evaluation file the training points at
    a, b = config_to_plain(default), config_to_plain(allmaps)
    for plain in (a, b):
        del plain["curriculum"]["phases"][LAST]["maps"]
    assert a["noise_free_eval"].pop("config_path") == "configs/rl/eval_train.yaml"
    assert b["noise_free_eval"].pop("config_path") == "configs/rl/eval_train_all_maps.yaml"
    assert a == b


def test_each_recipe_passes_the_startup_pool_check_with_its_own_evaluation_file() -> None:
    for recipe, eval_file in (("default", "eval_train"), ("all_maps", "eval_train_all_maps")):
        cfg = load_rl_config(CFG / f"{recipe}.yaml")
        validate_map_pools(cfg, load_base_env_config(cfg, ROOT), load_eval_config(CFG / f"{eval_file}.yaml"))
        assert cfg.noise_free_eval.config_path == f"configs/rl/{eval_file}.yaml"


def test_all_maps_has_nothing_held_out_and_the_default_still_holds_two_maps_out() -> None:
    assert load_eval_config(CFG / "eval_train_all_maps.yaml").held_out.maps == []
    assert [m.label for m in load_eval_config(CFG / "eval_train.yaml").held_out.maps] == ["center_gate", "long_wall_south"]


def test_the_two_recipes_cannot_be_mixed_up_silently() -> None:
    # the all-maps pool with the Run 2 evaluation file is exactly the mistake the startup check exists for
    cfg = load_rl_config(CFG / "all_maps.yaml")
    with pytest.raises(ConfigError, match="center_gate"):
        validate_map_pools(cfg, load_base_env_config(cfg, ROOT), load_eval_config(CFG / "eval_train.yaml"))


def test_all_maps_draws_all_seven_layouts() -> None:
    cfg = load_rl_config(CFG / "all_maps.yaml")
    pooled = phase_map_bases(load_base_env_config(cfg, ROOT), cfg.curriculum.phases[LAST])
    assert [label for label, _, _ in pooled] == _labels(cfg) and len(pooled) == 7
    assert [w for _, w, _ in pooled] == [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
