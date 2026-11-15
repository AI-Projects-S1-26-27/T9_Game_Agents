"""Stage B2b-2: the noise-free evaluation hook of the training loop (rl.train_eval, noise_free_eval config)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import yaml

torch = pytest.importorskip("torch")

from rl import train_eval  # noqa: E402
from rl.checkpoint import SECTION_COMPAT, load_checkpoint, load_policy, migrate_config, save_checkpoint  # noqa: E402
from rl.config import NoiseFreeEvalConfig, config_to_plain, load_base_env_config, load_rl_config  # noqa: E402
from rl.metrics import MetricsLogger, read_metrics  # noqa: E402
from rl.schema import ConfigError  # noqa: E402
from rl.train import main as train_main  # noqa: E402
from rl.trainer import Trainer  # noqa: E402
from rl_helpers import ROOT, rl_config, tiny_config  # noqa: E402


def _tiny_eval_yaml(tmp_path, name="eval_tiny.yaml") -> str:
    data = {
        "episodes": 2, "seed": 3, "confidence": 0.95, "sides": ["red", "blue"], "policy_mode": "argmax",
        "fixed_spawn_fraction": 0.0, "env_overrides": {"spawn_mode": "near_base", "max_episode_steps": 8},
        "route_roles": ["attacker"],
        "opponents": [{"kind": "random", "label": "random", "params": {}}, {"kind": "stay", "label": "stay", "params": {}}],
        "held_out": {"episodes": 2, "opponents": ["random"], "maps": []},
        "role_swap": {"enabled": True, "episodes": 2, "opponents": ["random"], "swaps": [["attacker", "defender"]],
                      "min_change_rate": 0.1},
        "policy_check": {"enabled": False, "episodes": 2, "opponents": ["random"]},
    }
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _with_eval(cfg, tmp_path, *, every_steps=16, at_end=True, enabled=True):
    nf = NoiseFreeEvalConfig(enabled=enabled, every_steps=every_steps, at_end=at_end,
                             config_path=_tiny_eval_yaml(tmp_path), overrides={})
    return replace(cfg, noise_free_eval=nf)


def _run(cfg, run_dir, resume=None):
    trainer = Trainer(cfg, load_base_env_config(cfg, ROOT), run_dir, resume)
    return trainer, trainer.run()


def _events(run_dir, name="noise_free_eval"):
    return [r for r in read_metrics(run_dir / "metrics.jsonl") if r.get("event") == name]


# ------------------------------------------------------------------------------------ config


def test_shipped_default_enables_the_hook_with_a_loadable_four_role_eval_config() -> None:
    cfg = load_rl_config(ROOT / "configs" / "rl" / "default.yaml")
    nf = cfg.noise_free_eval
    assert nf.enabled and nf.every_steps >= 1 and nf.at_end
    eval_cfg = train_eval.load_training_eval_config(cfg, ROOT)
    assert eval_cfg.episodes <= 40 and all(o.kind != "minimax" for o in eval_cfg.opponents)  # a hook must stay cheap
    assert "scripted_parked" in {o.label for o in eval_cfg.opponents}
    overridden = train_eval.load_training_eval_config(
        replace(cfg, noise_free_eval=replace(nf, overrides={"episodes": 4})), ROOT)
    assert overridden.episodes == 4


def test_hook_config_is_strict_and_validated_up_front(tmp_path) -> None:
    data = yaml.safe_load((ROOT / "configs" / "rl" / "default.yaml").read_text())
    del data["noise_free_eval"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="noise_free_eval"):
        load_rl_config(path)  # new configs must say what they want: nothing is defaulted in code
    with pytest.raises(ConfigError, match="every_steps"):
        load_rl_config(ROOT / "configs" / "rl" / "default.yaml", {"noise_free_eval.every_steps": 0})
    with pytest.raises(ConfigError, match="unknown key"):
        load_rl_config(ROOT / "configs" / "rl" / "default.yaml", {"noise_free_eval.nonsense": 1})
    # opponents written for 4-role teams do not fit a 3-role training config: caught before training starts
    cfg = replace(rl_config(), noise_free_eval=NoiseFreeEvalConfig(True, 10, True, "configs/rl/eval_train.yaml", {}))
    with pytest.raises(ConfigError, match="roles"):
        train_eval.load_training_eval_config(cfg, ROOT)
    bad = tiny_config(str(tmp_path))
    bad = replace(bad, noise_free_eval=NoiseFreeEvalConfig(True, 10, True, str(tmp_path / "missing.yaml"), {}))
    with pytest.raises(FileNotFoundError):
        Trainer(bad, load_base_env_config(bad, ROOT), tmp_path / "run")


def test_train_cli_reports_a_bad_eval_config_without_a_traceback(tmp_path, capsys) -> None:
    code = train_main(["--set", f"run.output_dir={tmp_path}", "--set", "noise_free_eval.config_path=nope.yaml"])
    assert code == 2
    assert "nope.yaml" in capsys.readouterr().err
    assert not (tmp_path / "mappo_default" / "metrics.jsonl").exists()


# ------------------------------------------------------------------ old checkpoints keep loading


def test_checkpoints_written_before_the_section_existed_load_with_the_hook_off(tmp_path) -> None:
    plain = config_to_plain(rl_config())
    del plain["noise_free_eval"]
    migrated = migrate_config(plain)
    assert migrated["noise_free_eval"] == SECTION_COMPAT["noise_free_eval"] and migrated["noise_free_eval"]["enabled"] is False
    assert "noise_free_eval" not in plain  # the input is not modified
    # a real checkpoint with the section removed from its stored config
    cfg = tiny_config(str(tmp_path))
    trainer = Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run")
    path = tmp_path / "ck.pt"
    save_checkpoint(path, **trainer._state())
    payload = torch.load(path, weights_only=True)
    del payload["config"]["noise_free_eval"]
    torch.save(payload, path)
    model, meta = load_policy(path, "cpu")
    assert meta["config"].noise_free_eval.enabled is False
    # ... and training can resume from it (the resumed run uses the NEW yaml's setting)
    resumed = Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run2", path)
    assert resumed.evaluator is None


def test_the_committed_legacy_checkpoint_loads() -> None:
    ckpt = ROOT / "runs" / "colab_run1" / "checkpoints" / "latest.pt"
    if not ckpt.exists():
        pytest.skip("legacy checkpoint not in this checkout")
    _, meta = load_policy(ckpt, "cpu")
    assert meta["config"].noise_free_eval.enabled is False
    assert load_checkpoint(ckpt, exact_layout=False)["counters"]["total_steps"] > 1_000_000


# --------------------------------------------------------------------------------- the hook


def test_training_logs_noise_free_eval_next_to_the_training_curve(tmp_path) -> None:
    cfg = _with_eval(tiny_config(str(tmp_path)), tmp_path, every_steps=16)
    trainer, counters = _run(cfg, tmp_path / "run")
    assert counters == {"total_steps": 48, "update": 3}
    run = tmp_path / "run"
    events = _events(run)
    assert [e["step"] for e in events] == [16, 32, 48]  # the end-of-run eval is not repeated at step 48
    for e in events:
        for key in ("eval/win_rate/random", "eval/win_rate/stay", "eval/draw_rate/random", "eval/mean_win_rate",
                    "eval/min_win_rate", "eval/stay_fraction", "eval/role_swap_changed", "eval/seconds"):
            assert key in e, key
        assert 0.0 <= e["eval/mean_win_rate"] <= 1.0
    assert sorted(p.name for p in (run / "eval").glob("*.json")) == [
        "step_0000000016.json", "step_0000000032.json", "step_0000000048.json"]
    report = json.loads((run / "eval" / "step_0000000048.json").read_text())
    assert report["meta"]["step"] == 48 and report["results"]["random"]["all"]["n"] == 2
    assert report["role_swap"]["all"]["n"] > 0 and report["held_out"] is None and report["policy_check"] is None
    # training update records are untouched by the extra rows (the report / viewer tools filter events out)
    updates = [r for r in read_metrics(run / "metrics.jsonl") if "event" not in r]
    assert [r["step"] for r in updates] == [16, 32, 48]
    from viz.report import load_run

    rows, _ = load_run(run)
    assert [r["step"] for r in rows] == [16, 32, 48]


def test_eval_only_at_the_end_when_every_steps_is_large(tmp_path) -> None:
    cfg = _with_eval(tiny_config(str(tmp_path)), tmp_path, every_steps=1_000_000)
    _run(cfg, tmp_path / "run")
    assert [e["step"] for e in _events(tmp_path / "run")] == [48]
    off = _with_eval(tiny_config(str(tmp_path / "off")), tmp_path, every_steps=1_000_000, at_end=False)
    _run(off, tmp_path / "off" / "run")
    assert _events(tmp_path / "off" / "run") == []


def test_disabled_hook_does_nothing(tmp_path) -> None:
    cfg = _with_eval(tiny_config(str(tmp_path)), tmp_path, enabled=False)
    trainer, _ = _run(cfg, tmp_path / "run")
    assert trainer.evaluator is None and _events(tmp_path / "run") == [] and not (tmp_path / "run" / "eval").exists()


def test_evaluation_does_not_change_training(tmp_path) -> None:
    plain = tiny_config(str(tmp_path / "a"))
    with_eval = _with_eval(tiny_config(str(tmp_path / "b")), tmp_path, every_steps=16)
    first, _ = _run(plain, tmp_path / "a" / "run")
    second, _ = _run(with_eval, tmp_path / "b" / "run")
    assert _events(tmp_path / "b" / "run")  # it did run
    for p, q in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(p, q)  # same weights: the eval touched no training RNG and left the model state alone


def test_evaluator_restores_the_models_train_or_eval_mode(tmp_path) -> None:
    cfg = _with_eval(tiny_config(str(tmp_path)), tmp_path)
    trainer = Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run")
    for mode in (True, False):
        trainer.model.train(mode)
        scalars, report = trainer.evaluator.run(trainer.model, trainer.value_norm, 0, trainer.device)
        assert trainer.model.training is mode
    assert "eval/mean_win_rate" in scalars and report["meta"]["eval_config"]["episodes"] == 2


def test_the_same_policy_gets_the_same_eval_numbers(tmp_path) -> None:
    cfg = _with_eval(tiny_config(str(tmp_path)), tmp_path)
    trainer = Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run")
    a, _ = trainer.evaluator.run(trainer.model, trainer.value_norm, 0, trainer.device)
    b, _ = trainer.evaluator.run(trainer.model, trainer.value_norm, 0, trainer.device)
    a.pop("eval/seconds"), b.pop("eval/seconds")
    assert a == b  # fixed seeds: curve points differ only because the policy differs


def test_a_failing_eval_does_not_stop_training(tmp_path, monkeypatch) -> None:
    def boom(self, *args, **kwargs):
        raise RuntimeError("eval exploded")

    monkeypatch.setattr(train_eval.NoiseFreeEvaluator, "run", boom)
    cfg = _with_eval(tiny_config(str(tmp_path)), tmp_path, every_steps=16)
    _, counters = _run(cfg, tmp_path / "run")
    assert counters["total_steps"] == 48  # training finished anyway
    failed = _events(tmp_path / "run", "noise_free_eval_failed")
    assert len(failed) == 3 and "eval exploded" in failed[0]["error"]
    assert (tmp_path / "run" / "checkpoints" / "latest.pt").exists()


def test_resume_continues_the_eval_schedule_without_repeating(tmp_path) -> None:
    run = tmp_path / "run"
    cfg = _with_eval(tiny_config(str(tmp_path), total_steps=32), tmp_path, every_steps=16, at_end=False)
    _run(cfg, run)
    assert [e["step"] for e in _events(run)] == [16, 32]
    longer = _with_eval(tiny_config(str(tmp_path), total_steps=64), tmp_path, every_steps=16, at_end=False)
    _run(longer, run, run / "checkpoints" / "latest.pt")
    assert [e["step"] for e in _events(run)] == [16, 32, 48, 64]


# ------------------------------------------------------------------------- scalars / logger


def _block(win, draw, loss, stay=0.2):
    return {"all": {"win": {"rate": win}, "draw": {"rate": draw}, "loss": {"rate": loss}, "stay_fraction": {"all": stay}}}


def test_flatten_scalars_covers_main_held_out_role_swap_and_policy_check() -> None:
    results = {"a": _block(1.0, 0.0, 0.0), "b": _block(0.5, 0.25, 0.25, 0.4)}
    extras = {
        "held_out": {"m1": {"a": _block(0.5, 0, 0.5), "b": _block(1.0, 0, 0)}, "m2": {"a": _block(0.0, 0, 1.0)}},
        "role_swap": {"all": {"rate": 0.3}, "pairs": {"attacker<->scout": {"rate": 0.25}}},
        "policy_check": {"a": {"win_difference": {"diff": 0.2}}, "b": {"win_difference": {"diff": 0.0}}},
    }
    out = train_eval.flatten_scalars(results, extras)
    assert out["eval/win_rate/b"] == 0.5 and out["eval/mean_win_rate"] == 0.75 and out["eval/min_win_rate"] == 0.5
    assert out["eval/stay_fraction"] == pytest.approx(0.3)
    assert out["eval_held_out/m1/win_rate"] == 0.75 and out["eval_held_out/mean_win_rate"] == pytest.approx(0.5)
    assert out["eval/role_swap_changed"] == 0.3 and out["eval/role_swap/attacker<->scout"] == 0.25
    assert out["eval/argmax_minus_sampled_win"] == pytest.approx(0.1)
    bare = train_eval.flatten_scalars(results, {"held_out": None, "role_swap": None, "policy_check": None})
    assert not any(k.startswith(("eval_held_out", "eval/role_swap", "eval/argmax")) for k in bare)


def test_log_eval_writes_an_event_record_that_curve_readers_skip(tmp_path) -> None:
    logger = MetricsLogger(tmp_path, tensorboard=False)
    logger.log(10, {"train/entropy": 1.0})
    logger.log_eval(10, {"eval/mean_win_rate": 0.5, "eval/nan": float("nan")})
    logger.close()
    rows = read_metrics(tmp_path / "metrics.jsonl")
    assert "event" not in rows[0] and rows[1]["event"] == "noise_free_eval"
    assert rows[1]["eval/mean_win_rate"] == 0.5 and rows[1]["eval/nan"] is None
