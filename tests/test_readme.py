"""The README must not rot: its links, images, paths and numbers are checked against the repository."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _local_targets(text: str) -> set[str]:
    found = set(re.findall(r"\]\(([^)\s]+)\)", text)) | set(re.findall(r'src="([^"]+)"', text))
    return {t.split("#")[0] for t in found if not t.startswith(("http://", "https://", "mailto:", "#")) and t.split("#")[0]}


def test_every_relative_link_and_image_exists() -> None:
    missing = sorted(t for t in _local_targets(README) if not (ROOT / t).exists())
    assert not missing, missing


def test_every_image_in_docs_img_is_used_by_the_readme() -> None:
    used = {Path(t).name for t in _local_targets(README) if t.startswith("docs/img/")}
    shipped = {p.name for p in (ROOT / "docs" / "img").iterdir() if p.is_file()}
    assert shipped == used, (shipped - used, used - shipped)


def test_repository_paths_named_in_code_spans_exist() -> None:
    skip = ("<", "*", "...", "outputs/viz", "runs/<")
    stale = []
    for span in re.findall(r"`([^`\n]+)`", README):
        for token in span.split():
            token = token.strip(",.;:()")
            if token.startswith(("configs/", "docs/", "scripts/", "src/", "tests/", "runs/", "outputs/")) and not any(s in token for s in skip):
                if not (ROOT / token.rstrip("/")).exists():
                    stale.append(token)
    assert not stale, sorted(set(stale))


def test_mermaid_diagrams_are_closed() -> None:
    assert README.count("```") % 2 == 0
    assert README.count("```mermaid") == 2


def _percent(count: int, n: int) -> str:
    return f"**{100 * count / n:.1f}%**"


@pytest.mark.parametrize("name", ["v2_run1_6.0M.json", "v2_run2_9M.json"])
def test_results_tables_match_the_evaluation_files(name: str) -> None:
    data = json.loads((ROOT / "outputs" / "eval" / name).read_text())
    column = 0 if "run1" in name else 1  # which of the two value columns of the tables
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]] for line in README.splitlines() if line.startswith("| ")}
    for opponent, res in data["results"].items():
        key = next(k for k in rows if k.startswith(opponent) or k.startswith(opponent.replace("classical", "classical AI")))
        a = res["all"]
        cell = rows[key][column]
        assert _percent(a["win"]["count"], a["n"]) in cell, (opponent, cell)
        assert f"D {a['draw']['count']}, L {a['loss']['count']}" in cell, (opponent, cell)
    for map_name, per_opponent in data["held_out"].items():
        wins = sum(r["all"]["win"]["count"] for r in per_opponent.values())
        games = sum(r["all"]["n"] for r in per_opponent.values())
        assert _percent(wins, games) in rows[map_name][column], (map_name, rows[map_name])


def test_unseen_maps_in_the_readme_match_the_evaluation_file() -> None:
    seen = json.loads((ROOT / "outputs" / "eval" / "v2_run2_9M.json").read_text())["held_out_seen"]
    rows = {line.split("|")[1].strip(): line for line in README.splitlines() if line.startswith("| ")}
    for map_name, was_seen in seen.items():
        assert ("| seen |" in rows[map_name]) == was_seen, map_name
        assert ("**UNSEEN**" in rows[map_name]) == (not was_seen), map_name


def test_report_experiment_numbers_are_in_the_text_and_sum_to_the_game_count() -> None:
    spec = importlib.util.spec_from_file_location("readme_assets", ROOT / "scripts" / "make_readme_assets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses with postponed annotations look their module up here
    try:
        spec.loader.exec_module(module)
        cfg = module.load_cfg(ROOT / "configs" / "plots" / "readme_assets.yaml")  # checks that every bar sums to its game count
    finally:
        sys.modules.pop(spec.name, None)
    for exp in cfg.experiments:
        for bar in exp.bars:
            pattern = rf"\b{bar.win}( wins)? / {bar.draw}( draws)? / {bar.loss}( losses)?"
            assert re.search(pattern, README), (exp.id, bar.label)
