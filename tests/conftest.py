"""Test configuration: skip RL/viz tests without their optional dependencies; run pygame headless."""

import importlib.util
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

collect_ignore_glob: list[str] = []
if importlib.util.find_spec("numpy") is None:
    collect_ignore_glob += ["test_rl_*.py", "test_viz_*.py"]
