"""Small statistics helpers for evaluation: Wilson intervals for rates, normal intervals for means."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence


def z_value(confidence: float) -> float:
    """Two-sided normal quantile, e.g. 1.96 for 0.95."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion. With ``n == 0`` nothing is known: (0, 1)."""
    if n < 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")
    if n == 0:
        return 0.0, 1.0
    z = z_value(confidence)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)  # the endpoints are exact, not 2.8e-17
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return lo, hi


def rate(successes: int, n: int, confidence: float) -> dict[str, float | int | None]:
    """A proportion with its Wilson interval (``rate`` is None when ``n == 0``)."""
    lo, hi = wilson_interval(successes, n, confidence)
    return {"count": successes, "n": n, "rate": successes / n if n else None, "lo": lo, "hi": hi}


def mean_stat(values: Sequence[float], confidence: float) -> dict[str, float | int | None]:
    """Mean with a normal-approximation interval (needs >= 2 values for an interval)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "lo": None, "hi": None}
    mean = math.fsum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "se": None, "lo": None, "hi": None}
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    half = z_value(confidence) * se
    return {"n": n, "mean": mean, "se": se, "lo": mean - half, "hi": mean + half}
