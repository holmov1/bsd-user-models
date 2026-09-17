"""Sign test and paired Wilcoxon test for steering significance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, wilcoxon

from user_distillation.steering.random_controls.flips import (
    BROKEN_THRESHOLD,
    load_flip_rates,
    pool_random_flip_rates,
)


@dataclass
class SignTestResult:
    n_cells:    int
    wins:       int
    p_value:    float
    mean_delta: float
    delta_ci:   tuple[float, float]


def sign_test(
    real: dict[str, float], random_draws: dict[str, list[float]],
    n_boot: int = 10_000, seed: int = 0,
) -> SignTestResult:
    """Exact binomial sign test: real vs random draws, pooled across cells."""
    keys = sorted(set(real) & set(random_draws))
    if not keys:
        raise ValueError("no (attribute, pair) cells in common between real and random results")
    n_random_per_cell = {len(random_draws[k]) for k in keys}
    if len(n_random_per_cell) != 1:
        raise ValueError(f"uneven random-seed count across cells: {n_random_per_cell}")
    n_random = n_random_per_cell.pop()

    wins = sum(1 for k in keys if real[k] > max(random_draws[k]))
    deltas = np.array([real[k] - np.mean(random_draws[k]) for k in keys])

    p = binomtest(wins, len(keys), 1 / (n_random + 1), alternative="greater").pvalue

    rng = np.random.default_rng(seed)
    boot_means = rng.choice(deltas, size=(n_boot, len(deltas)), replace=True).mean(axis=1)
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))

    return SignTestResult(len(keys), wins, float(p), float(deltas.mean()), ci)


@dataclass
class PairedTestResult:
    n_cells:    int
    wins:       int
    ties:       int
    p_value:    float
    mean_delta: float
    delta_ci:   tuple[float, float]


def paired_test(
    v: dict[str, float], h: dict[str, float], n_boot: int = 10_000, seed: int = 0,
) -> PairedTestResult:
    """Paired Wilcoxon signed-rank test between v and h."""
    keys = sorted(set(v) & set(h))
    if not keys:
        raise ValueError("no (attribute, pair) cells in common between v and h results")
    diffs = np.array([v[k] - h[k] for k in keys])
    wins = int((diffs > 0).sum())
    ties = int((diffs == 0).sum())

    if np.any(diffs != 0):
        _, p = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
    else:
        p = 1.0

    rng = np.random.default_rng(seed)
    boot_means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))

    return PairedTestResult(len(keys), wins, ties, float(p), float(diffs.mean()), ci)


@dataclass
class ModelArm:
    name:          str
    v_path:        str
    h_path:        str
    random_paths:  list[str]


def load_arm(
    arm: ModelArm, broken_threshold: float = BROKEN_THRESHOLD,
) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
    """(v_flip_rates, h_flip_rates, random_draws) for one model arm."""
    prefix = lambda d: {f"{arm.name}::{k}": v for k, v in d.items()}
    v = prefix(load_flip_rates(arm.v_path, "v", broken_threshold))
    h = prefix(load_flip_rates(arm.h_path, "h", broken_threshold))
    r = prefix(pool_random_flip_rates(arm.random_paths, broken_threshold))
    return v, h, r


def run(
    arms: list[ModelArm], space: str, broken_threshold: float = BROKEN_THRESHOLD,
) -> dict[str, SignTestResult]:
    """Sign test per arm plus one pooled test across all arms."""
    assert space in ("v", "h")
    per_arm: dict[str, tuple[dict, dict, dict]] = {arm.name: load_arm(arm, broken_threshold)
                                                    for arm in arms}

    results: dict[str, SignTestResult] = {}
    real_all: dict[str, float] = {}
    random_all: dict[str, list[float]] = {}
    for name, (v, h, r) in per_arm.items():
        real = v if space == "v" else h
        results[name] = sign_test(real, r)
        real_all.update(real)
        random_all.update(r)
    results["pooled"] = sign_test(real_all, random_all)
    return results


def run_v_vs_h(
    arms: list[ModelArm], broken_threshold: float = BROKEN_THRESHOLD,
) -> dict[str, PairedTestResult]:
    """Paired v-vs-h test per arm plus one pooled test across all arms."""
    per_arm: dict[str, tuple[dict, dict, dict]] = {arm.name: load_arm(arm, broken_threshold)
                                                    for arm in arms}
    results: dict[str, PairedTestResult] = {}
    v_all: dict[str, float] = {}
    h_all: dict[str, float] = {}
    for name, (v, h, _r) in per_arm.items():
        results[name] = paired_test(v, h)
        v_all.update(v)
        h_all.update(h)
    results["pooled"] = paired_test(v_all, h_all)
    return results
