"""Extract best-alpha flip rates per (attribute, pair) cell from eval_results*.json."""

import json
from pathlib import Path

BROKEN_THRESHOLD = 0.15


def select_best_alpha(
    pair: dict, space: str, broken_threshold: float = BROKEN_THRESHOLD,
) -> tuple[float, float, float] | None:
    """Best (alpha, flip_rate, broken_rate) for this pair, under the broken-rate threshold."""
    candidates = [
        (float(alpha), entry["flip_rate"], entry["broken_rate"])
        for alpha, spaces in pair.get("alphas", {}).items()
        for entry in [spaces.get(space)]
        if entry and entry.get("flip_rate") is not None and entry.get("broken_rate") is not None
    ]
    if not candidates:
        return None
    clean = [c for c in candidates if c[2] <= broken_threshold]
    return max(clean, key=lambda c: c[1]) if clean else min(candidates, key=lambda c: c[2])


def flip_rates_by_pair(
    results: dict, space: str, broken_threshold: float = BROKEN_THRESHOLD,
) -> dict[str, float]:
    """{"attr::src→tgt": flip_rate} for every pair with data in this space."""
    out: dict[str, float] = {}
    for attr, pairs in results.items():
        if attr.startswith("_"):
            continue
        for key, pair in pairs.items():
            if "→" not in key:
                continue
            best = select_best_alpha(pair, space, broken_threshold)
            if best is not None:
                out[f"{attr}::{key}"] = best[1]
    return out


def load_flip_rates(
    path: str | Path, space: str, broken_threshold: float = BROKEN_THRESHOLD,
) -> dict[str, float]:
    """flip_rates_by_pair loaded straight from an eval_results*.json file."""
    return flip_rates_by_pair(json.loads(Path(path).read_text()), space, broken_threshold)


def pool_random_flip_rates(
    paths: list[str | Path], broken_threshold: float = BROKEN_THRESHOLD,
) -> dict[str, list[float]]:
    """{"attr::src→tgt": [flip_rate per seed]}, restricted to pairs common to every seed."""
    per_seed = [load_flip_rates(p, "h", broken_threshold) for p in paths]
    if not per_seed:
        return {}
    common = set.intersection(*(set(d) for d in per_seed))
    return {k: [d[k] for d in per_seed] for k in common}
