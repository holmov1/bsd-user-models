"""Belief sharpness: per-attribute confidence and how much averaging shrinks it."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from user_distillation.data.attr_schema import ATTRIBUTE_SCHEMA


def compute_sharpness(
    beliefs_path: str, limit: int | None = None,
) -> tuple[int, list[dict], dict[str, list[float]]]:
    """Per-attribute belief-sharpness stats from a beliefs.jsonl file; returns (n, rows, top_prob)."""
    pmax_mean: dict[str, list[float]] = {a: [] for a in ATTRIBUTE_SCHEMA}
    margin: dict[str, list[float]] = {a: [] for a in ATTRIBUTE_SCHEMA}
    pmax_layout: dict[str, list[float]] = {a: [] for a in ATTRIBUTE_SCHEMA}
    agree: dict[str, list[float]] = {a: [] for a in ATTRIBUTE_SCHEMA}
    nres: dict[str, list[float]] = {a: [] for a in ATTRIBUTE_SCHEMA}
    top_prob: dict[str, list[float]] = {a: [] for a in ATTRIBUTE_SCHEMA}

    n = 0
    with open(beliefs_path) as f:
        for line in f:
            r = json.loads(line)
            bel, prb = r.get("beliefs", {}), r.get("probes", {})
            for a, v in bel.items():
                if a not in pmax_mean:
                    continue
                v = np.asarray(v, dtype=float)
                s = np.sort(v)[::-1]
                pmax_mean[a].append(s[0])
                margin[a].append(s[0] - s[1])
                top_prob[a].append(float(v.max()))
                nres[a].append(r.get("n_layouts_resolved", {}).get(a, 0))
                L = [np.asarray(x, dtype=float) for x in prb.get(a, []) if x is not None]
                if L:
                    pmax_layout[a].append(float(np.mean([x.max() for x in L])))
                    votes = [int(np.argmax(x)) for x in L]
                    agree[a].append(Counter(votes).most_common(1)[0][1] / len(votes))
            n += 1
            if limit and n >= limit:
                break

    rows = []
    for a, classes in ATTRIBUTE_SCHEMA.items():
        if not pmax_mean[a]:
            continue
        K = len(classes)
        pm = np.array(pmax_mean[a])
        pl = np.array(pmax_layout[a]) if pmax_layout[a] else np.array([np.nan])
        ag = np.array(agree[a]) if agree[a] else np.array([np.nan])
        head = (pm.mean() - 1 / K) / (1 - 1 / K)
        rows.append({
            "attr":        a,
            "K":           K,
            "pmax":        float(pm.mean()),
            "pmax_median": float(np.median(pm)),
            "pmax_p90":    float(np.percentile(pm, 90)),
            "pmax_layout": float(np.nanmean(pl)),
            "agree":       float(np.nanmean(ag)),
            "margin":      float(np.mean(margin[a])),
            "headroom":    float(head),
            "nlay":        float(np.mean(nres[a])),
        })

    return n, rows, top_prob
