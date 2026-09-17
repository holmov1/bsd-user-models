"""Evaluate saved linear probes against wilduser teacher pseudo-labels."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

from user_distillation.probes.utils import load_wilduser_features
from user_distillation.training.subspace_projector import LowRankUserSubspaceProjector


def eval_probes(
    data_path: str,
    h_path: str,
    probes_dir,
    projector: LowRankUserSubspaceProjector | None = None,
    beliefs_path: str | None = None,
    layer_idx: int = 0,
    limit: int | None = None,
    device: str = "cuda",
) -> dict:
    """Evaluate every probe_<attr>.pkl under probes_dir. Returns {attr: {n, probe_acc, probe_f1}}."""
    probes_dir = Path(probes_dir)
    wu_feats = load_wilduser_features(
        data_path, h_path, projector, device, limit,
        beliefs_path=beliefs_path, layer_idx=layer_idx,
    )

    results = {}
    for attr, (X, y_true) in wu_feats.items():
        probe_path = probes_dir / f"probe_{attr}.pkl"
        print(f"\n{'=' * 55}\nAttr: {attr}  ({len(y_true)} test samples)")
        if not probe_path.exists():
            print(f"  No probe at {probe_path}, skipping.")
            results[attr] = {"n": len(y_true), "error": "probe not found"}
            continue

        probe   = joblib.load(probe_path)
        classes = [str(c) for c in probe.classes_]
        mask    = np.isin(y_true, classes)
        X, y_true = X[mask], y_true[mask]
        if len(y_true) == 0:
            results[attr] = {"n": 0}
            continue

        probe_pred = probe.predict(X)
        probe_acc  = float((probe_pred == y_true).mean())
        present    = sorted(set(y_true.tolist()))
        probe_f1   = float(f1_score(y_true, probe_pred, labels=present, average="macro", zero_division=0))
        print(f"  probe_acc={probe_acc:.3f}  probe_f1={probe_f1:.3f}")

        results[attr] = {"n": len(y_true), "probe_acc": round(probe_acc, 4), "probe_f1": round(probe_f1, 4)}

    return results
