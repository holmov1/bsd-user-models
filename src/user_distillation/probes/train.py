"""Fit linear probes on v_user (projector output) or raw h_teacher (control baseline)."""
from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from user_distillation.data.attr_schema import ATTRIBUTE_SCHEMA
from user_distillation.probes.utils import load_wilduser_features
from user_distillation.training.subspace_projector import LowRankUserSubspaceProjector


def fit_probes(
    data_path: str,
    h_path: str,
    save_dir,
    projector: LowRankUserSubspaceProjector | None = None,
    beliefs_path: str | None = None,
    layer_idx: int = 0,
    limit: int | None = None,
    device: str = "cuda",
) -> dict:
    """Fit one LogisticRegression probe per eligible attribute"""
    save_dir.mkdir(parents=True, exist_ok=True)
    feat_dim = projector.A.weight.shape[0] if projector is not None else 4096

    wu_feats = load_wilduser_features(
        data_path, h_path, projector, device, limit,
        beliefs_path=beliefs_path, layer_idx=layer_idx,
    )

    results = {}
    for attr, (X, y) in wu_feats.items():
        classes = ATTRIBUTE_SCHEMA[attr]
        print(f"\n{'=' * 55}\nAttr: {attr}  classes={classes}  ({len(y)} samples)")
        if len(y) < 10:
            print("  Too few samples, skipping.")
            continue
        probe = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs")
        probe.fit(X, y)
        y_pred    = probe.predict(X)
        train_acc = float((y_pred == y).mean())
        counts    = {c: int((y == c).sum()) for c in set(y.tolist())}
        present   = sorted(c for c, n in counts.items() if n >= 2000)
        if not present:
            present = sorted(counts)
        train_f1  = float(f1_score(y, y_pred, labels=present, average="macro", zero_division=0))
        print(f"  train acc: {train_acc:.3f}  train f1: {train_f1:.3f}")
        joblib.dump(probe, save_dir / f"probe_{attr}.pkl")
        print(f"  saved → probe_{attr}.pkl")
        results[attr] = {"n_train": len(y), "train_acc": round(train_acc, 4), "train_f1": round(train_f1, 4)}

    return {"feat_dim": feat_dim, "attrs": list(wu_feats.keys()), "probe_results": results}
