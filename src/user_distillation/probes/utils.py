"""Shared utilities for probe training and evaluation."""

from __future__ import annotations

import json

import numpy as np
import torch
from tqdm import tqdm

from user_distillation.data.attr_schema import ATTRIBUTE_SCHEMA
from user_distillation.training.subspace_projector import LowRankUserSubspaceProjector
from user_distillation.training.utils import resolve_h_layer_idx  # noqa: F401  (re-exported)

_CHUNK = 512


def load_projector(path: str, device: str) -> LowRankUserSubspaceProjector:
    """Load a frozen, eval-mode LowRankUserSubspaceProjector from a saved state dict."""
    state = torch.load(path, map_location="cpu")
    rank = state["A.weight"].shape[0]
    dim  = state["A.weight"].shape[1]
    proj = LowRankUserSubspaceProjector(dim, rank)
    proj.load_state_dict(state)
    proj.to(device).eval()
    for p in proj.parameters():
        p.requires_grad = False
    return proj


def load_wilduser_features(
    records_path: str,
    h_path: str,
    projector: LowRankUserSubspaceProjector | None,
    device: str,
    limit: int | None = None,
    beliefs_path: str | None = None,
    layer_idx: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load h_teacher from npy (joined to beliefs_path by fingerprint), optionally project; returns {attr: (X, y)}."""
    h_mm = np.load(h_path, mmap_mode="r")

    records: list[dict] = []
    with open(records_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            records.append(json.loads(line))

    if beliefs_path is not None:
        beliefs_idx: dict[str, dict] = {}
        with open(beliefs_path) as f:
            for line in f:
                b = json.loads(line)
                fp = b.get("fingerprint") or str(b.get("idx"))
                beliefs_idx[fp] = b

        joined: list[dict] = []
        n_missing = 0
        for r in records:
            fp = r.get("fingerprint") or str(r.get("idx"))
            b = beliefs_idx.get(fp)
            if b is None:
                n_missing += 1
                continue
            joined.append({**r, "h_row": b["h_row"], "beliefs": b.get("beliefs", {})})
        if n_missing:
            print(f"  {n_missing}/{len(records)} records missing from {beliefs_path}, dropped")
        records = joined

    h_raw = np.stack(
        [h_mm[r["h_row"], layer_idx, :].astype(np.float32) for r in records], axis=0
    )

    if projector is not None:
        proj_device = next(projector.parameters()).device
        chunks = []
        for start in tqdm(range(0, len(h_raw), _CHUNK), desc="  project"):
            t = torch.from_numpy(h_raw[start : start + _CHUNK]).to(proj_device)
            with torch.no_grad():
                chunks.append(projector.A(t).float().cpu().numpy())
        X_all = np.vstack(chunks)
    else:
        X_all = h_raw

    by_attr: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for attr, classes in ATTRIBUTE_SCHEMA.items():
        mask = np.array([attr in r.get("beliefs", {}) for r in records])
        labels = np.array([
            classes[int(np.argmax(r["beliefs"][attr]))] for r, m in zip(records, mask) if m
        ])
        by_attr[attr] = (X_all[mask], labels)

    return by_attr


def print_eval_report(results: dict, title: str = "") -> None:
    """Print a probe_acc/probe_f1 table (with macro average) from eval_probes()'s output."""
    if title:
        print(f"\n{title}")
    header = f"{'Attr':<25} {'N':>5} {'ProbeAcc':>9} {'ProbeF1':>9}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for attr, m in results.items():
        if "probe_acc" not in m:
            continue
        f1_str = f"{m['probe_f1']:>9.3f}" if m.get("probe_f1") is not None else f"{'—':>9}"
        print(f"{attr:<25} {m['n']:>5} {m['probe_acc']:>9.3f} {f1_str}")
    print("=" * len(header))
    accs = [m["probe_acc"] for m in results.values() if "probe_acc" in m]
    f1s  = [m["probe_f1"]  for m in results.values() if m.get("probe_f1") is not None]
    f1_avg = f"{np.mean(f1s):>9.3f}" if f1s else f"{'—':>9}"
    print(f"{'MACRO AVG':<25} {'':>5} {np.mean(accs):>9.3f} {f1_avg}")
