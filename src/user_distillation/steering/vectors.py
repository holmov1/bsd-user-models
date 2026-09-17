"""Steering vector computation and storage."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from user_distillation.steering.config import SEED, STEERING_ATTRS
from user_distillation.training.subspace_projector import LowRankUserSubspaceProjector

_CHUNK = 512


@dataclass
class VectorStore:
    """Steering vectors in both h-space (hidden_dim) and v-space (rank-dim)."""

    steering_h:    dict[str, dict[str, Tensor]]
    steering_v:    dict[str, dict[str, Tensor]]
    class_means_h: dict[str, dict[str, Tensor]]
    class_means_v: dict[str, dict[str, Tensor]]
    source:        str
    layer:         int
    projector_path: str

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        data = torch.load(path, map_location="cpu")
        data.setdefault("class_means_h", {})
        data.setdefault("class_means_v", {})
        return cls(**data)

    def _global_h_norm(self, attr: str) -> float:
        """Mean class-activation norm for attr."""
        means = list(self.class_means_h[attr].values())
        global_h = torch.stack(means).mean(dim=0)
        return float(global_h.norm())

    def delta_h(
        self,
        attr:      str,
        tgt_class: str,
        alpha:     float,
        space:     str,
        projector: LowRankUserSubspaceProjector | None = None,
        src_class: str | None = None,
        relative:  bool = False,
    ) -> Tensor:
        """Return the h-space delta to inject for (attr, tgt_class, alpha, space)."""
        def _raw_v(cls: str) -> Tensor:
            if src_class is not None and self.class_means_v.get(attr):
                return self.class_means_v[attr][cls]
            return self.steering_v[attr][cls]

        def _raw_h(cls: str) -> Tensor:
            if src_class is not None and self.class_means_h.get(attr):
                return self.class_means_h[attr][cls]
            return self.steering_h[attr][cls]

        if space == "v":
            sv = _raw_v(tgt_class) - _raw_v(src_class) if src_class else _raw_v(tgt_class)
            if sv.norm() == 0:
                raise ValueError(f"Zero steering_v for {attr}/{tgt_class}")
            sv_unit = sv / sv.norm()
            sv_scaled = (alpha * sv_unit).to(next(projector.parameters()).device)
            sv_scaled = sv_scaled.to(next(projector.parameters()).dtype)
            with torch.no_grad():
                delta = projector.B(sv_scaled).float().cpu()
            if relative:
                if delta.norm() == 0:
                    raise ValueError(f"Zero projected delta for {attr}/{tgt_class}")
                delta = delta / delta.norm() * (alpha * self._global_h_norm(attr))
            return delta
        elif space == "h":
            sh = _raw_h(tgt_class) - _raw_h(src_class) if src_class else _raw_h(tgt_class)
            if sh.norm() == 0:
                raise ValueError(f"Zero steering_h for {attr}/{tgt_class}")
            sh_unit = sh / sh.norm()
            target_norm = alpha * self._global_h_norm(attr) if relative else alpha
            return (target_norm * sh_unit).float()
        else:
            raise ValueError(f"Unknown space: {space!r}. Use 'v' or 'h'.")

    def attrs(self) -> list[str]:
        return list(self.steering_h.keys())

    def classes(self, attr: str) -> list[str]:
        return list(self.steering_h.get(attr, {}).keys())


def _project_h(h_raw: np.ndarray, projector: LowRankUserSubspaceProjector, device: str) -> np.ndarray:
    """Apply projector.A to h_raw [N, 4096] to get v [N, rank]."""
    chunks = []
    dev = torch.device(device)
    for start in range(0, len(h_raw), _CHUNK):
        t = torch.from_numpy(h_raw[start : start + _CHUNK]).to(dev)
        with torch.no_grad():
            chunks.append(projector.A(t.to(next(projector.parameters()).dtype)).float().cpu().numpy())
    return np.vstack(chunks)


def _resolve_layer_slice(h_path: str | Path, inject_layer: int) -> int:
    """Second-dimension index in h_teacher.npy for inject_layer."""
    meta_path = Path(h_path).parent / "h_teacher_meta.json"
    if not meta_path.exists():
        return 0
    with open(meta_path) as f:
        meta = json.load(f)
    layers = meta.get("layers", [])
    if not layers:
        return 0
    if inject_layer not in layers:
        raise ValueError(
            f"inject_layer={inject_layer} not found in h_teacher_meta.json layers={layers}. "
            f"Available: {layers}"
        )
    idx = layers.index(inject_layer)
    print(f"  h_teacher slice: layer {inject_layer} → index {idx} of {layers}")
    return idx


def compute_centroid_vectors(
    steering_dir: str | Path,
    h_path: str | Path,
    projector: LowRankUserSubspaceProjector,
    device: str,
    projector_path: str = "",
    inject_layer: int = 0,
) -> VectorStore:
    """Compute class-centroid steering vectors from h_teacher.npy."""
    steering_dir = Path(steering_dir)
    h_mm = np.load(h_path, mmap_mode="r")
    layer_slice = _resolve_layer_slice(h_path, inject_layer)

    rank = projector.A.weight.shape[0]
    dim  = projector.A.weight.shape[1]

    steering_h:    dict[str, dict[str, Tensor]] = {}
    steering_v:    dict[str, dict[str, Tensor]] = {}
    class_means_h: dict[str, dict[str, Tensor]] = {}
    class_means_v: dict[str, dict[str, Tensor]] = {}

    header = f"{'Attr':<25} {'Class':<35} {'N':>6}  {'|steer_h|':>10}  {'|steer_v|':>10}"
    print(header)
    print("-" * len(header))

    for attr, viable in STEERING_ATTRS.items():
        ds_path = steering_dir / f"{attr}.json"
        if not ds_path.exists():
            print(f"  WARNING: {ds_path} not found, skipping {attr}")
            continue

        with open(ds_path) as f:
            ds = json.load(f)

        class_h: dict[str, np.ndarray] = {}
        class_v: dict[str, np.ndarray] = {}

        for cls in viable:
            if cls not in ds["classes"]:
                print(f"  WARNING: {attr}/{cls} not in steering dataset, skipping")
                continue
            h_rows = ds["classes"][cls]
            h_mat  = np.stack([h_mm[r, layer_slice, :].astype(np.float32) for r in h_rows])
            v_mat  = _project_h(h_mat, projector, device)
            class_h[cls] = h_mat.mean(axis=0)
            class_v[cls] = v_mat.mean(axis=0)

        if not class_h:
            continue

        global_mean_h = np.stack(list(class_h.values())).mean(axis=0)
        global_mean_v = np.stack(list(class_v.values())).mean(axis=0)

        steering_h[attr]    = {}
        steering_v[attr]    = {}
        class_means_h[attr] = {}
        class_means_v[attr] = {}
        for cls in class_h:
            sh = torch.from_numpy(class_h[cls] - global_mean_h)
            sv = torch.from_numpy(class_v[cls] - global_mean_v)
            steering_h[attr][cls]    = sh
            steering_v[attr][cls]    = sv
            class_means_h[attr][cls] = torch.from_numpy(class_h[cls])
            class_means_v[attr][cls] = torch.from_numpy(class_v[cls])
            n = len(ds["classes"][cls])
            print(f"  {attr:<25} {cls:<35} {n:>6}  {sh.norm():>10.4f}  {sv.norm():>10.4f}")

    return VectorStore(
        steering_h=steering_h,
        steering_v=steering_v,
        class_means_h=class_means_h,
        class_means_v=class_means_v,
        source="centroid",
        layer=inject_layer,
        projector_path=projector_path,
    )


def build_steering_datasets(
    records_path: str | Path,
    out_dir: str | Path,
    probe_layouts: dict,
    min_src_prob: float = 0.5,
    train_fps_path: str | Path = "data/wilduser/train.jsonl",
    seed: int = SEED,
) -> dict[str, dict]:
    """Stratify a model's beliefs.jsonl into balanced per-class steering datasets."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    with open(train_fps_path) as f:
        train_fps = {json.loads(l).get("fingerprint") for l in f}
    print(f"Train fingerprints: {len(train_fps):,} from {train_fps_path}")

    buckets: dict[str, dict[str, list[tuple[int, float]]]] = {
        attr: defaultdict(list) for attr in STEERING_ATTRS
    }

    print(f"Reading {records_path} ...  (min_src_prob={min_src_prob})")
    n_total = n_kept = n_skipped = 0
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("fingerprint") not in train_fps:
                n_skipped += 1
                continue
            h_row = r["h_row"]
            beliefs = r.get("beliefs", {})
            for attr, viable in STEERING_ATTRS.items():
                if attr not in beliefs:
                    continue
                probs = beliefs[attr]
                src_idx = int(np.argmax(probs))
                classes = probe_layouts[attr]["canonical_order"]
                label = classes[src_idx]
                n_total += 1
                if label in viable and probs[src_idx] >= min_src_prob:
                    buckets[attr][label].append((h_row, probs[src_idx]))
                    n_kept += 1
    print(f"  skipped {n_skipped} non-train records")
    print(f"  kept {n_kept}/{n_total} (attr,sample) pairs above threshold")

    out: dict[str, dict] = {}
    for attr, viable in STEERING_ATTRS.items():
        counts = {cls: len(buckets[attr][cls]) for cls in viable}
        n_per_class = min(counts.values())

        print(f"\n{attr}")
        for cls in viable:
            print(f"  {cls:<35} {counts[cls]:>7}  ->  {n_per_class}")

        classes_out = {}
        for cls in viable:
            rows = sorted(buckets[attr][cls], key=lambda x: x[1], reverse=True)[:n_per_class]
            rng.shuffle(rows)
            classes_out[cls] = [h_row for h_row, _ in rows]

        attr_out = {"attr": attr, "n_per_class": n_per_class, "classes": classes_out}
        path = out_dir / f"{attr}.json"
        with open(path, "w") as f:
            json.dump(attr_out, f)
        print(f"  saved -> {path}")
        out[attr] = attr_out

    return out


