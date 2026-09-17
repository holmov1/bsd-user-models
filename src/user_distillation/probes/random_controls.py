"""Random-direction control-baseline generators (isotropic, seeded) for probe training/eval."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_CHUNK = 4096


def random_unit_rows(n: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    rows = torch.randn(n, dim, generator=generator, dtype=torch.float32)
    return rows / rows.norm(dim=1, keepdim=True)


def make_random_projector(dim: int, rank: int, seed: int, out_path) -> None:
    """Save a {A.weight, B.weight} projector state dict with random unit-row A (and B, unused by probes)."""
    g = torch.Generator().manual_seed(seed)
    A_weight = random_unit_rows(rank, dim, g)
    B_weight = random_unit_rows(dim, rank, g)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"A.weight": A_weight, "B.weight": B_weight}, out_path)


def make_random_h(real_h_path: str, real_layer_idx: int, seed: int, out_path, progress=None) -> None:
    """Save a random-direction h_teacher.npy: real per-row magnitude, random unit direction."""
    real = np.load(real_h_path, mmap_mode="r")
    n, dim = real.shape[0], real.shape[-1]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, 1, dim))

    g = torch.Generator().manual_seed(seed)
    for start in range(0, n, _CHUNK):
        end = min(start + _CHUNK, n)
        real_chunk = torch.from_numpy(np.asarray(real[start:end, real_layer_idx, :], dtype=np.float32))
        real_norms = real_chunk.norm(dim=1, keepdim=True)

        rand = torch.randn(end - start, dim, generator=g, dtype=torch.float32)
        rand = rand / rand.norm(dim=1, keepdim=True) * real_norms

        out[start:end, 0, :] = rand.to(torch.float16).numpy()
        if progress is not None:
            progress(start, end, n)

    out.flush()
