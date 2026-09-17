"""Random-direction (isotropic h-space) steering control."""

import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from user_distillation.probes.utils import load_projector
from user_distillation.steering.config import PROBE_LAYOUTS_PATH, load_probe_layouts
from user_distillation.steering.evaluator import (
    SteeringConfig,
    SteeringEvaluator,
    load_steering_model,
    load_test_records,
    run_sweep,
)
from user_distillation.steering.random_controls.flips import load_flip_rates, pool_random_flip_rates
from user_distillation.steering.vectors import VectorStore


class RandomDirectionVectorStore:
    """VectorStore wrapper returning a magnitude-matched random h-space direction."""

    def __init__(self, real: VectorStore, seed: int):
        self._real = real
        self._seed = seed

    def __getattr__(self, name):
        return getattr(self._real, name)

    def delta_h(self, attr, tgt_class, alpha, space, projector=None,
                src_class=None, relative=False):
        real_delta = self._real.delta_h(
            attr, tgt_class, alpha, space, projector, src_class=src_class, relative=relative,
        )
        target_norm = real_delta.norm()
        key = f"{attr}|{src_class}|{tgt_class}|{alpha}|{space}"
        condition_seed = (self._seed ^ zlib.crc32(key.encode())) & 0xffff_ffff
        g = torch.Generator().manual_seed(condition_seed)
        rand = torch.randn(real_delta.shape, generator=g, dtype=torch.float32)
        rand = rand / rand.norm()
        return (rand * target_norm).to(real_delta.dtype)


def wilcoxon_report(real_results_path: Path, random_seed_paths: list[Path]) -> str:
    """Paired Wilcoxon signed-rank test, real vs random, per space."""
    real_v = load_flip_rates(real_results_path, "v")
    real_h = load_flip_rates(real_results_path, "h")
    random_flips = pool_random_flip_rates(random_seed_paths)

    lines = [
        "Paired Wilcoxon signed-rank test: real steering vs random-direction control",
        f"real results:   {real_results_path}",
        f"random seeds:   {[str(p) for p in random_seed_paths]}",
        "=" * 70,
    ]

    for space, real_flips in [("v", real_v), ("h", real_h)]:
        common = sorted(set(real_flips) & set(random_flips))
        if not common:
            lines.append(f"{space}-space: no overlapping pairs with random control, skipped")
            continue
        real_arr = np.array([real_flips[k] for k in common])
        random_arr = np.array([np.mean(random_flips[k]) for k in common])
        res = stats.wilcoxon(real_arr, random_arr, alternative="greater")
        sig = "significant (p<0.05)" if res.pvalue < 0.05 else "NOT significant"
        lines.append(
            f"{space}-space vs random  n_pairs={len(common)}  "
            f"mean_real={real_arr.mean():.4f}  mean_random={random_arr.mean():.4f}  "
            f"W={res.statistic:.1f}  p={res.pvalue:.4e}  -> {sig}"
        )
        lines.append(f"  pairs: {common}")

    return "\n".join(lines)


def run_direction_control(
    model_config_path: str,
    vectors_path: str,
    projector_path: str,
    beliefs_path: str,
    data_path: str,
    attrs: list[str],
    alphas: list[float],
    seeds: list[int],
    out_base: Path,
    config: SteeringConfig,
    pairwise: bool = True,
    relative: bool = False,
) -> None:
    """Run the random-direction control for one model across seeds."""
    print("Loading steering vectors (magnitude reference only, direction unused) ...", flush=True)
    real_vs = VectorStore.load(vectors_path)
    proj = load_projector(projector_path, config.device)
    probe_layouts = load_probe_layouts(PROBE_LAYOUTS_PATH)
    test_records = load_test_records(data_path, beliefs_path)
    sm = load_steering_model(model_config_path, config.device)
    if config.inject_layer is None:
        config = replace(config, inject_layer=real_vs.layer)
    evaluator = SteeringEvaluator(sm, proj, probe_layouts, config)

    for seed in seeds:
        out_path = out_base.with_stem(f"{out_base.stem}_seed{seed}")
        vs = RandomDirectionVectorStore(real_vs, seed=seed)
        print(f"\n{'#' * 60}\nSeed {seed}  source={real_vs.source}  layer={real_vs.layer}  "
              f"inject_layer={config.inject_layer}", flush=True)

        meta = {
            "control":      "random_direction_magnitude_matched",
            "spaces":       ["h"],
            "pairwise":     pairwise,
            "inject_layer": config.inject_layer,
            "model":        sm.model_id,
            "seed":         seed,
        }
        run_sweep(evaluator, vs, test_records, attrs, alphas, ["h"], out_path, meta,
                  pairwise=pairwise, relative=relative)
        print(f"Done seed {seed} -> {out_path}", flush=True)

    print(f"\nAll seeds done: {seeds}", flush=True)
