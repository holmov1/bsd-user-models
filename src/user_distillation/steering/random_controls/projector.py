"""Random-orthonormal-projector-B steering control."""

import gc
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from user_distillation.probes.utils import load_projector
from user_distillation.steering.config import STEERING_ATTRS
from user_distillation.steering.evaluator import (
    SteeringConfig,
    SteeringEvaluator,
    load_steering_model,
    load_test_records,
    run_sweep,
)
from user_distillation.steering.vectors import VectorStore

DEFAULT_ATTRS_ORDER = sorted(STEERING_ATTRS, key=lambda a: len(STEERING_ATTRS[a]) != 2)


@dataclass
class ModelSpec:
    key: str
    model_config: str
    projector: str
    vectors: str
    beliefs: str
    alphas: list[float]


MODELS = [
    ModelSpec(
        key="llama",
        model_config="user_distillation/config/models/llama-3.1-8b-instruct.yaml",
        projector="experiments/wilduser_distill/llama/layer16_iso_lasttok/w_user_best.pt",
        vectors="experiments/wilduser_distill/llama/layer16_iso_lasttok/steering/centroid_vectors.pt",
        beliefs="data/wilduser/llama/beliefs_corrected.jsonl",
        alphas=[0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0],
    ),
    ModelSpec(
        key="qwen",
        model_config="user_distillation/config/models/qwen3-8b-dense.yaml",
        projector="experiments/wilduser_distill/qwen3/layer18_iso_lasttok/w_user_best.pt",
        vectors="experiments/wilduser_distill/qwen3/layer18_iso_lasttok/steering/centroid_vectors.pt",
        beliefs="data/wilduser/qwen3-8b/beliefs_corrected.jsonl",
        alphas=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    ),
    ModelSpec(
        key="olmo",
        model_config="user_distillation/config/models/olmo-3-7b.yaml",
        projector="experiments/wilduser_distill/olmo/layer18_iso_baseline/w_user_best.pt",
        vectors="experiments/wilduser_distill/olmo/layer18_iso_baseline/steering/centroid_vectors_best.pt",
        beliefs="data/wilduser/olmo-3-7b/beliefs_corrected.jsonl",
        alphas=[0.0, 5.0, 10.0, 16.0, 22.0, 28.0],
    ),
]
MODELS_BY_KEY = {m.key: m for m in MODELS}


def random_orthonormal_B(dim: int, rank: int, seed: int,
                          device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Random orthonormal [dim, rank] matrix via QR."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(dim, rank, generator=g, dtype=torch.float32)
    Q, R = torch.linalg.qr(W)
    d = torch.diagonal(R)
    sign = torch.where(d == 0, torch.ones_like(d), torch.sign(d))
    Q = Q * sign.unsqueeze(0)
    return Q.to(device=device, dtype=dtype)


def run_model(
    spec: ModelSpec,
    probe_layouts: dict,
    out_root: Path,
    seeds: list[int],
    attrs: list[str],
    config: SteeringConfig,
    data_path: str = "data/wilduser/test.jsonl",
    alphas: list[float] | None = None,
    pairwise: bool = True,
    relative: bool = False,
) -> None:
    """Run the random-orthonormal-projector control for one model across seeds."""
    print("Loading steering vectors (real A / v-space directions, unchanged) ...", flush=True)
    vs = VectorStore.load(spec.vectors)
    print(f"  source={vs.source}  layer={vs.layer}", flush=True)

    proj = load_projector(spec.projector, config.device)
    rank, dim = proj.A.weight.shape
    b_device, b_dtype = proj.B.weight.device, proj.B.weight.dtype

    test_records = load_test_records(data_path, spec.beliefs)
    sm = load_steering_model(spec.model_config, config.device)
    print(f"\n{'#' * 70}\n# {spec.key}: {sm.model_id}\n{'#' * 70}", flush=True)

    if config.inject_layer is None:
        config = replace(config, inject_layer=vs.layer)
    resolved_alphas = alphas if alphas is not None else spec.alphas
    resolved_attrs = [a for a in attrs if a in probe_layouts and a in vs.steering_v]
    skipped = [a for a in attrs if a not in resolved_attrs]
    if skipped:
        print(f"WARNING: skipping attrs not in probe_layouts/VectorStore v-space: {skipped}", flush=True)

    out_dir = out_root / Path(spec.model_config).stem / "random_projector_steering"
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = SteeringEvaluator(sm, proj, probe_layouts, config)

    for seed in seeds:
        out_path = out_dir / f"eval_results_seed{seed}.json"
        B_rand = random_orthonormal_B(dim, rank, seed, b_device, b_dtype)
        with torch.no_grad():
            proj.B.weight.copy_(B_rand)

        ortho_err = (B_rand.T.float() @ B_rand.float() - torch.eye(rank, device=b_device)).abs().max().item()
        print(f"\n{'=' * 60}\n{spec.key}  seed={seed}  inject_layer={config.inject_layer}"
              f"  ||B_rand^T B_rand - I||_max={ortho_err:.2e}", flush=True)

        meta = {
            "control": "random_orthonormal_projector_B", "spaces": ["v"], "seed": seed,
            "pairwise": pairwise, "model": sm.model_id, "model_key": spec.key,
            "vectors": spec.vectors, "projector": spec.projector, "beliefs": spec.beliefs,
        }
        run_sweep(evaluator, vs, test_records, resolved_attrs, resolved_alphas, ["v"], out_path, meta,
                  pairwise=pairwise, relative=relative)
        print(f"Done {spec.key} seed {seed} -> {out_path}", flush=True)

    del sm
    gc.collect()
    torch.cuda.empty_cache()
