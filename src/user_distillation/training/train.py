import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from user_distillation.data.dataset import WildUserDataset
from user_distillation.training.trainer import DEFAULT_MIN_LAYOUTS_RESOLVED, DistillationTrainer
from user_distillation.training.utils import dtype_from_name, print_eval_report, resolve_h_layer_idx
from user_distillation.utils.reporting import RunLogger


def _resolve_eligible_classes_path(cfg: dict, h_path: str) -> str | None:
    if "eligible_classes_path" in cfg:
        return cfg["eligible_classes_path"]
    default = Path(h_path).parent / "eligible_classes.json"
    return str(default) if default.exists() else None


def _load_wilduser_dataset(
    records_paths: list[str],
    h_path: str,
    beliefs_path: str,
    layer_idx: int = 0,
    limit: int | None = None,
) -> WildUserDataset:
    return WildUserDataset(
        records_paths=records_paths,
        h_path=h_path,
        beliefs_path=beliefs_path,
        layer_idx=layer_idx,
        limit=limit,
    )


def _load_model_config(cfg: dict) -> dict:
    """Merge model-architecture config into cfg and return the merged dict."""
    model_cfg_path = cfg.get("model_config")
    if model_cfg_path is None:
        return cfg
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)
    # model_config keys are defaults; top-level cfg keys override them
    merged = {**model_cfg, **cfg}
    merged["model"] = model_cfg["model_id"]
    return merged


def _load_model(cfg: dict):
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=dtype_from_name(cfg.get("dtype", "bfloat16"))
    ).to(cfg["device"])
    model.eval()
    return tok, model


def _resolve_data_paths(cfg: dict) -> tuple[str, str, int]:
    h_path = cfg["h_teacher_path"]
    beliefs_path = str(Path(h_path).parent / cfg.get("beliefs_filename", "beliefs.jsonl"))
    layer_idx = resolve_h_layer_idx(h_path, cfg["teacher_layer_idx"])
    return h_path, beliefs_path, layer_idx


def _write_metrics_json(path: Path, metrics: dict) -> None:
    # default=... maps NaN to null: x != x is true only for NaN, json has no NaN literal
    path.write_text(json.dumps(metrics, default=lambda x: None if x != x else x, indent=2))


def train(cfg: dict) -> None:
    cfg = _load_model_config(cfg)
    print(
        f"Config: {json.dumps({k: v for k, v in cfg.items() if k != 'model'}, indent=2)}"
    )

    print("\nLoading model...")
    tok, model = _load_model(cfg)
    print(f"  {cfg['model']} on {cfg['device']}")

    print("Loading datasets...")
    h_path, beliefs_path, layer_idx = _resolve_data_paths(cfg)
    print(f"  h_teacher_path={h_path}  beliefs={beliefs_path}  layer_idx={layer_idx}")

    dataset = _load_wilduser_dataset(
        records_paths=cfg["train_records_paths"],
        h_path=h_path,
        beliefs_path=beliefs_path,
        layer_idx=layer_idx,
        limit=cfg.get("limit"),
    )
    print(f"  train: {len(dataset):,} samples")

    val_dataset = None
    if cfg.get("val_records_paths"):
        val_dataset = _load_wilduser_dataset(
            records_paths=cfg["val_records_paths"],
            h_path=h_path,
            beliefs_path=beliefs_path,
            layer_idx=layer_idx,
        )
        print(f"  val:   {len(val_dataset):,} samples")

    test_dataset = None
    if cfg.get("test_records_paths"):
        test_dataset = _load_wilduser_dataset(
            records_paths=cfg["test_records_paths"],
            h_path=h_path,
            beliefs_path=beliefs_path,
            layer_idx=layer_idx,
        )
        print(f"  test:  {len(test_dataset):,} samples")

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    inject_layer_idx = cfg["inject_layer_idx"]

    trainer = DistillationTrainer(
        model=model,
        tokenizer=tok,
        dataset=dataset,
        probe_layouts_path=cfg["probe_layouts_path"],
        lr=cfg["lr"],
        rank=cfg["rank"],
        weight_decay=cfg["weight_decay"],
        device=cfg["device"],
        save_dir=cfg["save_dir"],
        save_every=cfg["save_every"],
        inject_layer_idx=inject_layer_idx,
        layers_path=cfg.get("layers_path", "model.layers"),
        val_dataset=val_dataset,
        eval_every=cfg.get("eval_every", 400),
        train_batch_size=cfg.get("train_batch_size", 1),
        probe_chunk_size=cfg.get("probe_chunk_size", 10),
        min_layouts_resolved=cfg.get("min_layouts_resolved", DEFAULT_MIN_LAYOUTS_RESOLVED),
        eligible_classes_path=_resolve_eligible_classes_path(cfg, h_path),
        chat_template_kwargs=cfg.get("chat_template_kwargs"),
        isometric=cfg.get("isometric_bottleneck", False),
    )

    logger = RunLogger(incremental_path=save_dir / "results.json")
    print(f"\nTraining {cfg['epochs']} epochs...")
    trainer.train(cfg["epochs"], logger=logger)

    if test_dataset is not None:
        print("\nRunning test eval...")
        metrics = trainer.eval_metrics(test_dataset)
        logger.log_test(metrics["_overall"])
        print_eval_report(metrics, run_name="test")
        full_path = save_dir / "test_metrics_full.json"
        _write_metrics_json(full_path, metrics)
        print(f"Saved full test report → {full_path}")

    logger.save_json(save_dir / "results.json")
    logger.save_report(save_dir / "report.txt")
    print(f"\nSaved results → {save_dir}/results.json")
    print(f"Saved report  → {save_dir}/report.txt")


def evaluate(cfg: dict, splits: list[str] | None = None) -> None:
    """Load a saved projector and run eval_metrics on val + test datasets."""
    cfg = _load_model_config(cfg)
    save_dir = Path(cfg["save_dir"])
    checkpoint = save_dir / "w_user.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint}")

    print(f"Loading model ({cfg['model']}) on {cfg['device']} ...")
    tok, model = _load_model(cfg)

    h_path, beliefs_path, layer_idx = _resolve_data_paths(cfg)
    inject_layer_idx = cfg["inject_layer_idx"]

    trainer = DistillationTrainer(
        model=model,
        tokenizer=tok,
        dataset=None,
        probe_layouts_path=cfg["probe_layouts_path"],
        lr=cfg["lr"],
        rank=cfg["rank"],
        weight_decay=cfg["weight_decay"],
        device=cfg["device"],
        save_dir=str(save_dir),
        inject_layer_idx=inject_layer_idx,
        layers_path=cfg.get("layers_path", "model.layers"),
        min_layouts_resolved=cfg.get("min_layouts_resolved", DEFAULT_MIN_LAYOUTS_RESOLVED),
        eligible_classes_path=_resolve_eligible_classes_path(cfg, h_path),
        chat_template_kwargs=cfg.get("chat_template_kwargs"),
    )
    trainer.student.projector.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    trainer.student.projector.to(cfg["device"])
    print(f"Loaded projector from {checkpoint}")

    all_splits = [("val", "val_records_paths"), ("test", "test_records_paths")]
    if splits is not None:
        all_splits = [(s, k) for s, k in all_splits if s in splits]

    results: dict[str, dict] = {}
    for split, paths_key in all_splits:
        if not cfg.get(paths_key):
            continue
        dataset = _load_wilduser_dataset(
            records_paths=cfg[paths_key],
            h_path=h_path,
            beliefs_path=beliefs_path,
            layer_idx=layer_idx,
        )
        print(f"\nEval: {split} ({len(dataset):,} samples) ...")
        metrics = trainer.eval_metrics(dataset)
        print_eval_report(metrics, run_name=split)
        results[split] = metrics
        out = save_dir / f"{split}_metrics.json"
        _write_metrics_json(out, metrics)
        print(f"Saved → {out}")

    return results
