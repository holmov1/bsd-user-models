"""Checkpoint file I/O for DistillationTrainer: save/discover/load raw checkpoint dicts."""
from pathlib import Path

import torch


def save_projector(path: Path, projector_state: dict) -> None:
    torch.save(projector_state, path)


def save_checkpoint(
    save_dir: Path, projector_state: dict, optimizer_state: dict,
    epoch: int, global_step: int, best_val_f1: float,
) -> Path:
    ckpt = {
        "projector":   projector_state,
        "optimizer":   optimizer_state,
        "epoch":       epoch,        # last completed epoch (0-indexed)
        "global_step": global_step,
        "best_val_f1": best_val_f1,
    }
    path = Path(save_dir) / f"checkpoint_epoch{epoch + 1}.pt"
    torch.save(ckpt, path)
    return path


def latest_checkpoint(save_dir: Path) -> Path | None:
    ckpts = sorted(Path(save_dir).glob("checkpoint_epoch*.pt"))
    return ckpts[-1] if ckpts else None


def load_checkpoint(path: Path, device: str) -> dict:
    """Load a checkpoint dict with optimizer-state tensors already moved to `device`."""
    ckpt = torch.load(path, map_location="cpu")
    for state in ckpt["optimizer"]["state"].values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return ckpt
