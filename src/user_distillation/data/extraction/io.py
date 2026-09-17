"""Record/error/checkpoint I/O helpers shared across extraction modes."""

import json
from pathlib import Path

import numpy as np
import torch

from user_distillation.data.dataset import parse_context_to_messages


def get_messages(sample: dict) -> list[dict]:
    if sample.get("messages"):
        return sample["messages"]
    return parse_context_to_messages(sample["context"])


def user_text_length(messages: list[dict]) -> int:
    """Total character count across all user turns."""
    return sum(len(m["content"]) for m in messages if m["role"] == "user")


def is_oom(exc: BaseException) -> bool:
    return (
        isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
    )


def is_role_alternation_error(exc: BaseException) -> bool:
    """Some chat templates (e.g. Gemma's) hard-require strict user/assistant
    alternation and reject a record with consecutive same-role turns; skip
    only for models that don't tolerate it."""
    return "must alternate" in str(exc).lower()


def write_h_meta(
    path: Path,
    model: str,
    train: str,
    layers: list[int],
    hidden: int,
    threshold: float,
    created: str,
    h_rows: list[int],
    conv_idxs: list[int],
    fps: list[str],
    status: str,
) -> None:
    """Write h_teacher_meta.json — provenance + pointer index for h_teacher.npy."""
    with open(path, "w") as f:
        json.dump(
            {
                "model": model,
                "train": train,
                "layers": layers,
                "dtype": "float16",
                "hidden": hidden,
                "threshold": threshold,
                "created": created,
                "shape": [len(h_rows), len(layers), hidden],
                "status": status,
                "pointers": {
                    "h_row": h_rows,
                    "conv_idx": conv_idxs,
                    "fingerprint": fps,
                },
            },
            f,
        )


def load_h_meta(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def count_lines(path: Path) -> int:
    """Count newline-terminated lines without loading the file into memory."""
    if not path.exists():
        return 0
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def load_records_by_fingerprint(records_path: str) -> dict[str, dict]:
    by_fp: dict[str, dict] = {}
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            fp = r.get("fingerprint")
            if fp:
                by_fp[fp] = r
    return by_fp


def make_belief_record(
    idx,
    fingerprint: str,
    h_row: int,
    beliefs: dict,
    per_probe: dict,
    n_layouts_resolved: dict,
) -> dict:
    """Build one beliefs.jsonl row — shared by all extraction modes."""
    return {
        "idx": idx,
        "fingerprint": fingerprint,
        "h_row": h_row,
        "beliefs": beliefs,
        "probes": per_probe,
        "n_layouts_resolved": n_layouts_resolved,
    }


def _open_h_mm(h_path: Path, n_total: int, n_layers: int, hidden: int) -> np.memmap:
    """Allocate a fresh float16 memmap for h_teacher storage."""
    return np.lib.format.open_memmap(
        h_path, mode="w+", dtype=np.float16, shape=(n_total, n_layers, hidden)
    )


def _find_raw_skip(records_path: Path, last_fingerprint: str) -> int:
    """Row index + 1 of the last-written record in records.jsonl, by fingerprint.

    Not the same as len(beliefs.jsonl): OOM/role-alternation skips consume a raw
    line without writing output, so a plain line count under-counts consumption.
    """
    with open(records_path) as f:
        for i, line in enumerate(f):
            if json.loads(line).get("fingerprint") == last_fingerprint:
                return i + 1
    raise ValueError(f"fingerprint {last_fingerprint!r} not found in {records_path}")


def _load_resume_state(
    records_path: Path, beliefs_path: Path, meta_path: Path
) -> tuple[int, int, list[int], list[int], list[str]]:
    """Return (n_written, raw_skip, ptr_h_rows, ptr_conv_idxs, ptr_fps) for resuming.
    raw_skip is found via _find_raw_skip rather than assumed equal to n_written."""
    n_written = count_lines(beliefs_path)
    ptr_h_rows: list[int] = []
    ptr_conv_idxs: list[int] = []
    ptr_fps: list[str] = []
    raw_skip = 0
    if n_written > 0:
        with open(beliefs_path) as f:
            last_line = None
            for last_line in f:
                pass
        last_fp = json.loads(last_line)["fingerprint"]
        raw_skip = _find_raw_skip(records_path, last_fp)
        if meta_path.exists():
            with open(meta_path) as f:
                prior = json.load(f)
            ptr_h_rows = prior["pointers"]["h_row"]
            ptr_conv_idxs = prior["pointers"]["conv_idx"]
            ptr_fps = prior["pointers"]["fingerprint"]
    return n_written, raw_skip, ptr_h_rows, ptr_conv_idxs, ptr_fps
