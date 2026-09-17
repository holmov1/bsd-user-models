import json
from dataclasses import dataclass, field

import numpy as np
import torch


def parse_context_to_messages(context: str) -> list[dict]:
    """Reconstruct chat turns from the 'User: ... / Assistant: ...' flat string."""
    msgs: list[dict] = []
    role, buf = None, []
    for line in context.split("\n"):
        if line.startswith("User: "):
            if role is not None:
                msgs.append({"role": role, "content": "\n".join(buf).strip()})
            role, buf = "user", [line[len("User: ") :]]
        elif line.startswith("Assistant: "):
            if role is not None:
                msgs.append({"role": role, "content": "\n".join(buf).strip()})
            role, buf = "assistant", [line[len("Assistant: ") :]]
        else:
            buf.append(line)
    if role is not None:
        msgs.append({"role": role, "content": "\n".join(buf).strip()})
    return msgs


def strip_leading_assistant(messages: list[dict]) -> list[dict]:
    """Drop leading assistant turn(s)"""
    while messages and messages[0]["role"] == "assistant":
        messages = messages[1:]
    return messages


@dataclass
class WildUserSample:
    messages: list[dict]
    h_teacher: torch.Tensor
    beliefs: dict[str, list[float]]
    n_layouts_resolved: dict[str, int] = field(default_factory=dict)


class WildUserDataset(torch.utils.data.Dataset):
    """Training dataset: joins a split against a model-specific beliefs.jsonl by fingerprint."""

    def __init__(
        self,
        records_paths: list[str],
        h_path: str,
        beliefs_path: str,
        layer_idx: int = 0,
        limit: int | None = None,
    ):
        beliefs_idx = self._load_beliefs_index(beliefs_path)
        h_mm = np.lib.format.open_memmap(h_path, mode="r")
        self._memmaps = [h_mm]
        self._layer_idx = layer_idx
        self._items = self._build_items(records_paths, h_mm, beliefs_idx, limit)
        # item = (h_mm, h_row, messages, beliefs, n_layouts_resolved)

    def _build_items(
        self,
        records_paths: list[str],
        h_mm: np.memmap,
        beliefs_idx: dict[str, dict],
        limit: int | None = None,
    ) -> list[tuple]:
        items: list[tuple] = []
        for rec_path in records_paths:
            with open(rec_path) as f:
                for line in f:
                    rec = json.loads(line)
                    b = beliefs_idx.get(rec["fingerprint"])
                    if b is None:
                        continue
                    messages = parse_context_to_messages(rec["context"])
                    items.append(
                        (
                            h_mm,
                            b["h_row"],
                            messages,
                            b["beliefs"],
                            b.get("n_layouts_resolved", {}),
                        )
                    )
                    if limit and len(items) >= limit:
                        break
            if limit and len(items) >= limit:
                break
        return items

    def _load_beliefs_index(self, beliefs_path: str) -> dict[str, dict]:
        beliefs_idx: dict[str, dict] = {}
        with open(beliefs_path) as f:
            for line in f:
                r = json.loads(line)
                beliefs_idx[r["fingerprint"]] = r
        return beliefs_idx

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> WildUserSample:
        h_mm, h_row, messages, beliefs, n_layouts_resolved = self._items[idx]
        # h_mm shape: [N, n_layers, hidden_dim]; layer_idx selects which layer
        h_teacher = torch.from_numpy(h_mm[h_row, self._layer_idx, :].astype(np.float32))
        return WildUserSample(
            messages=messages,
            h_teacher=h_teacher,
            beliefs=beliefs,
            n_layouts_resolved=n_layouts_resolved,
        )
