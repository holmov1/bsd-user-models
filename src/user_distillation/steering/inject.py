"""Activation injection for CAA (Contrastive Activation Addition)."""

from __future__ import annotations

import torch
from torch import Tensor

from user_distillation.model_utils import get_injection_layer


def make_inject_hook(delta: Tensor, ctx_lengths: list[int], all_positions: bool = True):
    """Return a forward hook that adds delta to token positions."""
    def hook(_module, _input, output):
        is_tuple = isinstance(output, tuple)
        hidden   = output[0] if is_tuple else output

        B, seq_len, dim = hidden.shape
        if all_positions:
            mask = torch.ones(B, seq_len, 1, device=hidden.device, dtype=hidden.dtype)
        else:
            mask = torch.zeros(B, seq_len, 1, device=hidden.device, dtype=hidden.dtype)
            for i, ctx_len in enumerate(ctx_lengths):
                mask[i, :ctx_len, 0] = 1.0

        d = delta.to(hidden.device).to(hidden.dtype)
        d = d.unsqueeze(0).unsqueeze(0) if d.dim() == 1 else d.unsqueeze(1)
        new_hidden = hidden + d * mask
        return (new_hidden,) + output[1:] if is_tuple else new_hidden

    return hook


def steer_forward_batched(
    model:            "torch.nn.Module",
    input_ids:        Tensor,
    attn_mask:        Tensor,
    hidden_state_idx: int,
    delta:            Tensor | None,
    ctx_lengths:      list[int],
    all_positions:    bool = True,
    layers_path:      str = "model.layers",
) -> Tensor:
    """Run one forward pass with steering injection."""
    if delta is not None:
        hook   = make_inject_hook(delta, ctx_lengths, all_positions=all_positions)
        handle = get_injection_layer(model, layers_path, hidden_state_idx).register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
        finally:
            handle.remove()
    else:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)

    return out.logits
