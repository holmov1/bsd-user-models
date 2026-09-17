"""Shared model-architecture utilities."""

from __future__ import annotations

import yaml
import torch.nn as nn


def load_model_yaml(path: str) -> dict:
    """Load a model config YAML and return it as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_injection_layer(model: nn.Module, layers_path: str, hidden_state_idx: int) -> nn.Module:
    """Return model.layers[hidden_state_idx - 1].

    hidden_states[L] = output of model.layers[L-1].
    All callers pass a hidden_states index; the -1 lives here and nowhere else.
    """
    obj = model
    for attr in layers_path.split("."):
        obj = getattr(obj, attr)
    return obj[hidden_state_idx - 1]
