"""Steering configuration: which attributes/classes to steer, and shared constants."""

import json
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor

STEERING_ATTRS: dict[str, list[str]] = {
    "Gender":                  ["male", "female"],
    "Continent":                ["europe", "asia", "africa", "north_america", "south_america"],
    "EducationLevel":           ["school", "university"],
    "PoliticalOrientation":     ["apolitical", "green", "left", "right"],
    "IncomeLevel":              ["low", "middle", "high"],
    "AITrustLevel":             ["skeptical", "trusting"],
    "AIErrorTolerance":         ["low", "high"],
    "AIInteractionStyle":       ["transactional", "conversational", "hostile"],
    "UserIntent":               ["benign", "adversarial"],
    "UserReasoningComplexity":  ["simple", "moderate"],
    "UserTruthSeekingIntent":   ["truth_seeking", "confirmation_seeking", "persuasion_seeking"],
    "PerceivedEmotionalState":  ["neutral", "stressed", "sad", "excited"],
    "EvidencePreference":       ["anecdotal", "intuitive", "empirical", "theoretical"],
}

SEED              = 42
INJECT_LAYER      = 16   # Llama default
PROBE_LAYOUTS_PATH = "data/wilduser/probe_layouts_v2.json"


def load_probe_layouts(path: str | Path) -> dict:
    """Load probe_layouts.json."""
    with open(path) as f:
        return json.load(f)


@dataclass
class SteeringModel:
    """A loaded HF model/tokenizer plus its per-model chat-template config."""
    model:                object
    tok:                  object
    model_id:             str
    layers_path:          str
    probe_system:         str | None
    chat_template_kwargs: dict
    strip_leading_asst:   bool
    abcd_ids:             Tensor


@dataclass
class SteeringConfig:
    """Tunable knobs for a SteeringEvaluator run."""
    inject_layer:         int | None = None
    device:               str = "cuda:0"
    all_positions:        bool = True
    batch_size:           int = 8
    n_per_class:          int = 100
    seed:                 int = SEED
    min_src_prob:         float = 0.0
    min_src_margin:       float = 0.05
    baseline_oversample:  int = 5
