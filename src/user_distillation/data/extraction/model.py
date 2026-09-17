"""Model loading and per-model chat-template config."""

from dataclasses import dataclass, field

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, device: str):
    """Load tokenizer + model. Returns (tok, model, hidden_size)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # last_idx gather assumes right padding
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    return tok, model, model.config.hidden_size


@dataclass
class ModelExtractionConfig:
    chat_template_kwargs: dict = field(default_factory=dict)
    strip_leading_assistant: bool = False
    probe_system: str | None = None


def load_model_extraction_config(path: str | None) -> ModelExtractionConfig:
    """Read chat_template_kwargs / strip_leading_assistant / probe_system from a model YAML."""
    if not path:
        return ModelExtractionConfig()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return ModelExtractionConfig(
        chat_template_kwargs=cfg.get("chat_template_kwargs") or {},
        strip_leading_assistant=bool(cfg.get("strip_leading_assistant")),
        probe_system=cfg.get("probe_system"),
    )


def build_letter_ids(tokenizer, letters: list[str], device: str) -> torch.Tensor:
    """Answer-letter token ids (leading-space variant, like the trainer)."""
    ids = []
    for letter in letters:
        tid = None
        for prefix in ("", " "):
            t = tokenizer(prefix + letter, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                tid = t[0]
                break
        if tid is None:
            tid = tokenizer(letter, add_special_tokens=False)["input_ids"][0]
        ids.append(tid)
    return torch.tensor(ids, dtype=torch.long, device=device)
