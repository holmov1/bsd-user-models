import json
from pathlib import Path

import numpy as np
import torch

from user_distillation.data.attr_schema import ABCD_LETTERS, PROBE_SYSTEM
from user_distillation.data.extraction import build_letter_ids

HEADLINE_ATTRS = ("UserIntent", "UserExpertise", "Gender")

_LEGACY_SENTINEL = float("inf") 

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def dtype_from_name(name: str) -> torch.dtype:
    return _DTYPES[name]


def resolve_h_layer_idx(h_path: str, teacher_layer_idx: int) -> int:
    """Return the array index of teacher_layer_idx in the saved h_teacher.npy layers."""
    meta_path = Path(h_path).parent / "h_teacher_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["layers"].index(teacher_layer_idx)


def usable_indices(
    beliefs_list: list[dict], n_resolved_list: list[dict], attr: str, min_layouts_resolved: int,
) -> list[int]:
    """Sample indices usable for `attr`: belief present, averaged from enough layouts.
    """
    return [
        i for i, (b, nr) in enumerate(zip(beliefs_list, n_resolved_list))
        if attr in b and nr.get(attr, _LEGACY_SENTINEL) >= min_layouts_resolved
    ]


def build_abcd_ids(tokenizer, n: int, device: str) -> torch.Tensor:
    return build_letter_ids(tokenizer, ABCD_LETTERS[:n], device)


def build_probe_tensors(
    tokenizer, probe_layouts_path: str, chat_template_kwargs: dict | None, device: str,
) -> tuple[list[tuple[int, str, list[int]]], torch.Tensor, torch.Tensor]:
    """Pre-tokenize every probe layout, fixed across all training/eval samples.
    """
    with open(probe_layouts_path) as f:
        layouts = json.load(f)
    probe_meta: list[tuple[int, str, list[int]]] = []
    probe_texts: list[str] = []
    ct_kwargs = chat_template_kwargs or {}
    for attr, info in layouts.items():
        for lay in info["layouts"]:
            probe_texts.append(tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": PROBE_SYSTEM},
                    {"role": "user",   "content": lay["probe_text"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **ct_kwargs,
            ))
            probe_meta.append((info["n_cats"], attr, lay["perm"]))
    enc = tokenizer(probe_texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return probe_meta, enc["input_ids"].to(device), enc["attention_mask"].to(device)


def unpermute(canon, probs, perm: list[int]):
    """canon[:, perm[pos]] = probs[:, pos] for each pos; canon/probs are a tensor or array pair."""
    for pos in range(len(perm)):
        canon[:, perm[pos]] = probs[:, pos]
    return canon


def eligible_labels(eligible_classes: dict | None, attr: str, t_labels) -> list[int]:
    """Class labels to average metrics over: eligible ∩ observed (else observed-only)."""
    observed = sorted(set(t_labels.tolist()))
    if eligible_classes is None or attr not in eligible_classes:
        return observed
    eligible = sorted(int(idx) for idx, v in eligible_classes[attr].items() if v["eligible"])
    restricted = sorted(set(eligible) & set(observed))
    return restricted if restricted else observed


def print_eval_report(
    metrics: dict,
    headline_attrs: tuple[str, ...] = HEADLINE_ATTRS,
    run_name: str = "",
) -> None:
    """Pretty-print the output of DistillationTrainer.eval_metrics()."""
    sep = "=" * 62
    thin = "─" * 62
    header = f"  Run: {run_name}" if run_name else ""
    print(f"\n{sep}{header}")
    ov = metrics["_overall"]
    print(f"Macro-F1 (avg):      {ov['macro_f1']:.3f}")
    print(f"Balanced Acc (avg):  {ov['balanced_acc']:.3f}")
    print(f"Macro AUC (avg):     {ov['macro_auc']:.3f}")

    print(f"\n{thin}")
    print(f"{'Attribute':<26} {'Macro-F1':>9} {'Bal-Acc':>9} {'AUC':>9}")
    for attr, r in metrics.items():
        if attr.startswith("_"):
            continue
        auc = f"{r['macro_auc']:9.3f}" if not np.isnan(r["macro_auc"]) else "      nan"
        print(f"  {attr:<24} {r['macro_f1']:9.3f} {r['balanced_acc']:9.3f} {auc}")

    for attr in headline_attrs:
        if attr not in metrics:
            continue
        r = metrics[attr]
        print(f"\n{thin}")
        print(f"Headline: {attr}   Macro-F1={r['macro_f1']:.3f}  Bal-Acc={r['balanced_acc']:.3f}  AUC={r['macro_auc']:.3f}")
        print(f"  {'Class':<24} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        for cls in r["class_names"]:
            if cls in r["report"]:
                row = r["report"][cls]
                print(
                    f"  {cls:<24} {row['precision']:10.3f} {row['recall']:8.3f}"
                    f" {row['f1-score']:8.3f} {int(row['support']):9}"
                )
        cm = r["confusion"]
        names_short = [n[:9] for n in r["class_names"]]
        print(f"\n  Confusion ({attr}):")
        print("  " + "".join(f"{n:>10}" for n in names_short))
        for i, row in enumerate(cm):
            print(f"  {names_short[i]:<10}" + "".join(f"{v:>10}" for v in row))
    print(sep)
