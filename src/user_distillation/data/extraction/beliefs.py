"""Belief readout: multi-step ABCD letter resolution over probe layouts."""

import torch
import torch.nn.functional as F

from user_distillation.data.attr_schema import PROBE_SYSTEM

# maximum generation steps to read out a letter; if exceeded, the layout is unresolved
DEFAULT_MAX_READOUT_STEPS = 6


def _build_flat_probe_texts(
    messages: list[dict],
    layouts: dict,
    tokenizer,
    system_turn: list[dict],
    chat_template_kwargs: dict | None = None,
) -> list[tuple[str, int, list[int], str]]:
    """Build (attr, n_cats, perm, rendered_text) for every layout across all attributes."""
    chat_template_kwargs = dict(chat_template_kwargs or {})
    if not chat_template_kwargs.pop("system_role_supported", True):
        system_turn = []

    flat: list[tuple[str, int, list[int], str]] = []
    for attr, info in layouts.items():
        K = info["n_cats"]
        for lay in info["layouts"]:
            probe_text = lay["probe_text"]
            if not system_turn:
                probe_text = PROBE_SYSTEM + "\n\n" + probe_text
            probe_messages = (
                system_turn + messages + [{"role": "user", "content": probe_text}]
            )
            text = tokenizer.apply_chat_template(
                probe_messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            flat.append((attr, K, lay["perm"], text))
    return flat


def _pad_batch(
    token_lists: list[list[int]], pad_id: int, device: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of token-id lists into one (batch_ids, attn) pair."""
    maxlen = max(len(t) for t in token_lists)
    batch_ids = torch.full((len(token_lists), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(token_lists), maxlen), dtype=torch.long)
    for bi, t in enumerate(token_lists):
        L = len(t)
        batch_ids[bi, :L] = torch.tensor(t)
        attn[bi, :L] = 1
    if device is not None:
        batch_ids, attn = batch_ids.to(device), attn.to(device)
    return batch_ids, attn


def _try_resolve(
    top_id: int,
    logits: torch.Tensor,
    K: int,
    perm: list[int],
    bare_ids_k: torch.Tensor,
    space_ids_k: torch.Tensor,
    letter_set: set[int],
) -> dict | None:
    """If top_id (logits' argmax) is a letter token, return the resolved {"canon", ...}
    dict (un-permuted into canonical order); otherwise None.

    Takes raw logits rather than full-vocab-softmaxed probs: softmax(logits)[S] /
    sum(softmax(logits)[S]) == softmax(logits[S]) exactly (the full-vocab normalizer
    cancels), so slicing before softmax gives the identical canon distribution without
    computing a softmax over the whole vocabulary.
    """
    if top_id not in letter_set:
        return None
    chosen_ids = bare_ids_k if top_id in bare_ids_k.tolist() else space_ids_k
    p = F.softmax(logits[chosen_ids].float(), dim=-1).tolist()
    canon = [0.0] * K
    for pos, pr in zip(perm, p):
        canon[pos] = pr
    return {"resolved": True, "canon": canon}


def resolve_batch_multiround(
    model,
    tok,
    flat: list[tuple[str, int, list[int], str]],
    letter_ids: torch.Tensor,
    space_letter_ids: torch.Tensor,
    device: str,
    batch_size: int,
    max_steps: int,
) -> list[dict]:
    """Advance each probe token-by-token (up to max_steps) until the model's own
    argmax is a letter token. Returns a list aligned with flat:
    {"resolved": bool, "canon": list[float]|None, "steps": int}.
    """
    n = len(flat)
    results: list[dict | None] = [None] * n
    pad_id = tok.pad_token_id

    for chunk_start in range(0, n, batch_size):
        chunk_idx = list(range(chunk_start, min(chunk_start + batch_size, n)))
        id_lists = {
            i: tok(flat[i][3], add_special_tokens=False)["input_ids"] for i in chunk_idx
        }
        # letter-id sets depend only on K
        letter_sets = {}
        for i in chunk_idx:
            K = flat[i][1]
            bare_ids_k, space_ids_k = letter_ids[:K], space_letter_ids[:K]
            letter_sets[i] = (
                bare_ids_k,
                space_ids_k,
                set(bare_ids_k.tolist()) | set(space_ids_k.tolist()),
            )

        pending = list(chunk_idx)
        step = 0
        while pending:
            batch_ids, attn = _pad_batch([id_lists[i] for i in pending], pad_id, device)

            with torch.no_grad():
                logits = model(input_ids=batch_ids, attention_mask=attn).logits
            last = attn.sum(dim=1) - 1
            rows = logits[torch.arange(len(pending)), last, :]

            still_pending = []
            for bi, i in enumerate(pending):
                _attr, K, perm, _text = flat[i]
                bare_ids_k, space_ids_k, letter_set = letter_sets[i]
                row = rows[bi]
                top_id = int(torch.argmax(row))
                resolved = _try_resolve(
                    top_id, row, K, perm, bare_ids_k, space_ids_k, letter_set
                )
                if resolved is not None:
                    resolved["steps"] = step
                    results[i] = resolved
                elif step == max_steps:
                    results[i] = {"resolved": False, "canon": None, "steps": step}
                else:
                    id_lists[i].append(top_id)
                    still_pending.append(i)
            pending = still_pending
            step += 1

    return results


def label_sample(
    model,
    tok,
    messages: list[dict],
    layouts: dict,
    letter_ids,
    space_letter_ids,
    device: str,
    batch_size: int,
    max_steps: int,
    chat_template_kwargs: dict,
    probe_system: str | None = None,
) -> tuple[dict, dict, dict]:
    """Returns (beliefs, probes, n_layouts_resolved), all keyed by attribute.

    beliefs: {attr: [canonical probs]} — mean over resolved layouts; key omitted if
             none resolved. probes: {attr: [canon_or_None, ...]} per layout.
    """
    system_turn = [{"role": "system", "content": probe_system or PROBE_SYSTEM}]
    flat = _build_flat_probe_texts(
        messages, layouts, tok, system_turn, chat_template_kwargs
    )
    resolved = resolve_batch_multiround(
        model,
        tok,
        flat,
        letter_ids,
        space_letter_ids,
        device,
        batch_size,
        max_steps,
    )

    by_attr_canons: dict[str, list] = {a: [] for a in layouts}
    by_attr_probes: dict[str, list] = {a: [] for a in layouts}
    for (attr, _K, _perm, _text), r in zip(flat, resolved):
        by_attr_probes[attr].append(r["canon"])
        if r["resolved"]:
            by_attr_canons[attr].append(r["canon"])

    beliefs: dict[str, list] = {}
    n_layouts_resolved: dict[str, int] = {}
    for attr, canons in by_attr_canons.items():
        n_layouts_resolved[attr] = len(canons)
        if not canons:
            continue  # every layout unresolved -> omit this attribute entirely
        K = len(canons[0])
        mean = [sum(c[k] for c in canons) / len(canons) for k in range(K)]
        s = sum(mean)
        beliefs[attr] = [v / s for v in mean] if s > 0 else mean

    return beliefs, by_attr_probes, n_layouts_resolved
