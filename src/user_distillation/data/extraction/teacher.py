"""Teacher hidden-state capture.

Layer convention: hidden_states[L] is the output of decoder layer L-1 (index 0 =
embeddings). Pass the same L you read/inject at in the trainer.
"""

import torch


def _encode_context(
    tokenizer,
    messages: list[dict],
    device: str,
    chat_template_kwargs: dict | None = None,
) -> tuple[dict, int]:
    """Apply chat template + tokenize. Returns (enc, last_pos) — last_pos is the
    generation position (last non-padding token)."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    last = int(enc["attention_mask"].sum(dim=1).item()) - 1
    return enc, last


@torch.no_grad()
def teacher_hidden(
    model, tokenizer, messages, layers, device, chat_template_kwargs: dict | None = None
):
    """h_teacher at the generation position for each requested layer -> [n_layers, hidden]."""
    enc, last = _encode_context(tokenizer, messages, device, chat_template_kwargs)

    captures: dict[int, torch.Tensor] = {}
    hooks = []
    for L in layers:

        def _make_hook(l, pos):
            def _hook(_module, _inp, output):
                h = output[0] if isinstance(output, tuple) else output
                captures[l] = h[0, pos, :].float().cpu()

            return _hook

        hooks.append(
            model.model.layers[L - 1].register_forward_hook(_make_hook(L, last))
        )

    try:
        model(**enc)
    finally:
        for h in hooks:
            h.remove()

    return torch.stack([captures[L] for L in layers], dim=0)  # [n_layers, hidden]
