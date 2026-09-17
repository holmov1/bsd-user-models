"""implementation of the ABCD-letter readout.
"""

import torch
import torch.nn.functional as F


def build_space_letter_ids(tok, letters: list[str], device: str) -> torch.Tensor:
    """Space-prefixed letter ids for the given letters."""
    ids = []
    for letter in letters:
        t = tok(" " + letter, add_special_tokens=False)["input_ids"]
        ids.append(t[-1])
    return torch.tensor(ids, dtype=torch.long, device=device)


def resolve_letter(
    model, text: str, tok, letter_ids: torch.Tensor, space_letter_ids: torch.Tensor,
    n_cats: int, device: str, max_steps: int = 5,
) -> dict:
    """Advance until the argmax token is a letter.
    """
    letters = [chr(ord("A") + i) for i in range(n_cats)]
    bare_ids = letter_ids[:n_cats].tolist()
    space_ids = space_letter_ids[:n_cats].tolist()
    id_to_letter = {}
    for i, letter in enumerate(letters):
        id_to_letter[bare_ids[i]] = letter
        id_to_letter[space_ids[i]] = letter
    letter_id_set = set(bare_ids) | set(space_ids)

    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]

    for step in range(max_steps + 1):
        with torch.no_grad():
            out = model(input_ids=input_ids)
        probs = F.softmax(out.logits[0, -1, :].float(), dim=0)
        top_id = int(torch.argmax(probs))

        if top_id in letter_id_set:
            bare_p = probs[letter_ids[:n_cats]]
            space_p = probs[space_letter_ids[:n_cats]]
            use_bare = top_id in bare_ids
            chosen = bare_p if use_bare else space_p
            distribution = (chosen / chosen.sum().clamp(min=1e-9)).tolist()
            return {
                "resolved": True,
                "letter": id_to_letter[top_id],
                "distribution": distribution,
                "confidence": float(probs[top_id]),
                "steps": step,
            }

        if step == max_steps:
            return {
                "resolved": False, "letter": None, "distribution": None,
                "confidence": 0.0, "steps": step,
            }

        input_ids = torch.cat([input_ids, torch.tensor([[top_id]], device=device)], dim=1)

    raise RuntimeError("unreachable")
