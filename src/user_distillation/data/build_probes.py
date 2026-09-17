"""Build a fixed set of probe layouts."""

import json
import random

from user_distillation.data.attr_schema import (
    ABCD_LETTERS,
    ABCD_TEMPLATES,
    ATTR_DISPLAY_NAMES,
    ATTRIBUTE_SCHEMA,
    VALUE_DISPLAY_NAMES,
)


def build_layouts(seed: int = 0) -> dict:
    rng = random.Random(seed)
    T = len(ABCD_TEMPLATES)
    out: dict[str, dict] = {}

    for attr, cats in ATTRIBUTE_SCHEMA.items():
        K = len(cats)
        n_layouts = max(K, T)
        layouts = []
        for i in range(n_layouts):
            r = i % K  # cyclic rotation amount
            perm = [(pos + r) % K for pos in range(K)]  # perm[pos] = canonical idx at letter pos
            template_idx = i % T
            attr_display = rng.choice(ATTR_DISPLAY_NAMES[attr])

            options = []
            for pos in range(K):
                cidx = perm[pos]
                cval = cats[cidx]
                syn = rng.choice(VALUE_DISPLAY_NAMES[attr][cval])
                options.append(
                    {
                        "letter": ABCD_LETTERS[pos],
                        "canonical_idx": cidx,
                        "canonical_value": cval,
                        "display": syn,
                    }
                )
            options_str = "\n".join(f"{o['letter']}. {o['display']}" for o in options)
            probe_text = ABCD_TEMPLATES[template_idx].format(attr=attr_display, options=options_str)

            layouts.append(
                {
                    "layout_id": i,
                    "template_idx": template_idx,
                    "attr_display": attr_display,
                    "perm": perm,
                    "options": options,
                    "probe_text": probe_text,
                }
            )

        out[attr] = {
            "canonical_order": cats,
            "n_cats": K,
            "n_templates": T,
            "n_layouts": n_layouts,
            "layouts": layouts,
        }
    return out


def letters_to_canonical(letter_probs, perm, n_cats):
    """Map a letter-indexed prob vector to canonical value order."""
    canon = [0.0] * n_cats
    for pos in range(n_cats):
        canon[perm[pos]] = letter_probs[pos]
    return canon


if __name__ == "__main__":
    layouts = build_layouts(seed=0)
    with open("probe_layouts.json", "w") as f:
        json.dump(layouts, f, indent=2)
    print("\nsaved -> probe_layouts.json")
