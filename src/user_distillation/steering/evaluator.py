"""CAA steering evaluation via multi-step ABCD probing."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from user_distillation.data.attr_schema import ABCD_LETTERS, PROBE_SYSTEM
from user_distillation.data.dataset import (
    parse_context_to_messages,
    strip_leading_assistant,
)
from user_distillation.data.extraction.beliefs import _pad_batch, _try_resolve
from user_distillation.data.extraction.model import load_model, load_model_extraction_config
from user_distillation.model_utils import load_model_yaml
from user_distillation.steering.config import SEED, STEERING_ATTRS, SteeringConfig, SteeringModel
from user_distillation.steering.inject import steer_forward_batched
from user_distillation.steering.letter_readout import build_space_letter_ids
from user_distillation.steering.vectors import VectorStore
from user_distillation.training.subspace_projector import LowRankUserSubspaceProjector
from user_distillation.training.utils import build_abcd_ids

MAX_READOUT_STEPS = 5


def _build_abcd_ids(tok, device: str) -> Tensor:
    """ABCD-letter token ids for tok."""
    return build_abcd_ids(tok, len(ABCD_LETTERS), device)


def _build_batch(
    contexts: list[str],
    probe_text: str,
    tok,
    device: str,
    chat_template_kwargs: dict | None = None,
    probe_system: str | None = None,
    strip_leading_asst: bool = False,
) -> tuple[Tensor, Tensor, list[int], list[int]]:
    """Tokenize contexts + probe question into one right-padded batch."""
    ct_kwargs = dict(chat_template_kwargs or {})
    system_supported = ct_kwargs.pop("system_role_supported", True)
    sys_text = probe_system or PROBE_SYSTEM
    system_turn = [{"role": "system", "content": sys_text}] if system_supported else []

    full_encs: list[Tensor] = []
    ctx_lens_raw: list[int] = []

    for ctx in contexts:
        messages = parse_context_to_messages(ctx)
        if strip_leading_asst:
            messages = strip_leading_assistant(messages)

        ctx_text = tok.apply_chat_template(
            system_turn + messages,
            tokenize=False,
            add_generation_prompt=False,
            **ct_kwargs,
        )

        if system_supported:
            probe_content = probe_text
        else:
            probe_content = sys_text + "\n\n" + probe_text
        full_messages = system_turn + messages + [{"role": "user", "content": probe_content}]

        full_text = tok.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
            **ct_kwargs,
        )

        ctx_ids = tok(ctx_text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]
        full_ids = tok(full_text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]

        ctx_lens_raw.append(len(ctx_ids))
        full_encs.append(full_ids)

    B = len(full_encs)
    max_len = max(len(e) for e in full_encs)

    input_ids = torch.full(
        (B, max_len), tok.pad_token_id, dtype=torch.long, device=device
    )
    attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    ctx_lens = list(ctx_lens_raw)
    last_pos = [0] * B

    for i, full_ids in enumerate(full_encs):
        L = len(full_ids)
        input_ids[i, :L] = full_ids.to(device)
        attn_mask[i, :L] = 1
        last_pos[i] = L - 1

    return input_ids, attn_mask, ctx_lens, last_pos


def load_steering_model(model_config_path: str, device: str) -> SteeringModel:
    """Load the HF model/tokenizer and per-model chat-template config for steering eval."""
    model_cfg   = load_model_yaml(model_config_path)
    model_id    = model_cfg["model_id"]
    layers_path = model_cfg.get("layers_path", "model.layers")
    extraction_cfg = load_model_extraction_config(model_config_path)

    print(f"Loading model {model_id} ...", flush=True)
    tok, model, _ = load_model(model_id, device)
    abcd_ids = _build_abcd_ids(tok, device)

    return SteeringModel(
        model=model, tok=tok, model_id=model_id, layers_path=layers_path,
        probe_system=extraction_cfg.probe_system,
        chat_template_kwargs=extraction_cfg.chat_template_kwargs,
        strip_leading_asst=extraction_cfg.strip_leading_assistant,
        abcd_ids=abcd_ids,
    )


def load_test_records(data_path: str, beliefs_path: str) -> list[dict]:
    """Join context records with model-specific beliefs by fingerprint."""
    print(f"Loading beliefs from {beliefs_path} ...", flush=True)
    beliefs_idx: dict[str, dict] = {}
    with open(beliefs_path) as f:
        for line in f:
            r = json.loads(line)
            fp = r.get("fingerprint") or str(r.get("idx"))
            beliefs_idx[fp] = r
    print(f"  {len(beliefs_idx):,} belief records", flush=True)

    print(f"Loading context records from {data_path} ...", flush=True)
    records: list[dict] = []
    n_missing = 0
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            fp = r.get("fingerprint") or str(r.get("idx"))
            b = beliefs_idx.get(fp)
            if b is None:
                n_missing += 1
                continue
            records.append({
                "idx":         r.get("idx"),
                "fingerprint": fp,
                "context":     r["context"],
                "beliefs":     b.get("beliefs", {}),
                "h_row":       b.get("h_row"),
            })
    if n_missing:
        print(f"  dropped {n_missing} records not found in beliefs file", flush=True)
    print(f"  {len(records):,} records after join", flush=True)
    return records


class SteeringEvaluator:
    """CAA steering + multi-step ABCD readout evaluation for one loaded model/projector."""

    def __init__(
        self,
        sm: SteeringModel,
        projector: LowRankUserSubspaceProjector,
        probe_layouts: dict,
        config: SteeringConfig,
    ):
        self.model = sm.model
        self.tok = sm.tok
        self.projector = projector
        self.probe_layouts = probe_layouts
        self.abcd_ids = sm.abcd_ids
        self.layers_path = sm.layers_path
        self.chat_template_kwargs = sm.chat_template_kwargs
        self.probe_system = sm.probe_system
        self.strip_leading_asst = sm.strip_leading_asst
        self.inject_layer = config.inject_layer
        self.device = config.device
        self.all_positions = config.all_positions
        self.batch_size = config.batch_size
        self.n_per_class = config.n_per_class
        self.seed = config.seed
        self.min_src_prob = config.min_src_prob
        self.min_src_margin = config.min_src_margin
        self.baseline_oversample = config.baseline_oversample

    def abcd_probs(
        self, contexts: list[str], layout: dict, inject_delta: Tensor | None, pbar: tqdm | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run all contexts through one layout; return (canonical probs, broken mask)."""
        perm = layout["perm"]
        n_cats = len(perm)
        probe_text = layout["probe_text"]
        N = len(contexts)

        bare_ids = self.abcd_ids[:n_cats]
        space_ids = build_space_letter_ids(self.tok, ABCD_LETTERS[:n_cats], self.device)
        letter_id_set = set(bare_ids.tolist()) | set(space_ids.tolist())

        canon_all = np.zeros((N, n_cats), dtype=np.float32)
        broken_all = np.zeros(N, dtype=np.float32)

        for start in range(0, N, self.batch_size):
            batch_ctx = contexts[start : start + self.batch_size]
            B = len(batch_ctx)

            input_ids0, _, ctx_lens, last_pos0 = _build_batch(
                batch_ctx, probe_text, self.tok, self.device,
                self.chat_template_kwargs, self.probe_system, self.strip_leading_asst,
            )
            token_lists = [input_ids0[i, : last_pos0[i] + 1].tolist() for i in range(B)]
            resolved = [False] * B

            for step in range(MAX_READOUT_STEPS + 1):
                batch_input, batch_mask = _pad_batch(token_lists, self.tok.pad_token_id, self.device)
                cur_last_pos = (batch_mask.sum(dim=1) - 1).tolist()

                logits = steer_forward_batched(
                    self.model, batch_input, batch_mask, self.inject_layer, inject_delta,
                    ctx_lens, all_positions=self.all_positions, layers_path=self.layers_path,
                )

                still_going = False
                for i in range(B):
                    if resolved[i]:
                        continue
                    pos_logits = logits[i, cur_last_pos[i], :]
                    top_id = int(pos_logits.argmax())

                    result = _try_resolve(
                        top_id, pos_logits, n_cats, perm, bare_ids, space_ids, letter_id_set
                    )
                    if result is not None:
                        canon_all[start + i] = np.array(result["canon"], dtype=np.float32)
                        resolved[i] = True
                    elif step < MAX_READOUT_STEPS:
                        token_lists[i].append(top_id)
                        still_going = True

                if not still_going:
                    break

            for i in range(B):
                broken_all[start + i] = float(not resolved[i])

            if pbar is not None:
                pbar.update(1)

        return canon_all, broken_all

    def _group_by_src(self, attr: str, viable: list[str], canon_ord: list[str], test_records: list[dict]) -> dict:
        """Bucket test_records by teacher-belief src class, filtered by confidence margin."""
        by_src: dict[str, list[dict]] = defaultdict(list)
        n_filtered = 0
        for r in test_records:
            if attr not in r.get("beliefs", {}):
                continue
            probs = r["beliefs"][attr]
            src_idx = int(np.argmax(probs))
            lbl = canon_ord[src_idx]
            if lbl in viable:
                top2 = np.sort(probs)[::-1][:2]
                margin = float(top2[0] - top2[1]) if len(top2) > 1 else 1.0
                if probs[src_idx] >= self.min_src_prob and margin >= self.min_src_margin:
                    by_src[lbl].append(r)
                else:
                    n_filtered += 1
        if n_filtered:
            print(
                f"  prefiltered {n_filtered} ambiguous samples "
                f"(min_src_margin={self.min_src_margin}, min_src_prob={self.min_src_prob})",
                flush=True,
            )
        return by_src

    def _pbar_total(
        self, viable: list[str], by_src: dict, layouts: list[dict],
        alphas: list[float], spaces: list[str], alpha_plan: dict | None,
    ) -> int:
        """Total abcd_probs batch-units across the whole attribute sweep, for tqdm."""
        n_batches_per_pass = -(-self.n_per_class // self.batch_size)
        alphas_nonzero = [a for a in alphas if a != 0]
        n_src = sum(1 for c in viable if by_src.get(c))
        n_tgt = len(viable) - 1

        if alpha_plan:
            n_steering_combos = 0
            for src_class in viable:
                if not by_src.get(src_class):
                    continue
                for tgt_class in viable:
                    if tgt_class == src_class:
                        continue
                    pair_plan = alpha_plan.get(f"{src_class}→{tgt_class}", {})
                    for space in spaces:
                        space_alphas = pair_plan.get(space, alphas)
                        n_steering_combos += sum(1 for a in space_alphas if a != 0)
            return (
                n_src * len(layouts) * n_batches_per_pass
                + n_steering_combos * len(layouts) * n_batches_per_pass
            )
        return n_src * (
            len(layouts) * n_batches_per_pass
            + n_tgt * len(alphas_nonzero) * len(spaces) * len(layouts) * n_batches_per_pass
        )

    def _screen_baseline(
        self, pbar: tqdm, rng: random.Random, attr: str, src_class: str, pool: list[dict],
        layouts: list[dict], n_cats: int, baseline_cache: dict | None,
    ):
        """Screen an oversampled candidate pool down to n_per_class fully-resolved baseline samples."""
        candidate_take = min(len(pool), max(self.n_per_class * self.baseline_oversample, self.n_per_class))
        cached = baseline_cache.get((attr, src_class)) if baseline_cache is not None else None

        if cached is not None:
            samples, contexts, baseline_probs, baseline_broken, pass_rate = cached
            pbar.total -= len(layouts) * -(-candidate_take // self.batch_size)
            pbar.refresh()
            print(
                f"  {attr} {src_class}: reusing cached baseline screen "
                f"({len(samples)} samples, {100 * pass_rate:.0f}% pass)",
                flush=True,
            )
            return samples, contexts, baseline_probs, baseline_broken, pass_rate, candidate_take

        chunk_size = max(self.n_per_class, self.batch_size)
        pool_order = rng.sample(pool, candidate_take)

        samples, contexts, probs_list = [], [], []
        n_screened = 0
        for chunk_start in range(0, len(pool_order), chunk_size):
            if len(samples) >= self.n_per_class:
                break
            chunk = pool_order[chunk_start : chunk_start + chunk_size]
            chunk_contexts = [r["context"] for r in chunk]

            chunk_probs = np.zeros((len(chunk), n_cats), dtype=np.float32)
            chunk_broken = np.zeros(len(chunk), dtype=np.float32)
            for li, layout in enumerate(layouts):
                pbar.set_postfix_str(
                    f"{src_class} baseline-screen {len(samples)}/{self.n_per_class} found L{li + 1}/{len(layouts)}"
                )
                probs, broken = self.abcd_probs(chunk_contexts, layout, None, pbar=pbar)
                chunk_probs += probs
                chunk_broken += broken
            chunk_probs /= len(layouts)
            chunk_broken /= len(layouts)
            n_screened += len(chunk)

            for i in np.where(chunk_broken == 0)[0]:
                if len(samples) >= self.n_per_class:
                    break
                samples.append(chunk[i])
                contexts.append(chunk_contexts[i])
                probs_list.append(chunk_probs[i])

        take = len(samples)
        pass_rate = take / n_screened if n_screened else 0.0
        status = "stopped early" if take >= self.n_per_class else "exhausted candidate pool"
        print(
            f"  {attr} {src_class}: baseline screen {take}/{n_screened} resolve on all "
            f"{len(layouts)} layouts ({100 * pass_rate:.0f}% pass), {status}",
            flush=True,
        )
        if take == 0:
            print(f"  {attr} {src_class}: nothing survives the screen, skipping", flush=True)
            return None
        if take < self.n_per_class:
            print(
                f"  {attr} {src_class}: only {take} usable "
                f"(requested n_per_class={self.n_per_class}); using all of them",
                flush=True,
            )

        baseline_probs = np.stack(probs_list)
        baseline_broken = np.zeros(take, dtype=np.float32)
        if baseline_cache is not None:
            baseline_cache[(attr, src_class)] = (samples, contexts, baseline_probs, baseline_broken, pass_rate)
        return samples, contexts, baseline_probs, baseline_broken, pass_rate, candidate_take

    def _sweep_pair(
        self, pbar: tqdm, vectors: VectorStore, attr: str, src_class: str, tgt_class: str,
        canon_ord: list[str], layouts: list[dict], n_cats: int, contexts: list[str],
        baseline_probs: np.ndarray, baseline_broken: np.ndarray, take: int, pass_rate: float,
        candidate_take: int, samples: list[dict], alphas: list[float], spaces: list[str],
        pairwise: bool, relative: bool, alpha_plan: dict | None,
    ) -> dict:
        """Sweep every (alpha, space) for one (src_class, tgt_class) pair."""
        tgt_idx = canon_ord.index(tgt_class)
        pair_key = f"{src_class}→{tgt_class}"
        pair_res: dict = {
            "n_samples": take,
            "alphas": {},
            "baseline_pass_rate": round(pass_rate, 4),
            "n_screened": candidate_take,
            "mean_src_prob": round(
                float(np.mean([s["beliefs"][attr][canon_ord.index(src_class)] for s in samples])), 4,
            ),
            "mean_src_margin": round(
                float(np.mean([np.diff(np.sort(s["beliefs"][attr])[::-1][:2])[0] * -1 for s in samples])), 4,
            ),
        }
        pair_plan = alpha_plan.get(pair_key, {}) if alpha_plan else {}

        for space in spaces:
            space_alphas = pair_plan.get(space, alphas)
            for alpha in space_alphas:
                if alpha == 0:
                    after_probs = baseline_probs
                    after_broken = baseline_broken
                else:
                    try:
                        delta = vectors.delta_h(
                            attr, tgt_class, alpha, space, self.projector,
                            src_class=src_class if pairwise else None, relative=relative,
                        )
                        delta = delta.to(self.device)
                    except (KeyError, ValueError) as e:
                        print(f"    skip {pair_key} α={alpha} space={space}: {e}", flush=True)
                        continue

                    after_probs = np.zeros((take, n_cats), dtype=np.float32)
                    after_broken = np.zeros(take, dtype=np.float32)
                    for li, layout in enumerate(layouts):
                        pbar.set_postfix_str(
                            f"{src_class}→{tgt_class} α={alpha} [{space}] L{li + 1}/{len(layouts)}"
                        )
                        probs, broken = self.abcd_probs(contexts, layout, delta, pbar=pbar)
                        after_probs += probs
                        after_broken += broken
                    after_probs /= len(layouts)
                    after_broken /= len(layouts)

                valid = (after_broken < 1.0) & (baseline_broken < 1.0)
                n_valid = int(valid.sum())

                prob_deltas = after_probs[valid, tgt_idx] - baseline_probs[valid, tgt_idx]
                mean_pd = float(prob_deltas.mean()) if n_valid else float("nan")
                flip_rate = (
                    float((np.argmax(after_probs[valid], axis=1) == tgt_idx).mean())
                    if n_valid else float("nan")
                )
                broken_rate = float(after_broken.mean())
                alpha_res = {
                    "mean_prob_delta": round(mean_pd, 6) if n_valid else None,
                    "flip_rate": round(flip_rate, 4) if n_valid else None,
                    "broken_rate": round(broken_rate, 4),
                    "n_valid": n_valid,
                }
                tqdm.write(
                    f"  {pair_key:<35} α={alpha:>5} [{space}]  "
                    f"mean_Δp={mean_pd:+.4f}  flip={flip_rate:.3f}  broken={broken_rate:.3f}"
                )

                pair_res["alphas"].setdefault(str(alpha), {})[space] = alpha_res

                if alpha_res["flip_rate"] == 1.0 and alpha_res["broken_rate"] == 0.0:
                    tqdm.write(f"  {pair_key:<35} [{space}] saturated at α={alpha}, skipping higher alphas")
                    break

        return pair_res

    def evaluate_attr(
        self,
        vectors: VectorStore,
        test_records: list[dict],
        attr: str,
        alphas: list[float],
        spaces: list[str],
        pairwise: bool = True,
        relative: bool = False,
        alpha_plan: dict | None = None,
        baseline_cache: dict | None = None,
    ) -> dict:
        """Evaluate steering for one attribute across all (src→tgt) pairs, alphas, and spaces."""
        rng = random.Random(self.seed)
        viable = STEERING_ATTRS[attr]
        layouts = self.probe_layouts[attr]["layouts"]
        n_cats = self.probe_layouts[attr]["n_cats"]
        canon_ord = self.probe_layouts[attr]["canonical_order"]

        by_src = self._group_by_src(attr, viable, canon_ord, test_records)
        total_batches = self._pbar_total(viable, by_src, layouts, alphas, spaces, alpha_plan)
        pbar = tqdm(total=total_batches, desc=attr, unit="batch", ncols=100, leave=True)

        results: dict = {}
        for src_class in viable:
            pool = by_src.get(src_class, [])
            if not pool:
                print(f"  {attr} {src_class}: no test samples, skipping", flush=True)
                continue

            screened = self._screen_baseline(pbar, rng, attr, src_class, pool, layouts, n_cats, baseline_cache)
            if screened is None:
                continue
            samples, contexts, baseline_probs, baseline_broken, pass_rate, candidate_take = screened
            take = len(samples)

            for tgt_class in viable:
                if tgt_class == src_class:
                    continue
                pair_key = f"{src_class}→{tgt_class}"
                results[pair_key] = self._sweep_pair(
                    pbar, vectors, attr, src_class, tgt_class, canon_ord, layouts, n_cats,
                    contexts, baseline_probs, baseline_broken, take, pass_rate, candidate_take,
                    samples, alphas, spaces, pairwise, relative, alpha_plan,
                )

        pbar.close()
        return results


def run_sweep(
    evaluator: SteeringEvaluator,
    vectors: VectorStore,
    test_records: list[dict],
    attrs: list[str],
    alphas: list[float],
    spaces: list[str],
    out_path: Path,
    meta: dict,
    alpha_plan: dict | None = None,
    pairwise: bool = True,
    relative: bool = False,
) -> dict:
    """Evaluate every attr with evaluator, checkpointing results to out_path after each one."""
    all_results: dict = {"_meta": meta}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for attr in attrs:
        if attr not in evaluator.probe_layouts:
            print(f"WARNING: {attr} not in probe_layouts, skipping", flush=True)
            continue
        if attr not in vectors.steering_h:
            print(f"WARNING: {attr} not in VectorStore, skipping", flush=True)
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"Attribute: {attr}", flush=True)

        attr_plan = alpha_plan.get(attr) if alpha_plan else None
        all_results[attr] = evaluator.evaluate_attr(
            vectors, test_records, attr, alphas, spaces,
            pairwise=pairwise, relative=relative, alpha_plan=attr_plan,
        )
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  checkpoint -> {out_path}", flush=True)

    return all_results


def abcd_probs_batched(
    model, tok, contexts: list[str], layout: dict, abcd_ids: Tensor,
    inject_delta: Tensor | None, inject_layer: int, device: str,
    batch_size: int = 8, all_positions: bool = True, pbar: tqdm | None = None,
    layers_path: str = "model.layers", chat_template_kwargs: dict | None = None,
    probe_system: str | None = None, strip_leading_asst: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-compat free-function wrapper around SteeringEvaluator.abcd_probs."""
    sm = SteeringModel(
        model=model, tok=tok, model_id="", layers_path=layers_path,
        probe_system=probe_system, chat_template_kwargs=chat_template_kwargs or {},
        strip_leading_asst=strip_leading_asst, abcd_ids=abcd_ids,
    )
    config = SteeringConfig(
        inject_layer=inject_layer, device=device,
        all_positions=all_positions, batch_size=batch_size,
    )
    ev = SteeringEvaluator(sm, None, None, config)
    return ev.abcd_probs(contexts, layout, inject_delta, pbar=pbar)


def eval_attr(
    model, tok, vectors: VectorStore, projector: LowRankUserSubspaceProjector,
    test_records: list[dict], attr: str, alphas: list[float], spaces: list[str],
    probe_layouts: dict, abcd_ids: Tensor, inject_layer: int, device: str,
    n_per_class: int = 100, seed: int = SEED, batch_size: int = 8,
    min_src_prob: float = 0.0, min_src_margin: float = 0.05, pairwise: bool = True,
    all_positions: bool = True, layers_path: str = "model.layers",
    chat_template_kwargs: dict | None = None, probe_system: str | None = None,
    strip_leading_asst: bool = False, relative: bool = False, baseline_oversample: int = 5,
    alpha_plan: dict | None = None, baseline_cache: dict | None = None,
) -> dict:
    """Back-compat free-function wrapper around SteeringEvaluator.evaluate_attr."""
    sm = SteeringModel(
        model=model, tok=tok, model_id="", layers_path=layers_path,
        probe_system=probe_system, chat_template_kwargs=chat_template_kwargs or {},
        strip_leading_asst=strip_leading_asst, abcd_ids=abcd_ids,
    )
    config = SteeringConfig(
        inject_layer=inject_layer, device=device, all_positions=all_positions,
        batch_size=batch_size, n_per_class=n_per_class, seed=seed,
        min_src_prob=min_src_prob, min_src_margin=min_src_margin,
        baseline_oversample=baseline_oversample,
    )
    ev = SteeringEvaluator(sm, projector, probe_layouts, config)
    return ev.evaluate_attr(
        vectors, test_records, attr, alphas, spaces,
        pairwise=pairwise, relative=relative, alpha_plan=alpha_plan, baseline_cache=baseline_cache,
    )
