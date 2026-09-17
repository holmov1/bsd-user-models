import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from user_distillation.data.attr_schema import ABCD_LETTERS, ATTRIBUTE_SCHEMA
from user_distillation.data.dataset import WildUserDataset, WildUserSample
from user_distillation.training.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_projector,
)
from user_distillation.training.student_wrapper import StudentWrapper
from user_distillation.training.utils import (
    build_abcd_ids,
    build_probe_tensors,
    eligible_labels,
    unpermute,
    usable_indices,
)

DEFAULT_MIN_LAYOUTS_RESOLVED = 2


# ── trainer ───────────────────────────────────────────────────────────────────

class DistillationTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        dataset: WildUserDataset | None,
        probe_layouts_path: str,
        lr: float,
        rank: int,
        weight_decay: float = 0.0,
        device: str = "cuda",
        save_dir: str = "experiments/wilduser_distill",
        save_every: int = 1,
        inject_layer_idx: int | None = None,
        layers_path: str = "model.layers",
        val_dataset: WildUserDataset | None = None,
        eval_every: int = 400,
        train_batch_size: int = 1,
        probe_chunk_size: int = 10,
        min_layouts_resolved: int = DEFAULT_MIN_LAYOUTS_RESOLVED,
        eligible_classes_path: str | None = None,
        chat_template_kwargs: dict | None = None,
        isometric: bool = False,
    ):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset  # only needed by train(); None is fine for eval-only use
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.val_dataset = val_dataset
        self.eval_every = eval_every
        self.train_batch_size = train_batch_size
        self.probe_chunk_size = probe_chunk_size
        self.min_layouts_resolved = min_layouts_resolved

        self._eligible_classes: dict | None = None
        if eligible_classes_path is not None:
            with open(eligible_classes_path) as f:
                self._eligible_classes = json.load(f)
            print(f"  eligibility-restricted eval ON  ({eligible_classes_path})")

        hidden_dim = model.config.hidden_size
        model_dtype = next(model.parameters()).dtype
        self._model_dtype = model_dtype
        self.student = (
            StudentWrapper(
                model, hidden_dim, rank=rank,
                inject_layer_idx=inject_layer_idx,
                layers_path=layers_path,
                isometric=isometric,
            )
            .to(device)
            .to(model_dtype)
        )
        self.optimizer = torch.optim.AdamW(
            self.student.projector.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.abcd_id_tensor = build_abcd_ids(tokenizer, len(ABCD_LETTERS), device)

        self._probe_meta, self._probe_input_ids, self._probe_attn_mask = build_probe_tensors(
            tokenizer, probe_layouts_path, chat_template_kwargs, device,
        )
        print(
            f"  inject_layer_idx={inject_layer_idx}"
            f"  probes={len(self._probe_meta)}  seq_len={self._probe_input_ids.size(1)}"
        )

        self._best_val_f1: float = -float("inf")

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, num_epochs: int, logger=None):
        indices = list(range(len(self.dataset)))
        global_step = 0
        start_epoch = 0
        epoch_losses: list[float] = []

        latest_ckpt = self._latest_checkpoint()
        if latest_ckpt is not None:
            start_epoch, global_step = self._load_checkpoint(latest_ckpt)
            losses_file = self.save_dir / "losses.json"
            if losses_file.exists():
                with open(losses_file) as f:
                    epoch_losses = json.load(f)
            if start_epoch >= num_epochs:
                print(f"  Already completed {num_epochs} epochs — nothing to do.")
                return

        for epoch in range(start_epoch, num_epochs):
            random.shuffle(indices)
            loss, global_step = self._train_epoch(indices, epoch, global_step, logger)
            epoch_losses.append(loss)
            print(f"Epoch {epoch + 1}/{num_epochs}  loss={loss:.4f}")
            if (epoch + 1) % self.save_every == 0:
                self._save(f"w_user_{epoch + 1}.pt")
                self._save_checkpoint(epoch, global_step)
        self._save("w_user.pt")
        with open(self.save_dir / "losses.json", "w") as f:
            json.dump(epoch_losses, f)
        print(f"\nSaved W_user → {self.save_dir}/w_user.pt")

    def _train_epoch(
        self, indices: list[int], epoch: int, global_step: int, logger=None
    ) -> tuple[float, int]:
        total = 0.0
        n_skipped = 0
        bs = self.train_batch_size
        batches = [indices[i: i + bs] for i in range(0, len(indices), bs)]
        last_loss = float("nan")
        pbar = tqdm(batches, desc="  steps", leave=False, unit="batch")
        for batch_idx in pbar:
            samples = [self.dataset[i] for i in batch_idx]
            try:
                last_loss = self.train_step(samples)
                total += last_loss
                global_step += 1
                if logger is not None:
                    logger.log_loss(global_step, last_loss)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_skipped += len(batch_idx)

            pbar.set_postfix(loss=f"{last_loss:.4f}", oom=n_skipped)

            if self.val_dataset is not None and global_step > 0 and global_step % self.eval_every == 0:
                metrics = self.eval(self.val_dataset)
                is_best = metrics["macro_f1"] > self._best_val_f1
                if is_best:
                    self._best_val_f1 = metrics["macro_f1"]
                    self._save("w_user_best.pt")
                print(
                    f"\n  [val step={global_step}]"
                    f"  loss={metrics['loss']:.4f}  macro_f1={metrics['macro_f1']:.3f}"
                    + ("  ★ best" if is_best else "")
                )
                if logger is not None:
                    logger.log_val(global_step, epoch + 1, metrics["loss"], metrics["macro_f1"])
        n_processed = len(indices) - n_skipped
        return total / max(n_processed, 1), global_step

    def train_step(self, samples: list[WildUserSample]) -> float:
        self.optimizer.zero_grad()
        B = len(samples)

        h_teacher = torch.stack([s.h_teacher for s in samples]).to(self.device, dtype=self._model_dtype)

        beliefs_list = [s.beliefs for s in samples]
        n_resolved_list = [s.n_layouts_resolved for s in samples]
        attrs = {attr for (_, attr, _) in self._probe_meta}
        usable = {
            attr: usable_indices(beliefs_list, n_resolved_list, attr, self.min_layouts_resolved)
            for attr in attrs
        }
        targets = {
            attr: torch.stack([
                torch.tensor(samples[i].beliefs[attr], device=self.device, dtype=torch.float32)
                for i in idx
            ])
            for attr, idx in usable.items() if idx
        }

        total_n = len(self._probe_meta)
        total_loss = 0.0
        chunk = self.probe_chunk_size

        for start in range(0, total_n, chunk):
            chunk_meta = self._probe_meta[start: start + chunk]

            chunk_probs = []
            for i, (n_cats, attr, perm) in enumerate(chunk_meta):
                p_idx = start + i
                ids_p  = self._probe_input_ids[p_idx].unsqueeze(0).expand(B, -1)
                mask_p = self._probe_attn_mask[p_idx].unsqueeze(0).expand(B, -1)
                logits, _ = self.student(ids_p, mask_p, h_teacher)
                letter_p = F.softmax(logits[:, self.abcd_id_tensor[:n_cats]], dim=-1)
                canon = torch.zeros(B, n_cats, device=self.device, dtype=letter_p.dtype)
                chunk_probs.append(unpermute(canon, letter_p, perm))

            chunk_loss = torch.tensor(0.0, device=self.device)
            any_usable = False
            for s_b, (_, attr, _) in zip(chunk_probs, chunk_meta):
                idx = usable.get(attr, [])
                if not idx:
                    continue
                any_usable = True
                chunk_loss = chunk_loss + F.kl_div(
                    torch.log(s_b[idx].clamp(min=1e-9)), targets[attr], reduction="none"
                ).sum(dim=1).mean()
            chunk_loss = chunk_loss / total_n

            if not any_usable:
                continue
            chunk_loss.backward()
            total_loss += chunk_loss.item()

        torch.nn.utils.clip_grad_norm_(self.student.projector.parameters(), max_norm=1.0)
        self.optimizer.step()
        return total_loss

    # ── lightweight val eval (during training) ────────────────────────────────

    @torch.no_grad()
    def eval(self, dataset: WildUserDataset, batch_size: int = 32) -> dict:
        """Fast periodic eval: KL loss + macro-F1, used during training."""
        from sklearn.metrics import f1_score

        kl_vals: list[float] = []
        per_attr: dict[str, float] = {}
        for attr, _idx, t_probs, s_probs_sel, t_labels, s_preds in self._per_attr_predictions(
            dataset, batch_size
        ):
            per_attr[attr] = float(f1_score(t_labels, s_preds, average="macro", zero_division=0))
            kl = (t_probs * (np.log(t_probs.clip(1e-9)) - np.log(s_probs_sel.clip(1e-9)))).sum(axis=1)
            kl_vals.extend(kl.tolist())

        return {
            "loss":      float(np.mean(kl_vals)) if kl_vals else float("nan"),
            "macro_f1":  float(np.mean(list(per_attr.values()))) if per_attr else float("nan"),
            "per_attr":  per_attr,
        }

    # ── full post-training eval ───────────────────────────────────────────────

    @torch.no_grad()
    def eval_metrics(
        self,
        dataset: WildUserDataset,
        batch_size: int = 32,
    ) -> dict:
        """Macro-F1, balanced acc, AUC, per-class P/R/F1, confusion — against raw teacher beliefs."""
        import warnings

        from sklearn.metrics import (
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            roc_auc_score,
        )

        results: dict[str, dict] = {}
        for attr, idx, _t_probs, s_probs_sel, t_labels, s_preds in self._per_attr_predictions(
            dataset, batch_size
        ):
            labels = eligible_labels(self._eligible_classes, attr, t_labels)
            class_names = [ATTRIBUTE_SCHEMA[attr][i] for i in labels]

            macro_f1 = float(f1_score(t_labels, s_preds, labels=labels, average="macro", zero_division=0))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bal_acc = float(balanced_accuracy_score(t_labels, s_preds))
            try:
                auc = float(roc_auc_score(
                    t_labels, s_probs_sel[:, labels],
                    labels=labels, multi_class="ovr", average="macro",
                ))
            except ValueError:
                auc = float("nan")

            results[attr] = {
                "macro_f1":    macro_f1,
                "balanced_acc": bal_acc,
                "macro_auc":   auc,
                "n_samples":   len(idx),
                "class_names": class_names,
                "report":      classification_report(
                                   t_labels, s_preds,
                                   labels=labels,
                                   target_names=class_names,
                                   output_dict=True, zero_division=0),
                "confusion":   confusion_matrix(
                                   t_labels, s_preds,
                                   labels=labels).tolist(),
            }

        valid_aucs = [r["macro_auc"] for r in results.values() if not np.isnan(r["macro_auc"])]
        results["_overall"] = {
            "macro_f1":    float(np.mean([r["macro_f1"]     for r in results.values()])),
            "balanced_acc": float(np.mean([r["balanced_acc"] for r in results.values()])),
            "macro_auc":   float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
        }
        return results

    # ── internal helpers ──────────────────────────────────────────────────────

    def _per_attr_predictions(self, dataset: WildUserDataset, batch_size: int):
        """Yield (attr, idx, t_probs, s_probs_sel, t_labels, s_preds) for attrs with usable data."""
        all_h = np.stack([s.h_teacher.numpy() for s in dataset])
        all_beliefs = [s.beliefs for s in dataset]
        all_n_resolved = [s.n_layouts_resolved for s in dataset]
        attr_s = self._run_probes_batched(all_h, batch_size)

        for attr, s_probs in attr_s.items():
            idx = usable_indices(all_beliefs, all_n_resolved, attr, self.min_layouts_resolved)
            if not idx:
                continue
            t_probs = np.stack([np.array(all_beliefs[i][attr]) for i in idx])  # [n_usable, K]
            s_probs_sel = s_probs[idx]
            t_labels = t_probs.argmax(axis=1)
            s_preds = s_probs_sel.argmax(axis=1)
            yield attr, idx, t_probs, s_probs_sel, t_labels, s_preds

    def _run_probes_batched(
        self, all_h: np.ndarray, batch_size: int
    ) -> dict[str, np.ndarray]:
        """Run all pre-tokenized probes over N samples in batches; {attr: [N, n_cats]} canonical-averaged."""
        N = all_h.shape[0]
        attr_s_sum: dict[str, np.ndarray] = {}
        attr_n_probes: dict[str, int] = {}
        for n_cats, attr, _ in self._probe_meta:
            if attr not in attr_s_sum:
                attr_s_sum[attr] = np.zeros((N, n_cats), dtype=np.float64)
                attr_n_probes[attr] = 0
            attr_n_probes[attr] += 1

        for p_idx, (n_cats, attr, perm) in enumerate(
            tqdm(self._probe_meta, desc="  probes", leave=False)
        ):
            ids_p  = self._probe_input_ids[p_idx]
            mask_p = self._probe_attn_mask[p_idx]
            for b0 in range(0, N, batch_size):
                B = min(batch_size, N - b0)
                h_b = torch.from_numpy(all_h[b0: b0 + B]).to(
                    self.device, dtype=self._model_dtype
                )
                logits, _ = self.student(
                    ids_p.unsqueeze(0).expand(B, -1),
                    mask_p.unsqueeze(0).expand(B, -1),
                    h_b,
                )
                lp = F.softmax(logits[:, self.abcd_id_tensor[:n_cats]], dim=-1).float().cpu().numpy()
                attr_s_sum[attr][b0: b0 + B] += unpermute(np.zeros_like(lp), lp, perm)

        return {attr: attr_s_sum[attr] / attr_n_probes[attr] for attr in attr_s_sum}

    def _save(self, name: str):
        save_projector(self.save_dir / name, self.student.projector.plain_state_dict())

    def _save_checkpoint(self, epoch: int, global_step: int) -> Path:
        return save_checkpoint(
            self.save_dir, self.student.projector.state_dict(), self.optimizer.state_dict(),
            epoch, global_step, self._best_val_f1,
        )

    def _latest_checkpoint(self) -> Path | None:
        return latest_checkpoint(self.save_dir)

    def _load_checkpoint(self, path: Path) -> tuple[int, int]:
        """Restore projector + optimizer from a full checkpoint. Returns (start_epoch, global_step)."""
        ckpt = load_checkpoint(path, self.device)
        self.student.projector.load_state_dict(ckpt["projector"])
        self.student.projector.to(self.device)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        self._best_val_f1 = ckpt.get("best_val_f1", -float("inf"))
        print(f"  Resumed from {path.name}  (start_epoch={start_epoch}, global_step={global_step}, best_val_f1={self._best_val_f1:.3f})")
        return start_epoch, global_step
