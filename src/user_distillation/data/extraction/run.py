"""ExtractionRun: drives a teacher model to produce h_teacher.npy / beliefs.jsonl,
via extract() (h-caching, optionally with beliefs too) and label() (beliefs only,
against an existing pointer file). Both share the same per-record compute stages.
"""

import datetime
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from user_distillation.data.dataset import (
    parse_context_to_messages,
    strip_leading_assistant,
)
from user_distillation.data.extraction.beliefs import (
    DEFAULT_MAX_READOUT_STEPS,
    label_sample,
)
from user_distillation.data.extraction.io import (
    _load_resume_state,
    _open_h_mm,
    count_lines,
    is_oom,
    is_role_alternation_error,
    load_records_by_fingerprint,
    make_belief_record,
    write_h_meta,
)
from user_distillation.data.extraction.teacher import teacher_hidden


class ExtractionRun:
    """Runs a WildUser teacher model to produce h_teacher.npy / beliefs.jsonl."""

    def __init__(
        self, model_id: str, model, tok, hidden: int,
        letter_ids: torch.Tensor, space_letter_ids: torch.Tensor, layouts: dict,
        device: str, probe_batch_size: int = 8,
        max_readout_steps: int = DEFAULT_MAX_READOUT_STEPS,
        chat_template_kwargs: dict | None = None, probe_system: str | None = None,
        strip_leading_asst: bool = False,
    ):
        self.model_id = model_id
        self.model = model
        self.tok = tok
        self.hidden = hidden
        self.letter_ids = letter_ids
        self.space_letter_ids = space_letter_ids
        self.layouts = layouts
        self.device = device
        self.probe_batch_size = probe_batch_size
        self.max_readout_steps = max_readout_steps
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.probe_system = probe_system
        self.strip_leading_asst = strip_leading_asst

    def _messages(self, context: str) -> list[dict]:
        messages = parse_context_to_messages(context)
        if self.strip_leading_asst:
            messages = strip_leading_assistant(messages)
        return messages

    def _compute_h(self, messages: list[dict], layers: list[int]) -> torch.Tensor:
        return teacher_hidden(self.model, self.tok, messages, layers, self.device, self.chat_template_kwargs)

    def _compute_beliefs(self, messages: list[dict]) -> tuple[dict, dict, dict]:
        return label_sample(
            self.model, self.tok, messages, self.layouts, self.letter_ids, self.space_letter_ids,
            self.device, self.probe_batch_size, self.max_readout_steps, self.chat_template_kwargs,
            self.probe_system,
        )

    def extract(
        self, records_path: Path, out_dir: Path, layers: list[int], h_only: bool = False,
        resume: bool = False, limit: int = 0, checkpoint_every: int = 500,
    ) -> None:
        """Sequential pass over records.jsonl. Allocates h_teacher.npy; computes
        beliefs too unless h_only (then pair with label() afterward)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        beliefs_path = out_dir / "beliefs.jsonl"
        h_path = out_dir / "h_teacher.npy"
        meta_path = out_dir / "h_teacher_meta.json"
        meta_created = datetime.datetime.utcnow().isoformat() + "Z"

        N_total = count_lines(records_path)
        if limit:
            N_total = min(N_total, limit)

        if resume:
            n_written, raw_skip, ptr_h_rows, ptr_conv_idxs, ptr_fps = _load_resume_state(
                records_path, beliefs_path, meta_path,
            )
            h_mm = (
                np.lib.format.open_memmap(h_path, mode="r+") if h_path.exists()
                else _open_h_mm(h_path, N_total, len(layers), self.hidden)
            )
        else:
            n_written, raw_skip, ptr_h_rows, ptr_conv_idxs, ptr_fps = 0, 0, [], [], []
            h_mm = _open_h_mm(h_path, N_total, len(layers), self.hidden)
            write_h_meta(
                meta_path, self.model_id, str(records_path), layers, self.hidden,
                threshold=0.0, created=meta_created, h_rows=[], conv_idxs=[], fps=[], status="in_progress",
            )

        def _flush_meta(status: str) -> None:
            write_h_meta(
                meta_path, self.model_id, str(records_path), layers, self.hidden,
                threshold=0.0, created=meta_created, h_rows=ptr_h_rows, conv_idxs=ptr_conv_idxs,
                fps=ptr_fps, status=status,
            )

        n_oom = n_bad_role = 0
        beliefs_mode = "a" if n_written > 0 else "w"
        with open(records_path) as fin, open(beliefs_path, beliefs_mode) as fout:
            for _ in range(raw_skip):
                fin.readline()

            for global_row, line in enumerate(
                tqdm(fin, desc="extract", total=N_total - raw_skip, ncols=80), start=raw_skip
            ):
                if limit and n_written >= limit:
                    break
                sample = json.loads(line)
                messages = self._messages(sample["context"])

                try:
                    h = self._compute_h(messages, layers)
                    if h_only:
                        beliefs, per_probe, n_layouts_resolved = {}, {}, {}
                    else:
                        beliefs, per_probe, n_layouts_resolved = self._compute_beliefs(messages)
                except Exception as exc:
                    if is_oom(exc):
                        print(f"warning: OOM at row {global_row}; skipping", flush=True)
                        n_oom += 1
                    elif is_role_alternation_error(exc):
                        print(f"warning: non-alternating roles at row {global_row}; skipping", flush=True)
                        n_bad_role += 1
                    else:
                        raise
                    exc.__traceback__ = None
                    del exc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue

                h_row = n_written
                h_mm[h_row] = h.numpy().astype(np.float16)
                n_written += 1
                ptr_h_rows.append(h_row)
                ptr_conv_idxs.append(sample.get("idx", global_row))
                ptr_fps.append(sample.get("fingerprint", ""))

                fout.write(json.dumps(make_belief_record(
                    sample.get("idx", global_row), sample.get("fingerprint", ""),
                    h_row, beliefs, per_probe, n_layouts_resolved,
                )) + "\n")

                if checkpoint_every and n_written % checkpoint_every == 0:
                    h_mm.flush()
                    fout.flush()
                    _flush_meta("in_progress")

        h_mm.flush()
        _flush_meta("complete")
        print(f"\nDone. Written {n_written:,} records "
              f"({n_oom} OOM skipped, {n_bad_role} non-alternating-role skipped).", flush=True)
        print(f"  h_teacher.npy  shape={list(h_mm.shape)}  dtype=float16", flush=True)
        print(f"  beliefs.jsonl  {n_written:,} lines", flush=True)
        print(f"  out-dir: {out_dir}", flush=True)

    def label(
        self, pointer_path: Path, records_path: Path, out_path: Path,
        resume: bool = False, limit: int = 0, checkpoint_every: int = 200,
        source_filter: str | None = None,
    ) -> None:
        """Beliefs only, reading idx/fingerprint/h_row from pointer_path, joined
        against records_path by fingerprint for context."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records_by_fp = load_records_by_fingerprint(str(records_path))

        skip = count_lines(out_path) if resume else 0
        mode = "a" if skip > 0 else "w"

        n_total = count_lines(pointer_path)
        if limit:
            n_total = min(n_total, limit)

        n_written = skip
        n_missing_context = n_oom = 0
        n_resolved_total: dict[str, int] = defaultdict(int)
        n_layouts_total: dict[str, int] = defaultdict(int)

        with open(pointer_path) as fin, open(out_path, mode) as fout:
            for _ in range(skip):
                fin.readline()

            for line in tqdm(fin, desc="label", total=max(n_total - skip, 0), ncols=80):
                if limit and n_written >= limit:
                    break
                old = json.loads(line)
                fp = old.get("fingerprint")
                rec = records_by_fp.get(fp)
                if rec is None:
                    n_missing_context += 1
                    continue
                if source_filter and rec.get("source") != source_filter:
                    continue

                messages = self._messages(rec["context"])
                try:
                    beliefs, probes, n_resolved = self._compute_beliefs(messages)
                except Exception as exc:
                    if not is_oom(exc):
                        raise
                    print(f"warning: OOM at fingerprint={fp}; skipping", flush=True)
                    n_oom += 1
                    exc.__traceback__ = None
                    del exc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                fout.write(json.dumps(make_belief_record(
                    old.get("idx"), fp, old.get("h_row"), beliefs, probes, n_resolved,
                )) + "\n")
                n_written += 1
                for attr, probe_list in probes.items():
                    n_layouts_total[attr] += len(probe_list)
                    n_resolved_total[attr] += n_resolved.get(attr, 0)

                if checkpoint_every and n_written % checkpoint_every == 0:
                    fout.flush()

        print(f"\nDone. Written {n_written:,} records "
              f"({n_missing_context} missing context, {n_oom} OOM skipped).", flush=True)
        print(f"  out: {out_path}", flush=True)
        print(f"\nper-attribute layout resolution rate (max_steps={self.max_readout_steps}):", flush=True)
        for attr in self.layouts:
            t = n_layouts_total.get(attr, 0)
            r = n_resolved_total.get(attr, 0)
            pct = 100 * r / t if t else 0.0
            print(f"  {attr:<26} {r:6,}/{t:6,}  ({pct:.1f}% resolved)", flush=True)
