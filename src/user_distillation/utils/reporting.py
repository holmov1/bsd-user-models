import json
from datetime import datetime
from pathlib import Path


class RunLogger:
    def __init__(self, incremental_path: str | Path | None = None):
        self.step_losses: list[dict] = []
        self.val_metrics: list[dict] = []
        self.test_metrics: dict | None = None
        self._incremental_path = Path(incremental_path) if incremental_path else None

    def log_loss(self, step: int, loss: float):
        self.step_losses.append({"step": step, "loss": loss})

    def log_val(self, step: int, epoch: int, loss: float, macro_f1: float):
        self.val_metrics.append({"step": step, "epoch": epoch, "loss": loss, "macro_f1": macro_f1})
        if self._incremental_path:
            self.save_json(self._incremental_path)

    def log_test(self, metrics: dict):
        self.test_metrics = metrics

    def save_json(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(
                {
                    "step_losses": self.step_losses,
                    "val_metrics": self.val_metrics,
                    "test_metrics": self.test_metrics,
                },
                f,
                indent=2,
            )

    def save_report(self, path: str | Path):
        lines = [
            f"Run Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 50,
        ]

        if self.step_losses:
            avg = sum(x["loss"] for x in self.step_losses) / len(self.step_losses)
            lines.append(f"\nTraining: {len(self.step_losses)} steps, avg loss={avg:.4f}")

        if self.val_metrics:
            lines.append(f"\nValidation ({len(self.val_metrics)} evals):")
            for m in self.val_metrics:
                lines.append(
                    f"  step={m['step']:>6}  epoch={m['epoch']}"
                    f"  loss={m['loss']:.4f}  macro_f1={m['macro_f1']:.3f}"
                )

        if self.test_metrics:
            t = self.test_metrics
            lines.append(f"\nTest Set:")
            lines.append(f"  Macro-F1:     {t.get('macro_f1', float('nan')):.3f}")
            lines.append(f"  Balanced Acc: {t.get('balanced_acc', float('nan')):.3f}")
            lines.append(f"  Macro AUC:    {t.get('macro_auc', float('nan')):.3f}")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
