import torch

from user_distillation.model_utils import get_injection_layer
from user_distillation.training.subspace_projector import (
    LowRankUserSubspaceProjector,
    UserSubspaceProjector,
)


class StudentWrapper(torch.nn.Module):
    def __init__(
        self,
        student_model: torch.nn.Module,
        projector_dim: int,
        rank: int | None = None,
        init_projector_to: str = "small_random",
        inject_layer_idx: int | None = None,
        layers_path: str = "model.layers",
        isometric: bool = False,
    ):
        super().__init__()
        self.student_model = student_model
        self.inject_layer_idx = inject_layer_idx
        self._layers_path = layers_path
        if rank is not None:
            self.projector = LowRankUserSubspaceProjector(
                projector_dim, rank, init=init_projector_to, isometric=isometric
            )
        else:
            self.projector = UserSubspaceProjector(
                projector_dim, init=init_projector_to
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        h_teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the student model.
        """
        batch_size = input_ids.size(0)
        last_idx = attention_mask.sum(dim=1) - 1  # [batch]

        if self.inject_layer_idx is None:
            with torch.no_grad():
                h_student = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            h_last = h_student.hidden_states[-1][torch.arange(batch_size), last_idx, :]
            projection = self.projector(h_last, h_teacher)
            lm_heads = self.student_model.lm_head(projection.unsqueeze(1))
            return lm_heads.squeeze(1), projection

        # --- injection path ---
        delta = self.projector.delta(h_teacher)  # [batch, hidden]
        arange = torch.arange(batch_size, device=input_ids.device)

        def inject_hook(_module, _input, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output  # [batch, seq, hidden]
            mask = torch.zeros(
                batch_size, hidden.size(1), 1, device=hidden.device, dtype=hidden.dtype,
            )
            mask[arange, last_idx, 0] = 1.0
            new_hidden = hidden + delta.unsqueeze(1) * mask
            return (new_hidden,) + output[1:] if is_tuple else new_hidden

        layer = get_injection_layer(
            self.student_model, self._layers_path, self.inject_layer_idx
        )
        handle = layer.register_forward_hook(inject_hook)
        try:
            out = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        finally:
            handle.remove()

        logits = out.logits[arange, last_idx, :]  # [batch, vocab_size]
        return logits, delta
