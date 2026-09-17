import torch


def init_weights(module: torch.nn.Linear, mode: str) -> None:
    if mode == "zero":
        torch.nn.init.zeros_(module.weight)
    elif mode == "small_random":
        torch.nn.init.normal_(module.weight, std=1e-3)
    elif mode == "identity":
        torch.nn.init.eye_(module.weight)
    else:
        raise ValueError(f"Unknown init mode: {mode}")


class UserSubspaceProjector(torch.nn.Module):
    def __init__(self, dim: int, init: str = "small_random"):
        super().__init__()
        self.subspace_projector = torch.nn.Linear(dim, dim, bias=False)
        init_weights(self.subspace_projector, init)

    def delta(self, h_teacher: torch.Tensor) -> torch.Tensor:
        """Teacher-derived delta: the additive correction from h_teacher."""
        return self.subspace_projector(h_teacher)

    def forward(self, h_student: torch.Tensor, h_teacher: torch.Tensor) -> torch.Tensor:
        projection = h_student + self.subspace_projector(h_teacher)
        return projection


class _OrthonormalColumns(torch.nn.Module):
    """Parametrization giving W orthonormal columns, via QR"""

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        Q, R = torch.linalg.qr(W.float())
        d = torch.diagonal(R)
        sign = torch.where(d == 0, torch.ones_like(d), torch.sign(d))
        return (Q * sign.unsqueeze(0)).to(W.dtype)

    def right_inverse(self, Q: torch.Tensor) -> torch.Tensor:
        return Q


class LowRankUserSubspaceProjector(torch.nn.Module):
    def __init__(
        self, dim: int, rank: int, init: str = "small_random", isometric: bool = False
    ):
        super().__init__()
        self.A = torch.nn.Linear(dim, rank, bias=False)
        self.B = torch.nn.Linear(rank, dim, bias=False)
        init_weights(self.A, init)
        init_weights(self.B, init)
        self.isometric = isometric
        if isometric:
            torch.nn.utils.parametrize.register_parametrization(
                self.B, "weight", _OrthonormalColumns()
            )
        print(
            f"Initialized LowRankUserSubspaceProjector with rank {rank}, init '{init}'"
            f"{', isometric B' if isometric else ''}"
        )

    def plain_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "A.weight": self.A.weight.detach().cpu(),
            "B.weight": self.B.weight.detach().cpu(),
        }

    @property
    def v_user(self) -> torch.Tensor | None:
        return self._v_user

    def delta(self, h_teacher: torch.Tensor) -> torch.Tensor:
        """Teacher-derived delta: the additive correction from h_teacher."""
        self._v_user = self.A(h_teacher)
        return self.B(self._v_user)

    def forward(self, h_student: torch.Tensor, h_teacher: torch.Tensor) -> torch.Tensor:
        self._v_user = self.A(h_teacher)
        return h_student + self.B(self._v_user)
