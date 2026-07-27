import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        # scale by 1/sqrt(fan_in) so activation variance is preserved across layers.
        self.weight = nn.Parameter(
            torch.randn((out_features, in_features), dtype=dtype, device=device)
            / (in_features**0.5)
        )

    def forward(self, x: torch.Tensor):
        return x @ self.weight.T


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


class Swiglu(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, dtype: torch.dtype, device: torch.device
    ):
        super().__init__()
        self.up_proj = Linear(d_model, d_ff, dtype, device)
        self.gate_proj = Linear(d_model, d_ff, dtype, device)
        self.down_proj = Linear(d_ff, d_model, dtype, device)

    def forward(self, x: torch.Tensor):
        values = self.up_proj(x)
        gate = self.gate_proj(x)

        gated_values = silu(gate) * values

        return self.down_proj(gated_values)
