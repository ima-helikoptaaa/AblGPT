import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        epsilon: float = 1e-5,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        # gamma starts at 1 -> norm is an identity scale at init.
        self.gamma = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        return (x / rms) * self.gamma
