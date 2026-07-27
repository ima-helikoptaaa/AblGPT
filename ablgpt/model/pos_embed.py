import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        theta_base: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        theta = 1.0 / (
            theta_base
            ** (2 * torch.arange(head_dim // 2, dtype=dtype, device=device) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=dtype, device=device)
        cache = torch.outer(positions, theta)
        cos_cache = torch.cos(cache)
        sin_cache = torch.sin(cache)

        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.register_buffer("cos", cos_cache)
        self.register_buffer("sin", sin_cache)

    def forward(self, x: torch.Tensor):
        head_dim = x.shape[-1]
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]

        cos = self.cos[: x.shape[1]].unsqueeze(0).unsqueeze(2)
        sin = self.sin[: x.shape[1]].unsqueeze(0).unsqueeze(2)

        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos

        return torch.cat([x1_new, x2_new], dim=-1)
