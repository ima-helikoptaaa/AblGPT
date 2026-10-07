import torch
from torch import nn


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, max_seq, head_dim, device, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        t = torch.arange(max_seq, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int = 0):
        S = x.size(-2)
        cos = self.cos[position_offset : position_offset + S]
        sin = self.sin[position_offset : position_offset + S]
        x_f = x.float()
        return (x_f * cos + _rotate_half(x_f) * sin).to(x.dtype)
