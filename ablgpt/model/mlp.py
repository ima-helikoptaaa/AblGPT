import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, out_feat, in_feat, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_feat, in_feat, dtype=dtype, device=device) * in_feat**-0.5
        )

    def forward(self, x):
        return x @ self.weight.T


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_ff, d_model, dtype, device):
        super().__init__()

        self.up_proj = Linear(d_ff, d_model, dtype=dtype, device=device)
        self.gate_proj = Linear(d_ff, d_model, dtype=dtype, device=device)
        self.down_proj = Linear(d_model, d_ff, dtype=dtype, device=device)

    # shape of x -> (batch_size, seq_len, d_model)
    # shape of x_up -> (batch_size, seq_len, d_ff)
    # shape of x_gate -> (batch_size, seq_len, d_ff)
    # shape of output -> (batch_size, seq_len, d_model)
    def forward(self, x):
        x_up = self.up_proj(x)
        x_gate = silu(self.gate_proj(x))
        return self.down_proj(x_up * x_gate)
