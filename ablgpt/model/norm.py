import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, dtype, device, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.device = device
        self.gamma = nn.Parameter(torch.ones((dim), dtype=dtype, device=device))

    def forward(self, x: torch.Tensor):
        x_f = x.float()
        rms = torch.sqrt(torch.mean(torch.square(x_f), dim=-1, keepdim=True) + self.eps)
        x_f = (x_f / rms).to(x.dtype) * self.gamma
        return x_f


if __name__ == "__main__":
    norm = RMSNorm(1024, torch.float16, "cpu")
    x = torch.randn(2, 8, 1024) * 7.0
    out = norm(x)
