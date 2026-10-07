import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, dtype, device):
        super().__init__()

        self.embedding = nn.Parameter(
            torch.randn((vocab_size, d_model), dtype=dtype, device=device) * 0.02
        )

    def forward(self, x):
        return self.embedding[x]
