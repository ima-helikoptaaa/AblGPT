import math

import torch
from torch import nn

from ablgpt.model.mlp import Linear
from ablgpt.model.pos_embed import RotaryPositionalEmbeddings


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.dtype = dtype
        self.device = device

        self.q = Linear(d_model, n_heads * head_dim, dtype, device)
        self.k = Linear(d_model, n_kv_heads * head_dim, dtype, device)
        self.v = Linear(d_model, n_kv_heads * head_dim, dtype, device)
        self.out_proj = Linear(n_heads * head_dim, d_model, dtype, device)

        self.rope = RotaryPositionalEmbeddings(
            10000, max_seq_len, self.head_dim, self.dtype, self.device
        )

        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()

        self.causal_mask: torch.Tensor
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor):
        B, seq_len, _ = x.shape
        q = self.q(x).view(B, seq_len, self.n_heads, self.head_dim)
        q = self.rope(q).transpose(1, 2)  # (B, n_heads, T, head_dim)

        k = self.k(x).view(B, seq_len, self.n_kv_heads, self.head_dim)
        k = self.rope(k).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)

        groups = self.n_heads // self.n_kv_heads

        k = k.repeat_interleave(groups, dim=1)  # (B, n_heads, T, head_dim)

        attention = q @ k.transpose(
            -2, -1
        )  # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) = (B, n_heads, T, T)

        attention = attention.masked_fill(
            self.causal_mask[:seq_len, :seq_len], float("-inf")
        )

        attention_probs = torch.softmax((attention) / math.sqrt(self.head_dim), dim=-1)

        v = self.v(x).view(B, seq_len, self.n_kv_heads, self.head_dim)
        v = v.transpose(1, 2)
        v = v.repeat_interleave(groups, dim=1)  # (B, n_heads, seq_len, head_dim)

        values = (
            attention_probs @ v
        )  # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) = (B, n_heads, T, head_dim)
        values = values.transpose(1, 2).contiguous().view(B, seq_len, -1)

        return self.out_proj(values)
