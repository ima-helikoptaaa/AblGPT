import math

import torch
from torch import nn


class GQA(nn.Module):
    mask: torch.Tensor

    def __init__(
        self, max_seq, d_model, n_head, n_kv_head, head_dim, rope, dtype, device
    ):
        super().__init__()
        self.q = nn.Parameter(
            torch.randn(head_dim * n_head, d_model, dtype=dtype, device=device)
            * (d_model) ** -0.5
        )
        self.k = nn.Parameter(
            torch.randn(head_dim * n_kv_head, d_model, dtype=dtype, device=device)
            * (d_model) ** -0.5
        )
        self.v = nn.Parameter(
            torch.randn(head_dim * n_kv_head, d_model, dtype=dtype, device=device)
            * (d_model) ** -0.5
        )
        self.o = nn.Parameter(
            torch.randn(d_model, head_dim * n_head, dtype=dtype, device=device)
            * (head_dim * n_head) ** -0.5
        )

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.dtype = dtype

        self.rope = rope

        mask = torch.triu(torch.ones(max_seq, max_seq, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    # shape x -> (batch_size, seq_len, d_model)
    # shape q_proj -> (batch_size, seq_len, head_dim * n_head)
    # shape q_proj -> (batch_size, seq_len, n_head, head_dim)
    # shape q_proj -> (batch_size, n_head, seq_len, head_dim)
    # shape attention -> (batch_size, n_head, seq_len, seq_len)
    # shape attention -> (batch_size, n_head, seq_len, head_dim)
    def forward(self, x: torch.Tensor):
        q_proj = x @ self.q.T
        q_proj = q_proj.view(*x.shape[:-1], self.n_head, self.head_dim)
        q_proj = q_proj.transpose(1, 2)
        q_proj = self.rope(q_proj)

        n_reps = self.n_head // self.n_kv_head

        k_proj = x @ self.k.T
        k_proj = k_proj.view(*x.shape[:-1], self.n_kv_head, self.head_dim)
        k_proj = k_proj.transpose(1, 2)
        k_proj = self.rope(k_proj)
        k_proj = k_proj.repeat_interleave(n_reps, 1)

        v_proj = x @ self.v.T
        v_proj = v_proj.view(*x.shape[:-1], self.n_kv_head, self.head_dim)
        v_proj = v_proj.transpose(1, 2)
        v_proj = v_proj.repeat_interleave(n_reps, 1)

        scores = q_proj @ k_proj.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        S = x.size(1)
        scores = scores.masked_fill(self.mask[:S, :S], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v_proj.dtype)
        attention = attention @ v_proj
        attention = attention.transpose(1, 2)
        attention = attention.reshape(
            *attention.shape[:-2], self.n_head * self.head_dim
        )

        return attention @ self.o.T
