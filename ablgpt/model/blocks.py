from torch import nn

from ablgpt.model.attention import GroupedQueryAttention
from ablgpt.model.mlp import Swiglu
from ablgpt.model.norms import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_kv_heads, head_dim, max_seq_len, dtype, device):
        super().__init__()

        self.attn = GroupedQueryAttention(d_model, n_heads, head_dim, n_kv_heads, max_seq_len, dtype, device)
        self.ff = Swiglu(d_model, d_ff, dtype, device)
        self.attn_norm = RMSNorm(d_model, dtype=dtype, device=device)
        self.ff_norm = RMSNorm(d_model, dtype=dtype, device=device)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x
