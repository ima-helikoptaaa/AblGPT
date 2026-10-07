from torch import nn

from ablgpt.model.attention import GQA
from ablgpt.model.embedding import Embedding
from ablgpt.model.mlp import Linear, SwiGLU
from ablgpt.model.norm import RMSNorm
from ablgpt.model.pos_embed import RotaryEmbedding


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq,
        n_layers,
        d_model,
        d_ff,
        n_head,
        n_kv_head,
        head_dim,
        dtype,
        device,
    ):
        super().__init__()
        self.rope = RotaryEmbedding(max_seq, head_dim, device)
        self.embedding = Embedding(vocab_size, d_model, dtype, device)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    max_seq,
                    d_model,
                    d_ff,
                    n_head,
                    n_kv_head,
                    head_dim,
                    self.rope,
                    dtype,
                    device,
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = Linear(vocab_size, d_model, dtype, device)
        self.norm = RMSNorm(d_model, dtype, device)

        self.lm_head.weight = self.embedding.embedding

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.lm_head(self.norm(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self, max_seq, d_model, d_ff, n_head, n_kv_head, head_dim, rope, dtype, device
    ):
        super().__init__()
        self.attn = GQA(
            max_seq, d_model, n_head, n_kv_head, head_dim, rope, dtype, device
        )
        self.swiglu = SwiGLU(d_ff, d_model, dtype, device)
        self.attn_norm = RMSNorm(d_model, dtype, device)
        self.ffn_norm = RMSNorm(d_model, dtype, device)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.swiglu(self.ffn_norm(x))
        return x
