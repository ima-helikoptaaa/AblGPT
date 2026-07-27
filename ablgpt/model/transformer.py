import torch
from torch import nn
from ablgpt.model.blocks import TransformerBlock
from ablgpt.model.norms import RMSNorm
from ablgpt.model.mlp import Linear

class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        # small std: the head is tied to this, so it also sets the logit scale.
        self.embedding = nn.Parameter(
            torch.randn((vocab_size, d_model), dtype=dtype, device=device) * 0.02
        )

    def forward(self, token_ids: torch.Tensor):
        return self.embedding[token_ids]

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_len,
            n_layers,
            d_model,
            d_ff,
            n_heads,
            n_kv_heads,
            head_dim,
            dtype,
            device
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, dtype, device)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, d_ff, n_heads, n_kv_heads, head_dim, seq_len, dtype, device) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model, dtype=dtype, device=device)
        self.lm_head = Linear(d_model, vocab_size, dtype, device)
        self.lm_head.weight = self.embedding.embedding
        

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

