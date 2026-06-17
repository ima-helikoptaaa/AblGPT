"""Runtime tokenizer: vocabulary layout, special tokens, and encode/decode.

Vocabulary layout (VOCAB_SIZE = 49152):
    [0   .. 255]              raw bytes
    [256 .. 49152-64-1]       learned BPE merges
    [49152-64 .. 49151]       the 64 special tokens

Special tokens occupy the TOP ids so they stay stable regardless of how many
merges are learned. The named set is padded with <|reserved_k|> slots up to a
multiple of 64 — keeps VOCAB_SIZE a clean multiple of 64 for fast matmuls and
leaves headroom for future tokens without shifting existing ids.
"""

VOCAB_SIZE = 49152  # 64 * 768

# Named special tokens (order fixed — do not reorder, ids are positional).
_NAMED = [
    "<|endoftext|>",  # document separator / EOS / PAD
    "<|im_start|>",  # ChatML role-block start
    "<|im_end|>",  # ChatML role-block end
    "<|fim_prefix|>",  # fill-in-middle: code before the gap
    "<|fim_middle|>",  # fill-in-middle: the gap to predict
    "<|fim_suffix|>",  # fill-in-middle: code after the gap
]

# Pad the special-token block up to the next multiple of 64 with reserved slots.
_SPECIAL_BUDGET = 64
_n_reserved = _SPECIAL_BUDGET - len(_NAMED)
_RESERVED = [f"<|reserved_{i}|>" for i in range(_n_reserved)]

# Full ordered list of special tokens (named first, then reserved).
SPECIAL_TOKENS = _NAMED + _RESERVED

# Token string -> id (ids are the TOP `_SPECIAL_BUDGET` slots of the vocab).
SPECIAL_TOKEN_IDS = {
    tok: VOCAB_SIZE - _SPECIAL_BUDGET + i for i, tok in enumerate(SPECIAL_TOKENS)
}

# EOS doubles as the document separator and pad token; referenced widely
# (config, shard tokenizer, loss masking), so it gets a named constant. Other
# specials are looked up via SPECIAL_TOKEN_IDS at the sites that need them.
EOS_ID = SPECIAL_TOKEN_IDS["<|endoftext|>"]
PAD_ID = EOS_ID

# Number of vocab slots available to BPE merges (after bytes, before specials).
N_MERGES = VOCAB_SIZE - 256 - _SPECIAL_BUDGET
