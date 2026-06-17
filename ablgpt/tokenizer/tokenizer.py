"""Runtime tokenizer: load the trained BPE artifacts into a HuggingFace
`Tokenizer` and expose encode/decode.

The artifacts (data/vocab.json, data/merges.txt) are produced by
`ablgpt.tokenizer.bpe` in HF's byte-level encoding. To reproduce the exact
tokenization used at training time we must mirror the trainer's pipeline:

    1. split text on the same TOKEN_PAT regex (the trainer's pre-tokenization),
    2. map raw bytes through the GPT-2 byte->unicode table (ByteLevel),
    3. apply the learned BPE merges.

`ByteLevel(use_regex=False)` is important: it does only the byte mapping and
leaves the splitting to our `Split(TOKEN_PAT)` stage, instead of applying its
own GPT-2 regex (which would split differently and desync from training).

Usage:
For tokenizing text
    from ablgpt.tokenizer.tokenizer import load_tokenizer
    tok = load_tokenizer()
    ids = tok.encode("hello world").ids
    text = tok.decode(ids)

For running evaluation
    uv run python -m ablgpt.tokenizer.tokenizer
"""

from itertools import islice

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from ablgpt.data.build_tokenizer_corpus import tokenizer_mix

from ablgpt.data.utils import load_source
from ablgpt.tokenizer.bpe import TOKEN_PAT
from ablgpt.tokenizer.special_tokens import SPECIAL_TOKENS
from ablgpt.utils import REPO_ROOT


def load_tokenizer():
    vocab_path = REPO_ROOT / "tokenizer" / "vocab.json"
    merges_path = REPO_ROOT / "tokenizer" / "merges.txt"

    tokenizer = Tokenizer(BPE.from_file(str(vocab_path), str(merges_path)))

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(TOKEN_PAT),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    # Register the special tokens so they map to their reserved ids and are
    # never split into sub-pieces.
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    return tokenizer


def get_tokenizer_metrics(ds, tok, sample_size=2000):
    total_bytes = 0
    total_chars = 0
    total_words = 0
    total_tokens = 0
    for ex in islice(ds, sample_size):
        text = ex["text"]
        if not text:
            continue
        total_bytes += len(text.encode("utf-8"))
        total_chars += len(text)
        total_words += len(text.split())
        total_tokens += len(tok.encode(text).ids)

    if total_tokens == 0:
        return 0.0, 0.0, 0.0

    avg_bytes_per_token = total_bytes / total_tokens  # compression ratio
    avg_chars_per_token = total_chars / total_tokens
    # fertility = tokens per word; meaningful for prose, noisy for code
    # (whitespace word-counting breaks down on minified / dense source).
    avg_fertility = total_tokens / total_words if total_words else 0.0
    return (
        avg_bytes_per_token,
        avg_chars_per_token,
        avg_fertility,
    )


def eval_tokenizer(mix, tok, sample_size=2000):
    for repo_id, config, data_dir, text_field, _ in mix:
        ds = load_source(repo_id, config, data_dir, text_field)
        avg_bytes_per_token, avg_chars_per_token, avg_fertility = get_tokenizer_metrics(
            ds, tok, sample_size=sample_size
        )
        label = f"{repo_id}/{config or data_dir}" if (config or data_dir) else repo_id
        print(
            f"{label}: avg {avg_bytes_per_token:.2f} bytes per token",
            flush=True,
        )
        print(
            f"{label}: avg {avg_chars_per_token:.2f} chars per token",
            flush=True,
        )
        print(f"{label}: avg {avg_fertility:.2f} fertility", flush=True)


if __name__ == "__main__":
    tok = load_tokenizer()
    eval_tokenizer(tokenizer_mix, tok)
