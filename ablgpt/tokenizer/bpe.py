"""Train BPE from tokenizer training corpus and save tokenizer artifacts on disk

Usage:
    uv run python -m ablgpt.tokenizer.bpe
"""

import heapq
import json
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

import regex
from tqdm import tqdm

from ablgpt.tokenizer.special_tokens import N_MERGES, SPECIAL_TOKEN_IDS, SPECIAL_TOKENS
from ablgpt.utils import REPO_ROOT

TOKEN_PAT = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
SPECIAL_TOKEN_SPLIT_PAT = "|".join(map(re.escape, SPECIAL_TOKENS))


def bytes_to_unicode():
    """GPT-2 reversible byte -> printable-unicode map.

    Maps all 256 byte values to unicode chars that are never whitespace or
    control characters, so a token's string form is always safe to write into
    the space-delimited merges.txt. This is the exact mapping HuggingFace's
    ByteLevel BPE expects; vocab.json / merges.txt must be written in it.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()


def token_bytes_to_str(token_bytes):
    """Render a token's raw bytes as its HF byte-level string form."""
    return "".join(BYTE_ENCODER[b] for b in token_bytes)


def load_tokenizer_training_corpus(lines):
    local_counter = Counter()
    for line in lines:
        text = json.loads(line)["text"]
        for chunk in re.split(SPECIAL_TOKEN_SPLIT_PAT, text):
            tokens = regex.findall(TOKEN_PAT, chunk)
            local_counter.update(tokens)
    return local_counter


def get_token_freq():
    n_workers = os.cpu_count() or 1

    tokenizer_corpus_path = REPO_ROOT / "tokenizer" / "tokenizer_corpus.jsonl"
    tokenizer_corpus_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    with open(tokenizer_corpus_path, "r") as f:
        lines = f.readlines()

    chunk_size = max(1, len(lines) // (n_workers * 8))
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

    with Pool(n_workers) as pool:
        partial_counters = list(
            tqdm(
                pool.imap(load_tokenizer_training_corpus, chunks),
                total=len(chunks),
                desc="Processing chunks",
            )
        )

    # Merge all partial counters on the main process
    total = Counter()
    for c in partial_counters:
        total.update(c)
    return total


def merge_one_pair(seq, pair, new_id):
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


def merge_pair(
    token_bytes_freq,
    byte_pair_freq,
    byte_pair_to_merge,
    token_id,
    byte_pair_to_token,
    max_heap,
):
    for old_token in list(byte_pair_to_token[byte_pair_to_merge]):
        freq = token_bytes_freq.pop(old_token)
        new_token = merge_one_pair(old_token, byte_pair_to_merge, token_id)

        delta = Counter(zip(new_token, new_token[1:]))
        delta.subtract(zip(old_token, old_token[1:]))

        for pair, d in delta.items():
            if d == 0:
                continue
            byte_pair_freq[pair] += d * freq
            if byte_pair_freq[pair] <= 0:
                del byte_pair_freq[pair]
            else:
                heapq.heappush(
                    max_heap, (-byte_pair_freq[pair], pair)
                )  # push updated priority

        # re-index: old_word no longer exists; new_word takes its place
        for pair in set(zip(old_token, old_token[1:])):
            byte_pair_to_token[pair].discard(old_token)
        for pair in set(zip(new_token, new_token[1:])):
            byte_pair_to_token[pair].add(new_token)

        token_bytes_freq[new_token] += freq

    del byte_pair_to_token[byte_pair_to_merge]

    # pop until we find a non-stale top
    while max_heap:
        neg_freq, pair = heapq.heappop(max_heap)
        if byte_pair_freq.get(pair, 0) == -neg_freq:  # still current?
            return pair
    return None  # heap exhausted — no pairs left


def train_bpe():
    vocab = {i: bytes([i]) for i in range(256)}  # base byte tokens
    merges = []

    token_freq = get_token_freq()
    token_bytes_freq = defaultdict(int)
    byte_pair_freq = defaultdict(int)
    byte_pair_to_token = defaultdict(set)

    for token, freq in token_freq.items():
        token_bytes = token.encode("utf-8")
        token_bytes_freq[tuple(token_bytes)] += freq
        for pair in zip(token_bytes, token_bytes[1:]):
            byte_pair_freq[pair] += freq
            byte_pair_to_token[pair].add(tuple(token_bytes))

    del token_freq

    max_heap = [(-freq, pair) for pair, freq in byte_pair_freq.items()]
    heapq.heapify(max_heap)

    byte_pair_to_merge = None
    while max_heap:
        neg_freq, pair = heapq.heappop(max_heap)
        if byte_pair_freq.get(pair, 0) == -neg_freq:
            byte_pair_to_merge = pair
            break

    for token_id in tqdm(range(256, N_MERGES + 256), desc="Training BPE"):
        if byte_pair_to_merge is None:
            break
        left, right = byte_pair_to_merge
        merges.append(byte_pair_to_merge)
        vocab[token_id] = vocab[left] + vocab[right]
        byte_pair_to_merge = merge_pair(
            token_bytes_freq,
            byte_pair_freq,
            byte_pair_to_merge,
            token_id,
            byte_pair_to_token,
            max_heap,
        )

    # vocab.json is {token_string: id} in HF's byte-level encoding. Byte and
    # merge tokens are rendered via the GPT-2 byte map; special tokens are
    # written as their literal strings (e.g. "<|endoftext|>") so HF can match
    # them as added tokens.
    token_str_to_id = {token_bytes_to_str(b): tid for tid, b in vocab.items()}
    for tok, tid in SPECIAL_TOKEN_IDS.items():
        token_str_to_id[tok] = tid

    vocab_out_path = REPO_ROOT / "tokenizer" / "vocab.json"
    merges_out_path = REPO_ROOT / "tokenizer" / "merges.txt"

    with open(vocab_out_path, "w") as f:
        json.dump(token_str_to_id, f, ensure_ascii=False, indent=2)

    with open(merges_out_path, "w") as f:
        f.write("#version: 0.2\n")  # HF expects a version header
        for left, right in merges:
            f.write(
                f"{token_bytes_to_str(vocab[left])} {token_bytes_to_str(vocab[right])}\n"
            )


def main():
    train_bpe()


if __name__ == "__main__":
    main()
