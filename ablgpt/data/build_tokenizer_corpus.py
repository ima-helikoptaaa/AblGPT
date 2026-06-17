"""Stream + interleave a data mix to a single .jsonl on disk (for tokenizer training).

Usage:
    uv run python -m ablgpt.data.build_tokenizer_corpus
"""

import orjson
from datasets import interleave_datasets
from tqdm import tqdm

from ablgpt.data.utils import byte_weighted_probs
from ablgpt.utils import REPO_ROOT

# Each source is (repo_id, config, data_dir, text_field, weight).
_LANGS = [
    "python",
    "javascript",
    "typescript",
    "markdown",
    "c++",
    "c",
    "rust",
    "go",
    "java",
    "c-sharp",
    "ruby",
    "sql",
]
_STACK_SMOL = [
    ("bigcode/the-stack-smol", None, f"data/{lang}", "content", 0.01) for lang in _LANGS
]

tokenizer_mix = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", None, "text", 0.45),
    ("mlfoundations/dclm-baseline-1.0", None, None, "text", 0.30),
    *_STACK_SMOL,
    ("HuggingFaceTB/finemath", "finemath-4plus", None, "text", 0.08),
    ("wikimedia/wikipedia", "20231101.en", None, "text", 0.05),
]

TARGET_BYTES = 3 * 1024**3  # 3 GB for training tokenizer
WRITE_BUFFER = 64 * 1024 * 1024  # flush every 64 MB


def main():
    # Mix weights are byte-fraction TARGETS; convert to per-document sampling
    # probabilities so the realized byte mix matches them (interleave samples
    # per document, but docs differ in size across sources).
    datasets_list, probs = byte_weighted_probs(tokenizer_mix)

    print("[interleave] resolving features (may take a minute) ...", flush=True)
    mixed = interleave_datasets(
        datasets_list,
        probabilities=probs,
        seed=42,
        stopping_strategy="all_exhausted",
    )
    print("[interleave] ready", flush=True)

    pbar = tqdm(
        total=TARGET_BYTES,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Writing tokenizer_corpus",
    )

    out_path = REPO_ROOT / "tokenizer" / "tokenizer_corpus.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buf = []
    buf_size = 0
    written = 0

    with open(out_path, "wb") as f:
        for example in mixed:
            text = example["text"]
            if not text:
                continue
            line = orjson.dumps({"text": text}) + b"\n"
            buf.append(line)
            buf_size += len(line)
            written += len(line)
            pbar.update(len(line))
            if buf_size >= WRITE_BUFFER:
                f.writelines(buf)
                buf.clear()
                buf_size = 0
            if written >= TARGET_BYTES:
                break

    pbar.close()

    print(f"Wrote {written / 1024**3:.2f} GB to {out_path}")


if __name__ == "__main__":
    main()
