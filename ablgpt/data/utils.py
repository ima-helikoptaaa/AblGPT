import os
from importlib.resources import files
from pathlib import Path

from datasets import Features, Value, load_dataset
from huggingface_hub import hf_hub_download

# Wikipedia streams over HF's xet-bridge CDN, which times out mid-read.
# We only need ~5% of 3 GB (~150 MB), so download a couple of parquet shards
# to local disk first (hf_hub_download has robust resume/retry) and stream those.
WIKIPEDIA_SHARDS = [
    "20231101.en/train-00000-of-00041.parquet",
    "20231101.en/train-00001-of-00041.parquet",
]

os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")

TEXT_FEATURES = Features({"text": Value("string")})


REPO_ROOT = Path(str(files("ablgpt"))).parent


def load_source(repo_id, config, data_dir, text_field):
    label = f"{repo_id}/{config or data_dir}" if (config or data_dir) else repo_id
    print(f"[load] {label} ...", flush=True)

    if repo_id == "wikimedia/wikipedia":
        local_files = []
        for shard in WIKIPEDIA_SHARDS:
            print(f"[load] {label} downloading {shard} ...", flush=True)
            path = hf_hub_download(
                repo_id="wikimedia/wikipedia",
                filename=shard,
                repo_type="dataset",
            )
            local_files.append(path)
        ds = load_dataset(
            "parquet", data_files=local_files, split="train", streaming=True
        )
        print(f"[load] {label} OK (local)", flush=True)
        ds = ds.map(lambda ex, f=text_field: {"text": ex[f]}, features=TEXT_FEATURES)
        return ds.select_columns(["text"])

    ds = load_dataset(
        repo_id, config, data_dir=data_dir, split="train", streaming=True
    )
    print(f"[load] {label} OK", flush=True)
    ds = ds.map(lambda ex, f=text_field: {"text": ex[f]}, features=TEXT_FEATURES)
    return ds.select_columns(["text"])



