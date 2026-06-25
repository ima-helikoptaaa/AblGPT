import gzip
import os
from itertools import islice

import boto3
from botocore import UNSIGNED
from botocore.config import Config
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

# stack-edu (and other SmolLM/StarCoder-lineage datasets) ship only METADATA --
# blob_id, language, score, src_encoding -- not the source code. The bytes live
# in SoftwareHeritage's public S3 bucket, keyed by content/<blob_id>, gzip'd.
# Reads are anonymous (UNSIGNED) and the bucket is in us-east-1.
SWH_BUCKET = "softwareheritage"
SWH_REGION = "us-east-1"
_SWH_S3 = None  # lazily created per process (boto3 clients are not fork-safe)

# Datasets whose `text` must be resolved from SWH S3 rather than read inline.
SWH_REPOS = {"HuggingFaceTB/stack-edu"}


def _swh_client():
    global _SWH_S3
    if _SWH_S3 is None:
        _SWH_S3 = boto3.client(
            "s3", region_name=SWH_REGION, config=Config(signature_version=UNSIGNED)
        )
    return _SWH_S3


def _fetch_swh_content(blob_id, src_encoding):
    """Fetch one blob's source text from SoftwareHeritage S3. Returns "" on
    failure so a single bad/missing blob can't kill a streaming run."""
    try:
        obj = _swh_client().get_object(Bucket=SWH_BUCKET, Key=f"content/{blob_id}")
        raw = gzip.decompress(obj["Body"].read())
        return raw.decode(src_encoding or "utf-8", errors="replace")
    except Exception:
        return ""


def _swh_map(ex):
    """Map a stack-edu metadata row -> {"text": <fetched source>}."""
    return {"text": _fetch_swh_content(ex["blob_id"], ex.get("src_encoding"))}


def source_slug(repo_id, config):
    """Stable on-disk directory name for a source. Shared by the sharder (where
    it writes) and the loader (where it reads), so they must never diverge."""
    name = repo_id.split("/")[-1]
    return f"{name}-{config}" if config else name


def load_source(repo_id, config, data_dir, text_field, revision=None):
    label = f"{repo_id}/{config or data_dir}" if (config or data_dir) else repo_id
    print(f"[load] {label} ...", flush=True)

    if repo_id in SWH_REPOS:
        # Metadata-only dataset: stream the rows, then resolve each blob's code
        # from S3. The HF `text_field` is ignored (there is no inline text).
        ds = load_dataset(
            repo_id, config, data_dir=data_dir, split="train",
            streaming=True, revision=revision,
        )
        print(f"[load] {label} OK (metadata; code via SWH S3)", flush=True)
        # remove_columns already leaves only "text" — no select_columns needed.
        return ds.map(_swh_map, features=TEXT_FEATURES, remove_columns=ds.column_names)

    if repo_id == "wikimedia/wikipedia":
        local_files = []
        for shard in WIKIPEDIA_SHARDS:
            print(f"[load] {label} downloading {shard} ...", flush=True)
            path = hf_hub_download(
                repo_id="wikimedia/wikipedia",
                filename=shard,
                repo_type="dataset",
                revision=revision,
            )
            local_files.append(path)
        ds = load_dataset(
            "parquet", data_files=local_files, split="train", streaming=True
        )
        print(f"[load] {label} OK (local)", flush=True)
        return ds.map(
            lambda ex, f=text_field: {"text": ex[f]},
            features=TEXT_FEATURES,
            remove_columns=ds.column_names,
        )

    ds = load_dataset(
        repo_id, config, data_dir=data_dir, split="train",
        streaming=True, revision=revision,
    )
    print(f"[load] {label} OK", flush=True)
    # remove_columns is required: mapping to TEXT_FEATURES while the original
    # columns survive desyncs the schema from the declared features (and breaks
    # sources whose text lives in a non-"text" field, e.g. arxiver's `markdown`).
    return ds.map(
        lambda ex, f=text_field: {"text": ex[f]},
        features=TEXT_FEATURES,
        remove_columns=ds.column_names,
    )


def measure_avg_doc_bytes(ds, sample_size=2000):
    total = 0
    n = 0
    for ex in islice(ds, sample_size):
        text = ex["text"]
        if not text:
            continue
        total += len(text.encode("utf-8"))
        n += 1
    return (total / n) if n else 0.0


def byte_weighted_probs(mix, sample_size=2000):
    datasets_list = []
    raw = []  # target_i / avg_doc_bytes_i, pre-normalization
    for repo_id, config, data_dir, text_field, target in mix:
        ds = load_source(repo_id, config, data_dir, text_field)
        avg = measure_avg_doc_bytes(ds, sample_size=sample_size)
        label = f"{repo_id}/{config or data_dir}" if (config or data_dir) else repo_id
        print(f"[bytes] {label}: avg {avg:.0f} B/doc, target {target}", flush=True)
        datasets_list.append(ds)
        raw.append(target / avg if avg > 0 else 0.0)

    total = sum(raw)
    probs = [r / total for r in raw] if total > 0 else raw
    for (repo_id, config, data_dir, _, _), p in zip(mix, probs):
        label = f"{repo_id}/{config or data_dir}" if (config or data_dir) else repo_id
        print(f"[probs] {label}: sampling p = {p:.4f}", flush=True)
    return datasets_list, probs
