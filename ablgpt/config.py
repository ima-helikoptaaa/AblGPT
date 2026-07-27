"""Typed configuration dataclasses."""

from dataclasses import dataclass, field, fields

import yaml

from ablgpt.tokenizer.bpe import TOKEN_PAT
from ablgpt.tokenizer.special_tokens import EOS_ID, PAD_ID, VOCAB_SIZE
from ablgpt.utils import REPO_ROOT

CONFIGS_DIR = REPO_ROOT / "configs"


@dataclass
class TokenizerConfig:
    vocab_size: int = VOCAB_SIZE
    eos_id: int = EOS_ID
    pad_id: int = PAD_ID
    pattern: str = TOKEN_PAT


@dataclass
class MixSource:
    """One source in a data mix.

    `weight` is a byte-fraction TARGET (see ablgpt.data.utils.byte_weighted_probs);
    weights across a mix are renormalized, so they need not sum to exactly 1.
    """

    repo_id: str
    config: str | None
    data_dir: str | None
    text_field: str
    weight: float

    def as_tuple(self):
        """(repo_id, config, data_dir, text_field, weight) for the data pipeline."""
        return (self.repo_id, self.config, self.data_dir, self.text_field, self.weight)


@dataclass
class DataConfig:
    """A named data mix loaded from configs/data_mixes.yaml."""

    name: str
    sources: list[MixSource] = field(default_factory=list)

    def as_mix(self):
        """List of source tuples, the shape the data pipeline consumes."""
        return [s.as_tuple() for s in self.sources]


@dataclass
class ShardSource:
    """One source in the shard plan (configs/shard_plan.yaml).

    `tokens` is an UPPER-BOUND token budget: the sharder writes up to this many
    tokens of this source to disk (or fewer, if the source exhausts first). The
    loader enforces true per-category repetition limits at read time. `category`
    drives those limits (web|code|math|curated); `revision` pins the HF dataset
    commit so resume's stream-skip is exact and re-shards stay byte-identical.
    """

    repo_id: str
    config: str | None
    data_dir: str | None
    text_field: str
    category: str
    tokens: int
    revision: str | None = None

    def as_source(self):
        """(repo_id, config, data_dir, text_field) — the load_source signature."""
        return (self.repo_id, self.config, self.data_dir, self.text_field)


def load_shard_plan(path=None, category=None, repo_id=None):
    """Load the disk-inventory shard plan from configs/shard_plan.yaml."""
    path = path or (CONFIGS_DIR / "shard_plan.yaml")
    with open(path) as f:
        plan = yaml.safe_load(f)
    return [
        ShardSource(
            repo_id=s["repo_id"],
            config=s.get("config"),
            data_dir=s.get("data_dir"),
            text_field=s.get("text_field", "text"),
            category=s["category"],
            tokens=int(float(s["tokens"])),
            revision=s.get("revision"),
        )
        for s in plan["shards"]
        if category is None or s["category"] == category
        if repo_id is None or s["repo_id"] == repo_id
    ]


@dataclass
class TrainConfig:
    """A single training run's fully-resolved knobs (configs/train/<name>.yaml).

    One file per run IS the experiment record — stamp the resolved config
    (plus git + tokenizer_sha) into the run's output dir for reproducibility.
    `dtype`/`device` are strings here (YAML can't hold a torch object); the
    trainer maps them to torch types at load via torch_dtype().
    """

    # model
    vocab_size: int = VOCAB_SIZE
    seq_len: int = 1024
    n_layers: int = 12
    d_model: int = 768
    d_ff: int = 2048
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    # optim / run
    n_steps: int = 100
    batch_size: int = 8
    lr: float = 3e-4
    n_val_batches: int = 20
    # data
    mix_name: str = "smoke"
    # checkpointing
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    # runtime (strings -> torch objects at load)
    dtype: str = "float32"
    device: str = "cpu"

    def torch_dtype(self):
        import torch

        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]


def load_train_config(name, path=None):
    """Load a training run config from configs/train/<name>.yaml.

    YAML keys override TrainConfig defaults; unknown keys raise (typo guard).
    """
    path = path or (CONFIGS_DIR / "train" / f"{name}.yaml")
    with open(path) as f:
        overrides = yaml.safe_load(f) or {}
    by_name = {f.name: f for f in fields(TrainConfig)}
    unknown = set(overrides) - set(by_name)
    if unknown:
        raise KeyError(f"unknown TrainConfig keys in {path}: {sorted(unknown)}")
    # Coerce to the declared type: YAML reads bare sci-notation (e.g. 3e-4) as a
    # string, so cast int/float fields explicitly.
    coerced = {}
    for k, v in overrides.items():
        t = by_name[k].type
        if t in (int, float) and isinstance(v, str):
            v = t(float(v))
        coerced[k] = v
    return TrainConfig(**coerced)


def load_data_mix(name, path=None):
    """Load a named mix (e.g. "flagship", "ladder") from configs/data_mixes.yaml."""
    path = path or (CONFIGS_DIR / "data_mixes.yaml")
    with open(path) as f:
        all_mixes = yaml.safe_load(f)
    if name not in all_mixes:
        raise KeyError(f"mix {name!r} not in {path} (have: {sorted(all_mixes)})")
    sources = [
        MixSource(
            repo_id=s["repo_id"],
            config=s.get("config"),
            data_dir=s.get("data_dir"),
            text_field=s.get("text_field", "text"),
            weight=s["weight"],
        )
        for s in all_mixes[name]
    ]
    return DataConfig(name=name, sources=sources)
