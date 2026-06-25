"""Typed configuration dataclasses."""

from dataclasses import dataclass, field

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


def load_shard_plan(path=None, category=None):
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
    ]


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
