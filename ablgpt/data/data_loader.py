import hashlib
import warnings

import numpy as np

from ablgpt.data.shard_io import IndexedDataset
from ablgpt.utils import REPO_ROOT
from ablgpt.config import load_data_mix
from ablgpt.data.utils import source_slug

from torch.utils.data import Dataset

SHARDS_DIR = REPO_ROOT / "data" / "shards"


class GPTDataset:
    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        seq_len=1024,
        split="train",
        val_frac=0.001,
    ):
        self.indexed_dataset = indexed_dataset
        self.slug = indexed_dataset.slug
        self.seq_len = seq_len
        self.split = split

        total_tokens = indexed_dataset.n_tokens
        self.token_start, self.token_end = self._split_bounds(
            total_tokens, split, val_frac
        )
        split_tokens = self.token_end - self.token_start
        self.num_samples = max(0, (split_tokens - 1) // seq_len)
        if self.num_samples == 0:
            warnings.warn(
                f"'{self.slug}' ({split}) has 0 samples: "
                f"{split_tokens} tokens < seq_len+1 ({seq_len + 1}). "
                f"Check val_frac/seq_len vs shard size.",
                stacklevel=2,
            )

    def _split_bounds(self, total_tokens, split, val_frac):
        """Split at a seq_len-aligned token offset near val_frac from the end.

        No document straddles the split because we pack across EOS anyway;
        we just need the cut at a sample boundary so no train window reads
        val tokens.
        """
        split_token = total_tokens - int(total_tokens * val_frac)
        split_token = (split_token // self.seq_len) * self.seq_len
        if split == "train":
            return 0, split_token
        else:
            return split_token, total_tokens

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError(
                f"sample {idx} >= num_samples {self.num_samples} "
                f"for '{self.slug}' ({self.split})"
            )

        start = self.token_start + self.seq_len * idx
        tokens = self.indexed_dataset.get_span(start, self.seq_len + 1)
        tokens = np.asarray(tokens, dtype=np.int64)
        return tokens[:-1], tokens[1:]


class BlendedDataset(Dataset):
    def __init__(
        self,
        datasets: list[GPTDataset],
        weights: list[float],
        size: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.datasets = datasets
        self.weights = weights
        self.size = size
        self.rank = rank
        self.world_size = world_size
        self._build_indices()

    def _build_indices(self):
        cache = self._cache_path()
        if cache.exists():
            data = np.load(cache)
            self.dataset_index = data["dataset_index"]
            self.dataset_sample_index = data["dataset_sample_index"]
            return

        caps = np.array([d.num_samples for d in self.datasets], dtype=np.int64)

        raw = np.asarray(self.weights) * self.size
        num_per_ds = np.floor(raw).astype(np.int64)
        remainder = self.size - sum(num_per_ds)
        fractional = raw - num_per_ds
        order = np.argsort(-fractional)
        num_per_ds[order[:remainder]] += 1

        for i in range(len(num_per_ds)):
            if num_per_ds[i] > caps[i]:
                raise ValueError(
                    f"Dataset '{self.datasets[i].slug}': needs {num_per_ds[i]} samples "
                    f"(weight {self.weights[i]:.4f} × size {self.size}), "
                    f"but only {caps[i]} available. "
                    f"Reduce size, increase the shard budget, or adjust weights."
                )

        all_times = []
        all_ds = []
        all_samples = []
        for i, n in enumerate(num_per_ds):
            time = (np.arange(n, dtype=np.float64) + 0.5) / self.weights[i]
            all_times.append(time)
            all_ds.append(np.full(n, i, dtype=np.int16))
            all_samples.append(np.arange(n, dtype=np.int64))

        all_times = np.concatenate(all_times)
        all_ds = np.concatenate(all_ds)
        all_samples = np.concatenate(all_samples)

        sort_order = np.argsort(all_times, kind="stable")
        self.dataset_index = all_ds[sort_order]
        self.dataset_sample_index = all_samples[sort_order]

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            dataset_index=self.dataset_index,
            dataset_sample_index=self.dataset_sample_index,
        )

    def _cache_path(self):
        key = ",".join(d.slug for d in self.datasets)
        key += "|" + ",".join(f"{w:.6f}" for w in self.weights)
        key += f"|{self.size}"
        key += "|" + ",".join(
            f"{d.seq_len}:{d.split}:{d.num_samples}" for d in self.datasets
        )
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return SHARDS_DIR / "blend" / f"blend_{h}.npz"

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"sample {idx} >= num_samples {len(self)} for blend")
        global_idx = (idx * self.world_size) + self.rank
        ds_idx = int(self.dataset_index[global_idx])
        sample_idx = int(self.dataset_sample_index[global_idx])
        return self.datasets[ds_idx][sample_idx]

    def __len__(self):
        return self.size // self.world_size


def build_dataset(
    mix_name,
    seq_len=1024,
    split="train",
    val_frac=0.001,
    rank=0,
    world_size=1,
    size=None,
):

    data_config = load_data_mix(mix_name)

    datasets = []
    weights = []
    for source in data_config.sources:
        slug = source_slug(source.repo_id, source.config)
        gpt = GPTDataset(
            IndexedDataset(slug),
            seq_len=seq_len,
            split=split,
            val_frac=val_frac,
        )
        datasets.append(gpt)
        weights.append(source.weight)

    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    if size is None:
        caps = [d.num_samples for d in datasets]
        size = int(min(c / w for c, w in zip(caps, weights) if w > 0))

    return BlendedDataset(
        datasets, weights.tolist(), size, rank=rank, world_size=world_size
    )
