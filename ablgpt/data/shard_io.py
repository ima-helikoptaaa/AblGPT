import struct

import numpy as np

from ablgpt.utils import REPO_ROOT

SHARDS_DIR = REPO_ROOT / "data" / "shards"

MAGIC = b"MMIDIDX\x00\x00"
VERSION = 1
DTYPE_CODE = 8
DTYPE = np.dtype(np.uint16)

# Byte layout of the fixed header (magic + version + dtype + 2 counts).
HEADER_LEN = len(MAGIC) + 8 + 1 + 8 + 8


def write_index(idx_path, sequence_lengths):
    """Write a .idx for a .bin holding `sequence_lengths` back-to-back uint16 docs.

    Pointers are the cumulative byte offset of each doc (2 bytes/token), derived
    purely from the lengths -- so a merged shard can rebuild a correct index from
    its parts' lengths alone, without carrying any per-part byte offsets across.
    """
    lengths = np.asarray(sequence_lengths, dtype=np.int32)
    num_docs = len(lengths)
    pointers = np.zeros(num_docs, dtype=np.int64)
    pointers[1:] = np.cumsum(lengths[:-1].astype(np.int64) * 2)
    with open(idx_path, "wb") as f:
        f.write(MAGIC)  # magic
        f.write(struct.pack("<Q", VERSION))  # version
        f.write(struct.pack("<B", DTYPE_CODE))  # dtype_code
        f.write(struct.pack("<Q", num_docs))  # sequence_count
        f.write(struct.pack("<Q", num_docs))  # doc_count
        f.write(lengths.tobytes())  # sequence_lengths
        f.write(pointers.tobytes())  # sequence_pointers
        f.write(np.arange(num_docs + 1, dtype=np.int64).tobytes())  # document_indices


def read_index_lengths(idx_path):
    """Read just the per-doc token-length array from a .idx (for merging parts)."""
    with open(idx_path, "rb") as f:
        header = f.read(HEADER_LEN)
    if header[: len(MAGIC)] != MAGIC:
        raise RuntimeError(f"{idx_path} file integrity compromised")
    (sequence_count,) = struct.unpack_from("<Q", header, len(MAGIC) + 8 + 1)
    return np.memmap(
        idx_path, dtype=np.int32, mode="r", offset=HEADER_LEN, shape=(sequence_count,)
    )


class IndexedDatasetBuilder:
    def __init__(self, slug, name=None):
        self.slug = slug

        shard_dir = SHARDS_DIR / self.slug
        shard_dir.mkdir(parents=True, exist_ok=True)

        # `name` lets a parallel worker write a part (e.g. "<slug>.part3") into the
        # same shard dir; defaults to the slug for the single-process path.
        stem = name or slug
        self.idx_path = shard_dir / f"{stem}.idx"

        shards_path = shard_dir / f"{stem}.bin"
        self.shard_file = open(shards_path, "wb")

        self.sequence_lengths = []

    def add_document(self, ids):
        self.shard_file.write(np.array(ids, dtype=np.uint16).tobytes())
        self.sequence_lengths.append(len(ids))

    def finalize(self):
        self.shard_file.close()
        write_index(self.idx_path, self.sequence_lengths)


class IndexedDataset:
    def __init__(self, slug):
        self.slug = slug
        shard_dir = SHARDS_DIR / slug
        self.index_path = shard_dir / f"{slug}.idx"
        self.bin_path = shard_dir / f"{slug}.bin"

        self._bin = None
        self._index_loaded = False

    def _ensure_index(self):
        if not self._index_loaded:
            self._read_index()
            self._index_loaded = True

    @property
    def n_docs(self):
        self._ensure_index()
        return len(self.sequence_lengths)

    @property
    def n_tokens(self):
        self._ensure_index()
        return int(self.sequence_lengths.sum())

    def _validate_shard_integrity(self, shard_idx_path, magic):
        if magic != MAGIC:
            raise RuntimeError(f"{shard_idx_path} file integrity compromised")

    def _read_index(self):
        with open(self.index_path, "rb") as f:
            data = f.read(34)

        offset = 0
        magic = data[offset : offset + len(MAGIC)]
        offset += len(MAGIC)
        self._validate_shard_integrity(self.index_path, magic)

        (version,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        (dtype_code,) = struct.unpack_from("<B", data, offset)
        offset += 1
        (sequence_count,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        (doc_count,) = struct.unpack_from("<Q", data, offset)
        offset += 8

        sequence_lengths = np.memmap(
            self.index_path,
            dtype=np.int32,
            mode="r",
            offset=offset,
            shape=(sequence_count,),
        )
        offset += sequence_lengths.nbytes
        sequence_pointers = np.memmap(
            self.index_path,
            dtype=np.int64,
            mode="r",
            offset=offset,
            shape=(sequence_count,),
        )
        offset += sequence_pointers.nbytes
        document_indices = np.memmap(
            self.index_path, dtype=np.int64, mode="r", offset=offset, shape=(doc_count,)
        )

        self.sequence_pointers = sequence_pointers
        self.sequence_lengths = sequence_lengths
        self.document_indices = document_indices

    def get_document(self, i):
        self._ensure_index()
        ptr = int(self.sequence_pointers[i])
        start = ptr // DTYPE.itemsize
        length = int(self.sequence_lengths[i])
        return self.get_span(start, length)

    def _ensure_open(self):
        if self._bin is None:
            self._bin = np.memmap(self.bin_path, dtype=DTYPE, mode="r")
        return self._bin

    def get_span(self, start, length):
        bin = self._ensure_open()
        if start + length > len(bin):
            raise ValueError(
                f"span [{start}:{start+length}] exceeds bin length {len(bin)} "
                f"(requested {length} tokens from offset {start})"
            )
        return np.asarray(bin[start : start + length])
