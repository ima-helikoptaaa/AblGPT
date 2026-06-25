import struct

import numpy as np

from ablgpt.utils import REPO_ROOT

SHARDS_DIR = REPO_ROOT / "data" / "shards"

MAGIC = b"MMIDIDX\x00\x00"
VERSION = 1
DTYPE_CODE = 8
DTYPE = np.dtype(np.uint16)


class IndexedDatasetBuilder:
    def __init__(self, slug):
        self.slug = slug

        shard_dir = SHARDS_DIR / self.slug
        shard_dir.mkdir(parents=True, exist_ok=True)

        self.idx_path = shard_dir / f"{slug}.idx"

        shards_path = shard_dir / f"{slug}.bin"
        self.shard_file = open(shards_path, "wb")

        self.sequence_lengths = []
        self.sequence_pointers = [0]

    def add_document(self, ids):
        self.shard_file.write(np.array(ids, dtype=np.uint16).tobytes())
        self.sequence_lengths.append(len(ids))
        self.sequence_pointers.append(self.sequence_pointers[-1] + (len(ids) * 2))

    def finalize(self):
        self.shard_file.close()
        num_docs = len(self.sequence_lengths)
        with open(self.idx_path, "wb") as f:
            f.write(MAGIC)  # magic
            f.write(struct.pack("<Q", VERSION))  # version
            f.write(struct.pack("<B", DTYPE_CODE))  # dtype_code
            f.write(struct.pack("<Q", num_docs))  # sequence_count
            f.write(struct.pack("<Q", num_docs))  # doc_count
            f.write(
                np.array(self.sequence_lengths, dtype=np.int32).tobytes()
            )  # sequence_lengths
            f.write(
                np.array(self.sequence_pointers[:-1], dtype=np.int64).tobytes()
            )  # sequence_pointers
            f.write(
                np.arange(num_docs + 1, dtype=np.int64).tobytes()
            )  # document_indices


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
