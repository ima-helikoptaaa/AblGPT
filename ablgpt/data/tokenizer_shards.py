import argparse
import shutil
from multiprocessing import Pool

import numpy as np

from ablgpt.config import load_shard_plan
from ablgpt.data.shard_io import (
    SHARDS_DIR,
    IndexedDatasetBuilder,
    read_index_lengths,
    write_index,
)
from ablgpt.data.utils import load_source, source_slug
from ablgpt.tokenizer.special_tokens import EOS_ID
from ablgpt.tokenizer.tokenizer import load_tokenizer


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--all", action="store_true", help="Process all categories"
    )
    parser.add_argument(
        "-c",
        "--category",
        type=str,
        default=None,
        required=False,
        help="Category name to process",
    )
    parser.add_argument(
        "-r",
        "--repo_id",
        type=str,
        default=None,
        required=False,
        help="Repo id to process",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Parallel tokenizer workers (1 = single-process). Splits the source "
        "stream into N disjoint file-shards, tokenizes them concurrently, then "
        "concatenates the parts into one .bin + .idx.",
    )

    return parser.parse_args()


def tokenize_shards(slug, tok, make_dataset, max_tokens, name=None, log_label=None,
                    log_every=5000, max_retries=5):
    """Tokenize a stream to a single .bin/.idx, stopping past `max_tokens`.

    `make_dataset` is a zero-arg callable returning a fresh stream, so a transient
    HF streaming error (e.g. "Cannot send a request, as the client has been
    closed") is survivable: we rebuild the stream and .skip() past the docs already
    written, resuming without duplicating or losing data. Without this, one flaky
    worker's exception propagates through pool.starmap and discards the WHOLE
    source's work -- even at 95% done (observed in practice).

    If `log_label` is set, print a running token count every `log_every` docs so
    a long parallel run is trackable from the log (workers interleave, hence the
    label). Counts are per-worker; sum across labels for the mix total.
    """
    total_token_len = 0
    n_docs = 0
    shard_writer = IndexedDatasetBuilder(slug, name=name)
    attempt = 0
    while True:
        dataset = make_dataset()
        if n_docs:  # resuming after an error: skip what we've already written
            dataset = dataset.skip(n_docs)
        try:
            for ex in dataset:
                text = ex["text"]
                ids = tok.encode(text).ids
                total_token_len += len(ids) + 1
                n_docs += 1

                shard_writer.add_document(ids + [EOS_ID])
                if log_label and n_docs % log_every == 0:
                    print(
                        f"[{log_label}] {total_token_len / 1e6:.1f}M tok  "
                        f"(docs={n_docs})",
                        flush=True,
                    )
                if total_token_len > max_tokens:
                    break
            break  # stream exhausted or budget hit -> done
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            print(
                f"[{log_label or slug}] stream error after {n_docs} docs "
                f"(attempt {attempt}/{max_retries}), resuming: {e!r}",
                flush=True,
            )
    shard_writer.finalize()
    if log_label:
        print(
            f"[{log_label}] DONE {total_token_len / 1e6:.1f}M tok  (docs={n_docs})",
            flush=True,
        )


def _worker(cfg_dict, index, workers, max_tokens):
    """One parallel worker: tokenize shard `index` of `workers` to a part file.

    Runs in its own process (fork), so it loads its own tokenizer and opens its
    own HF stream, then keeps only every `workers`-th file via .shard(). Each
    worker gets an equal slice of the TOTAL budget so the parts sum to ~max_tokens.
    """
    tok = load_tokenizer()

    def make_dataset():
        ds = load_source(
            cfg_dict["repo_id"],
            cfg_dict["config"],
            cfg_dict["data_dir"],
            cfg_dict["text_field"],
            cfg_dict["revision"],
        )
        return ds.shard(num_shards=workers, index=index)

    slug = cfg_dict["slug"]
    part = f"{slug}.part{index}"
    tokenize_shards(
        slug, tok, make_dataset, max_tokens // workers, name=part,
        log_label=f"w{index:02d}",
    )
    return part


def merge_parts(slug, parts):
    """Concatenate part .bin files in order and rebuild one .idx from their lengths.

    The merged index is derived purely from the parts' per-doc token lengths (see
    write_index): byte pointers are a cumulative sum over the concatenated docs,
    so no per-part offset bookkeeping crosses the boundary. Parts are removed after.
    """
    shard_dir = SHARDS_DIR / slug
    out_bin = shard_dir / f"{slug}.bin"
    out_idx = shard_dir / f"{slug}.idx"

    all_lengths = []
    with open(out_bin, "wb") as dst:
        for part in parts:
            part_bin = shard_dir / f"{part}.bin"
            with open(part_bin, "rb") as src:
                shutil.copyfileobj(src, dst)
            all_lengths.append(read_index_lengths(shard_dir / f"{part}.idx"))

    write_index(out_idx, np.concatenate(all_lengths))

    for part in parts:
        (shard_dir / f"{part}.bin").unlink()
        (shard_dir / f"{part}.idx").unlink()


def shard_source(cfg, tok, workers):
    """Shard one source, single-process if workers==1 else fan out + merge."""
    slug = source_slug(cfg.repo_id, cfg.config)
    if workers <= 1:
        def make_dataset():
            return load_source(
                cfg.repo_id, cfg.config, cfg.data_dir, cfg.text_field, cfg.revision
            )

        tokenize_shards(slug, tok, make_dataset, cfg.tokens)
        return

    cfg_dict = {
        "repo_id": cfg.repo_id,
        "config": cfg.config,
        "data_dir": cfg.data_dir,
        "text_field": cfg.text_field,
        "revision": cfg.revision,
        "slug": slug,
    }
    tasks = [(cfg_dict, i, workers, cfg.tokens) for i in range(workers)]
    with Pool(workers) as pool:
        parts = pool.starmap(_worker, tasks)
    # Merge in worker order (0..N-1) so the byte layout is deterministic.
    merge_parts(slug, [f"{slug}.part{i}" for i in range(workers)])


def main():
    args = get_argparse()
    if args.all:
        shard_cfg = load_shard_plan()
    elif args.category:
        shard_cfg = load_shard_plan(category=args.category)
    elif args.repo_id:
        shard_cfg = load_shard_plan(repo_id=args.repo_id)
    else:
        print("Provide --all, --category or --repo_id. Use --help for usage.")
        return

    # A shared tokenizer is only used on the single-process path; workers each
    # load their own (tokenizers objects don't survive fork cleanly).
    tok = load_tokenizer() if args.workers <= 1 else None
    for cfg in shard_cfg:
        shard_source(cfg, tok, args.workers)


if __name__ == "__main__":
    main()
