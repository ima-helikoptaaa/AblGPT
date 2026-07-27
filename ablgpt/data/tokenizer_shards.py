import argparse

from ablgpt.config import load_shard_plan
from ablgpt.data.shard_io import IndexedDatasetBuilder
from ablgpt.data.utils import load_source, source_slug
from ablgpt.tokenizer.special_tokens import EOS_ID
from ablgpt.tokenizer.tokenizer import load_tokenizer
from ablgpt.utils import REPO_ROOT

SHARDS_DIR = REPO_ROOT / "data" / "shards"


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

    return parser.parse_args()


def tokenize_shards(slug, tok, dataset, max_tokens):
    total_token_len = 0
    shard_writer = IndexedDatasetBuilder(slug)
    for ex in dataset:
        text = ex["text"]
        ids = tok.encode(text).ids
        total_token_len += len(ids) + 1

        shard_writer.add_document(ids + [EOS_ID])
        if total_token_len > max_tokens:
            break
    shard_writer.finalize()


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

    tok = load_tokenizer()
    for cfg in shard_cfg:
        slug = source_slug(cfg.repo_id, cfg.config)
        dataset = load_source(
            cfg.repo_id, cfg.config, cfg.data_dir, cfg.text_field, cfg.revision
        )
        tokenize_shards(slug, tok, dataset, cfg.tokens)


if __name__ == "__main__":
    main()
