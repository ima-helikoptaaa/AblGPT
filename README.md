# AblGPT

A decoder-only language model and its pretraining stack, implemented from scratch in
PyTorch — no `transformers`, no `tokenizers`, no reference implementation copied in.
The model, the BPE tokenizer, the sharded data pipeline, and the training loop are all
written here.

The project exists to answer one question rigorously:

> Which architectural choices move the loss-per-FLOP frontier for small language
> models at fixed compute — and does a scaling law fit on a cheap dense ladder predict
> a larger MoE model trained at that frontier's extrapolation?

That framing drives the design: everything is config-driven so architecture and
data-mix ablations are reproducible and comparable, rather than a pile of one-off runs.

## Status

Implemented and working: model, tokenizer, data pipeline, trainer, correctness gates.
Both correctness gates pass, and a full-size 12L/768 smoke run trains on real
FineWeb-Edu data with train and validation loss falling from the `ln(vocab_size)`
baseline.

**Not yet run:** the scaling-law ladder, the architecture ablations, and the flagship
pretraining run. Those need datacenter GPUs. Their configs are committed
(`configs/data_mixes.yaml`, `configs/train/*_cuda.yaml`), but no results exist yet — the
only checkpoints on disk are from the smoke run.

### Choosing a rung

`configs/train/` holds three Chinchilla-optimal rungs (all at ~20 tokens/param, so
they sit on the same compute-optimal line and their loss curves are comparable):

| Config | Params | Tokens | Wall clock, 1x RTX PRO 6000 Blackwell |
|---|---|---|---|
| `run_124m.yaml` | 124M | 2.5B | well under a day |
| `run_246m.yaml` | 246M | 4.9B | **2.5–4.5 days** as written; ~2 with fused attention |
| `run_480m.yaml` | 480M | 10.0B | 6–10 days as written |

`run_246m.yaml` is the intended first real rung: it is the largest model that
finishes in a few days on one card rather than a week and a half.

Wall clock is dominated by `attention.py` materializing the full `B×nh×T×T` score
tensor and softmaxing it explicitly — there is no fused/flash kernel on that path
and no `torch.compile`, which puts effective MFU in the 12–18% band rather than
30–40%. Switching that module to `F.scaled_dot_product_attention` (with
`enable_gqa=True`, which also makes the `repeat_interleave` unnecessary) is the
single highest-leverage change in the repo and roughly halves every number above.

Two issues to fix before treating any of these as a reference run: the model is
built directly in bf16 with no autocast or fp32 master weights (AdamW updates bf16
params in place, and small updates round away over ~19k steps), and checkpoints do
not restore the data-stream position (so an interruption re-trains on seen tokens).

## Model

`ablgpt/model/` — a Llama-style decoder-only transformer, each piece hand-implemented:

| Component | File | Notes |
|---|---|---|
| Grouped-query attention | `attention.py` | 12 query heads, 4 KV heads, `head_dim` 64 |
| Rotary position embeddings | `pos_embed.py` | applied per-head to Q and K |
| RMSNorm | `norms.py` | pre-norm placement |
| SwiGLU feed-forward | `mlp.py` | `d_ff` 2048 |
| Blocks + LM head | `blocks.py`, `transformer.py` | embedding/output weight tying |

Default configuration is 12 layers, `d_model` 768, `seq_len` 1024.

Training uses a warmup-stable-decay LR schedule (`wsd_lambda` in `train/trainer.py`):
linear warmup, a stable plateau, then linear decay over the final 10% of steps — with
gradient-norm clipping at 1.0 and atomic checkpointing.

## Correctness gates

Bugs in a pretraining stack are expensive precisely because training still *looks* like
it works — the loss goes down and the model is quietly broken. Two gates must pass
before any real compute is spent:

1. **Step-0 loss** — at initialization, cross-entropy must equal `ln(vocab_size)` ≈ 10.80
   for the 49,152-token vocabulary. This one number catches bad initialization scale,
   broken weight tying, and logit/label reshape bugs.
2. **Overfit-tiny** — drive the loss to near zero on a handful of batches. Proves the
   optimizer, masking, and gradient path actually learn.

## Tokenizer

`ablgpt/tokenizer/` — byte-level BPE trained from scratch to a 49,152-token vocabulary,
with multiprocessing corpus frequency counting and HuggingFace-compatible
`vocab.json` / `merges.txt` output, so the artifacts work with standard tooling even
though the trainer does not depend on it.

The training corpus is sampled by **byte-fraction targets rather than document counts**
(`ablgpt/data/build_tokenizer_corpus.py`) — sampling by document count silently
over-weights sources whose documents happen to be short.

## Data pipeline

`ablgpt/data/` — streams HuggingFace datasets into an indexed shard format, then serves
blended batches to the trainer:

- **Sharding** (`tokenizer_shards.py`, `shard_io.py`) — tokenize and write memory-mapped
  shards alongside a separate index, resumable across interrupted streams.
- **Byte-weighted blending** (`data_loader.py`) — mix weights in `configs/data_mixes.yaml`
  are byte-fraction *targets*; the loader converts them into per-document sampling
  probabilities so the realized mix matches what was specified.
- **Named mixes** — `smoke` (single-source sanity), `ladder` (web-only, for scaling-law
  rungs and ablations), `ladder_plus_code` (the controlled ±code ablation), `flagship`,
  and `flagship_anneal`.

Keeping code and math out of the `ladder` mix is deliberate: those capabilities do not
emerge at proxy scale, so including them adds variance without adding signal.

## Layout

```
ablgpt/
  model/       transformer, GQA attention, RoPE, RMSNorm, SwiGLU
  tokenizer/   byte-level BPE implementation and special tokens
  data/        sharding, indexed shard IO, blended streaming loader
  train/       config-driven trainer, WSD schedule, checkpointing
  config.py    typed config schema for the YAML above
configs/
  data_mixes.yaml, shard_plan.yaml, train/*.yaml
```

## Running

```bash
uv sync

# correctness gates (config name only, no .yaml)
uv run python -m ablgpt.train.trainer gate_step0    --gate step0
uv run python -m ablgpt.train.trainer gate_overfit  --gate overfit

# real-data smoke run
uv run python -m ablgpt.train.trainer smoke
```

`configs/train/*_cuda.yaml` are the same runs targeted at CUDA rather than Apple MPS.

Design decisions and their rationale are logged in `history.md`.
