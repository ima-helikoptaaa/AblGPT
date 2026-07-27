import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from ablgpt.config import load_train_config
from ablgpt.data.data_loader import build_dataset
from ablgpt.model.transformer import TransformerLM


def wsd_lambda(step, warmup, total, decay_frac=0.1):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return step / max(1, warmup)          # linear warmup 0->1
    if step < decay_start:
        return 1.0                            # stable plateau
    # linear decay 1->0 over the last decay_frac of steps
    return max(0.0, (total - step) / max(1, total - decay_start))


def build_model(cfg):
    """Instantiate TransformerLM from a TrainConfig (dtype str -> torch.dtype)."""
    dtype = cfg.torch_dtype()
    model = TransformerLM(
        cfg.vocab_size, cfg.seq_len, cfg.n_layers, cfg.d_model, cfg.d_ff,
        cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, dtype, cfg.device,
    )
    model.to(cfg.device)
    return model


def _loss(model, x, y, vocab_size):
    logits = model(x)
    return F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1))


def save_checkpoint(cfg, model, optimizer, scheduler, step):
    """Atomic checkpoint of model + optimizer + scheduler + step.

    Writes to a .tmp then renames, so a crash mid-save never leaves a corrupt
    .pt. Data-stream position is NOT saved yet (bit-exact resume is a later
    task); this restores weights and optimizer/schedule state only.
    """
    out = Path(cfg.checkpoint_dir) / cfg.mix_name
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"step_{step}.pt.tmp"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        tmp,
    )
    tmp.replace(out / f"step_{step}.pt")


def train(cfg):
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda s: wsd_lambda(s, warmup=100, total=cfg.n_steps)
    )

    ds_train = build_dataset(cfg.mix_name, cfg.seq_len, "train")
    ds_val = build_dataset(cfg.mix_name, cfg.seq_len, "val")
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=cfg.batch_size)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=cfg.batch_size)

    for step, (x, y) in enumerate(dl_train):
        if step >= cfg.n_steps:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)

        loss = _loss(model, x, y, cfg.vocab_size)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"step {step}  loss {loss.item():.4f}")

        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for i, (x_val, y_val) in enumerate(dl_val):
                    if i >= cfg.n_val_batches:
                        break
                    x_val, y_val = x_val.to(cfg.device), y_val.to(cfg.device)
                    losses.append(_loss(model, x_val, y_val, cfg.vocab_size).item())
                print(f"step {step}  val loss {sum(losses) / len(losses):.4f}")
            model.train()

        if step > 0 and step % cfg.checkpoint_every == 0:
            save_checkpoint(cfg, model, optimizer, scheduler, step)
            print(f"step {step}  checkpoint saved")


def check_step0(cfg):
    """Gate 1: loss on a fresh model, ONE batch, BEFORE any optimizer step.
    Expect ~= ln(vocab_size). Far off => init / tying / loss-reshape bug."""
    import math

    model = build_model(cfg)
    model.eval()
    ds = build_dataset(cfg.mix_name, cfg.seq_len, "train")
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size)
    x, y = next(iter(dl))
    x, y = x.to(cfg.device), y.to(cfg.device)
    with torch.no_grad():
        loss = _loss(model, x, y, cfg.vocab_size).item()
    expected = math.log(cfg.vocab_size)
    print(f"step-0 loss {loss:.4f}  expected ~{expected:.4f} (ln vocab_size)")


def overfit_tiny(cfg):
    """Gate 2: loop ONE fixed batch; loss must drive toward ~0.
    Proves loader -> x/y shift -> model -> backward end-to-end."""
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ds = build_dataset(cfg.mix_name, cfg.seq_len, "train")
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size)
    x, y = next(iter(dl))  # ONE fixed batch, reused every step
    x, y = x.to(cfg.device), y.to(cfg.device)

    for step in range(cfg.n_steps):
        loss = _loss(model, x, y, cfg.vocab_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == cfg.n_steps - 1:
            print(f"step {step}  loss {loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config name under configs/train/ (no .yaml)")
    parser.add_argument(
        "--gate",
        choices=["step0", "overfit"],
        default=None,
        help="run a correctness gate instead of the full train loop",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    if args.gate == "step0":
        check_step0(cfg)
    elif args.gate == "overfit":
        overfit_tiny(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
