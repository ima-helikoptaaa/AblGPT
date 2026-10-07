import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from ablgpt.config import TrainConfig, load_train_config
from ablgpt.data.data_loader import build_dataset
from ablgpt.model.transformer import TransformerLM


def load_model(cfg):
    model = TransformerLM(
        cfg.vocab_size,
        cfg.max_seq,
        cfg.n_layers,
        cfg.d_model,
        cfg.d_ff,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        TrainConfig().torch_dtype(),
        cfg.device,
    )
    model = model.to(cfg.device)
    return model


def autocast_context(cfg):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=cfg.device.startswith("cuda") and cfg.dtype == "bfloat16",
    )


def warmup_decay(cfg, step):
    if step < cfg.warmup_steps:
        return step / (cfg.warmup_steps)
    elif step < cfg.n_steps * 0.9:
        return 1.0
    else:
        return (cfg.n_steps - step) / (cfg.n_steps * 0.1)


def save_model_checkpoint(cfg, model, optim, scheduler, step, train_seen, val_seen):
    out = Path(cfg.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "completed_steps": step,
        "train_batches_seen": train_seen,
        "val_batches_seen": val_seen,
        "config": asdict(cfg),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
    if cfg.device == "mps":
        checkpoint["mps_rng"] = torch.mps.get_rng_state()
    path = out / f"model_checkpoint_{step}.pth"
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    print(f"Training checkpoint saved at step {step}")


def restore_iterator(loader, batches_seen):
    # The current dataset is deterministic and shuffle=False. Skip only the
    # remainder of the current pass; prefetched batches were not consumed.
    iterator = iter(loader)
    for _ in range(batches_seen % len(loader)):
        next(iterator)
    return iterator


def check_step0(cfg):
    torch.manual_seed(42)
    dataset = build_dataset(cfg.mix_name, cfg.seq_len, "train")
    loader = DataLoader(dataset, cfg.batch_size, False, num_workers=2, pin_memory=True)
    model = load_model(cfg)
    model.eval()
    X, y = next(iter(loader))
    X = X.to(cfg.device, non_blocking=True)
    y = y.to(cfg.device, non_blocking=True)
    with torch.no_grad(), autocast_context(cfg):
        logits = model(X)
        loss = F.cross_entropy(
            logits.view(-1, cfg.vocab_size).float(), y.view(-1)
        ).item()

    expected = math.log(cfg.vocab_size)
    print(f"Step-0 loss: {loss:.4f}; expected near {expected:.4f}")

    # A sanity threshold, not proof that every component is correct.
    assert math.isfinite(loss), "Loss is not finite"
    assert abs(loss - expected) < 0.2, "Check initialization and loss computation"


def overfit_tiny(cfg):
    torch.manual_seed(42)
    model = load_model(cfg)
    model.train()
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)

    dataset = build_dataset(cfg.mix_name, cfg.seq_len, "train")
    loader = DataLoader(dataset, cfg.batch_size, False, num_workers=2, pin_memory=True)

    X, y = next(iter(loader))

    X = X.to(cfg.device, non_blocking=True)
    y = y.to(cfg.device, non_blocking=True)

    for step in range(cfg.n_steps):
        optim.zero_grad()

        with autocast_context(cfg):
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size).float(), y.view(-1))
        assert torch.isfinite(loss).item(), "Loss is not finite"

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}: loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad(), autocast_context(cfg):
        logits = model(X)
        final_loss = F.cross_entropy(
            logits.view(-1, cfg.vocab_size).float(), y.view(-1)
        ).item()

    print(f"Final fixed-batch loss: {final_loss:.4f}")
    assert final_loss < 0.1, "Fixed batch has not been overfit yet"


def train(cfg, resume=None):
    ds_train = build_dataset(cfg.mix_name, cfg.seq_len, "train")
    ds_val = build_dataset(cfg.mix_name, cfg.seq_len, "val")

    dl_train = DataLoader(
        ds_train, cfg.batch_size, False, num_workers=2, pin_memory=True
    )
    dl_val = DataLoader(ds_val, cfg.batch_size, False, num_workers=2, pin_memory=True)

    model = load_model(cfg)
    optim = AdamW(model.parameters(), lr=cfg.lr)

    scheduler = LambdaLR(
        optim,
        lr_lambda=lambda step: warmup_decay(cfg, step),
    )

    start_step = 0
    train_seen = val_seen = 0
    checkpoint = None
    if resume is not None:
        path = Path(cfg.checkpoint_dir) / f"model_checkpoint_{resume}.pth"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "completed_steps" not in checkpoint:
            raise ValueError(
                "This is a weights-only checkpoint; full resume requires a new training checkpoint."
            )
        saved_config = checkpoint["config"]
        for key, value in asdict(cfg).items():
            if (
                key not in {"device", "checkpoint_dir"}
                and saved_config.get(key) != value
            ):
                raise ValueError(f"Cannot resume with changed config field: {key}")
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["completed_steps"]
        train_seen = checkpoint["train_batches_seen"]
        val_seen = checkpoint["val_batches_seen"]

    train_iter = restore_iterator(dl_train, train_seen)
    val_iter = restore_iterator(dl_val, val_seen)

    # Iterator creation and skipping can consume RNG; restore it afterwards.
    if checkpoint is not None:
        torch.set_rng_state(checkpoint["torch_rng"])
        if "cuda_rng" in checkpoint and cfg.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if "mps_rng" in checkpoint and cfg.device == "mps":
            torch.mps.set_rng_state(checkpoint["mps_rng"])
        print(f"Resumed training after {start_step} optimizer steps")

    for step in range(start_step, cfg.n_steps):
        optim.zero_grad()
        training_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            try:
                X_tr, y_tr = next(train_iter)
            except StopIteration:
                train_iter = iter(dl_train)
                X_tr, y_tr = next(train_iter)

            train_seen += 1
            X_tr = X_tr.to(cfg.device, non_blocking=True)
            y_tr = y_tr.to(cfg.device, non_blocking=True)

            with autocast_context(cfg):
                logits = model(X_tr)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size).float(),
                    y_tr.reshape(-1),
                )

            (loss / cfg.grad_accum_steps).backward()
            training_loss += loss.item() / cfg.grad_accum_steps

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        print(f"training loss, step {step} = {training_loss:.4f}")

        model.eval()
        with torch.no_grad():
            if step > 0 and step % 500 == 0:
                loss = 0
                for _ in range(cfg.n_val_batches):
                    try:
                        X_val, y_val = next(val_iter)
                    except StopIteration:
                        val_iter = iter(dl_val)
                        X_val, y_val = next(val_iter)
                    val_seen += 1
                    X_val = X_val.to(cfg.device, non_blocking=True)
                    y_val = y_val.to(cfg.device, non_blocking=True)
                    with autocast_context(cfg):
                        logits = model(X_val)
                        loss += F.cross_entropy(
                            logits.view(-1, cfg.vocab_size).float(), y_val.view(-1)
                        )
                print(f"validation loss, step {step} = {loss/cfg.n_val_batches:.4f}")
            if (step + 1) % cfg.checkpoint_every == 0:
                save_model_checkpoint(
                    cfg, model, optim, scheduler, step + 1, train_seen, val_seen
                )

        model.train()
    save_model_checkpoint(
        cfg, model, optim, scheduler, cfg.n_steps, train_seen, val_seen
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="smoke")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gate", choices=["step0", "overfit"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_train_config(args.config)
    if args.gate == "step0":
        check_step0(cfg)
    elif args.gate == "overfit":
        overfit_tiny(cfg)
    else:
        train(cfg, args.resume)


if __name__ == "__main__":
    main()
