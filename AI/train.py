"""Training script: DDPM noise-prediction with text conditioning."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from AI.collate import collate_fn, random_prompt_dropout  # noqa: E402
from AI.dataset import VideoTextDataset  # noqa: E402
from AI.model import TextToVideoModel  # noqa: E402
from AI.scheduler import DDPMScheduler  # noqa: E402
from AI.tokenizer import SimpleTokenizer, build_default_tokenizer  # noqa: E402


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_profile(model_cfg, training_cfg, profile, auto_profile):
    if auto_profile:
        from AI.resource_detector import detect_profile
        profile = detect_profile()
    profile = profile or "low"
    m = dict(model_cfg.get("defaults", {}))
    m.update(model_cfg["profiles"].get(profile, {}))
    t = dict(training_cfg.get("defaults", {}))
    t.update(training_cfg["profiles"].get(profile, {}))
    return m, t, profile


def ensure_tokenizer(dataset_root: Path, checkpoint_dir: Path, max_length: int) -> SimpleTokenizer:
    vocab_path = checkpoint_dir / "tokenizer.json"
    if vocab_path.exists():
        return SimpleTokenizer.load(vocab_path)
    prompts = []
    if dataset_root.exists():
        for d in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
            p = d / "prompt.txt"
            if p.exists():
                prompts.append(p.read_text(encoding="utf-8").strip())
    tok = build_default_tokenizer(max_length=max_length)
    if prompts:
        tok.build_vocab(prompts)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tok.save(vocab_path)
    return tok


def _save_checkpoint(path: Path, model, optimizer, epoch: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "global_step": step}, path)


def train(args: argparse.Namespace, epochs_override: Optional[int] = None) -> int:
    cfg_dir = _ROOT / "config"
    model_cfg = load_yaml(cfg_dir / "model.yaml")
    training_cfg = load_yaml(cfg_dir / "training.yaml")
    m_cfg, t_cfg, profile = resolve_profile(model_cfg, training_cfg, args.profile, args.auto_profile)
    if epochs_override is not None:
        t_cfg["epochs"] = int(epochs_override)

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.data_dir) if args.data_dir else (_ROOT / "datasets" / "train")
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (_ROOT / t_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = _ROOT / t_cfg["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Profile: {profile} | device={device}")
    print(f"Dataset: {dataset_root}")

    tokenizer = ensure_tokenizer(dataset_root, checkpoint_dir, m_cfg["text_max_length"])
    print(f"Vocab size: {tokenizer.vocab_size}")

    dataset = VideoTextDataset(root=dataset_root, image_size=m_cfg["image_size"],
                               num_frames=m_cfg["num_frames"], tokenizer=tokenizer,
                               max_length=m_cfg["text_max_length"], validate=False)
    if len(dataset) == 0:
        print("ERROR: no valid samples in dataset.")
        return 1
    print(f"Samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=t_cfg["batch_size"], shuffle=True,
                        num_workers=t_cfg.get("num_workers", 0), collate_fn=collate_fn,
                        pin_memory=(device.type == "cuda"), drop_last=False)

    model = TextToVideoModel(
        vocab_size=tokenizer.vocab_size, image_size=m_cfg["image_size"],
        num_frames=m_cfg["num_frames"], latent_channels=m_cfg["latent_channels"],
        base_channels=m_cfg["base_channels"],
        channel_multipliers=tuple(m_cfg["channel_multipliers"]),
        num_res_blocks=m_cfg["num_res_blocks"], text_embed_dim=m_cfg["text_embed_dim"],
        text_max_length=m_cfg["text_max_length"], transformer_layers=m_cfg["transformer_layers"],
        transformer_heads=m_cfg["transformer_heads"], attention_heads=m_cfg["attention_heads"],
        dropout=m_cfg["dropout"], vae_downsample=model_cfg["vae"]["downsample_factor"],
    ).to(device)

    if t_cfg.get("use_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
        print("Gradient checkpointing enabled.")

    start_epoch = 0
    global_step = 0
    resume_path = Path(args.resume) if args.resume else None
    if resume_path and resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)
        print(f"Resumed from {resume_path} (epoch {start_epoch})")

    scheduler = DDPMScheduler(num_train_timesteps=t_cfg["num_train_timesteps"],
                              beta_schedule=t_cfg["scheduler_type"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t_cfg["learning_rate"]),
                                  betas=tuple(t_cfg["betas"]),
                                  weight_decay=float(t_cfg["weight_decay"]))
    if resume_path and resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
    except Exception:
        writer = None

    use_amp = bool(t_cfg.get("mixed_precision")) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    pad_id = tokenizer.pad_id
    prompt_dropout = float(t_cfg.get("prompt_dropout_prob", 0.1))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    accum = int(t_cfg.get("gradient_accumulation_steps", 1))

    print("Starting training...")
    for epoch in range(start_epoch, int(t_cfg["epochs"])):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{t_cfg['epochs']}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            frames = batch["frames"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            gen = torch.Generator(device="cpu")
            gen.manual_seed(args.seed + global_step)
            drop_mask = random_prompt_dropout(input_ids, attn_mask, pad_id, prompt_dropout, generator=gen)
            if drop_mask.any():
                drop_idx = drop_mask.bool().to(device)
                empty_enc = tokenizer.encode("")
                empty_ids = torch.tensor(empty_enc, device=device, dtype=input_ids.dtype)
                empty_mask = torch.tensor(tokenizer.attention_mask(empty_enc),
                                          device=device, dtype=attn_mask.dtype)
                input_ids = input_ids.clone()
                attn_mask = attn_mask.clone()
                input_ids[drop_idx] = empty_ids
                attn_mask[drop_idx] = empty_mask

            with torch.cuda.amp.autocast(enabled=use_amp):
                latents = model.encode_video(frames)
                noise = torch.randn_like(latents)
                t = scheduler.sample_timesteps(latents.shape[0], device=device)
                noisy = scheduler.add_noise(latents, noise, t)
                pred = model(noisy, t, input_ids, attn_mask)
                loss = F.mse_loss(pred, noise) / accum

            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            n_batches += 1
            epoch_loss += loss.item() * accum
            if global_step % t_cfg["log_every"] == 0:
                pbar.set_postfix({"loss": f"{loss.item() * accum:.4f}", "step": global_step})
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item() * accum, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            if global_step % t_cfg["save_every"] == 0:
                _save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, global_step)
                _save_checkpoint(checkpoint_dir / f"step_{global_step:06d}.pt", model, optimizer, epoch, global_step)

        avg = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch+1} average loss: {avg:.4f}")

    _save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, int(t_cfg["epochs"]) - 1, global_step)
    if writer is not None:
        writer.close()
    print("Training complete.")
    return 0


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Train text-to-video model")
    parser.add_argument("--profile", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--auto-profile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)
    return train(args, epochs_override=args.epochs)


if __name__ == "__main__":
    sys.exit(_main())
