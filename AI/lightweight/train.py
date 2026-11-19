"""Lightweight MVP training entry point (CPU / 4GB GPU)."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from AI.lightweight.model import LightTextToVideoModel  # noqa: E402
from AI.lightweight.tokenizer import build_mvp_tokenizer  # noqa: E402
from AI.scheduler import DDPMScheduler  # noqa: E402


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _synthetic_batch(batch_size: int, num_frames: int, image_size: int, device: torch.device):
    return torch.rand(batch_size, num_frames, 3, image_size, image_size, device=device) * 2 - 1


def train_mvp(steps: int = 5, seed: int = 42) -> int:
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MVP] Device: {device}")

    tokenizer = build_mvp_tokenizer(max_length=16)
    model = LightTextToVideoModel(vocab_size=tokenizer.vocab_size,
                                  image_size=64, num_frames=8).to(device)
    model.train()

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    t0 = time.time()
    for step in range(steps):
        frames = _synthetic_batch(1, 8, 64, device)
        enc = tokenizer.encode("a dragon flying above snowy mountains")
        input_ids = torch.tensor([enc], dtype=torch.long, device=device)
        attn = torch.tensor([tokenizer.attention_mask(enc)], dtype=torch.long, device=device)

        latents = model.encode_video(frames)
        noise = torch.randn_like(latents)
        t = scheduler.sample_timesteps(latents.shape[0], device=device)
        noisy = scheduler.add_noise(latents, noise, t)
        pred = model(noisy, t, input_ids, attn)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"[MVP] step {step+1}/{steps} loss={loss.item():.4f}")

    elapsed = time.time() - t0
    print(f"[MVP] trained {steps} steps in {elapsed:.2f}s")

    ckpt_dir = _ROOT / "AI" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(ckpt_dir / "tokenizer.json")
    torch.save({"model": model.state_dict()}, ckpt_dir / "lightweight_latest.pt")
    print(f"[MVP] saved checkpoint to {ckpt_dir / 'lightweight_latest.pt'}")
    return 0


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="MVP lightweight training")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    return train_mvp(steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    sys.exit(_main())
