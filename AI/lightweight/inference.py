"""Lightweight MVP inference: prompt -> MP4 video."""

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

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from AI.export import export_video  # noqa: E402
from AI.lightweight.model import LightTextToVideoModel  # noqa: E402
from AI.lightweight.tokenizer import build_mvp_tokenizer  # noqa: E402
from AI.scheduler import DDIMScheduler  # noqa: E402
from AI.tokenizer import SimpleTokenizer  # noqa: E402


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mvp_model(checkpoint: Optional[Path], device: torch.device,
                   num_frames: int = 8):
    ckpt_dir = _ROOT / "AI" / "checkpoints"
    tok_path = ckpt_dir / "tokenizer.json"
    tokenizer = SimpleTokenizer.load(tok_path) if tok_path.exists() else build_mvp_tokenizer(max_length=16)
    model = LightTextToVideoModel(vocab_size=tokenizer.vocab_size,
                                  image_size=64, num_frames=num_frames).to(device)
    candidate = ckpt_dir / "lightweight_latest.pt" if checkpoint is None else Path(checkpoint)
    if candidate.exists():
        state = torch.load(candidate, map_location=device)
        model.load_state_dict(state["model"])
        print(f"[MVP] loaded checkpoint {candidate}")
    else:
        print(f"[MVP] WARNING: no checkpoint at {candidate}; using random weights")
    return model, tokenizer


@torch.no_grad()
def generate_mvp(prompt: str, cfg_scale: float = 7.5, num_inference_steps: int = 20,
                 seed: int = 42, output: Optional[Path] = None,
                 device: Optional[torch.device] = None,
                 num_frames: int = 8, image_size: int = 64, fps: int = 8) -> str:
    set_global_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_mvp_model(None, device, num_frames=num_frames)
    model.eval()

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              device=device, eta=0.0)

    enc = tokenizer.encode(prompt)
    ids = torch.tensor([enc], dtype=torch.long, device=device)
    mask = torch.tensor([tokenizer.attention_mask(enc)], dtype=torch.long, device=device)
    uncond = tokenizer.encode("")
    u_ids = torch.tensor([uncond], dtype=torch.long, device=device)
    u_mask = torch.tensor([tokenizer.attention_mask(uncond)], dtype=torch.long, device=device)

    h = image_size // model.vae_downsample
    w = image_size // model.vae_downsample
    shape = (1, model.latent_channels, num_frames, h, w)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    latents = torch.randn(shape, generator=gen, device=device)

    num_inference_steps = max(1, min(int(num_inference_steps), scheduler.num_train_timesteps))
    ts = scheduler._make_timestep_list(num_inference_steps)
    t0 = time.time()
    for i, t in enumerate(ts.tolist()):
        prev_t = int(ts[i + 1]) if i + 1 < len(ts) else -1
        tb = torch.full((latents.shape[0],), t, dtype=torch.long, device=device)
        ctx_c = model.encode_text(ids, mask)
        ctx_u = model.encode_text(u_ids, u_mask)
        noise_c = model.unet(latents, tb, ctx_c)
        noise_u = model.unet(latents, tb, ctx_u)
        noise = noise_u + cfg_scale * (noise_c - noise_u)
        latents = scheduler.step(noise, t, latents, prev_timestep=prev_t, generator=gen)

    frames = model.decode_video(latents)
    frames = torch.clamp(frames, -1.0, 1.0)
    print(f"[MVP] inference took {time.time() - t0:.2f}s")

    out_path = Path(output) if output else (_ROOT / "static" / "videos" / "mvp_output.mp4")
    arr = frames[0].detach().cpu().numpy()
    return export_video(arr, out_path, fps=fps, bitrate="2M")


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="MVP lightweight inference")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)
    path = generate_mvp(prompt=args.prompt, cfg_scale=args.cfg_scale,
                        num_inference_steps=args.num_inference_steps, seed=args.seed,
                        output=Path(args.output) if args.output else None,
                        num_frames=args.num_frames, image_size=args.image_size, fps=args.fps)
    print(f"[MVP] video saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
