"""Inference script: text prompt -> MP4 video."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from AI.export import export_video  # noqa: E402
from AI.model import TextToVideoModel  # noqa: E402
from AI.scheduler import DDIMScheduler  # noqa: E402
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


def resolve_profile(model_cfg, inference_cfg, profile, auto_profile):
    if auto_profile:
        from AI.resource_detector import detect_profile
        profile = detect_profile()
    profile = profile or "low"
    m = dict(model_cfg.get("defaults", {}))
    m.update(model_cfg["profiles"].get(profile, {}))
    i = dict(inference_cfg.get("defaults", {}))
    i.update(inference_cfg["profiles"].get(profile, {}))
    return m, i, profile


def load_tokenizer(checkpoint_dir: Path, max_length: int) -> SimpleTokenizer:
    vocab_path = checkpoint_dir / "tokenizer.json"
    if vocab_path.exists():
        return SimpleTokenizer.load(vocab_path)
    return build_default_tokenizer(max_length=max_length)


@torch.no_grad()
def generate_video(prompt: str, model: TextToVideoModel, tokenizer: SimpleTokenizer,
                   scheduler: DDIMScheduler, image_size: int, num_frames: int,
                   num_inference_steps: int, cfg_scale: float, device: torch.device,
                   seed: int = 42,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> torch.Tensor:
    set_global_seed(seed)
    model.eval()
    cfg_scale = max(1.0, float(cfg_scale))

    enc = tokenizer.encode(prompt)
    ids = torch.tensor([enc], dtype=torch.long, device=device)
    mask = torch.tensor([tokenizer.attention_mask(enc)], dtype=torch.long, device=device)

    uncond_enc = tokenizer.encode("")
    uncond_ids = torch.tensor([uncond_enc], dtype=torch.long, device=device)
    uncond_mask = torch.tensor([tokenizer.attention_mask(uncond_enc)], dtype=torch.long, device=device)

    h = image_size // model.vae_downsample
    w = image_size // model.vae_downsample
    latent_shape = (1, model.latent_channels, num_frames, h, w)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    latents = torch.randn(latent_shape, generator=gen, device=device)

    num_inference_steps = max(1, min(int(num_inference_steps), scheduler.num_train_timesteps))
    ts = scheduler._make_timestep_list(num_inference_steps)
    total = len(ts)

    with tqdm(total=total, desc="Sampling", unit="step") as pbar:
        for i, t in enumerate(ts.tolist()):
            prev_t = int(ts[i + 1]) if i + 1 < len(ts) else -1
            t_batch = torch.full((latents.shape[0],), t, dtype=torch.long, device=device)

            if cfg_scale > 1.0:
                ctx_cond = model.encode_text(ids, mask)
                ctx_uncond = model.encode_text(uncond_ids, uncond_mask)
                noise_cond = model.unet(latents, t_batch, ctx_cond)
                noise_uncond = model.unet(latents, t_batch, ctx_uncond)
                noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                ctx = model.encode_text(ids, mask)
                noise = model.unet(latents, t_batch, ctx)

            latents = scheduler.step(noise, t, latents, prev_timestep=prev_t, generator=gen)
            pbar.update(1)
            if progress_callback is not None:
                progress_callback(i + 1, total)

    frames = model.decode_video(latents)
    return torch.clamp(frames, -1.0, 1.0)


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Text-to-video inference")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--auto-profile", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    cfg_dir = _ROOT / "config"
    model_cfg = load_yaml(cfg_dir / "model.yaml")
    inference_cfg = load_yaml(cfg_dir / "inference.yaml")
    m_cfg, i_cfg, profile = resolve_profile(model_cfg, inference_cfg, args.profile, args.auto_profile)

    image_size = args.image_size if args.image_size is not None else m_cfg["image_size"]
    num_frames = args.num_frames if args.num_frames is not None else m_cfg["num_frames"]
    steps = args.num_inference_steps if args.num_inference_steps is not None else i_cfg["num_inference_steps"]
    cfg = args.cfg_scale if args.cfg_scale is not None else i_cfg["cfg_scale"]
    fps = args.fps if args.fps is not None else i_cfg["fps"]
    output = Path(args.output) if args.output else (_ROOT / i_cfg["output_path"])
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_dir = _ROOT / "AI" / "checkpoints"
    tokenizer = load_tokenizer(checkpoint_dir, m_cfg["text_max_length"])

    model = TextToVideoModel(
        vocab_size=tokenizer.vocab_size, image_size=image_size, num_frames=num_frames,
        latent_channels=m_cfg["latent_channels"], base_channels=m_cfg["base_channels"],
        channel_multipliers=tuple(m_cfg["channel_multipliers"]),
        num_res_blocks=m_cfg["num_res_blocks"], text_embed_dim=m_cfg["text_embed_dim"],
        text_max_length=m_cfg["text_max_length"], transformer_layers=m_cfg["transformer_layers"],
        transformer_heads=m_cfg["transformer_heads"], attention_heads=m_cfg["attention_heads"],
        dropout=m_cfg["dropout"], vae_downsample=model_cfg["vae"]["downsample_factor"],
    ).to(device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (checkpoint_dir / "latest.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: checkpoint not found at {ckpt_path}; using random weights.")

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              device=device, eta=float(i_cfg.get("eta", 0.0)))

    print(f"Profile: {profile} | device={device} | size={image_size} | frames={num_frames}")
    frames = generate_video(prompt=args.prompt, model=model, tokenizer=tokenizer,
                            scheduler=scheduler, image_size=image_size, num_frames=num_frames,
                            num_inference_steps=steps, cfg_scale=cfg, device=device, seed=args.seed)

    arr = frames[0].detach().cpu().numpy()
    export_video(arr, output, fps=fps, bitrate=i_cfg.get("bitrate", "4M"))
    print(f"Video saved to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
