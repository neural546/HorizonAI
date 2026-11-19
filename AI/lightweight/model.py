"""Lightweight video diffusion model for the CPU / 4GB GPU MVP."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from AI.model import (  # noqa: E402
    ResBlock3D,
    SpatialCrossAttention,
    TemporalAttention,
    TextTransformerEncoder,
    TimeEmbed,
    TinyVAEDecoder,
    TinyVAEEncoder,
)


class LightVideoUNet(nn.Module):
    """Compact 3-level video U-Net with one skip per level."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 32,
        channel_multipliers: tuple = (1, 2, 4),
        context_dim: int = 128,
        time_dim: int = 128,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_embed = TimeEmbed(time_dim)
        self.time_proj = nn.Linear(time_dim * 4, time_dim * 4)
        self.in_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        mults = list(channel_multipliers)
        self.n_levels = len(mults)

        self.down_res = nn.ModuleList()
        self.down_cross = nn.ModuleList()
        self.down_temporal = nn.ModuleList()
        self.down_sample = nn.ModuleList()

        cur = base_channels
        level_out_channels = []
        for level, mult in enumerate(mults):
            out_ch = base_channels * mult
            self.down_res.append(ResBlock3D(cur, out_ch, time_dim * 4, dropout))
            cur = out_ch
            self.down_cross.append(SpatialCrossAttention(cur, context_dim, attention_heads))
            self.down_temporal.append(TemporalAttention(cur, attention_heads))
            level_out_channels.append(cur)
            if level < self.n_levels - 1:
                self.down_sample.append(nn.Conv3d(cur, cur, 3, stride=2, padding=1))
            else:
                self.down_sample.append(nn.Identity())

        self.mid1 = ResBlock3D(cur, cur, time_dim * 4, dropout)
        self.mid_spatial = SpatialCrossAttention(cur, context_dim, attention_heads)
        self.mid_temporal = TemporalAttention(cur, attention_heads)
        self.mid2 = ResBlock3D(cur, cur, time_dim * 4, dropout)

        self.up_res = nn.ModuleList()
        self.up_cross = nn.ModuleList()
        self.up_temporal = nn.ModuleList()
        self.up_sample = nn.ModuleList()

        for level in range(self.n_levels - 1, -1, -1):
            out_ch = base_channels * mults[level]
            skip_ch = level_out_channels[level]
            in_ch = cur + skip_ch
            self.up_res.append(ResBlock3D(in_ch, out_ch, time_dim * 4, dropout))
            cur = out_ch
            self.up_cross.append(SpatialCrossAttention(cur, context_dim, attention_heads))
            self.up_temporal.append(TemporalAttention(cur, attention_heads))
            if level > 0:
                self.up_sample.append(
                    nn.ConvTranspose3d(cur, cur, 3, stride=2, padding=1, output_padding=1)
                )
            else:
                self.up_sample.append(nn.Identity())

        self.out_norm = nn.GroupNorm(8, cur)
        self.out_conv = nn.Conv3d(cur, out_channels, 3, padding=1)
        self.context_dim = context_dim
        self.gradient_checkpointing = False

    def _project_context(self, ctx: torch.Tensor) -> torch.Tensor:
        if ctx.shape[-1] == self.context_dim:
            return ctx
        if ctx.shape[-1] < self.context_dim:
            pad = torch.zeros(ctx.shape[0], ctx.shape[1],
                              self.context_dim - ctx.shape[-1],
                              device=ctx.device, dtype=ctx.dtype)
            return torch.cat([ctx, pad], dim=-1)
        return ctx[..., : self.context_dim]

    def _res(self, rb, h, t_emb):
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(rb, h, t_emb, use_reentrant=False)
        return rb(h, t_emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = self._project_context(context)
        t_emb = F.silu(self.time_proj(self.time_embed(t)))

        h = self.in_conv(x)
        skips = []

        for level in range(self.n_levels):
            h = self._res(self.down_res[level], h, t_emb)
            h = self.down_cross[level](h, context)
            h = self.down_temporal[level](h)
            skips.append(h)
            h = self.down_sample[level](h)

        h = self._res(self.mid1, h, t_emb)
        h = self.mid_spatial(h, context)
        h = self.mid_temporal(h)
        h = self._res(self.mid2, h, t_emb)

        for level in range(self.n_levels):
            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=h.shape[2:], mode="nearest")
            h = self._res(self.up_res[level], torch.cat([h, skip], dim=1), t_emb)
            h = self.up_cross[level](h, context)
            h = self.up_temporal[level](h)
            h = self.up_sample[level](h)

        return self.out_conv(F.silu(self.out_norm(h)))


class LightTextToVideoModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_size: int = 64,
        num_frames: int = 8,
        latent_channels: int = 4,
        base_channels: int = 32,
        channel_multipliers: tuple = (1, 2, 4),
        text_embed_dim: int = 128,
        text_max_length: int = 16,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        attention_heads: int = 4,
        dropout: float = 0.1,
        vae_downsample: int = 8,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_frames = num_frames
        self.latent_channels = latent_channels
        self.vae_downsample = vae_downsample

        self.text_encoder = TextTransformerEncoder(
            vocab_size=vocab_size, embed_dim=text_embed_dim,
            num_layers=transformer_layers, num_heads=transformer_heads,
            max_length=text_max_length, dropout=dropout,
        )
        self.unet = LightVideoUNet(
            in_channels=latent_channels, out_channels=latent_channels,
            base_channels=base_channels, channel_multipliers=channel_multipliers,
            context_dim=text_embed_dim, time_dim=128,
            attention_heads=attention_heads, dropout=dropout,
        )
        self.vae_encoder = TinyVAEEncoder(
            in_ch=3, latent_ch=latent_channels, base_ch=16, downsample=vae_downsample
        )
        self.vae_decoder = TinyVAEDecoder(
            in_ch=latent_channels, out_ch=3, base_ch=16, upsample=vae_downsample
        )

    def encode_text(self, input_ids, attention_mask=None):
        return self.text_encoder(input_ids, attention_mask)

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        return self.vae_encoder(x)

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae_decoder(latents)
        return frames.permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, noisy_latents, timesteps, input_ids, attention_mask=None):
        ctx = self.encode_text(input_ids, attention_mask)
        return self.unet(noisy_latents, timesteps, ctx)
