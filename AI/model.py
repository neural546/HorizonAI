"""Latent video diffusion model with text conditioning.

Pipeline:
    prompt -> tokenizer -> transformer text encoder -> cross-attention
    -> 3D U-Net (video) with temporal + spatial attention
    -> latent video -> tiny VAE decoder -> frames
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TextTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int,
                 num_heads: int, max_length: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, L = input_ids.shape
        L = min(L, self.max_length)
        x = self.embed(input_ids[:, :L]) + self.pos_embed[:, :L]
        if attention_mask is not None:
            mask = attention_mask[:, :L].bool()
            x = self.encoder(x, src_key_padding_mask=~mask)
        else:
            x = self.encoder(x)
        return self.norm(x)


class TimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.dim))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(context_dim, dim)
        self.v = nn.Linear(context_dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.out(out)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class TemporalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        residual = x
        y = self.norm(x)
        y = y.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        return residual + y


class SpatialCrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross = CrossAttention(dim, context_dim, num_heads)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        residual = x
        y = self.norm(x)
        y = y.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        y, _ = self.self_attn(y, y, y, need_weights=False)
        y = y.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        y = y.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        ctx = context[:, None].expand(-1, t, -1, -1).reshape(b * t, context.shape[1], -1)
        y = self.cross(y, ctx)
        y = y.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return residual + y


class VideoUNet(nn.Module):
    """3D video U-Net with symmetric skip connections (one skip per level)."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        context_dim: int = 256,
        time_dim: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.num_res_blocks = num_res_blocks
        self.time_embed = TimeEmbed(time_dim)
        self.time_proj = nn.Linear(time_dim * 4, time_dim * 4)
        self.in_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        mults = list(channel_multipliers)
        self.n_levels = len(mults)

        self.down_res = nn.ModuleList()
        self.down_temporal = nn.ModuleList()
        self.down_cross = nn.ModuleList()
        self.down_sample = nn.ModuleList()

        cur = base_channels
        level_out_channels = []
        for level, mult in enumerate(mults):
            out_ch = base_channels * mult
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_blocks.append(ResBlock3D(cur, out_ch, time_dim * 4, dropout))
                cur = out_ch
            self.down_res.append(res_blocks)
            self.down_temporal.append(TemporalAttention(cur, attention_heads))
            self.down_cross.append(SpatialCrossAttention(cur, context_dim, attention_heads))
            level_out_channels.append(cur)
            if level < self.n_levels - 1:
                self.down_sample.append(nn.Conv3d(cur, cur, 3, stride=2, padding=1))
            else:
                self.down_sample.append(nn.Identity())

        self.mid_res1 = ResBlock3D(cur, cur, time_dim * 4, dropout)
        self.mid_temporal = TemporalAttention(cur, attention_heads)
        self.mid_cross = SpatialCrossAttention(cur, context_dim, attention_heads)
        self.mid_res2 = ResBlock3D(cur, cur, time_dim * 4, dropout)

        self.up_res = nn.ModuleList()
        self.up_temporal = nn.ModuleList()
        self.up_cross = nn.ModuleList()
        self.up_sample = nn.ModuleList()

        for level in range(self.n_levels - 1, -1, -1):
            out_ch = base_channels * mults[level]
            skip_ch = level_out_channels[level]
            res_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = (cur + skip_ch) if j == 0 else out_ch
                res_blocks.append(ResBlock3D(in_ch, out_ch, time_dim * 4, dropout))
                cur = out_ch
            self.up_res.append(res_blocks)
            self.up_temporal.append(TemporalAttention(cur, attention_heads))
            self.up_cross.append(SpatialCrossAttention(cur, context_dim, attention_heads))
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

    def _project_context(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] == self.context_dim:
            return context
        if context.shape[-1] < self.context_dim:
            pad = torch.zeros(context.shape[0], context.shape[1],
                              self.context_dim - context.shape[-1],
                              device=context.device, dtype=context.dtype)
            return torch.cat([context, pad], dim=-1)
        return context[..., : self.context_dim]

    def _res(self, rb: nn.Module, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(rb, h, t_emb, use_reentrant=False)
        return rb(h, t_emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = self._project_context(context)
        t_emb = self.time_embed(t)
        t_emb = F.silu(self.time_proj(t_emb))

        h = self.in_conv(x)
        skips = []

        for level in range(self.n_levels):
            for rb in self.down_res[level]:
                h = self._res(rb, h, t_emb)
            h = self.down_temporal[level](h)
            h = self.down_cross[level](h, context)
            skips.append(h)
            h = self.down_sample[level](h)

        h = self._res(self.mid_res1, h, t_emb)
        h = self.mid_temporal(h)
        h = self.mid_cross(h, context)
        h = self._res(self.mid_res2, h, t_emb)

        for level in range(self.n_levels):
            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=h.shape[2:], mode="nearest")
            for j, rb in enumerate(self.up_res[level]):
                if j == 0:
                    h = self._res(rb, torch.cat([h, skip], dim=1), t_emb)
                else:
                    h = self._res(rb, h, t_emb)
            h = self.up_temporal[level](h)
            h = self.up_cross[level](h, context)
            h = self.up_sample[level](h)

        return self.out_conv(F.silu(self.out_norm(h)))


class TinyVAEEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, latent_ch: int = 4, base_ch: int = 32,
                 downsample: int = 8):
        super().__init__()
        assert downsample in (2, 4, 8, 16)
        n_down = int(math.log2(downsample))
        layers = [nn.Conv3d(in_ch, base_ch, 3, padding=1), nn.SiLU(),
                  nn.Conv3d(base_ch, base_ch, 3, padding=1)]
        ch = base_ch
        for _ in range(n_down):
            layers += [nn.Conv3d(ch, ch * 2, 3, stride=2, padding=1), nn.SiLU(),
                       nn.Conv3d(ch * 2, ch * 2, 3, padding=1), nn.SiLU()]
            ch *= 2
        layers += [nn.Conv3d(ch, latent_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.latent_ch = latent_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyVAEDecoder(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 3, base_ch: int = 32,
                 upsample: int = 8):
        super().__init__()
        n_up = int(math.log2(upsample))
        ch = base_ch * (2 ** n_up)
        layers = [nn.Conv3d(in_ch, ch, 3, padding=1), nn.SiLU()]
        for _ in range(n_up):
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv3d(ch, ch // 2, 3, padding=1), nn.SiLU()]
            ch //= 2
        layers += [nn.Conv3d(ch, out_ch, 3, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TextToVideoModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        image_size: int = 64,
        num_frames: int = 8,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        text_embed_dim: int = 256,
        text_max_length: int = 32,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
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
        self.unet = VideoUNet(
            in_channels=latent_channels, out_channels=latent_channels,
            base_channels=base_channels, channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks, context_dim=text_embed_dim,
            time_dim=256, attention_heads=attention_heads, dropout=dropout,
        )
        self.vae_encoder = TinyVAEEncoder(
            in_ch=3, latent_ch=latent_channels, base_ch=32, downsample=vae_downsample
        )
        self.vae_decoder = TinyVAEDecoder(
            in_ch=latent_channels, out_ch=3, base_ch=32, upsample=vae_downsample
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
        context = self.encode_text(input_ids, attention_mask)
        return self.unet(noisy_latents, timesteps, context)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.unet.gradient_checkpointing = bool(enabled)
