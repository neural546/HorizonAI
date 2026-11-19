import torch

from AI.model import (
    CrossAttention, TemporalAttention, TextToVideoModel, TextTransformerEncoder,
    TinyVAEDecoder, TinyVAEEncoder, VideoUNet, timestep_embedding,
)


def test_timestep_embedding():
    assert timestep_embedding(torch.tensor([0, 10, 999]), 64).shape == (3, 64)


def test_text_encoder():
    enc = TextTransformerEncoder(vocab_size=50, embed_dim=64, num_layers=2,
                                 num_heads=4, max_length=16)
    out = enc(torch.randint(0, 50, (2, 16)), torch.ones(2, 16, dtype=torch.long))
    assert out.shape == (2, 16, 64)


def test_cross_attention():
    ca = CrossAttention(dim=64, context_dim=128, num_heads=4)
    assert ca(torch.randn(2, 16, 64), torch.randn(2, 8, 128)).shape == (2, 16, 64)


def test_temporal_attention():
    ta = TemporalAttention(dim=32, num_heads=4)
    x = torch.randn(1, 32, 4, 8, 8)
    assert ta(x).shape == x.shape


def test_vae_roundtrip():
    enc = TinyVAEEncoder(in_ch=3, latent_ch=4, base_ch=8, downsample=8)
    dec = TinyVAEDecoder(in_ch=4, out_ch=3, base_ch=8, upsample=8)
    x = torch.randn(1, 3, 8, 32, 32)
    z = enc(x)
    assert z.shape[1] == 4 and z.shape[2:] == (8, 4, 4)
    assert dec(z).shape == x.shape


def test_unet_shape_num_res_blocks_1():
    unet = VideoUNet(in_channels=4, out_channels=4, base_channels=16,
                     channel_multipliers=(1, 2), num_res_blocks=1,
                     context_dim=32, attention_heads=2)
    x = torch.randn(1, 4, 4, 8, 8)
    out = unet(x, torch.tensor([10]), torch.randn(1, 8, 32))
    assert out.shape == x.shape


def test_unet_shape_num_res_blocks_2():
    """Regression: up-path channel accounting must work for num_res_blocks >= 2."""
    unet = VideoUNet(in_channels=4, out_channels=4, base_channels=16,
                     channel_multipliers=(1, 2), num_res_blocks=2,
                     context_dim=32, attention_heads=2)
    x = torch.randn(1, 4, 4, 8, 8)
    out = unet(x, torch.tensor([10]), torch.randn(1, 8, 32))
    assert out.shape == x.shape


def test_unet_shape_three_levels():
    unet = VideoUNet(in_channels=4, out_channels=4, base_channels=8,
                     channel_multipliers=(1, 2, 4), num_res_blocks=2,
                     context_dim=16, attention_heads=2)
    x = torch.randn(1, 4, 4, 16, 16)
    out = unet(x, torch.tensor([10]), torch.randn(1, 8, 16))
    assert out.shape == x.shape


def test_full_model_forward():
    model = TextToVideoModel(
        vocab_size=32, image_size=32, num_frames=4,
        latent_channels=4, base_channels=16, channel_multipliers=(1, 2),
        num_res_blocks=1, text_embed_dim=32, text_max_length=8,
        transformer_layers=1, transformer_heads=2, attention_heads=2, vae_downsample=8,
    )
    out = model(torch.randn(1, 4, 4, 4, 4), torch.tensor([10]),
                torch.randint(0, 32, (1, 8)), torch.ones(1, 8, dtype=torch.long))
    assert out.shape == (1, 4, 4, 4, 4)


def test_full_model_num_res_blocks_2():
    model = TextToVideoModel(
        vocab_size=32, image_size=32, num_frames=4,
        latent_channels=4, base_channels=16, channel_multipliers=(1, 2),
        num_res_blocks=2, text_embed_dim=32, text_max_length=8,
        transformer_layers=1, transformer_heads=2, attention_heads=2, vae_downsample=8,
    )
    out = model(torch.randn(1, 4, 4, 4, 4), torch.tensor([10]),
                torch.randint(0, 32, (1, 8)), torch.ones(1, 8, dtype=torch.long))
    assert out.shape == (1, 4, 4, 4, 4)


def test_encode_decode_video():
    model = TextToVideoModel(
        vocab_size=32, image_size=32, num_frames=4,
        latent_channels=4, base_channels=16, channel_multipliers=(1, 2),
        num_res_blocks=1, text_embed_dim=32, text_max_length=8,
        transformer_layers=1, transformer_heads=2, attention_heads=2, vae_downsample=8,
    )
    frames = torch.randn(1, 4, 3, 32, 32)
    z = model.encode_video(frames)
    assert z.shape[1] == 4
    assert model.decode_video(z).shape == frames.shape


def test_gradient_checkpointing_toggle():
    model = TextToVideoModel(
        vocab_size=32, image_size=32, num_frames=4,
        latent_channels=4, base_channels=16, channel_multipliers=(1, 2),
        num_res_blocks=1, text_embed_dim=32, text_max_length=8,
        transformer_layers=1, transformer_heads=2, attention_heads=2, vae_downsample=8,
    )
    model.set_gradient_checkpointing(True)
    assert model.unet.gradient_checkpointing is True
    model.train()
    out = model(torch.randn(1, 4, 4, 4, 4), torch.tensor([10]),
                torch.randint(0, 32, (1, 8)), torch.ones(1, 8, dtype=torch.long))
    assert out.shape == (1, 4, 4, 4, 4)
