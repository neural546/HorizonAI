import torch

from AI.inference import set_global_seed
from AI.scheduler import DDIMScheduler
from AI.tokenizer import build_default_tokenizer


def test_set_global_seed_reproducible():
    set_global_seed(123); a = torch.randn(4)
    set_global_seed(123); b = torch.randn(4)
    assert torch.allclose(a, b)


def test_ddim_sampling_loop_shape():
    s = DDIMScheduler(num_train_timesteps=100, eta=0.0)
    gen = torch.Generator(); gen.manual_seed(0)
    out = s.sample(lambda x, t: torch.zeros_like(x), (1, 4, 4, 8, 8),
                   num_inference_steps=5, generator=gen)
    assert out.shape == (1, 4, 4, 8, 8)


def test_tokenizer_encode_for_inference():
    tok = build_default_tokenizer(max_length=16)
    ids = tok.encode("a dragon flying")
    assert len(ids) == 16 and ids[0] == tok.bos_id


def test_ddim_timestep_list_clamped():
    s = DDIMScheduler(num_train_timesteps=100)
    ts = s._make_timestep_list(5000)
    assert len(ts) == 100
