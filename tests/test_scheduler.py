import torch

from AI.scheduler import DDIMScheduler, DDPMScheduler


def test_ddpm_betas_linear():
    s = DDPMScheduler(num_train_timesteps=100, beta_schedule="linear")
    assert s.betas.shape == (100,) and s.betas[0] < s.betas[-1]


def test_ddpm_betas_cosine():
    s = DDPMScheduler(num_train_timesteps=100, beta_schedule="cosine")
    assert s.betas.shape == (100,) and float(s.betas.min()) > 0


def test_add_noise_shape():
    s = DDPMScheduler(num_train_timesteps=100)
    x = torch.randn(2, 4, 8, 8, 8)
    assert s.add_noise(x, torch.randn_like(x), torch.tensor([0, 50])).shape == x.shape


def test_add_noise_at_t0_close_to_original():
    s = DDPMScheduler(num_train_timesteps=100)
    x = torch.randn(1, 4, 8, 8, 8)
    assert torch.allclose(s.add_noise(x, torch.randn_like(x), torch.tensor([0])), x, atol=0.2)


def test_ddpm_step_shape():
    s = DDPMScheduler(num_train_timesteps=100)
    sample = torch.randn(1, 4, 8, 8, 8)
    assert s.step(torch.randn_like(sample), 10, sample).shape == sample.shape


def test_ddim_deterministic():
    s = DDIMScheduler(num_train_timesteps=100, eta=0.0)
    sample = torch.randn(1, 4, 8, 8, 8)
    noise = torch.randn_like(sample)
    assert torch.allclose(s.step(noise, 50, sample, prev_timestep=40),
                          s.step(noise, 50, sample, prev_timestep=40))


def test_ddim_timestep_list():
    s = DDIMScheduler(num_train_timesteps=1000)
    ts = s._make_timestep_list(10)
    assert len(ts) == 10 and ts[0] > ts[-1]


def test_ddim_sample_runs():
    s = DDIMScheduler(num_train_timesteps=100, eta=0.0)
    gen = torch.Generator(); gen.manual_seed(0)
    out = s.sample(lambda x, t: torch.zeros_like(x), (1, 4, 4, 8, 8),
                   num_inference_steps=5, generator=gen)
    assert out.shape == (1, 4, 4, 8, 8)


def test_ddim_alpha_prev_expand_safe():
    """Regression: 0-d alpha_prev must not crash on .expand()."""
    s = DDIMScheduler(num_train_timesteps=100, eta=0.0)
    sample = torch.randn(2, 4, 4, 4, 4)
    out = s.step(torch.randn_like(sample), 50, sample, prev_timestep=25)
    assert out.shape == sample.shape


def test_ddim_single_step_uses_last_timestep():
    """Regression: num_inference_steps=1 must return [T-1], not [0]."""
    s = DDIMScheduler(num_train_timesteps=100)
    ts = s._make_timestep_list(1)
    assert len(ts) == 1 and int(ts[0]) == 99


def test_ddim_timestep_list_clamped():
    s = DDIMScheduler(num_train_timesteps=100)
    ts = s._make_timestep_list(5000)
    assert len(ts) == 100
