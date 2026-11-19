"""Diffusion schedulers: DDPM (training) and DDIM (inference)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _make_beta_schedule(schedule: str, num_timesteps: int,
                        beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray:
    if schedule == "linear":
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    if schedule == "cosine":
        steps = num_timesteps + 1
        s = 0.008
        x = np.linspace(0, num_timesteps, steps, dtype=np.float64)
        alphas_cumprod = np.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 1e-4, 0.999)
    raise ValueError(f"Unknown beta schedule: {schedule}")


class DDPMScheduler:
    def __init__(self, num_train_timesteps: int = 1000, beta_schedule: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 2e-2,
                 device: str | torch.device = "cpu"):
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_schedule = beta_schedule
        self.device = torch.device(device)

        betas = _make_beta_schedule(beta_schedule, self.num_train_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])

        self._betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        self._alphas = torch.tensor(alphas, dtype=torch.float32, device=self.device)
        self._alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32, device=self.device)
        self._alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32, device=self.device)

        self._sqrt_alphas_cumprod = torch.sqrt(self._alphas_cumprod)
        self._sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self._alphas_cumprod)
        self._sqrt_recip_alphas = torch.sqrt(1.0 / self._alphas)
        self._posterior_variance = (
            self._betas * (1.0 - self._alphas_cumprod_prev) / (1.0 - self._alphas_cumprod)
        )
        self._posterior_mean_coef1 = (
            self._betas * torch.sqrt(self._alphas_cumprod_prev) / (1.0 - self._alphas_cumprod)
        )
        self._posterior_mean_coef2 = (
            (1.0 - self._alphas_cumprod_prev) * torch.sqrt(self._alphas) / (1.0 - self._alphas_cumprod)
        )

    @property
    def betas(self) -> torch.Tensor:
        return self._betas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return self._alphas_cumprod

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = arr.gather(0, t.to(arr.device))
        while out.dim() < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape).to(dtype=torch.float32)

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        shape = original.shape
        sqrt_alpha = self._extract(self._sqrt_alphas_cumprod, timesteps, shape).to(original.dtype)
        sqrt_one_minus = self._extract(self._sqrt_one_minus_alphas_cumprod, timesteps, shape).to(original.dtype)
        return sqrt_alpha * original + sqrt_one_minus * noise

    def sample_timesteps(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        return torch.randint(0, self.num_train_timesteps, (batch_size,),
                             device=device, dtype=torch.long)

    def _make_timestep_list(self, num_inference_steps: int) -> torch.Tensor:
        """Return a descending timestep list.

        For n >= 2 uses evenly-spaced timesteps. For n == 1 returns [T-1]
        (a single full denoise step starting from maximum noise).
        """
        n = max(1, min(int(num_inference_steps), self.num_train_timesteps))
        if n == 1:
            return torch.tensor([self.num_train_timesteps - 1], dtype=torch.long)
        step_ratio = self.num_train_timesteps // n
        ts = (torch.arange(0, n) * step_ratio).round().long()
        return torch.flip(ts, dims=[0])

    @torch.no_grad()
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        t = torch.full((sample.shape[0],), int(timestep), dtype=torch.long, device=sample.device)
        shape = sample.shape
        sqrt_one_minus = self._extract(self._sqrt_one_minus_alphas_cumprod, t, shape).to(sample.dtype)
        alpha_bar = self._extract(self._alphas_cumprod, t, shape).to(sample.dtype)
        coef1 = self._extract(self._posterior_mean_coef1, t, shape).to(sample.dtype)
        coef2 = self._extract(self._posterior_mean_coef2, t, shape).to(sample.dtype)
        var = self._extract(self._posterior_variance, t, shape).to(sample.dtype)

        pred_x0 = (sample - sqrt_one_minus * model_output) / torch.sqrt(alpha_bar)
        mean = coef1 * pred_x0 + coef2 * sample
        if timestep == 0:
            return mean
        noise = torch.randn(shape, generator=generator, device=sample.device, dtype=sample.dtype)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model_output_fn, shape, generator: Optional[torch.Generator] = None,
               device: Optional[torch.device] = None,
               num_steps: Optional[int] = None) -> torch.Tensor:
        device = device or self.device
        sample = torch.randn(shape, generator=generator, device=device)
        n_steps = num_steps or self.num_train_timesteps
        for t in self._make_timestep_list(n_steps).tolist():
            out = model_output_fn(sample, t)
            sample = self.step(out, t, sample, generator=generator)
        return sample


class DDIMScheduler(DDPMScheduler):
    def __init__(self, num_train_timesteps: int = 1000, eta: float = 0.0, **kwargs):
        super().__init__(num_train_timesteps=num_train_timesteps, **kwargs)
        self.eta = float(eta)

    @torch.no_grad()
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor,
             prev_timestep: int = -1,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        t = torch.full((sample.shape[0],), int(timestep), dtype=torch.long, device=sample.device)
        shape = sample.shape
        alpha_t = self._extract(self._alphas_cumprod, t, shape).to(sample.dtype)

        if prev_timestep >= 0:
            alpha_prev_scalar = self._alphas_cumprod[int(prev_timestep)]
        else:
            alpha_prev_scalar = torch.tensor(1.0, device=self._alphas_cumprod.device,
                                             dtype=self._alphas_cumprod.dtype)
        alpha_prev = alpha_prev_scalar.reshape(1, 1, 1, 1, 1).expand(shape).to(sample.dtype)

        pred_x0 = (sample - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        pred = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * model_output

        if self.eta > 0 and prev_timestep >= 0:
            sigma = self.eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            )
            noise = torch.randn(shape, generator=generator, device=sample.device, dtype=sample.dtype)
            pred = pred + sigma * noise
        return pred

    @torch.no_grad()
    def sample(self, model_output_fn, shape, num_inference_steps: int = 50,
               generator: Optional[torch.Generator] = None,
               device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        sample = torch.randn(shape, generator=generator, device=device)
        ts = self._make_timestep_list(num_inference_steps)
        for i, t in enumerate(ts.tolist()):
            prev_t = int(ts[i + 1]) if i + 1 < len(ts) else -1
            out = model_output_fn(sample, t)
            sample = self.step(out, t, sample, prev_timestep=prev_t, generator=generator)
        return sample
