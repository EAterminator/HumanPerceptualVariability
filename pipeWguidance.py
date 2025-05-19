import re
import torch
from tqdm import tqdm
import torch.nn as nn
import sys
from diffusion_prior import *
from diffusers.schedulers.scheduling_ddpm import *
from torch.cuda.amp import autocast, GradScaler


def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


class Pipe_guidance(Pipe):

    def __init__(self, diffusion_prior, scheduler=None, device="cuda"):
        super().__init__(diffusion_prior, scheduler, device)

    @torch.no_grad()
    def generate_guidance(
        self,
        loss_fn=None,
        N=None,
        num_inference_steps=50,
        timesteps=None,
        guidance_scale=1.0,
        generator=None,
        shape=None,
        num_resampling_steps=1,
        inversion_noise=None,
        use_ema=True,
        clamp_scale=None,
        ref_image=None,
    ):
        if inversion_noise is not None:
            assert N == inversion_noise.shape[0]
        model = self.ema if use_ema else self.diffusion_prior
        model.eval()

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
            retrieve_timesteps,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, timesteps
        )

        # 2. Prepare noise
        if inversion_noise is not None:
            x_t = inversion_noise
        else:
            x_t = torch.randn(N, *shape, generator=generator, device=self.device)

        # 3. Denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t_tensor = torch.ones(N, dtype=torch.float, device=self.device) * t
            for s in range(num_resampling_steps):
                with torch.enable_grad():
                    x_t.requires_grad_(True)
                    noise_pred = model(x_t, t_tensor)

                    # 3.1 Unconditional sampling x_t -> x_{t-1}
                    output = self.scheduler.step(
                        noise_pred, t, x_t, generator=generator
                    )
                    x_t_uncond, x_0t = output.prev_sample, output.pred_original_sample

                    # 3.2 Posterior sampling
                    sigma_t = self._get_variance(t) ** 0.5
                    sqrt_n_shape = np.prod(shape) ** 0.5
                    # Calculate posterior gradient
                    grad = torch.autograd.grad(loss_fn(x_0t).mean(), x_t)[0]
                    norm = torch.linalg.norm(grad.view(N, -1), dim=1).view(
                        -1, *([1] * len(shape))
                    )
                    grad = sqrt_n_shape * sigma_t * grad / norm
                    # Apply classifier guidance
                    x_t = x_t_uncond - guidance_scale * grad
                    if (
                        clamp_scale is not None
                    ):  # when there is a ref_image for constraint
                        change = x_t - ref_image
                        change = change.renorm(
                            p=2, dim=0, maxnorm=clamp_scale
                        ).requires_grad_()
                        x_t = ref_image + change

                # 3.3 Resampling trick/time travel
                # resampling for s-1 times
                if s < num_resampling_steps - 1:
                    x_t = self._forward_one_step(x_t, t, generator)  # q(h_t | h_{t-1})
                x_t = x_t.detach()

        return x_t

    # get generation step
    def _previous_timestep(self, timestep):
        num_inference_steps = self.scheduler.num_inference_steps
        prev_t = (
            timestep - self.scheduler.config.num_train_timesteps // num_inference_steps
        )
        return prev_t

    # get variance strength in each step
    def _get_variance(self, t):

        # get beta_t
        prev_t = self._previous_timestep(t)
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        )
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)
        return variance

    # forward process: add noise to x_{t-1} to get x_t, add noise
    def _forward_one_step(self, x_t, t, generator=None):
        # get beta_t
        prev_t = self._previous_timestep(t)
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        )
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # q(x_t | x_{t-1}):
        # DDPM: x_t = sqrt(1 - \beta_t) * x_{t-1} + \beta_t * N(0, 1)
        noise = torch.randn(x_t.shape, generator=generator, device=x_t.device)
        a, b = np.sqrt(1 - current_beta_t), current_beta_t
        x_t = a * x_t + b * noise

        return x_t

    def get_inversion(
        self,
        x,
        num_inference_steps=50,
    ):
        from diffusers.schedulers import DDIMInverseScheduler

        inverse_scheduler = DDIMInverseScheduler()
        model = self.ema
        model.eval()

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
            retrieve_timesteps,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, timesteps
        )

        # 2. Prepare noise
        x_t = x
        N = x.shape[0]
        shape = x.shape[1:]

        # 3. Denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t_tensor = torch.ones(N, dtype=torch.float, device=self.device) * t
            noise_pred = model(x_t, t_tensor)
            # 3.1 Unconditional sampling x_t -> x_{t-1}
            x_t = self.inverse_scheduler.step(noise_pred, t, x_t).prev_sample
            x_t = x_t.detach()
        return x_t
