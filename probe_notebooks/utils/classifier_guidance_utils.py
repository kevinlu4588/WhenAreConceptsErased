import os, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
# ============================================================
# 1️⃣  Model
# ============================================================
class FixedTimestepEncoding(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod)

    def forward(self, t):
        alpha_bar = self.alphas_cumprod[t]
        return torch.stack([alpha_bar.sqrt(), (1 - alpha_bar).sqrt()], dim=-1)


class LatentClassifierT(nn.Module):
    def __init__(self, latent_shape=(4, 64, 64), scheduler=None):
        super().__init__()
        c, h, w = latent_shape
        flat_dim = c * h * w
        self.t_embed = FixedTimestepEncoding(scheduler)
        self.fc_t = nn.Linear(2, 1024)
        self.fc_x = nn.Linear(flat_dim, 1024)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, z, t):
        z_flat = z.flatten(start_dim=1)
        return self.net(self.fc_x(z_flat) + self.fc_t(self.t_embed(t)))

def guided_generation_latent(pipe, prompt, classifier, device, target_class_prob=1.0,
                             num_inference_steps=50, guidance_scale=7.5,
                             classifier_scale=10.0, seed=42):
    """Performs classifier-guided diffusion through the latent space."""
    generator = torch.Generator(device=device).manual_seed(seed)

    # --- text embeddings ---
    text_inputs = pipe.tokenizer(
        prompt, return_tensors="pt", padding="max_length", truncation=True,
        max_length=pipe.tokenizer.model_max_length
    )
    text_emb = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
    uncond_emb = pipe.text_encoder(
        pipe.tokenizer([""], return_tensors="pt", padding="max_length",
                       max_length=pipe.tokenizer.model_max_length).input_ids.to(device)
    )[0]
    text_emb = torch.cat([uncond_emb, text_emb])

    # --- initial latents ---
    latents = torch.randn((1, pipe.unet.in_channels, 64, 64),
                          generator=generator, device=device, dtype=torch.float16)
    latents = latents * pipe.scheduler.init_noise_sigma
    pipe.scheduler.set_timesteps(num_inference_steps)

    # --- diffusion loop ---
    for t in pipe.scheduler.timesteps:
        latent_input = torch.cat([latents] * 2)
        latent_input = pipe.scheduler.scale_model_input(latent_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=text_emb).sample

        # classifier-free guidance
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # classifier guidance
        latents.requires_grad_(True)
        with torch.enable_grad():
            logits = classifier(latents.float(), t.expand(latents.size(0)).to(latents.device))
            target = torch.tensor(target_class_prob, device=device)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), target)
            grad = torch.autograd.grad(-loss, latents)[0]

        grad_scaled = grad * classifier_scale
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        noise_pred = noise_pred - beta_prod_t.sqrt() * grad_scaled

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # --- decode ---
    with torch.no_grad():
        image = pipe.vae.decode(latents / 0.18215).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype("uint8")
    return Image.fromarray(image)

def guided_generation_latent_noise_based(pipe, prompt, classifier, target_class_prob=1.0, num_inference_steps=50, guidance_scale=7.5, classifier_scale=10.0, seed=42, t_downsample=64, eta = 1.0, variance_scale = None, device='cuda'):
        """
        Using classifier guidance during the inference pipeline
        """

        generator = torch.Generator(device=device).manual_seed(seed)

        # --- text embeddings ---
        text_inputs = pipe.tokenizer(
            prompt, return_tensors="pt", padding="max_length", truncation=True,
            max_length=pipe.tokenizer.model_max_length
        )
        text_emb = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_emb = pipe.text_encoder(
            pipe.tokenizer([""], return_tensors="pt", padding="max_length",
                        max_length=pipe.tokenizer.model_max_length).input_ids.to(device)
        )[0]
        text_emb = torch.cat([uncond_emb, text_emb])

        # --- initial latents ---
        latents = torch.randn((1, pipe.unet.in_channels, 64, 64),
                            generator=generator, device=device, dtype=torch.float16)
        latents = latents * pipe.scheduler.init_noise_sigma
        pipe.scheduler.set_timesteps(num_inference_steps)

        # --- diffusion loop ---
        for i, t in enumerate(pipe.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_emb).sample

            # Classifier-free guidance
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            latents.requires_grad_(True)
            with torch.enable_grad():
                latents_down = latents.float()

                logit = classifier(latents_down, t.expand(latents.size(0)).to(latents.device))
                target = torch.tensor(target_class_prob, device=device)
                loss = F.binary_cross_entropy_with_logits(logit.squeeze(), target)
                grad = torch.autograd.grad(-loss, latents)[0]

            grad_scaled = grad * classifier_scale
            alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
            beta_prod_t = 1- alpha_prod_t
            noise_pred = noise_pred - beta_prod_t.sqrt() * grad_scaled

            latents = pipe.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

        with torch.no_grad():
            image = pipe.vae.decode(latents / 0.18215).sample


        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image.cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype("uint8")
        return Image.fromarray(image)