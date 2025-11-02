import os
import torch
import warnings
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from probes.base_probe import BaseProbe
from .utils import call_sdv14

warnings.filterwarnings("ignore", message=".*IProgress not found.*")
StableDiffusionPipeline.__call__ = call_sdv14
StableDiffusionPipeline._callback_tensor_inputs = [
    "dt_latents", "latents", "prompt_embeds", "negative_prompt_embeds"
]


# ============================================================
# 🔧 Diffusion Completion Helper Functions
# ============================================================
def decode_latent(pipe, latent, device):
    """Decode a latent tensor into an image using the pipeline's VAE."""
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    latent = latent.to(device) / 0.18215
    with torch.no_grad():
        decoded = pipe.vae.decode(latent).sample
    image = (decoded.clamp(-1, 1) + 1) / 2
    image = image[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((image * 255).astype("uint8"))


def generate_base_latents(pipe, prompt, interrupt_steps, seed, num_inference_steps, device):
    """Run the base pipeline until interruption and collect latents + dt images."""
    interrupted_latents, dt_images = [], []

    for t_step in interrupt_steps:
        gen = torch.Generator(device=device).manual_seed(seed)

        def interrupt_callback(pipeline, i, t, callback_kwargs):
            if i == t_step:
                pipeline._interrupt = True
                interrupted_latents.append(callback_kwargs["latents"].clone())
            return callback_kwargs

        pipe.scheduler.set_timesteps(num_inference_steps)
        _ = pipe(
            prompt,
            generator=gen,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            callback_on_step_end=interrupt_callback,
        )

        run_timesteps = pipe.scheduler.timesteps[t_step + 1:]
        pipe.scheduler.set_timesteps(num_inference_steps)

        dt = pipe(
            prompt,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_inference_steps=num_inference_steps,
            run_timesteps=run_timesteps,
            latents=interrupted_latents[-1].clone(),
        ).images[0]

        dt_images.append(dt)

    return interrupted_latents, dt_images


# ============================================================
# 🧠 Diffusion Completion Probe
# ============================================================
class DiffusionCompletionProbe(BaseProbe):
    """
    Probe for visualizing diffusion completion — running base SDv1.4 up to certain timesteps,
    then completing the diffusion using either the base model or the concept-erased model (self.pipe).
    """

    def run(self, num_images=None, interrupt_steps=None, debug=False):
        """
        Run diffusion completion for a set of prompts and save images.
        """
        device = self.device
        interrupt_steps = interrupt_steps or [5, 10]
        prompts = self._load_prompts(num_images)

        print(f"🚀 Running DiffusionCompletionProbe for concept='{self.concept}'")

        # === Load base pipeline ===
        base_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.bfloat16
        ).to(device).to(torch.bfloat16)
        base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
        base_pipe.set_progress_bar_config(disable=True)

        # === Iterate over prompts ===
        for i, (prompt, seed) in tqdm(
            enumerate(prompts),
            total=len(prompts),
            desc=f"DiffusionCompletion {self.concept}"
        ):
            # ------------------------------------------------------------
            # Step 1: Generate interrupted latents + base completions
            # ------------------------------------------------------------
            base_latents, dt_images = generate_base_latents(
                base_pipe, prompt, interrupt_steps, seed,
                self.num_inference_steps, device
            )

            if debug:
                for idx, dt_img in enumerate(dt_images):
                    fname = f"{self.concept}_{i:03d}_base_t{interrupt_steps[idx]}_seed{seed}.png"
                    self.save_image(dt_img, fname, subfolder="base_dt")

            # ------------------------------------------------------------
            # Step 2: Base reconstructions (only if debug)
            # ------------------------------------------------------------
            base_pipe.scheduler.set_timesteps(self.num_inference_steps)
            timesteps = base_pipe.scheduler.timesteps

            if debug:
                for idx, t_step in enumerate(interrupt_steps):
                    latent = base_latents[idx]
                    run_timesteps = timesteps[t_step + 1:]

                    base_recon = base_pipe(
                        prompt,
                        generator=torch.Generator(device=device).manual_seed(seed),
                        num_inference_steps=self.num_inference_steps,
                        output_type="latent",
                        run_timesteps=run_timesteps,
                        latents=latent
                    ).images[0]
                    decoded_base = decode_latent(base_pipe, base_recon, device)

                    fname = f"{self.concept}_{i:03d}_base_recon_t{t_step}_seed{seed}.png"
                    self.save_image(decoded_base, fname, subfolder="base_recon")

            # ------------------------------------------------------------
            # Step 3: Reconstructions with erased model (always saved)
            # ------------------------------------------------------------
            self.pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
            self.pipe.scheduler.set_timesteps(self.num_inference_steps)
            self.pipe.set_progress_bar_config(disable=True)

            for idx, t_step in enumerate(interrupt_steps):
                latent = base_latents[idx]
                run_timesteps = timesteps[t_step + 1:]

                recon = self.pipe(
                    prompt,
                    generator=torch.Generator(device=device),
                    num_inference_steps=self.num_inference_steps,
                    output_type="latent",
                    run_timesteps=run_timesteps,
                    latents=latent
                ).images[0]

                decoded_erased = decode_latent(self.pipe, recon, device)
                fname = f"{self.concept}_{i:03d}_erased_recon_t{t_step}_seed{seed}.png"
                self.save_image(decoded_erased, fname, subfolder="erased_recon")

        print(f"✅ Saved diffusion completions for {len(prompts)} prompts to {self.output_dir}")
