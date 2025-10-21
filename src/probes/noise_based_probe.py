from tqdm import tqdm
import torch
import torch.nn.functional as F
import types
import os
from noisy_diffuser_scheduling.schedulers.eta_ddim_scheduler import DDIMScheduler
from probes.base_probe import BaseProbe
import sys
from pathlib import Path
# Optional import for classifier guidance
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from classifier_guidance.create_latent_classifier import LatentClassifierT
from PIL import Image

class NoiseBasedProbe(BaseProbe):
    def guided_generation_latent(self, pipe, prompt, classifier, target_class_prob=1.0, num_inference_steps=50, guidance_scale=7.5, classifier_scale=10.0, seed=42, t_downsample=64, eta = 1.0, variance_scale = None):
        """
        Using classifier guidance during the inference pipeline
        """

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # --- text embeddings ---
        text_inputs = pipe.tokenizer(
            prompt, return_tensors="pt", padding="max_length", truncation=True,
            max_length=pipe.tokenizer.model_max_length
        )
        text_emb = pipe.text_encoder(text_inputs.input_ids.to(self.device))[0]
        uncond_emb = pipe.text_encoder(
            pipe.tokenizer([""], return_tensors="pt", padding="max_length",
                        max_length=pipe.tokenizer.model_max_length).input_ids.to(self.device)
        )[0]
        text_emb = torch.cat([uncond_emb, text_emb])

        # --- initial latents ---
        latents = torch.randn((1, pipe.unet.in_channels, 64, 64),
                            generator=generator, device=self.device, dtype=torch.float16)
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
                target = torch.tensor(target_class_prob, device=self.device)
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



    def run(self, num_images=None, debug=False, use_cls_guidance=False):
        """
        Generate noisy variants and keep the best-scoring image per prompt.

        If config["use_classifier_guidance"] is True, will load the corresponding
        latent classifier for the given concept and apply gradient guidance during sampling.
        """
        prompts = self._load_prompts(num_images)

        # Backup scheduler
        original_scheduler = self.pipe.scheduler
        self.pipe.scheduler = DDIMScheduler.from_pretrained(
            self.pipeline_path, subfolder="scheduler"
        )
        # ============================================================
        # 🚀 Run Generation Loop
        # ============================================================
        for i, (prompt, seed) in tqdm(enumerate(prompts), total=len(prompts), desc=f"Noisy {self.concept}"):
            generator = torch.manual_seed(int(seed))
            best_image, best_score = None, float("-inf")
            classifier_dir = self.config.get(
                    "classifier_root",
                    "/share/u/kevin/DiffusionConceptErasure/classifier_guidance/latent_classifiers",
                )

            classifier = None
            classifier_path = os.path.join(classifier_dir, f"{self.concept}.pt")
            if use_cls_guidance:
                self.output_dir = Path(self.output_dir).parent / "classifier_guidance"
                classifier = LatentClassifierT(scheduler=self.pipe.scheduler).to(self.device)
                checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
                classifier.load_state_dict(checkpoint["model_state_dict"])
                classifier.eval()
                # =======================================================
                # Classifier-guided mode: sweep over classifier_scales × etas
                # =======================================================
                classifier_scales = self.config.get("classifier_scales", [50, 100, 150, 175])
                etas = self.config.get("eta_values", [1.0, 1.17, 1.34, 1.51, 1.68, 1.85])

                for cls_scale in classifier_scales:
                    print(f"\n🧠 Classifier guidance scale: {cls_scale}")
                    for eta in etas:
                        image = self.guided_generation_latent(
                            self.pipe,
                            prompt=prompt,
                            classifier=classifier,
                            target_class_prob=1.0,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            classifier_scale=cls_scale,
                            seed=int(seed),
                            eta=eta,
                        )

                        score_prompt = "a picture of a dog" if self.concept == "english_springer_spaniel" else prompt
                        score = self.score(image, score_prompt)

                        if debug:
                            subfolder = f"debug/cls{cls_scale}"
                            fname = f"{self.concept}_{i:03d}_eta{eta:.2f}_cls{cls_scale}_seed{seed}.png"
                            self.save_image(image, fname, subfolder=subfolder)

                        if score > best_score:
                            best_score, best_image = score, image

            else:
                # =======================================================
                # Standard noise-based mode: sweep over variance_scales × etas
                # =======================================================
                etas = self.config.get("eta_values", [1.0, 1.17, 1.34, 1.51, 1.68, 1.85])
                variance_scales = self.config.get("variance_scales", [1.0, 1.02, 1.03, 1.04])

                for eta in etas:
                    for vscale in variance_scales:
                        image = self.pipe(
                            prompt,
                            generator=generator,
                            eta=eta,
                            variance_scale=vscale,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                        ).images[0]

                        score_prompt = "a picture of a dog" if self.concept == "english_springer_spaniel" else prompt
                        score = self.score(image, score_prompt)

                        if debug:
                            subfolder = f"debug/eta{eta:.2f}_vs{vscale:.2f}"
                            fname = f"{self.concept}_{i:03d}_eta{eta:.2f}_vs{vscale:.2f}_seed{seed}.png"
                            self.save_image(image, fname, subfolder=subfolder)

                        if score > best_score:
                            best_score, best_image = score, image

            # --- save best-scoring image ---
            self.save_image(best_image, f"{self.concept}_{i:03d}_seed{seed}_best.png")


    # ------------------------------------------------------------------
    def score(self, image, prompt):
        """Default scoring logic (CLIP or classifier)."""
        if "score_type" not in self.config:
            return 0.0
        mode = self.config["score_type"].lower()
        if mode == "clip":
            return self._clip_score(image, prompt)
        elif mode == "classification":
            return self._classifier_score(image)
        else:
            raise ValueError(f"Unknown score type: {mode}")
