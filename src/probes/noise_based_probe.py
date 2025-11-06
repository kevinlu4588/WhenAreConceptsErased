from tqdm import tqdm
import torch
import torch.nn.functional as F
import types
import os
from noisy_diffuser_scheduling.schedulers.eta_ddim_scheduler import DDIMScheduler
from probes.base_probe import BaseProbe
from pathlib import Path
from PIL import Image
from .utils import rank_with_resnet_in_memory, load_classifier

    # Add imports needed for classifier loading
import sys
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))
from classifier_guidance.create_latent_classifier import LatentClassifierT
from huggingface_hub import hf_hub_download


class NoiseBasedProbe(BaseProbe):

    def run(self, num_images=None, debug=False, use_classifier_guidance=False):
        """
        Generate noisy variants and keep the best-scoring image per prompt.

        If config["use_classifier_guidance"] is True, will load the corresponding
        latent classifier for the given concept and apply gradient guidance during sampling.
        """
        prompts = self._load_prompts(num_images)

        # Backup scheduler
        self.pipe.scheduler = DDIMScheduler.from_pretrained(
            self.pipeline_path, subfolder="scheduler"
        )
        # ============================================================
        # 🚀 Run Generation Loop
        # ============================================================
        for i, (prompt, seed) in tqdm(enumerate(prompts), total=len(prompts), desc=f"Noisy {self.concept}"):
            generator = torch.manual_seed(int(seed))
            best_image, best_score = None, float("-inf")
            variants = []

            if use_classifier_guidance:
                classifier = self._load_classifier()
                # =======================================================
                # Classifier-guided mode: sweep over classifier_scales × etas
                # =======================================================
                classifier_scales = self.config.get("classifier_scales", [30, 50,75, 125])
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
                        variants.append(image)

                        if debug:
                            subfolder = f"debug/cls{cls_scale}"
                            fname = f"{self.concept}_{i:03d}_eta{eta:.2f}_cls{cls_scale}_seed{seed}.png"
                            self.save_image(image, fname, subfolder=subfolder)

            else:
                # =======================================================
                # Standard noise-based mode: sweep over variance_scales × etas
                # =======================================================
                etas = self.config.get("eta_values", [1.0, 1.17, 1.34, 1.51, 1.68, 1.85])
                variance_scales = self.config.get("variance_scales", [1.0, 1.02, 1.03, 1.04])
                variants = []
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
                        variants.append(image)

                        # score_prompt = "a picture of a dog" if self.concept == "english_springer_spaniel" else prompt
                        # score = self.score(image, score_prompt)

                        if debug:
                            subfolder = f"debug/image_{i}"
                            fname = f"{self.concept}_{i:03d}_eta{eta:.2f}_vs{vscale:.2f}_seed{seed}.png"
                            self.save_image(image, fname, subfolder=subfolder)

                        # if score > best_score:
                        #     best_score, best_image = score, image

            # --- save best-scoring image ---
            best_image, best_prob = rank_with_resnet_in_memory(variants, concept=self.concept)
            self.save_image(best_image, f"{self.concept}_{i:03d}_seed{seed}_best.png")

    


    def load_classifier(self):
        """
        Load classifier from local path first, then fallback to HuggingFace.
        Uses class attributes: self.pipe, self.concept, self.config, self.device
        
        Returns:
            classifier: Loaded and evaluated LatentClassifierT model
        """
        classifier = LatentClassifierT(scheduler=self.pipe.scheduler).to(self.device)
        
        # Try local path first - use absolute path based on project root
        classifier_dir = self.config.get(
            "classifier_root",
            str(project_root / "classifier_guidance" / "latent_classifiers"),
        )
        classifier_path = os.path.join(classifier_dir, f"{self.concept}.pt")
        
        if os.path.exists(classifier_path):
            try:
                checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
                classifier.load_state_dict(checkpoint["model_state_dict"])
                print(f"✅ Loaded from local: {classifier_path}")
                classifier.eval()
                return classifier
            except Exception as e:
                print(f"⚠️ Failed to load from local path: {e}")
        else:
            print(f"📁 Local classifier not found at {classifier_path}, trying HuggingFace...")
        
        # Fallback to HuggingFace
        hf_sources = [
            f"DiffusionConceptErasure/latent-classifier-{self.concept}",  # Primary repo
        ]
        
        for hf_model_id in hf_sources:
            # Try different filename patterns
            for filename in ["model.pt", f"{self.concept}.pt", "classifier.pt"]:
                try:
                    model_file = hf_hub_download(
                        repo_id=hf_model_id,
                        filename=filename,
                        cache_dir=self.config.get("hf_cache_dir", None)
                    )
                    checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
                    classifier.load_state_dict(checkpoint["model_state_dict"])
                    print(f"✅ Loaded from HF: {hf_model_id}/{filename}")
                    classifier.eval()
                    return classifier
                except Exception as e:
                    # Try next filename or next repo
                    continue
        
        raise FileNotFoundError(f"Could not find classifier for {self.concept} locally or on HuggingFace")

    
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
