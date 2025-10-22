from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
import os, sys
from probes.base_probe import BaseProbe
from .utils import rank_with_resnet_in_memory
# Make classifier guidance importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from classifier_guidance.create_latent_classifier import LatentClassifierT


# ============================================================
# 🔧 Helper: Classifier-Guided Latent Generation
# ============================================================
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


# ============================================================
# 🧠 Standard Prompt Probe (with optional classifier guidance)
# ============================================================
class StandardPromptProbe(BaseProbe):
    
    def run(self, num_images=None, debug=False, use_classifier_guidance=False):
        """Generate standard diffusion outputs or classifier-guided outputs."""
        prompts = self._load_prompts(num_images)
        classifier = None

        if use_classifier_guidance:
            classifier_dir = self.config.get(
                "classifier_root",
                "/share/u/kevin/DiffusionConceptErasure/classifier_guidance/latent_classifiers"
            )
            classifier_path = os.path.join(classifier_dir, f"{self.concept}.pt")
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(f"❌ Classifier not found for '{self.concept}' at {classifier_path}")
            print(f"🔹 Loading classifier for '{self.concept}'...")
            classifier = LatentClassifierT(scheduler=self.pipe.scheduler).to(self.device)
            checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
            classifier.load_state_dict(checkpoint["model_state_dict"])
            classifier.eval()
            print(f"✅ Classifier loaded successfully.")

        # ============================================================
        # 🚀 Generation loop
        # ============================================================
        
        for i, (prompt, seed) in tqdm(enumerate(prompts), total=len(prompts),
                                      desc=f"Standard {self.concept}"):
        
            generator = torch.manual_seed(int(seed))
            best_image, best_score = None, float("-inf")
            variants = []
            if use_classifier_guidance:
                # sweep over 24 scales
                classifier_scales = self.config.get(
                    "classifier_scales",
                    list(range(25, 200, 7))  # 24 scales (25 → 200)
                )
                for cls_scale in classifier_scales:
                    image = guided_generation_latent(
                        self.pipe,
                        prompt=prompt,
                        classifier=classifier,
                        device=self.device,
                        classifier_scale=cls_scale,
                        seed=int(seed),
                        num_inference_steps=50,
                        guidance_scale=7.5,
                    )
                    variants.append(image)
                    # score_prompt = (
                    #     "a picture of a dog" if self.concept == "english_springer_spaniel" else prompt
                    # )
                    # score = self.score(image, score_prompt)

                    if debug:
                        fname = f"{self.concept}_{i:03d}_cls{cls_scale}_seed{seed}.png"
                        self.save_image(image, fname, subfolder="debug")

                    # if score > best_score:
                    #     best_score, best_image = score, image
                best_image, best_prob = rank_with_resnet_in_memory(variants, concept=self.concept)

            else:
                generator = torch.manual_seed(int(seed))
                image = self.pipe(prompt, generator=generator).images[0]
                best_image = image

            self.save_image(best_image, f"{self.concept}_{i:03d}_seed{seed}.png")

        print(f"✅ Saved {len(prompts)} images to {self.output_dir}")

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
