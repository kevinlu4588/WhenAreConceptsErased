from tqdm import tqdm
import torch
from noisy_diffuser_scheduling.schedulers.eta_ddim_scheduler import DDIMScheduler
from probes.base_probe import BaseProbe


class NoiseBasedProbe(BaseProbe):
    def run(self, num_images=None, debug=False):
        """Generate noisy variants and keep the best-scoring image per prompt.

        Args:
            num_images (int, optional): number of images to generate.
            debug (bool): if True, save all noisy variants in a 'debug' subfolder
                          inside this probe's output directory.
        """
        prompts = self._load_prompts(num_images)

        # Backup and replace scheduler
        original_scheduler = self.pipe.scheduler
        self.pipe.scheduler = DDIMScheduler.from_pretrained(
            self.pipeline_path, subfolder="scheduler"
        )

        for i, (prompt, seed) in tqdm(
            enumerate(prompts), total=len(prompts), desc=f"Noisy {self.concept}"
        ):
            generator = torch.manual_seed(int(seed))
            best_image, best_score = None, float("-inf")
            print(self.config)
            for eta in self.config['eta_values']:
                for vscale in self.config['variance_scales']:
                    image = self.pipe(
                        prompt, generator=generator, eta=eta, variance_scale=vscale
                    ).images[0]
                    if self.concept == "english_springer_spaniel":
                        score_prompt = "a picture of a dog"
                    else:
                        score_prompt = prompt
                    score = self.score(image, score_prompt)

                    # Save all variants to the debug subfolder (using BaseProbe.save_image)
                    if debug:
                        debug_filename = (
                            f"{self.concept}_{i:03d}_eta{eta:.2f}_vs{vscale:.2f}_seed{seed}.png"
                        )
                        self.save_image(image, debug_filename, subfolder="debug")

                    # Track best-scoring image
                    if score > best_score:
                        best_score, best_image = score, image

            # Save only best image to the main probe output folder
            self.save_image(
                best_image, f"{self.concept}_{i:03d}_seed{seed}_best.png"
            )

        # Restore scheduler
        self.pipe.scheduler = original_scheduler
        print(f"✅ Finished noise-based probing for {self.concept}")

    # ------------------------------------------------------------------
    def score(self, image, prompt):
        """Default scoring logic — subclasses can override."""
        if "score_type" not in self.config:
            return 0.0
        if self.config['score_type'].lower() == "clip":
            return self._clip_score(image, prompt)
        elif self.config['score_type'].lower() == "classification":
            return self._classifier_score(image)
        else:
            raise ValueError(f"Unknown score type: {self.config['score_type']}")
