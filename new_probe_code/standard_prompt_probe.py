from tqdm import tqdm
from base_probe import BaseProbe
import torch

class StandardPromptProbe(BaseProbe):
    def run(self, num_images=None):
        """Generate standard diffusion outputs for the given number of prompts."""
        prompts = self._load_prompts(num_images)
        for i, (prompt, seed) in tqdm(enumerate(prompts), total=len(prompts), desc=f"Standard {self.concept}"):
            generator = torch.manual_seed(int(seed))
            image = self.pipe(prompt, generator=generator).images[0]
            self.save_image(image, f"{self.concept}_{i:03d}_seed{seed}.png")

        print(f"✅ Saved {len(prompts)} images to {self.output_dir}")
