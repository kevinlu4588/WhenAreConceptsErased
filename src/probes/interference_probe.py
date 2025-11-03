import os
import torch
from tqdm import tqdm
import pandas as pd
from PIL import Image
from probes.base_probe import BaseProbe

# ============================================================
# 🧠 Interference Probe
# ============================================================
class InterferenceProbe(BaseProbe):
    """
    Probe to test interference — generates images from prompts of *other* concepts
    using the current erased model (self.pipe).
    """

    def run(self, num_images=10, debug=False):
        """
        Generate images for prompts from other concept CSVs and (optionally) from the erased concept itself.
        Each other concept contributes up to `num_images` images.
        """
        device = self.device
        prompt_dir = self.root_data_dir / "prompts"
        all_csvs = [
            os.path.join(prompt_dir, f)
            for f in os.listdir(prompt_dir)
            if f.endswith(".csv")
        ]

        # ============================================================
        # Load prompts from other concepts (first N per concept)
        # ============================================================
        target_csv = os.path.join(prompt_dir, f"{self.concept}.csv")
        other_csvs = [csv for csv in all_csvs if csv != target_csv]

        print(f"🚀 Running InterferenceProbe for concept='{self.concept}'")
        print(f"Found {len(other_csvs)} other concept CSVs.")

        # ============================================================
        # Run inference on erased model
        # ============================================================
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

        for csv in other_csvs:
            other_concept = os.path.splitext(os.path.basename(csv))[0]

            try:
                df = pd.read_csv(csv)
                if "prompt" in df.columns and "seed" in df.columns:
                    subset = df.head(min(num_images, len(df)))[["prompt", "seed"]].values.tolist()
                else:
                    prompts = df.iloc[:, 0].dropna().tolist()
                    subset = [(p, 0) for p in prompts[:num_images]]
            except Exception as e:
                print(f"⚠️ Could not load {csv}: {e}")
                continue

            print(f"🔹 Generating {len(subset)} images for concept '{other_concept}'")

            for i, (prompt, seed) in tqdm(
                enumerate(subset),
                total=len(subset),
                desc=f"Interference {self.concept} → {other_concept}"
            ):
                generator = torch.manual_seed(int(seed))
                image = self.pipe(
                    prompt,
                    generator=generator,
                    num_inference_steps=self.num_inference_steps
                ).images[0]

                # Save under subfolder for each other concept
                fname = f"{self.concept}_{other_concept}_{i:03d}.png"
                self.save_image(image, fname, subfolder=f"{other_concept}")

        # ============================================================
        # (Debug) Also test prompts from the erased concept itself
        # ============================================================
        if debug and os.path.exists(target_csv):
            print(f"🔍 Debug mode ON — generating first {num_images} samples for erased concept '{self.concept}'")

            df = pd.read_csv(target_csv)
            if "prompt" in df.columns and "seed" in df.columns:
                debug_prompts = df.head(min(num_images, len(df)))[["prompt", "seed"]].values.tolist()
            else:
                prompts = df.iloc[:, 0].dropna().tolist()
                debug_prompts = [(p, 0) for p in prompts[:num_images]]

            for j, (prompt, seed) in tqdm(
                enumerate(debug_prompts),
                total=len(debug_prompts),
                desc=f"Debug (original concept: {self.concept})"
            ):
                generator = torch.manual_seed(int(seed))
                print(prompt)
                image = self.pipe(
                    prompt,
                    generator=generator,
                ).images[0]

                fname = f"{self.concept}_debug_{j:03d}.png"
                self.save_image(image, fname, subfolder="debug_original")

        print(f"✅ Saved interference images to {self.output_dir}")
