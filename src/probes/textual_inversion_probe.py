import os
import subprocess
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
from probes.base_probe import BaseProbe


class TextualInversionProbe(BaseProbe):
    """
    Probe that:
      1️⃣ Trains a textual inversion embedding on an erased model (self.pipeline_path)
      2️⃣ Evaluates that embedding on the same model
    """

    # ================================================================
    # 🧠 TRAINING
    # ================================================================
    def train_textual_inversion(self):
        """Run textual inversion training for the given concept."""
        concept = self.concept
        erasing_type = self.erasing_type
        initializer_token = getattr(self.config, "initializer_token", "object")

        # ---- Paths ----
        script_path = (
            "/share/u/kevin/DiffusionConceptErasure/new_probe_code/noisy_diffuser_scheduling/"
            "examples/textual_inversion/a_textual_inversion.py"
        )
        erased_model_path = self.pipeline_path  # ✅ pipeline_path = erased model path

        # ✅ Use base_images_root for training data, just like InpaintProbe
        base_images_root = getattr(self.config, "base_images_path", None)
        if base_images_root is None:
            base_images_root = "/share/u/kevin/DiffusionConceptErasure/new_probe_code/results/base_model"

        data_dir = os.path.join(base_images_root, concept, "standardpromptprobe")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Training images folder not found: {data_dir}")

        # ✅ Nested output directory: textual_inversion/<erasing_type>/<concept>
        output_dir = os.path.join(self.output_dir, "textual_inversion", erasing_type, concept)
        os.makedirs(output_dir, exist_ok=True)

        print(f"🚀 Training textual inversion for '{concept}' using erased model:\n  {erased_model_path}")
        print(f"📂 Training data: {data_dir}")

        cmd = [
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            script_path,
            f"--pretrained_model_name_or_path={erased_model_path}",
            f"--train_data_dir={data_dir}",
            "--learnable_property=object",
            f"--placeholder_token=<{concept}>",
            f"--initializer_token={initializer_token}",
            "--resolution=512",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=4",
            "--max_train_steps=3000",
            "--learning_rate=5e-4",
            "--scale_lr",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            f"--output_dir={output_dir}",
        ]

        subprocess.run(cmd, check=True)
        print(f"✅ Finished training textual inversion for '{concept}'")
        return output_dir

    # ================================================================
    # 🧪 EVALUATION
    # ================================================================
    def run(self, num_images=None, retrain=False):
        """
        Runs evaluation using a textual inversion embedding trained on the erased model.
        If retrain=True, forces retraining even if the embedding already exists.
        """
        concept = self.concept
        erasing_type = self.erasing_type

        # ✅ Updated directory layout
        ti_dir = os.path.join(self.output_dir, "textual_inversion", erasing_type, concept)
        learned_embeds_path = os.path.join(ti_dir, "learned_embeds.safetensors")

        # ---- Train if needed ----
        if retrain or not os.path.exists(learned_embeds_path):
            print("🧩 No existing embedding found — starting training...")
            ti_dir = self.train_textual_inversion()
            learned_embeds_path = os.path.join(ti_dir, "learned_embeds.safetensors")
        else:
            print(f"📦 Using cached embedding: {learned_embeds_path}")

        # ---- Load the erased model (pipeline) ----
        print(f"🔹 Loading erased model from {self.pipeline_path}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.pipeline_path, torch_dtype=torch.float16
        ).to(self.device)
        self.pipe.load_textual_inversion(learned_embeds_path)
        self.pipe.safety_checker = None

        # ---- Generate images (same as StandardPromptProbe) ----
        prompts = self._load_prompts(num_images)
        for i, (prompt, seed) in tqdm(
            enumerate(prompts), total=len(prompts), desc=f"Textual Inversion {self.concept}"
        ):
            generator = torch.manual_seed(int(seed))
            prompt = prompt.replace(concept, f"<{concept}>")

            image = self.pipe(prompt, generator=generator).images[0]
            self.save_image(image, f"{self.concept}_{i:03d}_seed{seed}_ti.png")

        print(f"✅ Saved {len(prompts)} textual inversion eval images to {self.output_dir}")
