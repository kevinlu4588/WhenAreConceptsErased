import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class Evaluator:
    def __init__(self, results_root, output_dir=None, device=None):
        self.results_root = results_root
        self.output_dir = output_dir or os.path.join(results_root, "evaluations")
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔹 Using device: {self.device}")

        # Load CLIP model + processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # --- Default mask size (matches your InpaintProbe) ---
        self.default_mask_size = (256, 256)

    # ================================================================
    # 🧮 Compute CLIP score
    # ================================================================
    def clip_score(self, image, prompt):
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # (1, 1)
        return logits_per_image.item()

    # ================================================================
    # 🚀 Main evaluation
    # ================================================================
    def evaluate(self, erasing_types=None, concepts=None, probes=None, overwrite=False):
        erasing_types = erasing_types or self._list_dirs(self.results_root)
        all_rows = []

        for erasing_type in erasing_types:
            et_path = os.path.join(self.results_root, erasing_type)
            if not os.path.isdir(et_path):
                continue

            concepts_to_eval = concepts or self._list_dirs(et_path)
            for concept in concepts_to_eval:
                concept_path = os.path.join(et_path, concept)
                if not os.path.isdir(concept_path):
                    continue

                probes_to_eval = probes or self._list_dirs(concept_path)
                for probe_name in probes_to_eval:
                    probe_path = os.path.join(concept_path, probe_name)
                    if not os.path.isdir(probe_path):
                        continue

                    print(f"📂 Evaluating {erasing_type}/{concept}/{probe_name}")

                    prompt = self._concept_to_prompt(concept)
                    for fname in tqdm(sorted(os.listdir(probe_path))):
                        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                            continue

                        fpath = os.path.join(probe_path, fname)
                        image = Image.open(fpath).convert("RGB")

                        try:
                            # --- Special handling for inpainting probes ---
                            if "inpaint" in probe_name.lower():
                                image = self._crop_to_mask(image)

                            score = self.clip_score(image, prompt)
                            all_rows.append({
                                "erasing_type": erasing_type,
                                "concept": concept,
                                "probe": probe_name,
                                "filename": fname,
                                "clip_score": score,
                            })
                        except Exception as e:
                            print(f"⚠️ Failed on {fpath}: {e}")

        # --- Save raw CSV ---
        raw_csv_path = os.path.join(self.output_dir, "clip_scores_raw.csv")
        pd.DataFrame(all_rows).to_csv(raw_csv_path, index=False)
        print(f"✅ Raw CLIP scores saved to: {raw_csv_path}")

        # --- Save averaged CSV ---
        if all_rows:
            df = pd.DataFrame(all_rows)
            df_avg = (
                df.groupby(["erasing_type", "concept", "probe"], as_index=False)
                .agg({"clip_score": "mean"})
                .sort_values(["concept", "erasing_type", "probe"])
            )
            avg_csv_path = os.path.join(self.output_dir, "clip_scores_averaged.csv")
            df_avg.to_csv(avg_csv_path, index=False)
            print(f"✅ Averaged CLIP scores saved to: {avg_csv_path}")

    # ================================================================
    # 🖼️ Mask cropping for inpainting
    # ================================================================
    def _crop_to_mask(self, image):
        """Crop image to the center masked region (same as inpainting probe)."""
        w, h = image.size
        mask_w, mask_h = self.default_mask_size
        left = (w - mask_w) // 2
        top = (h - mask_h) // 2
        right = left + mask_w
        bottom = top + mask_h
        cropped = image.crop((left, top, right, bottom))
        return cropped

    # ================================================================
    # 🔧 Helpers
    # ================================================================
    def _list_dirs(self, path):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    def _concept_to_prompt(self, concept):
        """Map concept name to a natural prompt."""
        concept_words = concept.replace("_", " ")
        style_keywords = ["van gogh", "picasso", "andy warhol", "monet", "vangogh"]
        if any(k in concept_words.lower() for k in style_keywords):
            return f"a painting in the style of {concept_words}"
        else:
            return f"a picture of a {concept_words}"


if __name__ == "__main__":
    evaluator = Evaluator(results_root="/share/u/kevin/DiffusionConceptErasure/new_probe_code/results")
    evaluator.evaluate()  # defaults to everything
