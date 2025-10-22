import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision.models import resnet50, ResNet50_Weights


class Evaluator:
    def __init__(self, results_root, output_dir=None, device=None):
        self.results_root = results_root
        self.output_dir = output_dir or os.path.join(results_root, "evaluations")
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔹 Using device: {self.device}")

        # ---- CLIP ----
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # ---- ResNet-50 ----
        self.resnet_weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=self.resnet_weights).to(self.device)
        self.resnet.eval()
        self.resnet_preproc = self.resnet_weights.transforms()

        # ---- Defaults ----
        self.default_mask_size = (256, 256)
        self.style_concepts = ["van_gogh", "picasso", "andy_warhol", "monet", "vangogh"]

    # ================================================================
    # 🧮 CLIP scoring
    # ================================================================
    def clip_score(self, image, prompt):
        inputs = self.clip_proc(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item()

    # ================================================================
    # 🧠 Classifier scoring (Top-1 and Top-5)
    # ================================================================
    def classifier_topk(self, image, concept, topk=5):
        """Return tuple (top1_hit, top5_hit)"""
        if any(k in concept for k in self.style_concepts):
            return None, None  # skip style-based concepts

        x = self.resnet_preproc(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.resnet(x).softmax(dim=1)
            top_probs, top_indices = torch.topk(preds, topk, dim=1)

        labels = [self.resnet_weights.meta["categories"][i].lower() for i in top_indices[0].cpu().numpy()]
        concept_clean = concept.replace("_", " ").lower()
        top1_hit = int(concept_clean in labels[0])  # exact Top-1
        top5_hit = int(any(concept_clean in label for label in labels))
        return top1_hit, top5_hit

    # ================================================================
    # 🚀 Evaluation loop
    # ================================================================
    def evaluate(self, erasing_types=None, concepts=None, probes=None):
        erasing_types = erasing_types or self._list_dirs(self.results_root)
        all_rows = []

        for erasing_type in erasing_types:
            et_path = os.path.join(self.results_root, erasing_type)
            if not os.path.isdir(et_path):
                continue

            for concept in (concepts or self._list_dirs(et_path)):
                concept_path = os.path.join(et_path, concept)
                if not os.path.isdir(concept_path):
                    continue

                for probe_name in (probes or self._list_dirs(concept_path)):
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

                        # Crop for inpainting probes
                        if "inpaint" in probe_name.lower():
                            image = self._crop_to_mask(image)

                        try:
                            clip = self.clip_score(image, prompt)
                            top1, top5 = self.classifier_topk(image, concept)
                            all_rows.append({
                                "erasing_type": erasing_type,
                                "concept": concept,
                                "probe": probe_name,
                                "filename": fname,
                                "clip_score": clip,
                                "classifier_top1": top1,
                                "classifier_top5": top5,
                            })
                        except Exception as e:
                            print(f"⚠️ Failed on {fpath}: {e}")

        df = pd.DataFrame(all_rows)
        raw_csv = os.path.join(self.output_dir, "evaluation_raw.csv")
        df.to_csv(raw_csv, index=False)
        print(f"✅ Raw results saved to {raw_csv}")

        if not df.empty:
            df_avg = (
                df.groupby(["erasing_type", "concept", "probe"], as_index=False)
                .agg({
                    "clip_score": "mean",
                    "classifier_top1": lambda x: x.dropna().mean() * 100,
                    "classifier_top5": lambda x: x.dropna().mean() * 100,
                })
                .rename(columns={
                    "classifier_top1": "classifier_top1_acc",
                    "classifier_top5": "classifier_top5_acc",
                })
                .sort_values(["concept", "erasing_type", "probe"])
            )
            avg_csv = os.path.join(self.output_dir, "evaluation_averaged.csv")
            df_avg.to_csv(avg_csv, index=False)
            print(f"✅ Averaged results saved to {avg_csv}")

    # ================================================================
    # 🔧 Helpers
    # ================================================================
    def _crop_to_mask(self, image):
        w, h = image.size
        mw, mh = self.default_mask_size
        left, top = (w - mw) // 2, (h - mh) // 2
        right, bottom = left + mw, top + mh
        return image.crop((left, top, right, bottom))

    def _list_dirs(self, path):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    def _concept_to_prompt(self, concept):
        words = concept.replace("_", " ")
        if any(k in words.lower() for k in self.style_concepts):
            return f"a painting in the style of {words}"
        else:
            return f"a photo of a {words}"


if __name__ == "__main__":
    evaluator = Evaluator("classifier_results")
    evaluator.evaluate()
