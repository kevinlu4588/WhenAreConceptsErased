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

        self.default_mask_size = (256, 256)
        self.style_concepts = ["van_gogh", "picasso", "andy_warhol", "monet", "vangogh"]

    # ================================================================
    # 🧮 Scoring methods
    # ================================================================
    def clip_score(self, image, prompt):
        inputs = self.clip_proc(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item()

    def classifier_topk(self, image, concept, topk=5):
        """Return tuple (top1_hit, top5_hit) with concept-specific matching rules."""
        if any(k in concept for k in self.style_concepts):
            return None, None

        x = self.resnet_preproc(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.resnet(x).softmax(dim=1)
            top_probs, top_indices = torch.topk(preds, topk, dim=1)

        labels = [self.resnet_weights.meta["categories"][i].lower() for i in top_indices[0].cpu().numpy()]
        concept_clean = concept.replace("_", " ").lower()

        def match(concept_name, label):
            if concept_name == "airliner":
                return "plane" in label or "airliner" in label
            elif concept_name == "garbage_truck":
                return "truck" in label or "garbage" in label
            elif concept_name == "golf_ball":
                return "ball" in label
            return concept_name in label

        top1_hit = int(match(concept_clean, labels[0]))
        top5_hit = int(any(match(concept_clean, lbl) for lbl in labels))
        return top1_hit, top5_hit

    # ================================================================
    # 🚀 Evaluation entry point
    # ================================================================
    def evaluate(self, erasing_types=None, concepts=None, probes=None):
        all_rows = []
        erasing_types = erasing_types or self._list_dirs(self.results_root)

        for erasing_type in erasing_types:
            for concept in (concepts or self._list_dirs(os.path.join(self.results_root, erasing_type))):
                concept_path = os.path.join(self.results_root, erasing_type, concept)
                for probe_name in (probes or self._list_dirs(concept_path)):
                    probe_path = os.path.join(concept_path, probe_name)
                    if not os.path.isdir(probe_path):
                        continue

                    print(f"📂 Evaluating {erasing_type}/{concept}/{probe_name}")
                    if self._is_interference_probe(probe_name):
                        rows = self._evaluate_interference_probe(erasing_type, concept, probe_name, probe_path)
                    else:
                        rows = self._evaluate_normal_probe(erasing_type, concept, probe_name, probe_path)
                    all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        raw_csv = os.path.join(self.output_dir, "evaluation_raw.csv")
        df.to_csv(raw_csv, index=False)
        print(f"✅ Raw results saved to {raw_csv}")

        if not df.empty:
            self._aggregate_results(df)

    # ================================================================
    # 🧩 Probe-level evaluators
    # ================================================================
    def _evaluate_normal_probe(self, erasing_type, concept, probe_name, probe_path):
        prompt = self._concept_to_prompt(concept)
        rows = []

        for fname in tqdm(sorted(os.listdir(probe_path)), desc=f"{concept}/{probe_name}"):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            image = Image.open(os.path.join(probe_path, fname)).convert("RGB")
            if "inpaint" in probe_name.lower():
                image = self._crop_to_mask(image)
            try:
                clip = self.clip_score(image, prompt)
                top1, top5 = self.classifier_topk(image, concept)
                rows.append({
                    "erasing_type": erasing_type,
                    "concept": concept,
                    "probe": probe_name,
                    "filename": fname,
                    "clip_score": clip,
                    "classifier_top1": top1,
                    "classifier_top5": top5,
                })
            except Exception as e:
                print(f"⚠️ Failed on {fname}: {e}")
        return rows

    def _evaluate_interference_probe(self, erasing_type, erased_concept, probe_name, probe_path):
        rows = []
        for target_concept in self._list_dirs(probe_path):
            target_path = os.path.join(probe_path, target_concept)
            print(f"  ↳ Interference target: {target_concept}")

            for fname in tqdm(sorted(os.listdir(target_path)), desc=f"Interf {erased_concept}→{target_concept}"):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                image = Image.open(os.path.join(target_path, fname)).convert("RGB")
                try:
                    prompt = self._concept_to_prompt(target_concept)
                    clip = self.clip_score(image, prompt)
                    top1, top5 = self.classifier_topk(image, target_concept)
                    rows.append({
                        "erasing_type": erasing_type,
                        "erased_concept": erased_concept,
                        "target_concept": target_concept,
                        "probe": probe_name,
                        "filename": fname,
                        "clip_score": clip,
                        "classifier_top1": top1,
                        "classifier_top5": top5,
                    })
                except Exception as e:
                    print(f"⚠️ Failed on {fname}: {e}")
        return rows

    # ================================================================
    # 📊 Aggregation
    # ================================================================
    def _aggregate_results(self, df):
        interference_df = df[df["probe"].str.contains("interference", case=False, na=False)]
        normal_df = df[~df["probe"].str.contains("interference", case=False, na=False)]
        results = []

        def _aggregate(df, group_cols):
            return (
                df.groupby(group_cols, as_index=False)
                .agg({
                    "clip_score": "mean",
                    "classifier_top1": lambda x: x.dropna().mean() * 100,
                    "classifier_top5": lambda x: x.dropna().mean() * 100,
                })
                .rename(columns={
                    "classifier_top1": "classifier_top1_acc",
                    "classifier_top5": "classifier_top5_acc",
                })
            )

        if not normal_df.empty:
            results.append(_aggregate(normal_df, ["erasing_type", "concept", "probe"]))
        if not interference_df.empty:
            results.append(_aggregate(interference_df, ["erasing_type", "erased_concept", "target_concept", "probe"]))

        if results:
            df_avg = pd.concat(results, ignore_index=True)
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
        return f"a photo of a {words}"

    def _is_interference_probe(self, probe_name):
        return "interference" in probe_name.lower()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate diffusion model results using CLIP and ResNet classifiers.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the root directory containing model outputs to evaluate."
    )
    args = parser.parse_args()

    evaluator = Evaluator(results_root=args.results_dir)
    evaluator.evaluate()
