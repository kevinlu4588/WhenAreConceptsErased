import os
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
from torchvision.models import resnet50, ResNet50_Weights

def classify_image(images_path, model, preprocess, weights, topk=5, device="cuda:0"):
    """Classifies an image using ResNet50 and returns top-k categories and scores."""
    image = Image.open(images_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor).softmax(dim=1)
        top_probs, top_indices = torch.topk(predictions, topk, dim=1)

    top_categories = [weights.meta['categories'][idx] for idx in top_indices[0].cpu().numpy()]
    top_scores = top_probs[0].cpu().numpy().tolist()
    
    return top_categories, top_scores

def classification(root_path, output_csv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.DEFAULT
    resnet_model = resnet50(weights=weights).to(device)
    resnet_model.eval()
    preprocess = weights.transforms()

    results = []
    scheduler_agg_counts = defaultdict(lambda: {"top1_hits": 0, "top5_hits": 0, "total": 0})
    scheduler_method_counts = defaultdict(lambda: defaultdict(lambda: {"top1_hits": 0, "top5_hits": 0, "total": 0}))

    root = Path(root_path)
    for method_path in root.iterdir():
        if not method_path.is_dir():
            continue
        method = method_path.name
        for concept_path in method_path.iterdir():
            if not concept_path.is_dir():
                continue
            concept = concept_path.name
            concept_cleaned = concept.replace("_", " ").lower()
            if concept_cleaned == "chainsaw":
                concept_cleaned = "chain saw"
            for scheduler_path in concept_path.iterdir():
                if not scheduler_path.is_dir():
                    continue
                scheduler = scheduler_path.name

                image_files = list(scheduler_path.glob("*.png"))
                for img_path in image_files:
                    try:
                        top_categories, top_scores = classify_image(img_path, resnet_model, preprocess, weights, topk=5, device=device)

                        in_top1 = concept_cleaned == top_categories[0].lower()
                        in_top5 = any(concept_cleaned == cat.lower() for cat in top_categories)

                        results.append({
                            "method": method,
                            "concept": concept,
                            "scheduler": scheduler,
                            "image": img_path.name,
                            "in_top1": in_top1,
                            "in_top5": in_top5,
                        })

                        # === Count stats ===
                        scheduler_agg_counts[scheduler]["total"] += 1
                        scheduler_agg_counts[scheduler]["top1_hits"] += int(in_top1)
                        scheduler_agg_counts[scheduler]["top5_hits"] += int(in_top5)

                        scheduler_method_counts[scheduler][method]["total"] += 1
                        scheduler_method_counts[scheduler][method]["top1_hits"] += int(in_top1)
                        scheduler_method_counts[scheduler][method]["top5_hits"] += int(in_top5)

                    except Exception as e:
                        print(f"Failed on {img_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Classification results saved to: {output_csv}")

    print("\n=== Classification Accuracy by Scheduler ===")
    for scheduler, counts in scheduler_agg_counts.items():
        total = counts["total"]
        top1_pct = 100.0 * counts["top1_hits"] / total if total else 0.0
        top5_pct = 100.0 * counts["top5_hits"] / total if total else 0.0
        print(f"{scheduler:<10} Top-1: {top1_pct:.2f}% | Top-5: {top5_pct:.2f}%")

    print("\n=== Classification Accuracy by Scheduler and Method ===")
    for scheduler in scheduler_method_counts:
        print(f"\n{scheduler}")
        for method in scheduler_method_counts[scheduler]:
            stats = scheduler_method_counts[scheduler][method]
            total = stats["total"]
            top1_pct = 100.0 * stats["top1_hits"] / total if total else 0.0
            top5_pct = 100.0 * stats["top5_hits"] / total if total else 0.0
            print(f"  {method:<6} → Top-1: {top1_pct:.2f}% | Top-5: {top5_pct:.2f}%")


if __name__ == "__main__":
    root_path = "/path/to/your/ablation"
    output_csv = "ablation_classification.csv"

    classification(
        root_path=root_path,
        output_csv=output_csv
    )
