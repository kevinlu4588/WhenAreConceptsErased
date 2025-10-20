import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Root directory
root = Path("/path/to/your/ablation")

# Collect scores: (method, concept, scheduler) -> [scores]
clip_scores = defaultdict(list)

# Walk through structure
for method_path in root.iterdir():
    if not method_path.is_dir():
        continue
    method = method_path.name
    for concept_path in method_path.iterdir():
        if not concept_path.is_dir():
            continue
        concept = concept_path.name
        for scheduler_path in concept_path.iterdir():
            if not scheduler_path.is_dir():
                continue
            scheduler = scheduler_path.name

            image_files = list(scheduler_path.glob("*.png"))
            for img_path in tqdm(image_files, desc=f"{method}/{concept}/{scheduler}"):
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = clip_processor(text=[concept], images=image, return_tensors="pt", padding=True).to("cuda")
                    outputs = clip_model(**inputs)
                    score = outputs.logits_per_image[0].item()
                    clip_scores[(method, concept, scheduler)].append(score)
                except Exception as e:
                    print(f"Failed on {img_path}: {e}")

# Compute averages
print("\n=== Average CLIP Scores ===")
for key, scores in clip_scores.items():
    method, concept, scheduler = key
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"{method:<8} {concept:<25} {scheduler:<10} {avg_score:.4f}")

# Compute averages per scheduler
scheduler_aggregate = defaultdict(list)
for (method, concept, scheduler), scores in clip_scores.items():
    scheduler_aggregate[scheduler].extend(scores)

print("\n=== Average CLIP Scores by Scheduler ===")
for scheduler, scores in scheduler_aggregate.items():
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"{scheduler:<10} {avg_score:.4f}")


method_scheduler_aggregate = defaultdict(list)
for (method, concept, scheduler), scores in clip_scores.items():
    method_scheduler_aggregate[(method, scheduler)].extend(scores)

print("\n=== Average CLIP Scores by Method and Scheduler ===")
for (method, scheduler), scores in sorted(method_scheduler_aggregate.items()):
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"{method:<8} {scheduler:<10} {avg_score:.4f}")