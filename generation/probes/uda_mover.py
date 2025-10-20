import os
import csv
import argparse
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

def calculate_clip_score(image_path, prompt, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return outputs.logits_per_image.item()

def main():
    parser = argparse.ArgumentParser(description="Select and save best CLIP-scored images.")
    parser.add_argument("--methods", nargs='+', required=True, help="List of method names")
    parser.add_argument("--concepts", nargs='+', required=True, help="List of concept names")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images per concept")
    parser.add_argument("--prompts_dir", type=str, required=True, help="Directory containing prompt CSVs")
    parser.add_argument("--image_base_dir", type=str, required=True, help="Base directory containing generated images")
    parser.add_argument("--results_dir", type=str, default="./final_results_big", help="Directory to save best images")
    parser.add_argument("--results_file", type=str, default="./results/unlearned_results.txt", help="File to save average scores")
    args = parser.parse_args()

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    average_scores = []

    for method in args.methods:
        for concept in args.concepts:
            prompts_path = os.path.join(args.prompts_dir, f"{concept}.csv")
            with open(prompts_path, "r") as f:
                reader = csv.DictReader(f)
                prompts = [row['prompt'] for row in reader]

            highest_scores = []

            for i in range(args.num_images):
                folder_path = os.path.join(
                    args.image_base_dir, method, concept, f"{method}_{concept}_attack_idx_{i}", "images"
                )
                if not os.path.exists(folder_path):
                    print(f"Folder does not exist: {folder_path}. Skipping.")
                    continue

                prompt = prompts[i]
                best_image = None
                highest_score = float("-inf")

                for image_file in os.listdir(folder_path):
                    if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    image_path = os.path.join(folder_path, image_file)
                    clip_score = calculate_clip_score(image_path, prompt, model, processor)
                    if clip_score > highest_score:
                        highest_score = clip_score
                        best_image = image_path

                if best_image:
                    concept_result_dir = os.path.join(args.results_dir, method, concept)
                    Path(concept_result_dir).mkdir(parents=True, exist_ok=True)
                    best_image_save_path = os.path.join(concept_result_dir, f"image_{i}_{prompt}.png")
                    Image.open(best_image).save(best_image_save_path)
                    highest_scores.append(highest_score)

            if highest_scores:
                avg_score = sum(highest_scores) / len(highest_scores)
                average_scores.append(f"{method}/{concept}: {avg_score:.4f}")

    with open(args.results_file, "w") as f:
        f.write("\n".join(average_scores))

    print("Processing completed. Results saved.")

if __name__ == "__main__":
    main()
