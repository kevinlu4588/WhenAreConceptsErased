import os
import csv
from pathlib import Path
from PIL import Image
import torch
import argparse

from eta_diffusion import FineTunedModel, StableDiffuser
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from eta_diffusers.src.diffusers.schedulers.eta_ddim_scheduler import DDIMScheduler

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion and evaluate with CLIP.")
    parser.add_argument("--methods", nargs="+", required=True, help="List of methods to use.")
    parser.add_argument("--concepts", nargs="+", required=True, help="List of concepts.")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images per concept.")
    parser.add_argument("--prompts_dir", type=str, required=True, help="Directory containing concept prompt CSVs.")
    parser.add_argument("--results_dir", type=str, default="./noisy_attack_results", help="Directory to save results.")
    parser.add_argument("--etas", nargs="+", type=float, default=[1.3, 1.5, 1.7, 1.75, 1.8, 1.85, 1.9], help="List of eta values.")
    parser.add_argument("--variance_scales", nargs="+", type=float, default=[0.97, 1.0, 1.02, 1.03, 1.035, 1.04], help="List of variance scale values.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the script on (e.g., 'cuda', 'cpu').")
    return parser.parse_args()

def calculate_clip_score(image_path, prompt, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.item()

def save_image(image, folder_path, file_name):
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    image.save(folder_path / file_name)

def main():
    args = parse_arguments()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Generate images and save them
    for method in args.methods:
        for concept in args.concepts:
            if method == "noised":
                pipeline = StableDiffusionPipeline.from_pretrained(
                "/path/to/your/models/noised_vangogh_0.0093",
                torch_dtype=torch.float16,
            )
            elif method == "noised":
                pipeline = StableDiffusionPipeline.from_pretrained(
                "/path/to/your/models/noised_vangogh_0.01",
                torch_dtype=torch.float16,
            )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                f"/path/to/your/models/{method}_{concept}",
                torch_dtype=torch.float16,
            )
            pipeline.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
            pipeline.to(args.device)
            pipeline.safety_checker = None

            prompts_path = os.path.join(args.prompts_dir, f"{concept}.csv")
            prompts = []
            with open(prompts_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompts.append(row["prompt"])

            for i in range(args.num_images):
                for eta in args.etas:
                    for variance_scale in args.variance_scales:
                        image = pipeline(prompts[i], eta=eta, variance_scale=variance_scale).images[0]
                        folder_path = f"{args.results_dir}/{method}/{concept}/images_{i}"
                        file_name = f"eta_{eta}_vscale_{variance_scale}.png"
                        save_image(image, folder_path, file_name)

if __name__ == "__main__":
    main()
