import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models with CLIP scores.")
    parser.add_argument("--methods", nargs="+", required=True, help="List of methods (e.g., 'tv').")
    parser.add_argument("--concepts", nargs="+", required=True, help="List of concepts (e.g., 'garbage_truck', 'van_gogh').")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate and evaluate.")
    parser.add_argument("--base_ground_truth_path", type=str, required=True, help="Base path for ground truth datasets.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Base path for model checkpoints.")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results.")
    return parser.parse_args()

def load_dataset(file_path):
    return pd.read_csv(file_path)

def generate_images(dataset, pipeline, output_folder):
        images = []
        for idx, row in enumerate(tqdm(dataset.iterrows(), desc=f"Generating images in {output_folder}")):
            prompt = row[1]["prompt"]
            seed = row[1]["seed"]
            generator = torch.manual_seed(seed)
            image = pipeline(prompt, generator=generator).images[0]
            last_two_words = "_".join(prompt.split()[-2:])
            image_path = os.path.join(output_folder, f"{last_two_words}_{idx}.png")
            image.save(image_path)
            images.append((image, prompt))
        return images

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    for method in args.methods:
        for concept in args.concepts:
            ground_truth_path = os.path.join(args.base_ground_truth_path, f"{concept}.csv")
            model_path = ""
            if method == "stable_diffusion":
                model_path = "CompVis/stable-diffusion-v1-4"
            else:
                model_path = os.path.join(args.base_model_path, f"{method}_{concept}")
                if not os.path.exists(ground_truth_path) or not os.path.exists(model_path):
                    print(f"Missing ground truth or model for concept: {concept}, method: {method}")
                    continue

            model_type = method

            output_dir = os.path.join(args.results_dir, model_type, concept)
            ground_truth_dir = os.path.join(output_dir, "ground_truth")
            os.makedirs(ground_truth_dir, exist_ok=True)

            pipeline = StableDiffusionPipeline.from_pretrained(model_path).to("cuda")
            pipeline.safety_checker = None
            ground_truth_dataset = load_dataset(ground_truth_path)
            ground_truth_images = generate_images(ground_truth_dataset.head(args.num_images), pipeline, ground_truth_dir)

if __name__ == "__main__":
    main()
