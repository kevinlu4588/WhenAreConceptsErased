import os
import argparse
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline, AutoPipelineForInpainting
from PIL import Image
from tqdm import tqdm
import numpy as np


def load_model_inpaint_pipeline(model_folder_path, foundation_path):
    from diffusers import StableDiffusionInpaintPipeline

    fine_tuned_pipeline = StableDiffusionPipeline.from_pretrained(
        model_folder_path, torch_dtype=torch.float16
    ).to("cuda")

    fine_tuned_unet = fine_tuned_pipeline.unet
    fine_tuned_text_encoder = fine_tuned_pipeline.text_encoder

    foundation_pipeline = AutoPipelineForInpainting.from_pretrained(
        foundation_path, torch_dtype=torch.float16
    ).to("cuda")

    foundation_pipeline.unet = fine_tuned_unet
    foundation_pipeline.text_encoder = fine_tuned_text_encoder
    foundation_pipeline.safety_checker = None
    return foundation_pipeline


def load_dataset(file_path):
    return pd.read_csv(file_path)


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(folder_path, filename))
                images.append(img)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return images


def load_single_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def generate_inpainting_images(pipeline, base_images_path, mask_path, prompts_path, output_folder, num_images):
    inpaint_images = []
    dataset = load_dataset(prompts_path).head(num_images)
    images_set = load_images_from_folder(base_images_path)
    mask = load_single_image(mask_path)
    for idx, row in enumerate(tqdm(dataset.iterrows(), desc=f"Generating images in {output_folder}")):
        prompt = row[1]["prompt"]
        seed = row[1]["seed"]
        initial_image = images_set[idx]
        generator = torch.manual_seed(seed)
        image = pipeline(prompt=prompt, image=initial_image, mask_image=mask, generator=generator).images[0]
        last_two_words = "_".join(prompt.split()[-2:])
        image.save(os.path.join(output_folder, f"{last_two_words}_{idx}.png"))
        inpaint_images.append((image, prompt))
    return inpaint_images


def log_results(file_path, content):
    with open(file_path, "a") as file:
        file.write(content + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--foundation_model_path", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--results_dir", type=str, default="./testing_ga")
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--results_file", type=str, default="./results/inpaint_scores.txt")
    args = parser.parse_args()

    concept = args.concept
    method = args.method
    model_path = f"/path/to/your/models/{method}_{concept}"
    if method == "stable_diffusion":
        model_path = "CompVis/stable-diffusion-v1-4"
    if method == "noised":
        model_path = "/path/to/your/models/noised_vangogh_0.0093"

    init_images_path = f"/path/to/your/results/stable_diffusion/{concept}/ground_truth"
    ground_truth_dataset_path = f"/path/to/your/prompts/{concept}.csv"

    output_dir = os.path.join(args.results_dir, method, concept)
    inpaint_dir = os.path.join(output_dir, "inpainting")
    os.makedirs(inpaint_dir, exist_ok=True)

    pipeline = load_model_inpaint_pipeline(model_path, args.foundation_model_path)
    inpaint_images = generate_inpainting_images(
        pipeline, init_images_path, args.mask_path, ground_truth_dataset_path, inpaint_dir, args.num_images
    )

    result_summary = (
        f"Model Type: {method}\n"
        f"Concept: {concept}\n"
        f"Number of iterations: {args.num_images}\n"
        f"---------------------------------------------"
    )
    print(result_summary)
    log_results(args.results_file, result_summary)


if __name__ == "__main__":
    main()