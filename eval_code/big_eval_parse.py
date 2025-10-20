import os
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
objects_list = [
    "english_springer_spaniel",
    "airliner",
    "garbage_truck",
    "parachute",
    "cassette_player",
    "chainsaw",
    "tench",
    "french_horn",
    "golf_ball",
    "church"
]
def compute_clip_score(image_path, prompt):
    """
    Computes the CLIP score for an image and a prompt.
    """
    image = Image.open(image_path).convert("RGB")
    mask_path = "/path/to/your/masks/center_square_mask.png"
    if "inpaint" in image_path:
        mask = Image.open(mask_path).convert("L")  # Load mask as grayscale
        mask_array = np.array(mask)
        white_coords = np.where(mask_array == 255)
        y_min, y_max = white_coords[0].min(), white_coords[0].max()
        x_min, x_max = white_coords[1].min(), white_coords[1].max()

        clip_scores = []            # Crop the image to the masked area
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        # Prepare input for the CLIP model
        inputs = processor(text=[prompt], images=cropped_image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image.item()
    else:
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        # print(prompt)
        # print(logits_per_image.item())
        return logits_per_image.item()
def process_clip_scores(root_path, prompts_path, methods, concepts, evals, output_dir, cases=5):
    """
    Processes CLIP scores for the given methods, concepts, and evaluations.
    """
    os.makedirs(output_dir, exist_ok=True)

    for method in methods:
        method_data = []  # Collect rows for this method
        
        for concept in concepts:
            print("processing", method, concept)
            # Load prompts for the current concept
            prompt_file = os.path.join(prompts_path, f"{concept}.csv")
            if not os.path.exists(prompt_file):
                print(f"Prompts file not found: {prompt_file}")
                continue

            # Read the prompts into a list
            prompts = pd.read_csv(prompt_file)["prompt"].tolist()[:cases]

            # Build paths for evaluation folders
            eval_paths = {eval_name: os.path.join(root_path, method, concept, eval_name) for eval_name in evals}

            # Create a mapping of eval_name to their image paths
            eval_images = {
                eval_name: sorted([os.path.join(eval_path, image) for image in os.listdir(eval_path)]) 
                if os.path.exists(eval_path) else [] 
                for eval_name, eval_path in eval_paths.items()
            }

            # Process the specified number of cases
            for case_number in range(cases):
                row = {
                    "method": method,
                    "concept": concept,
                    "case_number": case_number + 1,
                }

                prompt = prompts[case_number]

                # Compute scores and classifications for each eval
                for eval_name in evals:
                    images = eval_images[eval_name]
                    if case_number < len(images):
                        image_path = images[case_number]
                        score = compute_clip_score(image_path, prompt)
                        row[f"{eval_name}_score"] = score
                        # if concept in objects_list:
                        #     top1, top5 = classify_image(image_path, concept)
                            # row[f"{eval_name}_top_1_classification"] = top1
                            # row[f"{eval_name}_top_5_classification"] = top1
                        # else:
                            # row[f"{eval_name}_top_1_classification"] = None
                            # row[f"{eval_name}_top_5_classification"] = None

                    else:
                        row[f"{eval_name}_score"] = None  # No image for this case number
                        # row[f"{eval_name}_top_1_classification"] = None
                        # row[f"{eval_name}_top_5_classification"] = None
                
                # Append the row to the method's data list
                method_data.append(row)

        # Save the method's data to a CSV file
        method_df = pd.DataFrame(method_data)
        output_file = os.path.join(output_dir, f"{method}_clip_scores.csv")
        method_df.to_csv(output_file, index=False)
        print(f"Data for method '{method}' saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP scores and classifications for images.")
    parser.add_argument("--root_path", type=str, required=True, help="Root path to the evaluation folders.")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to the prompts CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output CSV files.")
    parser.add_argument("--cases", type=int, default=5, help="Number of cases to process per concept.")
    parser.add_argument("--methods", nargs="+", default=["esdu", "esdx", "uce", "ga", "stable_diffusion"], help="List of methods to evaluate.")
    parser.add_argument("--concepts", nargs="+", default=[
        "english_springer_spaniel",
        "airliner",
        "garbage_truck",
        "parachute",
        "cassette_player",
        "chainsaw",
        "tench",
        "french_horn",
        "golf_ball",
        "church",
        "van_gogh",
        "picasso",
        "andy_warhol",
    ], help="List of concepts to evaluate.")
    parser.add_argument("--evals", nargs="+", default=["textual_inversion", "noisy", "uda", "inpainting", "ground_truth"], help="List of evaluation types.")

    args = parser.parse_args()

    process_clip_scores(
        root_path=args.root_path,
        prompts_path=args.prompts_path,
        methods=args.methods,
        concepts=args.concepts,
        evals=args.evals,
        output_dir=args.output_dir,
        cases=args.cases,
    )
