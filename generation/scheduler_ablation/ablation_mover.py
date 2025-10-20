import os
import csv
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse


# Helper function to calculate CLIP score
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


def main(args):
    # Initialize CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Parse input methods and concepts
    methods = args.methods.split(",")
    concepts = args.concepts.split(",")
    num_images = args.num_images
    results_dir = args.results_dir
    prompts_dir = args.prompts_dir
    results_file = args.results_file
    schedulers = args.schedulers

    # Ensure results directory exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    average_scores = []
    for scheduler in schedulers:
        for method in methods:
            for concept in concepts:
                prompts_path = os.path.join(prompts_dir, f"{concept}.csv")
                prompts = []
                # Load prompts for the concept
                if not os.path.exists(prompts_path):
                    print(f"Prompts file not found: {prompts_path}. Skipping concept.")
                    continue

                with open(prompts_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        prompts.append(row["prompt"])

                highest_scores = []  # Track the highest scores for averaging

                # Process images for the given method and concept
                for i in range(num_images):
                    folder_base = args.image_base_dir
                    folder_path = os.path.join(folder_base, scheduler, method, concept, f"images_{i}")

                    if not os.path.exists(folder_path):
                        print(f"Folder does not exist: {folder_path}. Skipping.")
                        continue

                    prompt = prompts[i]
                    best_image = None
                    highest_score = float("-inf")

                    # Find the best image in the folder
                    for image_file in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, image_file)
                        if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                            continue

                        clip_score = calculate_clip_score(image_path, prompt, model, processor)
                        if clip_score > highest_score:
                            highest_score = clip_score
                            best_image = image_path

                    if best_image:
                        # Save the best image to the results folder
                        concept_result_dir = os.path.join(results_dir, method, concept, scheduler)
                        print("saved to", concept_result_dir)
                        Path(concept_result_dir).mkdir(parents=True, exist_ok=True)

                        # Extract the filename from the best image path
                        best_image_filename = os.path.basename(best_image)
                        best_image_save_path = os.path.join(concept_result_dir, f"image_{i}_{best_image_filename}")

                        # Save the image
                        Image.open(best_image).save(best_image_save_path)
                        highest_scores.append(highest_score)

                # Calculate the average score for this method-concept pair
                if highest_scores:
                    avg_score = sum(highest_scores) / len(highest_scores)
                    average_scores.append(f"{method}/{concept}: {avg_score:.4f}")

        # Save average scores to the results file
        with open(results_file, "w") as f:
            f.write("\n".join(average_scores))
            print("Average scores saved:", average_scores)

        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and calculate CLIP scores for images.")

    parser.add_argument("--methods", type=str, required=True, help="Comma-separated list of methods (e.g., 'esdx,esdu,ga,uce').")
    parser.add_argument("--concepts", type=str, required=True, help="Comma-separated list of concepts (e.g., 'dog,cat,airliner').")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images per concept (default: 5).")
    parser.add_argument("--prompts_dir", type=str, required=True, help="Directory containing prompts CSV files.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--results_file", type=str, required=True, help="File to save the average scores.")
    parser.add_argument("--image_base_dir", type=str, required=True, help="Base directory containing image folders.")
    parser.add_argument("--schedulers", nargs = "+")
    args = parser.parse_args()

    main(args)
