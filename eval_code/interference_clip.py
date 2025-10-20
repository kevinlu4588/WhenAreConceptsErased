import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Define function to calculate CLIP score
def calculate_clip_score(image_path, text_prompt, clip_model, clip_processor):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[text_prompt], images=image, return_tensors="pt", padding=True).to("cuda:0")
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.item()

# Define paths
results_path = "/path/to/your/results"  # Path to generated images
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda:0")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Define methods and concepts
methods = ["rece"]
concepts = [
    "english_springer_spaniel", "airliner", "garbage_truck", "parachute", "cassette_player", 
    "chainsaw", "tench", "french_horn", "golf_ball", "church", "van_gogh", "picasso", "andy_warhol"
]
artists = ["van_gogh", "picasso", "andy_warhol"]
# Initialize results list
clip_results = []

# Iterate through methods and concepts
for method in methods:
    for concept in concepts:
        save_folder = os.path.join(results_path, method, concept, "interference")
        if not os.path.exists(save_folder):
            print(f"Image folder not found: {save_folder}, skipping.")
            continue

        print(f"Processing images for method: {method}, concept: {concept}")

        # Iterate through generated images
        for image_name in sorted(os.listdir(save_folder)):
            if not image_name.endswith(".png"):
                continue  # Skip non-image files

            # Extract the concept name and prompt index from filename
            parts = image_name.split("_prompt_")
            if len(parts) < 2:
                continue  # Skip malformed filenames

            prompt_text = parts[0]  # Extract the concept name
            prompt_idx = parts[1].split(".")[0]  # Extract the numerical index
            
            # Skip the first image of each concept (i.e., *_prompt_1.png)
            if prompt_idx == "1":
                print(f"Skipping first image: {image_name}")
                continue
            prepend = ""
            if prompt_text not in artists:
                prepend = "a picture of a "
            else:
                prepend = "art in the style of "

            prompt_text = prepend + " ".join(prompt_text.split("_"))
            image_path = os.path.join(save_folder, image_name)
            clip_score = calculate_clip_score(image_path, prompt_text, clip_model, clip_processor)
            clip_results.append({"method": method, "concept": concept, "image": image_name, "clip_score": clip_score})

# Convert results to DataFrame and save
results_df = pd.DataFrame(clip_results)
results_df.to_csv("stereo_inter_clip.csv", index=False)
print("CLIP score evaluation completed and saved to all_esdab_interference.csv")


# Group by concept and compute mean and std of CLIP scores
concept_summary = results_df.groupby("concept")["clip_score"].agg(["mean", "std"]).reset_index()
concept_summary = concept_summary.round(2)

print("\n=== Per-Concept CLIP Score Summary (Mean ± Std) ===")
print(concept_summary)

# Compute overall mean and std
overall_mean = results_df["clip_score"].mean()
overall_std = results_df["clip_score"].std()

print("\n=== Overall CLIP Score Summary ===")
print(f"Mean: {overall_mean:.2f}")
print(f"Std: {overall_std:.2f}")

# Optionally save summary to CSV
concept_summary.to_csv("stereo_inter_clip_summary.csv", index=False)
