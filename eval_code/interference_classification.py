import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights

# Function to classify an image using ResNet50
def classify_image(image_path, model, preprocess, weights, topk=5, device="cuda:0"):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor).softmax(dim=1)
        top_probs, top_indices = torch.topk(predictions, topk, dim=1)

    top_categories = [weights.meta['categories'][idx] for idx in top_indices[0].cpu().numpy()]
    top_scores = top_probs[0].cpu().numpy().tolist()
    
    return top_categories, top_scores

# Define paths
results_path = "/path/to/your/stereo_interference"
# results_path = "testing_esda"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load ResNet50 model for classification
weights = ResNet50_Weights.DEFAULT
resnet_model = resnet50(weights=weights).to(device)
resnet_model.eval()
preprocess = weights.transforms()

# Define methods and concepts
methods = ["rece"]
# methods = ["esdas", "esda", "esdus"]
concepts = [
    "english_springer_spaniel", "airliner", "garbage_truck", "parachute", "cassette_player",
    "chainsaw", "tench", "french_horn","church", "golf_ball", "van_gogh", "picasso", "andy_warhol"
]

# concepts = ["english_springer_spaniel"]
# Initialize results list
results = []

artists = ["andy_warhol", "van_gogh", "picasso"]
# Iterate through methods and concepts
for method in methods:
    for concept in concepts:
        save_folder = os.path.join(results_path, method, concept, "interference")
        if not os.path.exists(save_folder):
            print(f"Image folder not found: {save_folder}, skipping.")
            continue

        print(f"Processing images for method: {method}, concept: {concept}")

        # Iterate through images
        for image_name in sorted(os.listdir(save_folder)):
            if not image_name.endswith(".png"):
                continue  # Skip non-image files
            image_path = os.path.join(save_folder, image_name)
            if any(artist in image_name for artist in artists):
                continue  # Skip if image_name contains any artist name            # Classify the image using ResNet50
            top_categories, top_scores = classify_image(image_path, resnet_model, preprocess, weights, topk=5, device=device)

            # Check if the ground-truth concept is in Top-1 or Top-5
            before_prompt = image_name.split("_prompt")[0]
            concept_cleaned = " ".join(before_prompt.split("_")).lower()

            if concept_cleaned == "chainsaw":
                concept_cleaned = "chain saw"
            in_top1 = concept_cleaned == top_categories[0].lower()
            in_top5 = any(concept_cleaned == category.lower() for category in top_categories)

            # Get logits (score) for Top-1 and Top-5
            top1_score = top_scores[0] if in_top1 else 0.0
            top5_score = next((score for cat, score in zip(top_categories, top_scores) if cat.lower() == concept_cleaned), 0.0)

            # Store results
            results.append({
                    "method": method,
                    "concept": concept,
                    "image": image_name,
                    "in_top1": in_top1,
                    "top1_score": top1_score,
                    "in_top5": in_top5,
                    "top5_score": top5_score,
                    # "category_top1": top_categories[0],
                    # "score_top1": top_scores[0],
                    # "category_top2": top_categories[1],
                    # "score_top2": top_scores[1],
                    # "category_top3": top_categories[2],
                    # "score_top3": top_scores[2],
                    # "category_top4": top_categories[3],
                    # "score_top4": top_scores[3],
                    # "category_top5": top_categories[4],
                    # "score_top5": top_scores[4],
                })

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(f"{methods[0]}_inter_class.csv", index=False)

print("Classification completed and saved to esda_lassification_results.csv")
