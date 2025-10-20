import os
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
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

def classification(evals, root_path, methods, concepts, output_csv, cases):
    """
    Processes classification scores for the given methods and concepts.
    Saves results to a CSV file.
    """
    # Ensure directory exists only if output_csv has a directory component
    if os.path.dirname(output_csv):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results = []

    # Load ResNet50
    weights = ResNet50_Weights.DEFAULT
    resnet_model = resnet50(weights=weights).to('cuda')
    resnet_model.eval()
    preprocess = weights.transforms()
    artists = ["andy_warhol", "van_gogh", "picasso"]

    for method in methods:
        for concept in concepts:
            if any(artist in concept for artist in artists):
                continue  # Skip if image_name contains any artist name     
            print(f"Processing: {method} - {concept}")
            for eval in evals:
                im_path = os.path.join(root_path, method, concept, eval)

                # Check if the folder exists
                if not os.path.exists(im_path):
                    print(f"Skipping {im_path} (folder not found)")
                    continue
                print(im_path)
                # Get image filenames
                ground_truth_images = os.listdir(im_path)
                print(len(ground_truth_images))
                for image in ground_truth_images:
                    images_path = os.path.join(im_path, image)

                    if not os.path.isfile(images_path):
                        continue  # Skip if it's not a file

                    # Handle concept name formatting
                    concept_cleaned = " ".join(concept.split("_")).lower()
                    if concept_cleaned == "chainsaw":
                        concept_cleaned = "chain saw"

                    # Classify the image (pass full image path)
                    top_categories, top_scores = classify_image(images_path, resnet_model, preprocess, weights, topk=5, device='cuda')

                    in_top1 = concept_cleaned == top_categories[0].lower()
                    in_top5 = any(concept_cleaned == category.lower() for category in top_categories)

                    # Get logits (score) for Top-1 and Top-5
                    top1_score = top_scores[0] if in_top1 else 0.0
                    top5_score = next((score for cat, score in zip(top_categories, top_scores) if cat.lower() == concept_cleaned), 0.0)

                    # Store results
                    results.append({
                        "method": method,
                        "concept": concept,
                        "eval": eval,
                        "image": image,
                        "in_top1": in_top1,
                        "top1_score": top1_score,
                        "in_top5": in_top5,
                        "top5_score": top5_score,
                    })

    # Convert results to a DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    # Print summary per method and eval
    print("\n=== Classification Summary ===")
    for method in methods:
        method_df = df[df["method"] == method]
        print(f"\nMethod: {method}")
        for eval in evals:
            eval_df = method_df[method_df["eval"] == eval]
            if not eval_df.empty:
                top1_acc = eval_df["in_top1"].mean() * 100
                top5_acc = eval_df["in_top5"].mean() * 100
                top1_score = eval_df["top1_score"].mean()
                top5_score = eval_df["top5_score"].mean()
                print(f"{eval}: Top-1 Acc = {top1_acc:.2f}%, Top-1 Score = {top1_score:.4f}, "
                    f"Top-5 Acc = {top5_acc:.2f}%, Top-5 Score = {top5_score:.4f}")
            else:
                print(f"{eval}: No data found.")
    print("=================================\n")
    print(f"Classification results saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classification results for images.")
    parser.add_argument("--root_path", type=str, required=True, help="Root path to the evaluation folders.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--cases", type=int, default=5, help="Number of cases to process per concept.")
    parser.add_argument("--methods", nargs="+", default=["esdus", "esda", "esdas"], help="List of methods to evaluate.")
    parser.add_argument("--evals", nargs="+", default=["ground_truth", "textual_inversion"], help="List of methods to evaluate.")
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

    args = parser.parse_args()

    classification(
        evals=args.evals,
        root_path=args.root_path,
        methods=args.methods,
        concepts=args.concepts,
        output_csv=args.output_csv,  # ✅ Now correctly using user input
        cases=args.cases,  # ✅ Now correctly passing cases
    )
