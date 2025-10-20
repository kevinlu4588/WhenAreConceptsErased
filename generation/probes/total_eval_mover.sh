#!/bin/bash

# Define the list of methods and concepts
methods=("rece")
concepts=(
    "english_springer_spaniel"
    "airliner"
    "garbage_truck"
    "parachute"
    "cassette_player"
    "chainsaw"
    "tench"
    "french_horn"
    "golf_ball"
    "church"
    "van_gogh"
    "picasso"
    "andy_warhol"
)

num_images=30
base_ground_truth_path="/path/to/your/prompts"
base_model_path="/path/to/your/models"
results_dir="./rece_results"
# # Run Python script
# python3 eval.py \
#     --methods "${methods[@]}" \
#     --concepts "${concepts[@]}" \
#     --num_images "$num_images" \
#     --base_ground_truth_path "$base_ground_truth_path" \
#     --base_model_path "$base_model_path" \
#     --results_dir "$results_dir"


# Shared paths
# FOUNDATION_MODEL_PATH="stable-diffusion-v1-5/stable-diffusion-v1-5"
# RESULTS_DIR="./rece_results"
# MASK_PATH="/path/to/your/masks/center_square_mask.png"
# RESULTS_FILE="inpaint_scores.txt"

# Loop through each method and concept
# for METHOD in "${methods[@]}"; do
#   for CONCEPT in "${concepts[@]}"; do
#     echo "Running inpainting for method: $METHOD, concept: $CONCEPT"

#     python inpaint_eval.py \
#       --method "$METHOD" \
#       --concept "$CONCEPT" \
#       --num_images "$num_images" \
#       --foundation_model_path "$FOUNDATION_MODEL_PATH" \
#       --results_dir "$RESULTS_DIR" \
#       --mask_path "$MASK_PATH" \
#       --results_file "$RESULTS_FILE"
#   done
# done

# echo "Inpainting complete"


prompts_dir="/path/to/your/prompts"
intermediate_results_dir="./noisy_attack_results"
device="cuda:0"  # Change to "cpu" or another device if needed

# IMAGE_BASE_DIR="./noisy_attack_results"
ACTUAL_RESULTS_DIR="./rece_results"
# # Path to the Python script
# script="noisy_eval.py"
# RESULTS_FILE="noising_attack_clip_scores.txt"
# # Loop through each method and concept
# for method in "${methods[@]}"; do
#     for concept in "${concepts[@]}"; do
#         echo "Running for method: $method, concept: $concept on device: $device"
#         python "$script" \
#             --methods "$method" \
#             --concepts "$concept" \
#             --num_images "$num_images" \
#             --prompts_dir "$prompts_dir" \
#             --results_dir "$intermediate_results_dir" \
#             --device "$device"
#     done
# done
# echo "Completed Generated All Noisy Images"


# Loop through each method and concept
for method in "${methods[@]}"; do
  for concept in "${concepts[@]}"; do
    echo "Processing method: $method, concept: $concept"

    python new_noisy_mover.py \
      --methods "$method" \
      --concepts "$concept" \
      --num_images "$num_images" \
      --prompts_dir "$prompts_dir" \
      --results_dir "$ACTUAL_RESULTS_DIR" \
      --results_file "$RESULTS_FILE" \
      --image_base_dir "$intermediate_results_dir"
  done
done

echo "Completed moving best noisy images to results folder"

