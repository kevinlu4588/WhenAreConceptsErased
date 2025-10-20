#!/bin/bash

# List of methods
METHODS=("esdx" "ga" "uce")

# List of concepts
CONCEPTS=(
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

# Number of images to process
NUM_IMAGES=100

# Shared paths
FOUNDATION_MODEL_PATH="stable-diffusion-v1-5/stable-diffusion-v1-5"
RESULTS_DIR="./testing_results"
MASK_PATH="/path/to/your/masks/center_square_mask.png"
RESULTS_FILE="inpaint_scores.txt"

# Loop through each method and concept
for METHOD in "${METHODS[@]}"; do
  for CONCEPT in "${CONCEPTS[@]}"; do
    echo "Running inpainting for method: $METHOD, concept: $CONCEPT"

    python inpaint_eval.py \
      --method "$METHOD" \
      --concept "$CONCEPT" \
      --num_images "$NUM_IMAGES" \
      --foundation_model_path "$FOUNDATION_MODEL_PATH" \
      --results_dir "$RESULTS_DIR" \
      --mask_path "$MASK_PATH" \
      --results_file "$RESULTS_FILE"
  done
done

echo "Inpainting complete"
