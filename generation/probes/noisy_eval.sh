#!/bin/bash

# Define the list of methods and concepts
methods=("esdx")
concepts=(
    "airliner"
    # "english_springer_spaniel"
    # "garbage_truck"
    # "parachute"
    # "cassette_player"
    # "chainsaw"
    # "tench"
    # "french_horn"
    # "golf_ball"
    # "church"
    # "van_gogh"
    # "picasso"
    # "andy_warhol"
)

# Define additional arguments
num_images=2
prompts_dir="/path/to/your/prompts"
intermediate_results_dir="./noisy_attack_results"
device="cuda:0"  # Change to "cpu" or another device if needed

IMAGE_BASE_DIR="./noisy_attack_results"
ACTUAL_RESULTS_DIR="./testing_results"
# Path to the Python script
script="noisy_eval.py"
RESULTS_FILE="noising_attack_clip_scores.txt"
# Loop through each method and concept
for method in "${methods[@]}"; do
    for concept in "${concepts[@]}"; do
        echo "Running for method: $method, concept: $concept on device: $device"
        python "$script" \
            --methods "$method" \
            --concepts "$concept" \
            --num_images "$num_images" \
            --prompts_dir "$prompts_dir" \
            --results_dir "$intermediate_results_dir" \
            --device "$device"
    done
done
echo "Completed Generated All Noisy Images"


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