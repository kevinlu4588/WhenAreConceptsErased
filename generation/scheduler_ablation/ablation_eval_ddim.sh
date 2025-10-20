#!/bin/bash

# Define the list of methods and concepts
methods=("esdx" "esdu" "uce")
concepts=(
    "airliner"
    "english_springer_spaniel"
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

schedulers=('DDIM1')
# Define additional arguments
num_images=100
prompts_dir="/path/to/your/prompts"
intermediate_results_dir="./ablation_attack_results"
device="cuda:0"  # Change to "cpu" or another device if needed

IMAGE_BASE_DIR="./ablation_attack_results"
ACTUAL_RESULTS_DIR="./ablation"
# Path to the Python script
script="ablation_study.py"
RESULTS_FILE="ablation_results.txt"
# Loop through each method and concept
for scheduler in "${schedulers[@]}"; do
  for method in "${methods[@]}"; do
      for concept in "${concepts[@]}"; do
          echo "Running for method: $method, concept: $concept on device: $device"
          python "$script" \
              --methods "$method" \
              --concepts "$concept" \
              --num_images "$num_images" \
              --prompts_dir "$prompts_dir" \
              --results_dir "$intermediate_results_dir" \
              --device "$device"\
              --scheduler_type "$scheduler"
      done
  done
done
echo "Completed Generated All Noisy Images"


# # Loop through each method and concept
for method in "${methods[@]}"; do
  for concept in "${concepts[@]}"; do
    echo "Processing method: $method, concept: $concept"

    python ablation_mover.py \
      --methods "$method" \
      --concepts "$concept" \
      --num_images "$num_images" \
      --prompts_dir "$prompts_dir" \
      --results_dir "$ACTUAL_RESULTS_DIR" \
      --results_file "$RESULTS_FILE" \
      --image_base_dir "$intermediate_results_dir"\
      --schedulers "${schedulers[@]}"
  done
done

# echo "Completed moving best noisy images to results folder"