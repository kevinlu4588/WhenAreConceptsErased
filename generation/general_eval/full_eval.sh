#!/bin/bash

# Define the list of methods and concepts
methods=("uce")
concepts=(
    "van_gogh"
)

num_images=100
base_ground_truth_path="/path/to/your/prompts"
base_model_path="/path/to/your/models"
results_dir="./sdxl_results"
# # Run Python script
# python3 eval.py \
#     --methods "${methods[@]}" \
#     --concepts "${concepts[@]}" \
#     --num_images "$num_images" \
#     --base_ground_truth_path "$base_ground_truth_path" \
#     --base_model_path "$base_model_path" \
#     --results_dir "$results_dir"

#!/bin/bash

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

# Paths (update these if needed)
PROMPTS_PATH="/path/to/your/prompts"
DEVICE="cuda:0"
# Run the Python script with arguments
python interference.py \
  --models_path "$base_model_path" \
  --methods "${methods[@]}" \
  --concepts "${CONCEPTS[@]}" \
  --prompts_path "$PROMPTS_PATH" \
  --results_path "$results_dir" \
  --device "$DEVICE" \
  --num_images "$num_images"

num_images=1
prompts_dir="/path/to/your/prompts"
intermediate_results_dir="./sdxl_noisy_attack_results"
device="cuda:0"  # Change to "cpu" or another device if needed

IMAGE_BASE_DIR="./sdxl_noisy_attack_results"
ACTUAL_RESULTS_DIR="./sdxl_results"
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
