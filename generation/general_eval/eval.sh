#!/bin/bash

# Define the list of methods and concepts
methods=("ga")
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

num_images=100
base_ground_truth_path="/path/to/your/prompts"
base_model_path="/path/to/your/models"
results_dir="./testing_results"
# Run Python script
python3 eval.py \
    --methods "${methods[@]}" \
    --concepts "${concepts[@]}" \
    --num_images "$num_images" \
    --base_ground_truth_path "$base_ground_truth_path" \
    --base_model_path "$base_model_path" \
    --results_dir "$results_dir"
