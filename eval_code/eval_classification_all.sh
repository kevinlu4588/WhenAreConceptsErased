#!/bin/bash

# Define the list of methods and concepts
methods=("stereo")
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

# Create output directory if it doesn't exist
ROOT_PATH="/path/to/your/evaluation/folders"        # Path to evaluation folders
PROMPTS_PATH="final_data/prompts"    # Path to prompts CSV files
OUTPUT_DIR="/path/to/your/output/stereo.csv"  # Directory to save output files

# Ensure output directory exists

# Run Python script with correct method handling
python eval_classification.py \
    --root_path "$ROOT_PATH" \
    --output_csv "$OUTPUT_DIR" \
    --methods "${methods[@]}" \
    --evals "inpainting"\
    --cases 100
