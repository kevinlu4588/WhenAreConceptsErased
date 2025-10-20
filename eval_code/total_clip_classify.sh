#!/bin/bash

# source /path/to/your/miniconda3/etc/profile.d/conda.sh
# cd /path/to/your/ErasingDiffusionModels

# # Activate the asdf_env environment
# conda activate asdf_env
# echo "Active Conda Environment: $(conda env list | grep '*' | awk '{print $1}')" >> /tmp/test_script.log


# Paths to required resources
ROOT_PATH="/path/to/your/evaluation/folders"       # Replace with the root path to evaluation folders
PROMPTS_PATH="/path/to/your/prompts"   # Replace with the path to the prompts CSV files
OUTPUT_DIR="stereo_dataframes"       # Replace with the directory to save output files
SCRIPT_PATH="/path/to/your/eval_code/big_eval_parse.py" # Replace with the path to your Python script

# Methods to evaluate
METHODS=("rece")

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through methods
for METHOD in "${METHODS[@]}"; do
    # Print progress
    echo "Processing method: $METHOD"
    
    # Run the Python script for the current method
    python "$SCRIPT_PATH" \
        --root_path "$ROOT_PATH" \
        --prompts_path "$PROMPTS_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --methods "$METHOD" \
        --evals "ground_truth" "uda" "textual_inversion" "noisy" "inpainting"\
        --cases 100
done

echo "All tasks completed!"