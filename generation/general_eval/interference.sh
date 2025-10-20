#!/bin/bash

# Paths (update these if needed)
MODELS_PATH="/path/to/your/models"
PROMPTS_PATH="/path/to/your/prompts"
RESULTS_PATH="stereo_interference"
DEVICE="cuda:0"
METHODS=("stereo")
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
# Run the Python script with arguments
python interference.py \
  --models_path "$MODELS_PATH" \
  --methods "${METHODS[@]}" \
  --concepts "${CONCEPTS[@]}" \
  --prompts_path "$PROMPTS_PATH" \
  --results_path "$RESULTS_PATH" \
  --device "$DEVICE"
