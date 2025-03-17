#!/bin/bash

# List of objects and styles (one at a time)
OBJECTS=(
    # "English springer spaniel"
    # "airliner"
    # "garbage Truck"
    # "parachute"
    # "cassette player"
    # "chainsaw"
    # "tench"
    # "French horn"
    # "golf ball"
    # "church"
)
# OBJECTS=(
#     "church"
# )

STYLES=(
    "Van Gogh"
    "Picasso"
    "Andy Warhol"
#     "Thomas Kinkaide"
#     "Killian Eng"
)

BASE_OUTPUT_DIR="/share/u/kevin/ErasingDiffusionModels/testing_ga"
NUM_TRAIN_IMAGES=500

# Uncomment one of these lines based on the desired mode (objects or styles)
# TARGETS=("${OBJECTS[@]}")
TARGETS=("${STYLES[@]}")

# Loop through the selected list (either objects or styles)
for target in "${TARGETS[@]}"; do
    if [[ " ${OBJECTS[*]} " =~ " ${target} " ]]; then
        PROMPT="a picture of a ${target}"
    else
        PROMPT="a painting in the style of ${target}"
    fi

    OUTPUT_DIR="$BASE_OUTPUT_DIR/${target// /_}"
    
    echo "Generating images for: $PROMPT"
    
    # Call the Python script with the parameters
    python3 generate_training_images.py \
        --output_dir "$OUTPUT_DIR" \
        --prompt "$PROMPT" \
        --mode train \
        --num_train_images "$NUM_TRAIN_IMAGES"
done
