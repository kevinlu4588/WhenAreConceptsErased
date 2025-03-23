#!/bin/bash

# Exit immediately if a command exits with a non-zero status, except where explicitly handled
set -e

# Environment variables
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
BASE_TRAIN_DIR="/share/u/kevin/ErasingDiffusionModels/testing_ga"
BASE_OUTPUT_DIR="/share/u/kevin/ErasingDiffusionModels/final_models"

# List of concepts
OBJECTS=(
    "english_springer_spaniel"
    "garbage_truck"
    # "airliner"
    # "french_horn"
    # "church"
    # "cassette player"
    # "chainsaw"
    # "tench"
    # "parachute"
    # "golf ball"
)

# Iterate through each concept and train the model
for concept in "${OBJECTS[@]}"; do
    prompts=(
        "picture of a $concept"
        "photo of a $concept"
        "$concept"
        "portrait of a $concept"
        "a picture of a $concept"
        "a picture of a car"
    )

    # Join prompts into a single string separated by semicolons
    prompt_args=$(IFS=";"; echo "${prompts[*]}")

    echo "Starting training for concept: $concept"

    # Convert concept name to lowercase and replace spaces with underscores
    concept_safe=$(echo "$concept" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

    # Set concept-specific paths
    TRAIN_DIR="${BASE_TRAIN_DIR}/${concept_safe}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/ga_${concept_safe}"

    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR"

    # Check if training data directory exists
    if [ ! -d "$TRAIN_DIR" ]; then
        echo "Training directory $TRAIN_DIR does not exist. Skipping concept: $concept."
        continue
    fi

    # Run training command, passing the joined prompts as a single argument
    echo "Running training for $concept_safe..."
    if ! accelerate launch --mixed_precision="fp16" train_text_to_image.py \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --train_data_dir="$TRAIN_DIR" \
        --use_ema \
        --resolution=512 --center_crop --random_flip \
        --train_batch_size=5 \
        --gradient_accumulation_steps=4 \
        --gradient_checkpointing \
        --max_train_steps=80 \
        --learning_rate=1e-05 \
        --max_grad_norm=1 \
        --lr_scheduler="constant" --lr_warmup_steps=0 \
        --validation_epochs=1 \
        --output_dir="$OUTPUT_DIR" \
        --validation_prompts="$prompt_args" \
        --step_finisher=80; then
        echo "Training failed for concept: $concept. Skipping to the next one."
        continue
    fi

    echo "Training completed successfully for concept: $concept"
done

echo "Training completed for all concepts."
