#!/bin/bash

# Exit immediately if a command exits with a non-zero status, except where explicitly handled
set -e

# Environment variables
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
BASE_TRAIN_DIR="training_images"
BASE_OUTPUT_DIR="ga_models"

# List of objects and styles
OBJECTS=(
    "airliner"
    # "english_springer_spaniel"
    # "garbage_truck"
    # "french_horn"
    # "church"
    # "cassette player"
    # "chainsaw"
    # "tench"
    # "parachute"
    # "golf ball"
)

STYLES=(
    "Van Gogh"
    # "Picasso"
    # "Andy Warhol"
    # "Thomas Kinkaide"
    # "Killian Eng"
)

NUM_TRAIN_IMAGES=10

# Generate training images
for target in "${OBJECTS[@]}" "${STYLES[@]}"; do
    if [[ " ${OBJECTS[*]} " =~ " ${target} " ]]; then
        PROMPT="a picture of a ${target}"
    else
        PROMPT="a painting in the style of ${target}"
    fi

    OUTPUT_DIR="$BASE_TRAIN_DIR/$(echo "$target" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')"
    
    echo "Generating images for: $PROMPT"

    # Call the Python script with the parameters
    python3 generate_training_images.py \
        --output_dir "$OUTPUT_DIR" \
        --prompt "$PROMPT" \
        --mode train \
        --num_train_images "$NUM_TRAIN_IMAGES"

    # Confirm image generation was successful
    if [ ! -d "$OUTPUT_DIR" ] || [ -z "$(ls -A "$OUTPUT_DIR")" ]; then
        echo "Image generation failed or empty folder for: $target. Skipping training."
        continue
    fi
done

# Train models on generated images
for concept in "${OBJECTS[@]}" "${STYLES[@]}"; do
    echo "Starting training for concept: $concept"

    concept_safe=$(echo "$concept" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    TRAIN_DIR="${BASE_TRAIN_DIR}/${concept_safe}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/ga_${concept_safe}"

    mkdir -p "$OUTPUT_DIR"

    if [ ! -d "$TRAIN_DIR" ]; then
        echo "Training directory $TRAIN_DIR does not exist. Skipping concept: $concept."
        continue
    fi

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

echo "Pipeline completed successfully for all concepts and styles."
