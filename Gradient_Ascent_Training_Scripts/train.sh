#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Environment variables
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
BASE_TRAIN_DIR="/share/u/kevin/ErasingDiffusionModels/final_data/"
BASE_OUTPUT_DIR="/share/u/kevin/ErasingDiffusionModels/testing_ga"

# List of concepts
concepts=(
    "airliner"
)

prompts=(
    "picture of an airliner"
    "photo of an airliner"
    "airliner"
    "portrait of an airliner"
    "a picture of a house"
)

# Iterate through each concept and train the model
for concept in "${concepts[@]}"; do
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

    # Convert prompt array into a space-separated string with proper quoting
    prompt_args=()
    for prompt in "${prompts[@]}"; do
        prompt_args+=(--validation_prompts "$prompt")
    done

    # Run training command
    echo "Running training for $concept_safe..."
    accelerate launch --mixed_precision="fp16" train_text_to_image.py \
      --pretrained_model_name_or_path="$MODEL_NAME" \
      --train_data_dir="$TRAIN_DIR" \
      --use_ema \
      --resolution=512 --center_crop --random_flip \
      --train_batch_size=5 \
      --gradient_accumulation_steps=4 \
      --gradient_checkpointing \
      --max_train_steps=100 \
      --learning_rate=1e-05 \
      --max_grad_norm=1 \
      --lr_scheduler="constant" --lr_warmup_steps=0 \
      --validation_epochs=1 \
      --output_dir="$OUTPUT_DIR" \
      "${prompt_args[@]}"

    # Check for success and provide feedback
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for concept: $concept"
    else
        echo "Training failed for concept: $concept. Check logs for details."
        exit 1
    fi
done

echo "Training completed for all concepts."
