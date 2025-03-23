#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -eu

# Environment variables
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
BASE_TRAIN_DIR="/share/u/kevin/ErasingDiffusionModels/testing_ga"
BASE_OUTPUT_DIR="/share/u/kevin/ErasingDiffusionModels/final_models"

# List of artistic styles
STYLES=(
    # "van_gogh"
    # "picasso"
    "andy_warhol"
    # "thomas_kinkaide"
    # "killian eng"
)

# Iterate through each style and train the model
for style in "${STYLES[@]}"; do
    prompts=(
        "A painting in the style of $style"
        "An artwork inspired by $style"
        "A portrait in the style of $style"
        "A landscape painted like $style"
        "A sketch reminiscent of $style"
        "A house"
    )

    # Join prompts into a single string separated by semicolons
    prompt_args=$(IFS=";"; echo "${prompts[*]}")

    echo "Starting training for style: $style"

    # Convert style name to lowercase and replace spaces with underscores
    style_safe=$(echo "$style" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

    # Set style-specific paths
    TRAIN_DIR="${BASE_TRAIN_DIR}/${style_safe}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/ga_${style_safe}"

    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR"

    # Check if training data directory exists
    if [ ! -d "$TRAIN_DIR" ]; then
        echo "Training directory $TRAIN_DIR does not exist. Skipping style: $style."
        continue
    fi

    # Run training command, passing the joined prompts as a single argument
    echo "Running training for $style_safe..."
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
        --step_finisher=60

    echo "Training completed successfully for style: $style"
done

echo "Training completed for all styles."
