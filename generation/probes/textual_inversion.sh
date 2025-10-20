#!/bin/bash

# Define methods and concepts arrays
methods=("uce")
concepts=(
    "picasso"
)
# Define initializer tokens (ensure the order matches the concepts array)
initializer_tokens=("spa" "air" "tru" "par" "cas" "cha" "ten" "fre" "gol" "chu" "van" "pic" "and")
# Loop over methods and concepts
for method in "${methods[@]}"; do
    for j in "${!concepts[@]}"; do
        concept=${concepts[$j]}
        initializer_token=${initializer_tokens[$j]}
        
        # Construct the model path
        MODEL_PATH="/path/to/your/models/${method}_${concept}"

        # Define the data directory
        DATA_DIR="/path/to/your/data/${concept}/train"
        
        # Define the output directory
        OUTPUT_DIR="textual_inversion_${method}_${concept}"
        
        echo "Starting training for model: ${MODEL_PATH} and concept: ${concept}"
        
        # Run the command
        accelerate launch --mixed_precision=fp16 textual_inversion.py \
          --pretrained_model_name_or_path=$MODEL_PATH \
          --train_data_dir=$DATA_DIR \
          --learnable_property="object" \
          --placeholder_token="<${concept}>" \
          --initializer_token=$initializer_token \
          --resolution=512 \
          --train_batch_size=1 \
          --gradient_accumulation_steps=4 \
          --max_train_steps=3000 \
          --learning_rate=5.0e-04 \
          --scale_lr \
          --lr_scheduler="constant" \
          --lr_warmup_steps=0 \
          --output_dir=$OUTPUT_DIR
        
        echo "Completed training for model: ${MODEL_PATH} and concept: ${concept}"
    done
done
