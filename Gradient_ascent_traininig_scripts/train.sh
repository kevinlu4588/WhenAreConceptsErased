export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="./generation/english_springer_erasure"

accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=5 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="./checkpoint/english_springer"