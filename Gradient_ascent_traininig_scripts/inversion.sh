export MODEL_NAME="./checkpoint/van_gogh_erasure"
export DATA_DIR="./data/van_gogh_ti/train"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<art-style>" \
  --initializer_token="art" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --push_to_hub \
  --output_dir="./checkpoint/van_gogh_inversion"

export MODEL_NAME="./checkpoint/kilian_eng_erasure"
export DATA_DIR="./data/kilian_eng_ti/train"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<art-style>" \
  --initializer_token="art" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --push_to_hub \
  --output_dir="./checkpoint/kilian_eng_inversion"

export MODEL_NAME="./checkpoint/thomas_kinkade_erasure"
export DATA_DIR="./data/thomas_kinkade_ti/train"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<art-style>" \
  --initializer_token="art" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --push_to_hub \
  --output_dir="./checkpoint/thomas_kinkade_inversion"

