# OUTPUT_DIR="./data/english_springer_ti"
# PROMPT="a photo of an english springer"
# NUM_TRAIN_IMAGES=100

# python3 generate_training_images.py \
#     --output_dir $OUTPUT_DIR \
#     --prompt "$PROMPT" \
#     --mode train \
#     --num_train_images $NUM_TRAIN_IMAGES

# OUTPUT_DIR="./data/garbage_truck_ti"
# PROMPT="a photo of a garbage truck"
# NUM_TRAIN_IMAGES=100

# python3 generate_training_images.py \
#     --output_dir $OUTPUT_DIR \
#     --prompt "$PROMPT" \
#     --mode train \
#     --num_train_images $NUM_TRAIN_IMAGES

# OUTPUT_DIR="./data/kilian_eng_ti"
# PROMPT="a painting in the style of Kilian Eng"
# NUM_TRAIN_IMAGES=100

# python3 generate_training_images.py \
#     --output_dir $OUTPUT_DIR \
#     --prompt "$PROMPT" \
#     --mode train \
#     --num_train_images $NUM_TRAIN_IMAGES

# OUTPUT_DIR="./data/kilian_eng_ti"
# PROMPT="a painting in the style of Thomas Kinkade"
# NUM_TRAIN_IMAGES=100

# python3 generate_training_images.py \
#     --output_dir $OUTPUT_DIR \
#     --prompt "$PROMPT" \
#     --mode train \
#     --num_train_images $NUM_TRAIN_IMAGES

# OUTPUT_DIR="./data/van_gogh_ti"
# PROMPT="a painting in the style of Van Gogh"
# NUM_TRAIN_IMAGES=100

# python3 generate_training_images.py \
#     --output_dir $OUTPUT_DIR \
#     --prompt "$PROMPT" \
#     --mode train \
#     --num_train_images $NUM_TRAIN_IMAGES

OUTPUT_DIR="./generation/english_springer_erasure"
PROMPT="a photo of an english springer spaniel"
NUM_TRAIN_IMAGES=100

python3 generate_training_images.py \
    --output_dir $OUTPUT_DIR \
    --prompt "$PROMPT" \
    --mode train \
    --num_train_images $NUM_TRAIN_IMAGES

# PROMPT="a photo of a golden retriever"
# python3 generate_training_images.py \
#     --output_dir $OUTPUT_DIR \
#     --prompt "$PROMPT" \
#     --mode test \
#     --model_path "./checkpoint/english_springer_erasure" \
#     --num_train_images $NUM_TRAIN_IMAGES

# PROMPT="a photo of a american pit bull terrier"
# python3 generate_training_images.py \
#     --output_dir $OUTPUT_DIR \
#     --prompt "$PROMPT" \
#     --mode test \
#     --model_path "./checkpoint/english_springer_erasure" \
#     --num_train_images $NUM_TRAIN_IMAGES
