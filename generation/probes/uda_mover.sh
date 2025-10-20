# List of methods and concepts
METHODS=("esdx" "esdu" "ga" "uce")
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

# Paths and config
NUM_IMAGES=5
PROMPTS_DIR="/path/to/your/prompts"
IMAGE_BASE_DIR="/path/to/your/results"
RESULTS_DIR="./testing_results"
RESULTS_FILE="unlearned_results.txt"

# Run the script
python uda_mover.py \
  --methods "${METHODS[@]}" \
  --concepts "${CONCEPTS[@]}" \
  --num_images "$NUM_IMAGES" \
  --prompts_dir "$PROMPTS_DIR" \
  --image_base_dir "$IMAGE_BASE_DIR" \
  --results_dir "$RESULTS_DIR" \
  --results_file "$RESULTS_FILE"