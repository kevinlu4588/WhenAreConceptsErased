# Define methods and concepts
methods=("esdu" "uce")
concepts=(
    "airliner"
    "english_springer_spaniel"
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

schedulers=('DDPM')
# Define additional arguments
num_images=100
prompts_dir="/path/to/your/prompts"
intermediate_results_dir="./ablation_attack_results"
device="cuda:0"  # Change to "cpu" or another device if needed

IMAGE_BASE_DIR="./ablation_attack_results"
ACTUAL_RESULTS_DIR="./ablation"
# Path to the Python script
script="ablation_mover.py"
RESULTS_FILE="ablation_results.txt"

# Ensure results directory exists
mkdir -p $ACTUAL_RESULTS_DIR

for method in "${methods[@]}"; do
  for concept in "${concepts[@]}"; do
    echo "Processing method: $method, concept: $concept"

    python ablation_mover.py \
      --methods "$method" \
      --concepts "$concept" \
      --num_images "$num_images" \
      --prompts_dir "$prompts_dir" \
      --results_dir "$ACTUAL_RESULTS_DIR" \
      --results_file "$RESULTS_FILE" \
      --image_base_dir "$intermediate_results_dir"\
      --schedulers "${schedulers[@]}"
  done
done

echo "All tasks completed!"
