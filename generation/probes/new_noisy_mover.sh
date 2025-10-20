# Define methods and concepts
methods=("esdx")
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


# Define common parameters
NUM_IMAGES=1
PROMPTS_DIR="/path/to/your/prompts"
RESULTS_DIR="./testing_results"
RESULTS_FILE="noising_attack_clip_scores.txt"
IMAGE_BASE_DIR="./noisy_attack_results"
PYTHON_SCRIPT="new_noisy_mover.py"

# Ensure results directory exists
mkdir -p $RESULTS_DIR

# Loop through each method and concept
for method in "${methods[@]}"; do
    for concept in "${concepts[@]}"; do
        echo "Processing method: $method, concept: $concept"
        
        # Run the Python script with arguments
        python $PYTHON_SCRIPT \
            --methods "$method" \
            --concepts "$concept" \
            --num_images $NUM_IMAGES \
            --prompts_dir $PROMPTS_DIR \
            --results_dir $RESULTS_DIR \
            --results_file $RESULTS_FILE \
            --image_base_dir $IMAGE_BASE_DIR
    done
done

echo "All tasks completed!"
