#!/bin/bash
# ================================================================
# 🧠 Wrapper script for running all concept-erasure probes
# ================================================================

# Activate your conda or venv if needed
# source ~/miniconda3/bin/activate try_env

# --- Arguments you can easily modify ---
CONCEPTS=("van_gogh" "airliner")          # or ("all")
ERASING_TYPES=("esdx")                    # or ("esdx" "esdu" "uce" "ga")
PROBES=("standardpromptprobe" "inpaintingprobe" "noisebasedprobe")                            # or ("standardpromptprobe" "inpaintprobe" "textualinversionprobe")
NUM_IMAGES=10
DEVICE="cuda"
CONFIG="configs/default.yaml"
PIPELINE_PATH=""                          # optional: set to explicit model path to override structure

# --- Build the command ---
CMD="python new_runner.py \
  --concepts ${CONCEPTS[*]} \
  --erasing_types ${ERASING_TYPES[*]} \
  --probes ${PROBES[*]} \
  --num_images $NUM_IMAGES \
  --device $DEVICE \
  --config $CONFIG"

# Add pipeline path if provided
if [ -n "$PIPELINE_PATH" ]; then
  CMD+=" --pipeline_path $PIPELINE_PATH"
fi

# --- Print and run ---
echo "🚀 Running probe suite with command:"
echo "$CMD"
echo

eval $CMD
