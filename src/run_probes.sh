#!/bin/bash
# ================================================================
# 🧠 Wrapper script for running concept-erasure probes
# ================================================================

# --- Arguments you can easily modify ---
CONCEPTS=("english_springer_spaniel")            # or ("all")
ERASING_TYPES=("esdx")                            # list of erasing methods
PROBES=("noisebasedprobe")                         # list of probe types
NUM_IMAGES=10
DEVICE="cuda"
CONFIG="configs/default.yaml"

PIPELINE_PATH="/share/u/kevin/ErasingDiffusionModels/final_models/esdx_airliner"    # optional, set explicit pipeline path
UNET_PATH=""        # optional, set UNet path instead

# --- Build the command ---
CMD="python runner.py \
  --concepts ${CONCEPTS[*]} \
  --erasing_types ${ERASING_TYPES[*]} \
  --probes ${PROBES[*]} \
  --num_images $NUM_IMAGES \
  --device $DEVICE \
  --config $CONFIG"

if [[ -n "$PIPELINE_PATH" ]]; then
  CMD+=" --pipeline_path $PIPELINE_PATH"
fi

if [[ -n "$UNET_PATH" ]]; then
  CMD+=" --unet_path $UNET_PATH"
fi

# Validate
if [[ -z "$PIPELINE_PATH" && -z "$UNET_PATH" ]]; then
  echo "❌ You must specify either PIPELINE_PATH or UNET_PATH."
  exit 1
fi

echo "🚀 Running probe suite with command:"
echo "$CMD"
echo

eval $CMD
