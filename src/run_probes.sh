#!/bin/bash
# ================================================================
# 🧠 Wrapper script for running and evaluating a single model
# ================================================================

# --- User-configurable arguments ---
CONCEPT="airliner"               
ERASING_TYPE="esdx"              
PROBES=("all")       
NUM_IMAGES=10
DEVICE="cuda"
CONFIG="configs/default.yaml"

PIPELINE_PATH="kevinlu4588/esdx_airliner"
UNET_PATH=""

# --- Build runner command ---
RUN_CMD="python runner.py \
  --concepts $CONCEPT \
  --erasing_type $ERASING_TYPE \
  --probes ${PROBES[*]} \
  --num_images $NUM_IMAGES \
  --device $DEVICE \
  --config $CONFIG"

if [[ -n "$PIPELINE_PATH" ]]; then
  RUN_CMD+=" --pipeline_path $PIPELINE_PATH"
elif [[ -n "$UNET_PATH" ]]; then
  RUN_CMD+=" --unet_path $UNET_PATH"
else
  echo "❌ You must specify either PIPELINE_PATH or UNET_PATH."
  exit 1
fi

# --- Run probes ---
echo "🚀 Running probe suite:"
echo "$RUN_CMD"
echo
eval $RUN_CMD

# ================================================================
# 📊 Run evaluation right after
# ================================================================
echo
echo "🧮 Starting evaluation of all results..."
EVAL_CMD="python evaluator.py"
echo "$EVAL_CMD"
echo
eval $EVAL_CMD

echo "✅ All done!"
