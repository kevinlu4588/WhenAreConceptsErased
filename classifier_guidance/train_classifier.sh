#!/usr/bin/env bash

# ==========================
# 🧠 Configurable arguments
# ==========================
CONCEPT="airliner"
BASE_DIR="/share/u/kevin/DiffusionConceptErasure/classifier_guidance"
EPOCHS=10
BATCH_SIZE=8

# ==========================
# 🚀 Run training
# ==========================
python create_latent_classifier.py \
    "$CONCEPT" \
    --base_dir "$BASE_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"
