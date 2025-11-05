#!/bin/bash
# Example usage of the portable e2e concept classifier script

# Set custom base directory if needed (optional)
# export DCE_BASE_DIR="/path/to/your/DiffusionConceptErasure"

# Run the end-to-end classifier training for multiple concepts
python e2e_concept_classifier.py \
  "church, church building" "airliner" \
  --epochs 70 \
  --batch-size 8 \
  --lr 1e-4 \
  --n-neg 5000 \
  --output-dir "./trained_classifiers" \
  --force-recreate \
  --force-retrain

# The script will:
# 1. Download ImageNet data (if not present) using download_imagenet_classes.py
# 2. Create class subsets using create_class_subsets.py
# 3. Train latent classifiers using create_latent_classifier.py
# 4. Evaluate classifiers using evaluate_latent_classifier.py
# 5. Save final models and metadata

echo "Training complete! Check ./trained_classifiers/ for results."