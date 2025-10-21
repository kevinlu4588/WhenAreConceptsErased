#!/usr/bin/env python3
"""
🧠 Train latent classifiers for multiple ImageNet concepts.

Pipeline:
1. Creates balanced subsets (positives + capped negatives) for ALL concepts
2. Trains a latent classifier for each concept
3. Saves each model as latent_classifiers/<concept>.pt
"""

import os
import subprocess
import sys

# Fix PIL issue globally for this process and subprocesses
import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 10MB

# ============================================================
# 1️⃣ Configuration
# ============================================================
CONCEPTS = ["garbage truck, dustcart", "golf ball", "tench, Tinca tinca", "airliner"]

ROOT_DIR = "/share/u/kevin/DiffusionConceptErasure/local_imagenet_full"
OUT_ROOT = "/share/u/kevin/DiffusionConceptErasure/local_imagenet_fixed_subsets"
BASE_DIR = "/share/u/kevin/DiffusionConceptErasure/classifier_guidance"

CREATE_SCRIPT = os.path.join(BASE_DIR, "create_class_subsets.py")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "create_latent_classifier.py")
CLASSIFIER_DIR = os.path.join(BASE_DIR, "latent_classifiers")

# ============================================================
# 2️⃣ Ensure output dirs
# ============================================================
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(CLASSIFIER_DIR, exist_ok=True)

# ============================================================
# 3️⃣ Create all subsets first
# ============================================================
print(f"\n{'='*70}")
print("📂 PHASE 1: Creating all subsets")
print(f"{'='*70}\n")

failed_subsets = []
successful_subsets = []

for concept in CONCEPTS:
    print(f"\n📂 Creating subset for '{concept}'...")
    
    # Convert concept name to filesystem-safe format
    concept_safe = concept.replace(", ", "_").replace(" ", "_")
    
    # Check if subset already exists
    subset_path = os.path.join(OUT_ROOT, concept_safe)
    if os.path.exists(subset_path):
        print(f"   ✅ Already exists at {subset_path}")
        successful_subsets.append(concept)
        continue
    
    try:
        result = subprocess.run(
            [
                sys.executable, CREATE_SCRIPT,
                "--target_label", concept,
                "--root_dir", ROOT_DIR,
                "--out_root", OUT_ROOT,
                "--n_neg", "5000"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"   ✅ Successfully created subset for '{concept}'")
        successful_subsets.append(concept)
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to create subset for '{concept}'")
        print(f"      Error: {e}")
        if e.stderr:
            print(f"      Details: {e.stderr}")
        failed_subsets.append(concept)

# Summary of subset creation
print(f"\n📊 Subset creation summary:")
print(f"   ✅ Successful: {len(successful_subsets)}/{len(CONCEPTS)}")
if failed_subsets:
    print(f"   ❌ Failed: {', '.join(failed_subsets)}")
    print(f"\n⚠️  Continuing with successful subsets only...")

# ============================================================
# 4️⃣ Train classifiers for all successful subsets
# ============================================================
print(f"\n{'='*70}")
print("🎓 PHASE 2: Training classifiers")
print(f"{'='*70}\n")

failed_training = []
successful_training = []

for concept in successful_subsets:
    print(f"\n🎓 Training classifier for '{concept}'...")
    
    # Convert concept name to filesystem-safe format
    concept_safe = concept.replace(", ", "_").replace(" ", "_")
    
    # Check if classifier already exists
    classifier_path = os.path.join(CLASSIFIER_DIR, f"{concept_safe}.pt")
    if os.path.exists(classifier_path):
        print(f"   ℹ️  Classifier already exists at {classifier_path}")
        print(f"   ⚠️  Retraining anyway (delete file to skip)")
    
    try:
        result = subprocess.run(
            [
                sys.executable, TRAIN_SCRIPT,
                concept_safe,
                "--subset_dir", OUT_ROOT,
                "--save_dir", CLASSIFIER_DIR,
                "--epochs", "70",
                "--batch_size", "8",
            ],
            check=True,
            capture_output=False,  # Show training progress
            text=True
        )
        print(f"   ✅ Successfully trained classifier for '{concept}'")
        successful_training.append(concept)
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to train classifier for '{concept}'")
        print(f"      Error: {e}")
        failed_training.append(concept)

# ============================================================
# 5️⃣ Final summary
# ============================================================
print(f"\n{'='*70}")
print("🏁 FINAL SUMMARY")
print(f"{'='*70}\n")

print("📂 Subset Creation:")
print(f"   ✅ Successful: {len(successful_subsets)}/{len(CONCEPTS)}")
if successful_subsets:
    for concept in successful_subsets:
        concept_safe = concept.replace(", ", "_").replace(" ", "_")
        print(f"      - {concept} → {OUT_ROOT}/{concept_safe}")

if failed_subsets:
    print(f"   ❌ Failed: {len(failed_subsets)}")
    for concept in failed_subsets:
        print(f"      - {concept}")

print("\n🎓 Classifier Training:")
print(f"   ✅ Successful: {len(successful_training)}/{len(successful_subsets)}")
if successful_training:
    for concept in successful_training:
        concept_safe = concept.replace(", ", "_").replace(" ", "_")
        print(f"      - {concept} → {CLASSIFIER_DIR}/{concept_safe}.pt")

if failed_training:
    print(f"   ❌ Failed: {len(failed_training)}")
    for concept in failed_training:
        print(f"      - {concept}")

if not failed_subsets and not failed_training:
    print("\n🎉 All tasks completed successfully! ✅")
else:
    print(f"\n⚠️  Completed with {len(failed_subsets)} subset failures and {len(failed_training)} training failures")

print(f"\n📁 Outputs:")
print(f"   - Subsets: {OUT_ROOT}")
print(f"   - Models:  {CLASSIFIER_DIR}")