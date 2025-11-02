#!/usr/bin/env python3
"""
🧠 Train latent classifiers for multiple ImageNet concepts with progress display.
"""

import os
import sys
import subprocess
from time import time

import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 10MB

# ============================================================
# 1️⃣ Configuration
# ============================================================
CONCEPTS = ["church, church building"]

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
# 3️⃣ Create all subsets with progress
# ============================================================
print(f"\n{'='*70}")
print("📂 PHASE 1: Creating all subsets")
print(f"{'='*70}\n")

failed_subsets, successful_subsets = [], []

for i, concept in enumerate(CONCEPTS, 1):
    concept_safe = concept.replace(", ", "_").replace(" ", "_")
    subset_path = os.path.join(OUT_ROOT, concept_safe)

    print(f"\n[{i}/{len(CONCEPTS)}] 📂 Creating subset for '{concept}'...")
    if os.path.exists(subset_path):
        print(f"   ✅ Already exists → {subset_path}")
        successful_subsets.append(concept)
        continue

    start = time()
    cmd = [
        sys.executable, CREATE_SCRIPT,
        "--target_label", concept,
        "--root_dir", ROOT_DIR,
        "--out_root", OUT_ROOT,
        "--n_neg", "5000",
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print("   " + line.strip())
    ret = process.wait()
    elapsed = time() - start

    if ret == 0:
        print(f"   ✅ Successfully created subset in {elapsed:.1f}s")
        successful_subsets.append(concept)
    else:
        print(f"   ❌ Failed to create subset for '{concept}' (exit {ret})")
        failed_subsets.append(concept)

print(f"\n📊 Subset creation summary:")
print(f"   ✅ {len(successful_subsets)}/{len(CONCEPTS)} successful")
if failed_subsets:
    print(f"   ❌ Failed: {', '.join(failed_subsets)}")
    print("⚠️  Continuing with successful subsets only...")

# ============================================================
# 4️⃣ Train classifiers with progress
# ============================================================
print(f"\n{'='*70}")
print("🎓 PHASE 2: Training classifiers")
print(f"{'='*70}\n")

failed_training, successful_training = [], []

for i, concept in enumerate(successful_subsets, 1):
    concept_safe = concept.replace(", ", "_").replace(" ", "_")
    classifier_path = os.path.join(CLASSIFIER_DIR, f"{concept_safe}.pt")

    print(f"\n[{i}/{len(successful_subsets)}] 🎓 Training classifier for '{concept}'...")
    if os.path.exists(classifier_path):
        print(f"   ⚠️  File exists → {classifier_path}")
        print(f"   ℹ️  Retraining anyway (delete file to skip)\n")

    cmd = [
        sys.executable, TRAIN_SCRIPT, concept_safe,
        "--subset_dir", OUT_ROOT,
        "--save_dir", CLASSIFIER_DIR,
        "--epochs", "70",
        "--batch_size", "8",
    ]

    start = time()
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print("   " + line.strip())
    ret = process.wait()
    elapsed = time() - start

    if ret == 0:
        print(f"   ✅ Successfully trained classifier in {elapsed/60:.1f} min")
        successful_training.append(concept)
    else:
        print(f"   ❌ Failed to train classifier for '{concept}' (exit {ret})")
        failed_training.append(concept)

# ============================================================
# 5️⃣ Final summary
# ============================================================
print(f"\n{'='*70}")
print("🏁 FINAL SUMMARY")
print(f"{'='*70}\n")

print("📂 Subset Creation:")
print(f"   ✅ {len(successful_subsets)}/{len(CONCEPTS)} succeeded")
if failed_subsets:
    for c in failed_subsets:
        print(f"   ❌ {c}")

print("\n🎓 Classifier Training:")
print(f"   ✅ {len(successful_training)}/{len(successful_subsets)} succeeded")
if failed_training:
    for c in failed_training:
        print(f"   ❌ {c}")

if not failed_subsets and not failed_training:
    print("\n🎉 All tasks completed successfully!")
else:
    print(f"\n⚠️  Done with {len(failed_subsets)} subset failures, {len(failed_training)} training failures")

print(f"\n📁 Outputs:\n   - Subsets: {OUT_ROOT}\n   - Models:  {CLASSIFIER_DIR}")
