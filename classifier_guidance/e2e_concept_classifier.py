#!/usr/bin/env python3
"""
End-to-end script for training concept classifiers on ImageNet data.
This script handles:
1. Downloading ImageNet dataset (if needed)
2. Creating class subsets for given concepts
3. Training latent classifiers
4. Evaluating and saving the classifiers
"""

import argparse
import os
import sys
import subprocess
import json
from time import time
from pathlib import Path
import shutil
import glob

import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 10MB


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end concept classifier training')
    
    # Required arguments
    parser.add_argument('concepts', nargs='+', help='List of concepts to train classifiers for')
    
    # Data arguments
    base_dir = os.environ.get('DCE_BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--imagenet-dir', type=str, 
                        default=os.path.join(base_dir, 'local_imagenet_full'),
                        help='Path to ImageNet dataset (will download if not exists)')
    parser.add_argument('--subset-dir', type=str, 
                        default=os.path.join(base_dir, 'local_imagenet_fixed_subsets'),
                        help='Directory to store created subsets')
    parser.add_argument('--output-dir', type=str, default='./concept_classifiers',
                        help='Directory to save trained classifiers')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n-neg', type=int, default=5000, help='Number of negative samples per concept')
    parser.add_argument('--timestep-power', type=float, default=3.0, 
                        help='Power-law bias for timestep sampling')
    
    # Script paths
    parser.add_argument('--base-dir', type=str, 
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Base directory containing training scripts')
    
    # Options
    parser.add_argument('--skip-download', action='store_true', 
                        help='Skip ImageNet download check')
    parser.add_argument('--force-recreate', action='store_true',
                        help='Force recreation of existing subsets')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force retraining of existing classifiers')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip classifier evaluation step')
    
    return parser.parse_args()


def check_imagenet_exists(imagenet_dir):
    """Check if ImageNet dataset exists at the specified path."""
    if not os.path.exists(imagenet_dir):
        return False
    
    # Check for sharded structure created by download script
    shard_dirs = glob.glob(os.path.join(imagenet_dir, 'imagenet_train_*'))
    return len(shard_dirs) > 0


def download_imagenet(imagenet_dir, base_dir):
    """Download ImageNet dataset using the download script."""
    print(f"\n{'='*70}")
    print("DOWNLOADING IMAGENET")
    print(f"{'='*70}\n")
    
    print("WARNING: ImageNet download will:")
    print("   - Download from HuggingFace (no authentication required)")
    print("   - Require ~150GB of disk space")
    print("   - Take several hours depending on connection speed")
    
    response = input("\nDo you want to proceed with download? [y/N]: ")
    if response.lower() != 'y':
        print("Download cancelled. Please provide ImageNet data manually.")
        sys.exit(1)
    
    download_script = os.path.join(base_dir, "download_imagenet_classes.py")
    if not os.path.exists(download_script):
        print(f"ERROR: Download script not found: {download_script}")
        sys.exit(1)
    
    # Set environment variable for the download script
    env = os.environ.copy()
    env['DCE_BASE_DIR'] = os.path.dirname(base_dir)
    
    cmd = [sys.executable, download_script]
    print(f"\nRunning: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        text=True, env=env
    )
    
    for line in process.stdout:
        print(line.strip())
    
    ret = process.wait()
    if ret == 0:
        print("\nImageNet download completed successfully!")
    else:
        print(f"\nERROR: Download failed with exit code {ret}")
        sys.exit(1)


def create_subset(concept, imagenet_dir, subset_dir, n_neg, base_dir, force=False):
    """Create a subset for a given concept."""
    concept_safe = concept.replace(", ", "_").replace(" ", "_")
    subset_path = os.path.join(subset_dir, concept_safe)
    
    if os.path.exists(subset_path) and not force:
        print(f"   Subset already exists -> {subset_path}")
        return True
    
    create_script = os.path.join(base_dir, "create_class_subsets.py")
    if not os.path.exists(create_script):
        print(f"   ERROR: Create script not found: {create_script}")
        return False
    
    cmd = [
        sys.executable, create_script,
        "--target_label", concept,
        "--root_dir", imagenet_dir,
        "--out_root", subset_dir,
        "--n_neg", str(n_neg),
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
        print(f"   Successfully created subset in {elapsed:.1f}s")
        return True
    else:
        print(f"   ERROR: Failed to create subset (exit code {ret})")
        return False


def train_classifier(concept, subset_dir, save_dir, epochs, batch_size, lr, 
                    timestep_power, base_dir, force=False):
    """Train a latent classifier for a given concept."""
    concept_safe = concept.replace(", ", "_").replace(" ", "_")
    classifier_path = os.path.join(save_dir, f"{concept_safe}.pt")
    
    if os.path.exists(classifier_path) and not force:
        print(f"   WARNING: Classifier already exists -> {classifier_path}")
        print(f"   INFO: Use --force-retrain to retrain")
        return True
    
    train_script = os.path.join(base_dir, "create_latent_classifier.py")
    if not os.path.exists(train_script):
        print(f"   ERROR: Train script not found: {train_script}")
        return False
    
    cmd = [
        sys.executable, train_script, concept_safe,
        "--subset_dir", subset_dir,
        "--save_dir", save_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--timestep_power", str(timestep_power),
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
        print(f"   Successfully trained classifier in {elapsed/60:.1f} min")
        return True
    else:
        print(f"   ERROR: Failed to train classifier (exit code {ret})")
        return False


def evaluate_classifier(concept, subset_dir, classifier_path, save_dir, base_dir):
    """Evaluate a trained classifier across timesteps."""
    evaluate_script = os.path.join(base_dir, "evaluate_latent_classifier.py")
    if not os.path.exists(evaluate_script):
        print(f"   ERROR: Evaluate script not found: {evaluate_script}")
        return False
    
    concept_safe = concept.replace(", ", "_").replace(" ", "_")
    
    cmd = [
        sys.executable, evaluate_script,
        "--classifier_path", classifier_path,
        "--subset_dir", subset_dir,
        "--concept", concept_safe,
        "--save_dir", save_dir,
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
        print(f"   Successfully evaluated classifier in {elapsed:.1f}s")
        return True
    else:
        print(f"   ERROR: Failed to evaluate classifier (exit code {ret})")
        return False


def save_final_classifiers(concepts, save_dir, output_dir):
    """Copy classifiers to final output directory with simple names."""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    for concept in concepts:
        concept_safe = concept.replace(", ", "_").replace(" ", "_")
        src = os.path.join(save_dir, f"{concept_safe}.pt")
        dst = os.path.join(output_dir, f"{concept_safe}.pt")
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            saved_files.append(dst)
            print(f"   {concept} -> {dst}")
        else:
            print(f"   ERROR: Not found: {src}")
    
    return saved_files


def main():
    args = parse_args()
    
    # Print configuration
    print(f"\n{'='*70}")
    print("END-TO-END CONCEPT CLASSIFIER TRAINING")
    print(f"{'='*70}\n")
    
    print("Configuration:")
    print(f"   Concepts: {', '.join(args.concepts)}")
    print(f"   ImageNet dir: {args.imagenet_dir}")
    print(f"   Subset dir: {args.subset_dir}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Negative samples: {args.n_neg}")
    
    # Create output directories
    os.makedirs(args.subset_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Check/Download ImageNet
    if not args.skip_download:
        print(f"\n{'='*70}")
        print("PHASE 1: Checking ImageNet dataset")
        print(f"{'='*70}\n")
        
        if not check_imagenet_exists(args.imagenet_dir):
            print(f"ERROR: ImageNet not found at {args.imagenet_dir}")
            download_imagenet(args.imagenet_dir, args.base_dir)
        else:
            print(f"ImageNet found at {args.imagenet_dir}")
    
    # Step 2: Create subsets
    print(f"\n{'='*70}")
    print("PHASE 2: Creating class subsets")
    print(f"{'='*70}\n")
    
    successful_subsets = []
    for i, concept in enumerate(args.concepts, 1):
        print(f"\n[{i}/{len(args.concepts)}] Creating subset for '{concept}'...")
        if create_subset(concept, args.imagenet_dir, args.subset_dir, 
                        args.n_neg, args.base_dir, args.force_recreate):
            successful_subsets.append(concept)
    
    if not successful_subsets:
        print("\nERROR: No subsets created successfully. Exiting.")
        sys.exit(1)
    
    # Step 3: Train classifiers
    print(f"\n{'='*70}")
    print("PHASE 3: Training latent classifiers")
    print(f"{'='*70}\n")
    
    # Use a temporary directory for training outputs
    temp_save_dir = os.path.join(args.base_dir, "latent_classifiers")
    os.makedirs(temp_save_dir, exist_ok=True)
    
    successful_training = []
    for i, concept in enumerate(successful_subsets, 1):
        print(f"\n[{i}/{len(successful_subsets)}] Training classifier for '{concept}'...")
        if train_classifier(concept, args.subset_dir, temp_save_dir,
                          args.epochs, args.batch_size, args.lr,
                          args.timestep_power, args.base_dir, args.force_retrain):
            successful_training.append(concept)
    
    # Step 4: Evaluate classifiers (optional)
    if not args.skip_evaluation:
        print(f"\n{'='*70}")
        print("PHASE 4: Evaluating classifiers")
        print(f"{'='*70}\n")
        
        for i, concept in enumerate(successful_training, 1):
            print(f"\n[{i}/{len(successful_training)}] Evaluating classifier for '{concept}'...")
            concept_safe = concept.replace(", ", "_").replace(" ", "_")
            classifier_path = os.path.join(temp_save_dir, f"{concept_safe}.pt")
            evaluate_classifier(concept, args.subset_dir, classifier_path, 
                              temp_save_dir, args.base_dir)
    
    # Step 5: Save final classifiers
    print(f"\n{'='*70}")
    print("PHASE 5: Saving final classifiers")
    print(f"{'='*70}\n")
    
    saved_files = save_final_classifiers(successful_training, temp_save_dir, args.output_dir)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Results:")
    print(f"   Subsets created: {len(successful_subsets)}/{len(args.concepts)}")
    print(f"   Classifiers trained: {len(successful_training)}/{len(successful_subsets)}")
    print(f"   Files saved: {len(saved_files)}")
    
    if saved_files:
        print(f"\nClassifiers saved to: {args.output_dir}")
        for f in saved_files:
            print(f"   - {os.path.basename(f)}")
    
    # Save metadata
    metadata = {
        'concepts': args.concepts,
        'successful_concepts': successful_training,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'n_negative_samples': args.n_neg,
            'timestep_power': args.timestep_power,
        },
        'paths': {
            'imagenet_dir': args.imagenet_dir,
            'subset_dir': args.subset_dir,
            'output_dir': args.output_dir,
        }
    }
    
    metadata_path = os.path.join(args.output_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    
    if len(successful_training) == len(args.concepts):
        print("\nAll tasks completed successfully!")
    else:
        failed = set(args.concepts) - set(successful_training)
        print(f"\nWARNING: Failed concepts: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()