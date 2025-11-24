#!/usr/bin/env python3
import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 10MB

import os
import random
import argparse
from collections import Counter
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from datasets import get_dataset_config_info


def create_subset(target_label, root_dir, out_root, n_neg=5000):
    """Create a balanced subset for a specific ImageNet class."""
    
    # Load label mapping metadata
    info = get_dataset_config_info("imagenet-1k", trust_remote_code=True)
    label_names = info.features["label"].names
    
    try:
        target_id = label_names.index(target_label)
    except ValueError:
        print(f"Error: Label '{target_label}' not found in ImageNet-1k")
        print(f"   Available labels include: {label_names[:10]}...")
    
    print(f"\nCreating '{target_label}' dataset (all positives / {n_neg} negatives)")
    
    # Create output path using filesystem-safe name
    safe_label = target_label.replace(", ", "_").replace(" ", "_")
    out_path = os.path.join(out_root, safe_label)
    
    if os.path.exists(out_path):
        print(f"Already exists at {out_path}")
    
    pos_collected, neg_candidates = [], []
    neg_label_counts = Counter()
    
    # --- Pass 1: collect *all* positives ---
    print("Collecting all positive samples...")
    total_pos = 0
    
    shard_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for shard_name in tqdm(sorted(shard_dirs), desc="Scanning for positives"):
        shard_path = os.path.join(root_dir, shard_name)
        
        try:
            ds = Dataset.load_from_disk(shard_path)
        except Exception as e:
            print(f"   Warning: Skipping {shard_name}: {e}")
            continue
        
        # Fast index-based filtering
        labels = ds["label"]  # This is fast - just reads the label column
        positive_indices = [i for i, label in enumerate(labels) if label == target_id]
        
        if positive_indices:
            ds_pos = ds.select(positive_indices)
            pos_collected.append(ds_pos)
            total_pos += len(ds_pos)
            if total_pos % 100 == 0 or len(ds_pos) > 50:  # Only print for significant finds
                print(f"   Found {len(ds_pos)} in {shard_name}, total so far: {total_pos}")
    
    if not pos_collected:
        print(f"Warning: No positives found for '{target_label}' (id={target_id}). Exiting.")
        sys.exit(1)
    
    pos_ds = concatenate_datasets(pos_collected)
    print(f"Total positives collected: {len(pos_ds)}")
    
    # --- Pass 2: collect negatives ---
    print("Sampling negatives...")
    total_neg = 0
    
    for shard_name in tqdm(sorted(shard_dirs), desc="Sampling negatives"):
        shard_path = os.path.join(root_dir, shard_name)
        
        try:
            ds = Dataset.load_from_disk(shard_path)
        except Exception as e:
            print(f"   Warning: Skipping {shard_name}: {e}")
            continue
        
        # Fast index-based filtering for negatives
        labels = ds["label"]
        negative_indices = [i for i, label in enumerate(labels) if label != target_id]
        
        if not negative_indices:
            continue
        
        # Sample from the negative indices directly
        k = min(500, len(negative_indices))
        sample_idx = random.sample(negative_indices, k)
        ds_sample = ds.select(sample_idx)
        
        neg_candidates.append(ds_sample)
        # Update counts using the sampled labels
        sampled_labels = [labels[i] for i in sample_idx]
        neg_label_counts.update(sampled_labels)
        total_neg += k
        
        if total_neg % 2500 == 0:  # Only print every 2500 samples
            print(f"   Collected {total_neg} negatives so far...")
        
        if total_neg >= 5 * n_neg:  # Collect 5x to ensure diversity
            break
    
    if not neg_candidates:
        print(f"Error: No negative samples found. Exiting.")
        return
    
    neg_ds_all = concatenate_datasets(neg_candidates)
    
    # Final sampling to get exactly n_neg samples
    if len(neg_ds_all) > n_neg:
        final_sample_indices = random.sample(range(len(neg_ds_all)), n_neg)
        neg_ds = neg_ds_all.select(final_sample_indices)
    else:
        neg_ds = neg_ds_all
        print(f"   Warning: Only found {len(neg_ds)} negatives (requested {n_neg})")
    
    print(f"Using {len(neg_ds)} negative samples.")
    
    # --- Combine & label ---
    full_ds = concatenate_datasets([pos_ds, neg_ds])
    
    def add_label_bin(batch):
        """Process a batch of examples"""
        batch["label_bin"] = [int(label == target_id) for label in batch["label"]]
        return batch
    
    print("Adding binary labels...")
    full_ds = full_ds.map(add_label_bin, batched=True, batch_size=1000)
    
    # --- Save ---
    print(f"Saving dataset to {out_path}...")
    full_ds.save_to_disk(out_path)
    print(f"Saved dataset to {out_path}")
    
    # --- Summary ---
    pos_count = sum(full_ds["label_bin"])
    neg_count = len(full_ds) - pos_count
    print(f"\nDistribution:")
    print(f"   Positives: {pos_count}")
    print(f"   Negatives: {neg_count}")
    print(f"   Ratio: {pos_count / (pos_count + neg_count):.3f} positives")
    
    print("\nTop 5 negative labels:")
    for label_id, count in neg_label_counts.most_common(5):
        if label_id < len(label_names):
            print(f"   {label_names[label_id]:<30} : {count}")


def main():
    parser = argparse.ArgumentParser(description="Create balanced ImageNet subsets for concept erasure.")
    parser.add_argument("--target_label", type=str, required=True, 
                        help="Target class name (e.g., 'airliner', 'church')")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory containing ImageNet shards")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Output directory for subsets")
    parser.add_argument("--n_neg", type=int, default=5000,
                        help="Number of negative samples (default: 5000)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_root, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create the subset
    create_subset(
        target_label=args.target_label,
        root_dir=args.root_dir,
        out_root=args.out_root,
        n_neg=args.n_neg
    )


if __name__ == "__main__":
    main()