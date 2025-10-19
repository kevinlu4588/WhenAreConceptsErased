import os
import random
from collections import Counter
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from datasets import get_dataset_config_info

# ---- Configuration ----
root_dir = "/share/u/kevin/DiffusionConceptErasure/local_imagenet_full"
out_root = "/share/u/kevin/DiffusionConceptErasure/local_imagenet_fixed_subsets"
os.makedirs(out_root, exist_ok=True)

TARGET_LABEL = "airliner"
N_NEG = 5000  # only negatives are capped

def create_subset():
    # Load label mapping metadata
    info = get_dataset_config_info("imagenet-1k", trust_remote_code=True)
    label_names = info.features["label"].names
    target_id = label_names.index(TARGET_LABEL)

    print(f"\n🎯 Creating '{TARGET_LABEL}' dataset (all positives / {N_NEG} neg)")
    out_path = os.path.join(out_root, TARGET_LABEL.replace(", ", "_").replace(" ", "_"))
    if os.path.exists(out_path):
        print(f"✅ Already exists at {out_path}")
        return

    pos_collected, neg_candidates = [], []
    neg_label_counts = Counter()

    # --- Pass 1: collect *all* positives ---
    print("🔍 Collecting all positive samples...")
    total_pos = 0
    for shard_name in tqdm(sorted(os.listdir(root_dir)), desc="Scanning for positives"):
        shard_path = os.path.join(root_dir, shard_name)
        if not os.path.isdir(shard_path):
            continue
        ds = Dataset.load_from_disk(shard_path)
        ds_pos = ds.filter(lambda ex: ex["label"] == target_id)
        if len(ds_pos) > 0:
            pos_collected.append(ds_pos)
            total_pos += len(ds_pos)
            print(f"   ➕ Found {len(ds_pos)} in {shard_name}, total so far: {total_pos}")

    if not pos_collected:
        print(f"⚠️ No positives found for '{TARGET_LABEL}'. Exiting.")
        return
    pos_ds = concatenate_datasets(pos_collected)
    print(f"📦 Total positives collected: {len(pos_ds)}")

    # --- Pass 2: collect negatives ---
    print("🔍 Sampling negatives...")
    total_neg = 0
    for shard_name in tqdm(sorted(os.listdir(root_dir)), desc="Sampling negatives"):
        shard_path = os.path.join(root_dir, shard_name)
        if not os.path.isdir(shard_path):
            continue
        ds = Dataset.load_from_disk(shard_path)
        ds_neg = ds.filter(lambda ex: ex["label"] != target_id)
        if len(ds_neg) == 0:
            continue
        k = min(500, len(ds_neg))  # small sample per shard
        sample_idx = random.sample(range(len(ds_neg)), k)
        ds_sample = ds_neg.select(sample_idx)
        neg_candidates.append(ds_sample)
        neg_label_counts.update(ds_sample["label"])
        total_neg += k
        print(f"   ➖ Sampled {k} from {shard_name}, total negatives so far: {total_neg}")
        if total_neg >= 5 * N_NEG:
            break

    neg_ds_all = concatenate_datasets(neg_candidates)
    neg_ds = neg_ds_all.select(random.sample(range(len(neg_ds_all)), N_NEG))
    print(f"📦 Using {len(neg_ds)} negative samples.")

    # --- Combine & label ---
    full_ds = concatenate_datasets([pos_ds, neg_ds])

    def add_label_bin(ex):
        ex["label_bin"] = int(ex["label"] == target_id)
        return ex

    full_ds = full_ds.map(add_label_bin)

    # --- Save ---
    full_ds.save_to_disk(out_path)
    print(f"💾 Saved dataset to {out_path}")

    # --- Summary ---
    pos_count = sum(full_ds["label_bin"])
    neg_count = len(full_ds) - pos_count
    print(f"\n📊 Distribution:")
    print(f"   ➕ Positives: {pos_count}")
    print(f"   ➖ Negatives: {neg_count}")
    print(f"   ⚖️ Ratio: {pos_count / (pos_count + neg_count):.3f} positives")

    print("\n🔝 Top 5 negative labels:")
    for label_id, count in neg_label_counts.most_common(5):
        print(f"   {label_names[label_id]:<30} : {count}")

if __name__ == "__main__":
    create_subset()
