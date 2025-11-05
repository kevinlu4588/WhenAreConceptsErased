import os
import itertools
from datasets import load_dataset, Dataset
from tqdm import tqdm

# ---- Configuration ----
dataset_name = "imagenet-1k"
# Use environment variable or default to relative path
base_dir = os.environ.get('DCE_BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
local_dir = os.path.join(base_dir, "local_imagenet_full")
num_shards = 100   # ~1.28M / 100 ≈ 12.8k samples per shard
os.makedirs(local_dir, exist_ok=True)

print(f"📦 Preparing to stream {dataset_name} and save to {local_dir} ...")

# ---- Initialize streaming dataset ----
streaming_ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
iterator = iter(streaming_ds)

# ---- Determine shard size ----
approx_total = 1_281_167  # known ImageNet train size
shard_size = approx_total // num_shards + 1

# ---- Detect already-completed shards ----
existing_shards = {
    int(d.split("_")[-1])
    for d in os.listdir(local_dir)
    if d.startswith("imagenet_train_") and os.path.isdir(os.path.join(local_dir, d))
}
start_idx = max(existing_shards) + 1 if existing_shards else 0
print(f"🔁 Resuming from shard {start_idx} (found {len(existing_shards)} complete shards)")

# ---- Skip already-processed samples ----
# Advance the iterator to skip previously completed shards
samples_to_skip = start_idx * shard_size
if samples_to_skip > 0:
    print(f"⏩ Skipping {samples_to_skip:,} samples already processed...")
    for _ in tqdm(range(samples_to_skip), desc="Skipping processed samples"):
        next(iterator, None)

# ---- Download & save remaining shards ----
for shard_idx in range(start_idx, num_shards):
    shard_path = os.path.join(local_dir, f"imagenet_train_{shard_idx:05d}")
    if os.path.exists(shard_path):
        print(f"✅ Shard {shard_idx} already exists, skipping.")
        continue

    # Take next chunk of samples
    samples = list(itertools.islice(iterator, shard_size))
    if not samples:
        print("📉 No more samples left to stream.")
        break

    # Convert to Hugging Face Dataset and save
    shard = Dataset.from_list(samples)
    shard.save_to_disk(shard_path)
    print(f"💾 Saved shard {shard_idx+1}/{num_shards} ({len(samples)} samples)")

print("✅ All available shards processed and saved.")
