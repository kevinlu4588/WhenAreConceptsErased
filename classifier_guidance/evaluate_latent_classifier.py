from sklearn.metrics import roc_auc_score, recall_score
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk
from diffusers import AutoencoderKL, DDPMScheduler
from PIL import Image

# ---- Import model and dataset wrapper ----
from create_latent_classifier import LatentClassifierT, TimestepEmbedder, BinDataset

# ============================================================
# 1️⃣ Config
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/stable-diffusion-v1-4"
subset_path = "/share/u/kevin/DiffusionConceptErasure/local_imagenet_fixed_subsets/airliner"
save_path = "./latent_classifier_airliner.pt"
image_size = 512
batch_size = 8

# ============================================================
# 2️⃣ Load dataset + dataloader
# ============================================================
print(f"📂 Loading dataset from {subset_path}")
ds = load_from_disk(subset_path)
ds = ds.shuffle(seed=42)
test_ds = ds.select(range(int(0.9 * len(ds)), len(ds)))
print(f"✅ Loaded {len(test_ds)} test samples")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])
test_dl = DataLoader(BinDataset(test_ds, transform), batch_size=batch_size, shuffle=False, num_workers=2)

# ============================================================
# 3️⃣ Load VAE + Scheduler + Classifier
# ============================================================
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).eval()
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

classifier = LatentClassifierT().to(device)
classifier.load_state_dict(torch.load(save_path, map_location=device))
classifier.eval()
print(f"✅ Loaded classifier weights from {save_path}")

# ============================================================
# 4️⃣ Pass 1: Encode all test images once
# ============================================================
print("🧠 Encoding all test images into latents (1× pass)...")
all_latents, all_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_dl, desc="Encoding"):
        imgs = imgs.to(device)
        latents = vae.encode(imgs).latent_dist.sample() * 0.18215
        all_latents.append(latents.cpu())
        all_labels.append(labels)

all_latents = torch.cat(all_latents)
all_labels = torch.cat(all_labels)
print(f"✅ Cached {len(all_latents)} latents of shape {tuple(all_latents[0].shape)}")

# ============================================================
# 5️⃣ Pass 2: Evaluate across timesteps using cached latents
# ============================================================
num_timesteps = scheduler.config.num_train_timesteps
print(num_timesteps)
step_interval = max(1, num_timesteps // 50)  # ~50 eval points
timesteps_to_eval = list(range(0, num_timesteps, step_interval))
# lists for metrics
acc_list, auc_list, tpr_list = [], [], []

print(f"\n🔍 Evaluating classifier over {len(timesteps_to_eval)} timesteps...")
with torch.no_grad():
    for t_eval in tqdm(timesteps_to_eval, desc="Timesteps"):
        latents = all_latents.to(device)
        labels = all_labels.to(device)

        noise = torch.randn_like(latents)
        ts = torch.full((latents.size(0),), t_eval, device=device, dtype=torch.long)
        noisy = scheduler.add_noise(latents, noise, ts)

        logits = classifier(noisy, ts).squeeze(-1)
        preds = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy()

        # --- standard metrics ---
        acc = np.mean((preds > 0.5) == labels_np)
        try:
            auc = roc_auc_score(labels_np, preds)
        except ValueError:
            auc = np.nan

            # --- true positive recall ---
            # recall = TP / (TP + FN)
            tpr = recall_score(labels_np, (preds > 0.5), pos_label=1)
    
            acc_list.append(acc)
            auc_list.append(auc)
            tpr_list.append(tpr)
    
    print(f"\n✅ Evaluation complete! Peak metrics:")
    print(f"   🎯 Best accuracy: {max(acc_list):.3f}")
    print(f"   📊 Best AUC: {max(auc_list):.3f}")
    print(f"   🎨 Best TPR: {max(tpr_list):.3f}")
    
    # ============================================================
    # 6️⃣ Plot and Save
    # ============================================================
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps_to_eval, acc_list, label="Accuracy (All)", linewidth=2)
    plt.plot(timesteps_to_eval, auc_list, label="AUROC", linewidth=2)
    plt.plot(timesteps_to_eval, tpr_list, label="True Positive Recall", linestyle="--", linewidth=2, color="orange")
    plt.xlabel("Timestep")
    plt.ylabel("Performance")
    plt.title("Classifier Performance vs Diffusion Timestep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot_path = os.path.join(args.save_dir, f"{args.concept}_performance_vs_timestep.png")
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved performance plot to {save_plot_path}")
    
    # Save metrics CSV
    import pandas as pd
    metrics_path = os.path.join(args.save_dir, f"{args.concept}_metrics.csv")
    metrics_df = pd.DataFrame({
        'timestep': timesteps_to_eval,
        'accuracy': acc_list,
        'auc': auc_list,
        'tpr': tpr_list
    })
    metrics_df.to_csv(metrics_path, index=False)
    print(f"📊 Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
