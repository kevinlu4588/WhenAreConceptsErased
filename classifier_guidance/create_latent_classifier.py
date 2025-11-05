#!/usr/bin/env python3
import os, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from diffusers import AutoencoderKL, DDIMScheduler
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from PIL import Image
import PIL.PngImagePlugin
from PIL import Image
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # Fix PNG issue


# ============================================================
# 1️⃣  Model
# ============================================================
class FixedTimestepEncoding(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod)

    def forward(self, t):
        alpha_bar = self.alphas_cumprod[t]
        return torch.stack([alpha_bar.sqrt(), (1 - alpha_bar).sqrt()], dim=-1)


class LatentClassifierT(nn.Module):
    def __init__(self, latent_shape=(4, 64, 64), scheduler=None):
        super().__init__()
        c, h, w = latent_shape
        flat_dim = c * h * w
        self.t_embed = FixedTimestepEncoding(scheduler)
        self.fc_t = nn.Linear(2, 1024)
        self.fc_x = nn.Linear(flat_dim, 1024)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, z, t):
        z_flat = z.flatten(start_dim=1)
        return self.net(self.fc_x(z_flat) + self.fc_t(self.t_embed(t)))


class BinDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transform):
        self.ds, self.transform = ds, transform

    def __getitem__(self, i):
        try:
            img = self.ds[i]["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = self.transform(img)
            label = torch.tensor(self.ds[i]["label_bin"], dtype=torch.float32)
        except Exception as e:
            print(f"[WARN] Error loading idx {i}: {e}")
            img = Image.new("RGB", (512, 512), "black")
            img = self.transform(img)
            label = torch.tensor(0.0)
        return img, label

    def __len__(self):
        return len(self.ds)


# ============================================================
# 2️⃣  Utility — Precompute and cache VAE latents
# ============================================================
def precompute_latents(vae, dataloader, device, cache_path):
    if os.path.exists(cache_path):
        print(f"⚡ Using cached latents at {cache_path}")
        return torch.load(cache_path)

    print(f"💾 Precomputing and caching latents to {cache_path}")
    all_latents, all_labels = [], []
    for imgs, labels in tqdm(dataloader, desc="Encoding latents"):
        imgs = imgs.to(device)
        with torch.no_grad():
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215
        all_latents.append(latents.cpu())
        all_labels.append(labels)
    all_latents = torch.cat(all_latents)
    all_labels = torch.cat(all_labels)
    torch.save({"latents": all_latents, "labels": all_labels}, cache_path)
    print(f"✅ Saved latents ({len(all_latents)} samples)")
    return {"latents": all_latents, "labels": all_labels}


# ============================================================
# 3️⃣  Utility — Power-law timestep sampling
# ============================================================
def sample_timesteps(num_train_timesteps, batch_size, device, power=3.0):
    """Sample timesteps biased toward higher (noisier) values."""
    u = torch.rand(batch_size, device=device)
    t = (u ** (1.0 / power)) * (num_train_timesteps - 1)
    return t.long().clamp(0, num_train_timesteps - 1)


# ============================================================
# 4️⃣  Training Loop
# ============================================================
def train_classifier(concept, subset_dir, save_dir, num_epochs=10, batch_size=8, lr=1e-4,
                     k_timesteps=7, timestep_power=3.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"

    subset_path = os.path.join(subset_dir, concept)
    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Subset not found at {subset_path}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"📂 Loading dataset from {subset_path}")
    ds = load_from_disk(subset_path)
    print(f"✅ Loaded {len(ds)} samples")

    pos_count = sum(ds["label_bin"])
    neg_count = len(ds) - pos_count
    print(f"📊 Class distribution - Pos: {pos_count}, Neg: {neg_count}")
    pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32, device=device)
    print(f"⚖️ Using pos_weight={pos_weight.item():.2f} in BCE")

    ds = ds.shuffle(seed=42)
    n_train = int(0.9 * len(ds))
    train_ds, val_ds = ds.select(range(n_train)), ds.select(range(n_train, len(ds)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_dl_raw = DataLoader(BinDataset(train_ds, transform), batch_size=batch_size,
                              shuffle=False, num_workers=4)
    val_dl_raw = DataLoader(BinDataset(val_ds, transform), batch_size=batch_size,
                            shuffle=False, num_workers=2)

    print(f"🔧 Loading VAE + scheduler...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    classifier = LatentClassifierT(scheduler=scheduler).to(device)

    # ---- Precompute and cache latents ----
    cache_train = os.path.join(save_dir, f"{concept}_train_latents.pt")
    cache_val = os.path.join(save_dir, f"{concept}_val_latents.pt")
    train_data = precompute_latents(vae, train_dl_raw, device, cache_train)
    val_data = precompute_latents(vae, val_dl_raw, device, cache_val)

    train_dl = DataLoader(TensorDataset(train_data["latents"], train_data["labels"]),
                          batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(val_data["latents"], val_data["labels"]),
                        batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss, best_pr_auc = float("inf"), 0.0
    save_path = os.path.join(save_dir, f"{concept}.pt")
    import matplotlib.pyplot as plt
    import pandas as pd

    metrics = {"epoch": [], "val_loss": [], "pr_auc": [], "roc_auc": [], "tpr5": [], "guidance_score": []}

    # ============================================================
    for epoch in range(1, num_epochs+1):
        classifier.train()
        total_loss = 0.0
        for latents, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{num_epochs} [train]"):
            latents, labels = latents.to(device), labels.to(device)
            noisy, lbls, ts = [], [], []
            for _ in range(k_timesteps):
                t = sample_timesteps(scheduler.config.num_train_timesteps,
                                     latents.size(0), device, power=timestep_power)
                n = torch.randn_like(latents)
                noisy.append(scheduler.add_noise(latents, n, t))
                lbls.append(labels)
                ts.append(t)
            noisy, lbls, ts = torch.cat(noisy), torch.cat(lbls), torch.cat(ts)
            logits = classifier(noisy, ts).squeeze(-1)
            loss = criterion(logits, lbls)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"🧮 Epoch {epoch} Train Loss: {total_loss/len(train_dl):.4f}")

        # ---------------- Validation ----------------
        classifier.eval()
        all_labels, all_logits = [], []
        val_loss = 0.0
        with torch.no_grad():
            for latents, labels in tqdm(val_dl, desc=f"Epoch {epoch}/{num_epochs} [val]"):
                latents, labels = latents.to(device), labels.to(device)
                logits_all = []
                for _ in range(3):
                    t = sample_timesteps(scheduler.config.num_train_timesteps,
                                         latents.size(0), device, power=timestep_power)
                    n = torch.randn_like(latents)
                    logits_all.append(classifier(scheduler.add_noise(latents, n, t), t).squeeze(-1))
                logits = torch.stack(logits_all).mean(0)
                val_loss += criterion(logits, labels).item()
                all_labels.append(labels.cpu())
                all_logits.append(logits.cpu())

        all_labels = torch.cat(all_labels).numpy()
        all_logits = torch.cat(all_logits).numpy()
        probs = 1 / (1 + np.exp(-all_logits))
        avg_val_loss = val_loss / len(val_dl)

        pr_auc = average_precision_score(all_labels, probs)
        roc_auc = roc_auc_score(all_labels, probs)
        fpr, tpr, _ = roc_curve(all_labels, probs)
        tpr5 = tpr[np.abs(fpr - 0.05).argmin()]

        metrics["epoch"].append(epoch)
        metrics["val_loss"].append(avg_val_loss)
        metrics["pr_auc"].append(pr_auc)
        metrics["roc_auc"].append(roc_auc)
        metrics["tpr5"].append(tpr5)

        print(f"✅ Epoch {epoch}")
        print(f"   Val Loss={avg_val_loss:.4f} | PR AUC={pr_auc:.3f} | "
              f"ROC AUC={roc_auc:.3f} | TPR@5%FPR={tpr5:.3f}")

                # ---------------- Save model (classifier-guidance priority) ----------------
    
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "tpr5": tpr5,
            }, save_path)
            print(f"💾 Saved best model  val_loss={val_loss}, "
                  f"PR AUC={pr_auc:.3f}, TPR@5%FPR={tpr5:.3f}) → {save_path}")

        print()

    print("="*60)
    print(f"🏁 Training finished for {concept}")
    print(f"Best Val Loss={best_val_loss:.4f}, Best PR AUC={best_pr_auc:.3f}")
    print(f"Model saved at: {save_path}")
    print("="*60)

    df = pd.DataFrame(metrics)
    csv_path = os.path.join(save_dir, f"{concept}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"📈 Saved metrics to {csv_path}")

    plt.figure(figsize=(10,6))
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", color="red")
    plt.plot(df["epoch"], df["pr_auc"], label="PR AUC", color="blue")
    plt.plot(df["epoch"], df["roc_auc"], label="ROC AUC", color="green")
    plt.plot(df["epoch"], df["tpr5"], label="TPR@5%FPR", color="purple")
    plt.plot(df["epoch"], df["guidance_score"], label="Guidance Score", color="orange", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title(f"Validation Metrics — {concept}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"{concept}_metrics.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"📊 Metrics plot saved to {plot_path}")
# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train latent classifier with cached VAE latents and biased timestep sampling.")
    p.add_argument("concept", type=str)
    p.add_argument("--subset_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timestep_power", type=float, default=3.0,
                   help="Power-law bias for timestep sampling (1=uniform, >1 favors noisy latents).")
    args = p.parse_args()
    train_classifier(args.concept, args.subset_dir, args.save_dir,
                     args.epochs, args.batch_size, args.lr, timestep_power=args.timestep_power)
