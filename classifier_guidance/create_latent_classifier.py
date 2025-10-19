#!/usr/bin/env python3
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, DDIMScheduler
from datasets import load_from_disk
from PIL import Image
import argparse

# ============================================================
# 1️⃣  Define Model: Timestep Embedder + Classifier
# ============================================================
# ============================================================
# 🧩 Frozen DDIM-consistent timestep encoding
# ============================================================
class FixedTimestepEncoding(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod)

    def forward(self, t):
        alpha_bar = self.alphas_cumprod[t]
        signal_scale = alpha_bar.sqrt()
        noise_scale = (1 - alpha_bar).sqrt()
        # shape: (batch, 2)
        return torch.stack([signal_scale, noise_scale], dim=-1)



class LatentClassifierT(nn.Module):
    def __init__(self, latent_shape=(4, 64, 64), scheduler=None):
        super().__init__()
        c, h, w = latent_shape
        flat_dim = c * h * w

        # --- Frozen encoding ---
        self.t_embed = FixedTimestepEncoding(scheduler)
        self.fc_t = nn.Linear(2, 1024)  # project (signal, noise) → 1024

        # --- Latent + fusion layers ---
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
        x_proj = self.fc_x(z_flat)
        t_proj = self.fc_t(self.t_embed(t))
        return self.net(x_proj + t_proj)



class BinDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.ds[i]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(self.ds[i]["label_bin"], dtype=torch.float32)


# ============================================================
# 2️⃣  Training Loop
# ============================================================
def train_classifier(concept, base_dir=".", num_epochs=10, batch_size=8, lr=1e-4, k_timesteps=7):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"

    subset_path = os.path.join(base_dir, "subsets", concept)
    save_dir = os.path.join(base_dir, "classifiers", concept)
    os.makedirs(save_dir, exist_ok=True)

    print(f"📂 Loading dataset from {subset_path}")
    ds = load_from_disk(subset_path)
    print(f"✅ Loaded {len(ds)} samples")

    # Split dataset
    ds = ds.shuffle(seed=42)
    n = len(ds)
    train_ds = ds.select(range(0, int(0.8 * n)))
    val_ds = ds.select(range(int(0.8 * n), int(0.9 * n)))
    test_ds = ds.select(range(int(0.9 * n), n))
    print(f"📊 Split — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_dl = DataLoader(BinDataset(train_ds, transform), batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(BinDataset(val_ds, transform), batch_size=batch_size, shuffle=False, num_workers=2)

    # Load VAE + scheduler
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Init classifier
    classifier = LatentClassifierT(scheduler=scheduler).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # ============================================================
    # 3️⃣  Training loop
    # ============================================================
    for epoch in range(1, num_epochs + 1):
        classifier.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch} [train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            all_noisy, all_labels, all_t = [], [], []
            for _ in range(k_timesteps):
                ts = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device)
                noise = torch.randn_like(latents)
                noisy = scheduler.add_noise(latents, noise, ts)
                all_noisy.append(noisy)
                all_labels.append(labels)
                all_t.append(ts)

            noisy = torch.cat(all_noisy)
            labels = torch.cat(all_labels)
            ts = torch.cat(all_t)

            logits = classifier(noisy, ts).squeeze(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dl)

        # --- Validation ---
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_dl, desc=f"Epoch {epoch} [val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215
                ts = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device)
                noise = torch.randn_like(latents)
                noisy = scheduler.add_noise(latents, noise, ts)
                logits = classifier(noisy, ts).squeeze(-1)
                val_loss += loss_fn(logits, labels).item()

        avg_val_loss = val_loss / len(val_dl)
        print(f"✅ Epoch {epoch}/{num_epochs} — Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}\n")

    # ============================================================
    # 4️⃣  Save model
    # ============================================================
    save_path = os.path.join(save_dir, f"latent_classifier_{concept}.pt")
    torch.save(classifier.state_dict(), save_path)
    print(f"💾 Saved classifier to {save_path}")


# ============================================================
# 5️⃣  CLI Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train latent classifier for concept erasure.")
    parser.add_argument("concept", type=str, help="Concept name (e.g., 'airliner', 'church', etc.)")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing subsets/ and classifiers/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    train_classifier(
        concept=args.concept,
        base_dir=args.base_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
