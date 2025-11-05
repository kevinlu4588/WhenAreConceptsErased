#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uploads two latent classifiers to the Hugging Face org "DiffusionConceptErasure":
- latent-classifier-airliner
- latent-classifier-church

Requires: huggingface_hub (pip install huggingface_hub)
Auth: run `huggingface-cli login` or set HF_TOKEN env var.
"""
import io
import os
from datetime import datetime
from huggingface_hub import HfApi, create_repo, whoami

ORG = "DiffusionConceptErasure"

MODELS = [
    {
        "repo_name": "latent-classifier-airliner",
        "title": "Airliner Latent Classifier (Stable Diffusion v1.4)",
        "pt_path": "/share/u/kevin/DiffusionConceptErasure/classifier_guidance/latent_classifiers/airliner.pt",
        "filename_in_repo": "airliner.pt",
        "concept": "airliner",
    },
    {
        "repo_name": "latent-classifier-church",
        "title": "Church Latent Classifier (Stable Diffusion v1.4)",
        "pt_path": "/share/u/kevin/DiffusionConceptErasure/classifier_guidance/latent_classifiers/church.pt",
        "filename_in_repo": "church.pt",
        "concept": "church",
    },
]

README_TMPL = """---
license: mit
library_name: pytorch
tags:
- latent-classifier
- stable-diffusion
- diffusion
- concept-probing
- classifier-guidance
- SD1.4
pipeline_tag: text-to-image
language:
- en
---

# {title}

**Latent-space binary classifier** trained on **Stable Diffusion v1.4** VAE latents (shape `4×64×64`) with a simple MLP head and a timestep embedding (from the DDIM scheduler).  
Intended for **concept probing** and **classifier guidance** in diffusion workflows.

- **Concept:** `{concept}`
- **Input:** latent tensor `z ∈ ℝ^{{4×64×64}}` and a diffusion timestep `t`
- **Output:** logit/probability that `z` contains the concept at timestep `t`
- **Author/Org:** DiffusionConceptErasure
- **Date:** {date}

## Usage (PyTorch)

```python
import torch
from diffusers import DDIMScheduler

# ---- model definition (must match training) ----
import torch.nn as nn
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

# ---- load weights ----
repo_id = "{org}/{repo_name}"
ckpt_name = "{filename_in_repo}"

scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
model = LatentClassifierT(scheduler=scheduler)

state = torch.hub.load_state_dict_from_url(
    f"https://huggingface.co/{{repo_id}}/resolve/main/{{ckpt_name}}",
    map_location="cpu"
)
model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
model.eval()

# Example inference:
z = torch.randn(1, 4, 64, 64)           # latent
t = torch.randint(0, scheduler.config.num_train_timesteps, (1,))  # timestep
with torch.no_grad():
    logit = model(z, t)                 # shape [1, 1]
    prob = torch.sigmoid(logit)
print(prob.item())
```

## Notes

- Trained with DDIM power-law timestep sampling biased to noisier latents.
- For classifier guidance, average logits across a few noisy t samples if desired.
- Expectation: highest discriminability at moderate noise; extreme noise reduces signal.

## Citation

If you use this, please cite:

```bibtex
@inproceedings{{lu2025concepts,
  title={{When Are Concepts Erased From Diffusion Models?}},
  author={{Kevin Lu and Nicky Kriplani and Rohit Gandikota and Minh Pham and David Bau and Chinmay Hegde and Niv Cohen}},
  booktitle={{NeurIPS}},
  year={{2025}}
}}
```
"""

def main():
    api = HfApi()
    
    # Sanity: confirm auth
    try:
        me = whoami()
        print(f"🔐 Logged in as: {me['name']} ({me.get('type','user')})")
    except Exception as e:
        raise SystemExit("You must huggingface-cli login or set HF_TOKEN") from e
    
    for m in MODELS:
        repo_id = f"{ORG}/{m['repo_name']}"
        print(f"\n=== Upserting repo: {repo_id} ===")
        
        # Create or ensure repo exists in org
        create_repo(
            repo_id=repo_id,  # Full repo ID includes org/name
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        
        # Upload weights
        if not os.path.exists(m["pt_path"]):
            raise FileNotFoundError(f"Missing file: {m['pt_path']}")
        
        print(f"⬆️  Uploading weights: {m['pt_path']} → {repo_id}/{m['filename_in_repo']}")
        api.upload_file(
            path_or_fileobj=m["pt_path"],
            path_in_repo=m["filename_in_repo"],
            repo_id=repo_id,
            repo_type="model",
        )
        
        # Build README (model card)
        readme_str = README_TMPL.format(
            title=m["title"],
            concept=m["concept"],
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            org=ORG,
            repo_name=m["repo_name"],
            filename=m["filename_in_repo"],
            filename_in_repo=m["filename_in_repo"],
        )
        
        buf = io.BytesIO(readme_str.encode("utf-8"))
        print(f"📝 Uploading README.md → {repo_id}/README.md")
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
    
    print("\n✅ Done. Check your repos on the Hub.")

if __name__ == "__main__":
    main()