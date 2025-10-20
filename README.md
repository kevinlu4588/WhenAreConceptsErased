# 🧠 Diffusion Concept Erasure — Probes & Evaluation

This repository provides a unified interface for running **concept erasure probes** (e.g., noise-based, inpainting, textual inversion) on Stable Diffusion models, and for **automatically evaluating** their results using CLIP similarity and classification accuracy.

---

# Environment Setup

Create and activate the provided Conda environment:

conda env create -f environment_erasing_env.yaml
conda activate erasing_env

## ⚡ Quick Start Example

Run a full probe and evaluation in one command:

```bash
bash run_single_model.sh

This will:

1. Load an SD1.4 model with "airliner" erased via esd-x, 
2. Run all available probes
3. Save generated images under results/
4. Automatically compute CLIP and classiifer-based evaluation metrics


Using Your Own Models (Direct Python Usage)

You can run runner.py directly instead of the shell script.

If your model directory contains a complete pipeline (e.g., UNet, VAE, text encoder):

python runner.py \
  --concept airliner \
  --erasing_type esdx \
  --probes noisebasedprobe \
  --num_images 10 \
  --device cuda \
  --config configs/default.yaml \
  --pipeline_path kevinlu4588/airliner

Or a path to a custom UNET checkpoint

python runner.py \
  --concept airliner \
  --erasing_type esdx \
  --probes noisebasedprobe \
  --num_images 10 \
  --device cuda \
  --config configs/default.yaml \
  --unet_path /path/to/your/custom_unet


  Running evaluator 

  python evaluator.py
