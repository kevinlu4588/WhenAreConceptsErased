#!/usr/bin/env python3
import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path
import torch
import yaml
from tqdm import tqdm

from probes.base_probe import BaseProbe


# ============================================================
# 🧠 1️⃣ Utility: Discover all probe subclasses
# ============================================================
def discover_probes(probes_dir: str):
    """Auto-discovers all subclasses of BaseProbe in a folder."""
    probe_classes = {}
    for file in Path(probes_dir).glob("*_probe.py"):
        module_name = file.stem
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseProbe) and obj is not BaseProbe:
                probe_classes[name.lower()] = obj
    return probe_classes


# ============================================================
# ⚙️ 2️⃣ Utility: Load config YAML or defaults
# ============================================================
def load_config(config_path="configs/default.yaml"):
    default_config = {
        "base_images_path": "results/base_model",
        "mask_size": [256, 256],
        "initializer_token": "object",
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f)
        default_config.update(user_cfg)
    return default_config


# ============================================================
# 🚀 3️⃣ Core Runner Logic
# ============================================================
def run_all(probes_to_run, concepts, num_images, device, config, erasing_type, pipeline_path=None, unet_path=None):
    probes_dir = os.path.join(os.path.dirname(__file__), "probes")
    probe_classes = discover_probes(probes_dir)

    available_probes = list(probe_classes.keys())
    if "all" in probes_to_run:
        probes_to_run = available_probes

    print(f"🧩 Available probes: {available_probes}")
    print(f"✅ Selected probes: {probes_to_run}")

    if not pipeline_path and not unet_path:
        raise ValueError(
            "❌ You must provide either --pipeline_path or --unet_path to load a model."
        )

    for concept in concepts:
        print(f"\n⚙️ Running {probes_to_run} on {concept}")
        print(f"📦 Using {'pipeline' if pipeline_path else 'UNet'} path: {pipeline_path or unet_path}")

        for probe_name in probes_to_run:
            ProbeClass = probe_classes[probe_name]
            probe = ProbeClass(
                pipeline_path=pipeline_path,
                unet_path=unet_path,
                erasing_type=erasing_type,
                concept=concept,
                num_images=num_images,
                device=device,
                config=config,
            )
            try:
                probe.run(num_images=num_images, debug=True)
            except Exception as e:
                print(f"❌ {probe_name} failed for {concept}: {e}")


# ============================================================
# 🧮 4️⃣ CLI Interface
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Run concept erasure probes.")
    parser.add_argument("--concepts", nargs="+", default=["van_gogh"], help="Concepts to test")
    parser.add_argument("--probes", nargs="+", default=["all"], help="Which probes to run (default: all)")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images per probe")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--pipeline_path", type=str, help="Path to a pretrained pipeline folder.")
    parser.add_argument("--unet_path", type=str, help="Path to a UNet checkpoint for SD1.4 base.")
    parser.add_argument("--erasing_type", type=str, help="Name of the concept erasure method")

    args = parser.parse_args()

    if args.pipeline_path and args.unet_path:
        parser.error("❌ Please provide only one of --pipeline_path or --unet_path, not both.")
    if not args.pipeline_path and not args.unet_path:
        parser.error("❌ You must provide either --pipeline_path or --unet_path.")

    config = load_config(args.config)

    run_all(
        probes_to_run=args.probes,
        concepts=args.concepts,
        num_images=args.num_images,
        device=args.device,
        config=config,
        erasing_type = args.erasing_type,
        pipeline_path=args.pipeline_path,
        unet_path=args.unet_path,
    )
