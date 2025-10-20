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

from base_probe import BaseProbe


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
def load_config(config_path="/share/u/kevin/DiffusionConceptErasure/new_probe_code/configs/default.yaml"):
    default_config = {
        "base_images_path": "/share/u/kevin/DiffusionConceptErasure/new_probe_code/results/base_model",
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
def run_all(probes_to_run, erasing_types, concepts, num_images, device, config, pipeline_path=None):
    probes_dir = os.path.dirname(__file__)
    probe_classes = discover_probes(probes_dir)

    available_probes = list(probe_classes.keys())
    if "all" in probes_to_run:
        probes_to_run = available_probes

    print(f"🧩 Available probes: {available_probes}")
    print(f"✅ Selected probes: {probes_to_run}")

    for erasing_type in erasing_types:
        for concept in concepts:
            # Allow user override
            if pipeline_path:
                model_path = pipeline_path
            else:
                model_path = f"/share/u/kevin/ErasingDiffusionModels/final_models/{erasing_type}_{concept}"

            print(f"\n⚙️ Running {probes_to_run} on {concept} ({erasing_type})")
            print(f"📦 Using pipeline path: {model_path}")

            for probe_name in probes_to_run:
                ProbeClass = probe_classes[probe_name]
                probe = ProbeClass(
                    pipeline_path=model_path,
                    erasing_type=erasing_type,
                    concept=concept,
                    num_images=num_images,
                    device=device,
                    config=config,
                )
                try:
                    probe.run(num_images=num_images, debug=True)
                except Exception as e:
                    print(f"❌ {probe_name} failed for {concept} ({erasing_type}): {e}")


# ============================================================
# 🧮 4️⃣ CLI Interface
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Run concept erasure probes.")
    parser.add_argument("--concepts", nargs="+", default=["all"], help="Concepts to test (default: all)")
    parser.add_argument("--erasing_types", nargs="+", default=["esdx"], help="Erasing methods")
    parser.add_argument("--probes", nargs="+", default=["all"], help="Which probes to run (default: all)")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images per probe")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--pipeline_path", type=str, default=None,
                        help="Optional explicit path to a model pipeline (overrides erasing_type/concept structure).")

    args = parser.parse_args()
    config = load_config(args.config)
    print(config)
    # Default list of concepts if "all" passed
    if "all" in args.concepts:
        args.concepts = [
            "van_gogh"
        ]

    run_all(
        probes_to_run=args.probes,
        erasing_types=args.erasing_types,
        concepts=args.concepts,
        num_images=args.num_images,
        device=args.device,
        config=config,
        pipeline_path=args.pipeline_path,
    )


if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    main()
