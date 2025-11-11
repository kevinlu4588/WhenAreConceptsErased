#!/usr/bin/env python3
# ================================================================
# 🚀 Runner for Concept Erasure Probes
# ================================================================
# Usage examples:
#   python runner.py --concept church \
#       --pipeline_path DiffusionConceptErasure/uce_church \
#       --erasing_type uce --probes all --num_images 30
#
#   python runner.py --concept church --unet_path checkpoints/unet.pt \
#       --erasing_type stereo --probes standard noise --num_images 10
# ================================================================

import argparse
import os
from pathlib import Path
import torch
import yaml

# Import probes explicitly for readability
from probes.standard_prompt_probe import StandardPromptProbe
from probes.noise_based_probe import NoiseBasedProbe
from probes.diffusion_completion_probe import DiffusionCompletionProbe
from probes.interference_probe import InterferenceProbe
from probes.inpainting_probe import InpaintingProbe
from probes.textual_inversion_probe import TextualInversionProbe
from evaluator import Evaluator

# ================================================================
# ⚙️ Utility Functions
# ================================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "data" / "demo_results"

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

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

# ================================================================
# 🧩 Available Probes
# ================================================================
ALL_PROBES = {
    "standardpromptprobe": StandardPromptProbe,
    "noisebasedprobe": NoiseBasedProbe,
    "diffusioncompletionprobe": DiffusionCompletionProbe,
    "interferenceprobe": InterferenceProbe,
    "inpaintingprobe": InpaintingProbe,
    "textualinversionprobe": TextualInversionProbe,
}

# ================================================================
# 🚀 Core Runner Logic
# ================================================================
def run_probes(
    probes_to_run,
    concepts,
    num_images,
    device,
    config,
    erasing_type,
    pipeline_path=None,
    unet_path=None,
):
    if not pipeline_path and not unet_path:
        raise ValueError("❌ You must provide either --pipeline_path or --unet_path.")

    if "all" in [p.lower() for p in probes_to_run]:
        probes_to_run = list(ALL_PROBES.keys())

    print(f"\n🧠 Running probes: {probes_to_run}")
    print(f"📦 Model source: {'Pipeline' if pipeline_path else 'UNet'}")
    print(f"⚙️ Erasing type: {erasing_type}")
    print(f"🎯 Concepts: {concepts}\n")

    for concept in concepts:
        for probe_name in probes_to_run:
            probe_key = probe_name.lower()
            if probe_key not in ALL_PROBES:
                print(f"⚠️ Unknown probe '{probe_name}' — skipping.")
                continue

            ProbeClass = ALL_PROBES[probe_key]
            output_dir = RESULTS_DIR / erasing_type / concept / probe_key
            ensure_dir(output_dir)

            print(f"\n➡️ Running {probe_key} on '{concept}' ({erasing_type})")
            probe = ProbeClass(
                pipeline_path=pipeline_path,
                unet_path=unet_path,
                erasing_type=erasing_type,
                concept=concept,
                num_images=num_images,
                device=device,
                config=config,
            )
            probe.output_dir = str(output_dir)

            try:
                probe.run(num_images=num_images, debug=True)
                print(f"✅ {probe_key} completed successfully.")
            except Exception as e:
                print(f"❌ {probe_key} failed for {concept}: {e}")

# ================================================================
# 🏁 CLI Entry Point
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Run concept erasure probes.")
    parser.add_argument("--concept", nargs="+", help="Concepts to test")
    parser.add_argument("--probes", nargs="+", default=["all"], help="Which probes to run (default: all)")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images per probe")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--pipeline_path", type=str, help="Path to a pretrained pipeline folder.")
    parser.add_argument("--unet_path", type=str, help="Path to a UNet checkpoint for SD1.4 base.")
    parser.add_argument("--erasing_type", type=str, help="Name of the concept erasure method")

    args = parser.parse_args()

    # Validation logic
    if args.pipeline_path and args.unet_path:
        parser.error("❌ Please provide only one of --pipeline_path or --unet_path.")
    if not args.pipeline_path and not args.unet_path:
        parser.error("❌ You must provide either --pipeline_path or --unet_path.")
    if not args.erasing_type:
        parser.error("❌ You must provide --erasing_type to name the method (e.g., uce, stereo, rece).")
    if not args.concept:
        parser.error("❌ You must specify at least one --concept.")

    config = load_config(args.config)

    run_probes(
        probes_to_run=args.probes,
        concepts=args.concept,
        num_images=args.num_images,
        device=args.device,
        config=config,
        erasing_type=args.erasing_type,
        pipeline_path=args.pipeline_path,
        unet_path=args.unet_path,
    )

    print(f"\n{'='*70}")
    print(f"📊 Running Evaluator on results in '{RESULTS_DIR}'")
    print(f"{'='*70}")
    try:
        evaluator = Evaluator(RESULTS_DIR)
        evaluator.evaluate()
        print("✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

    print("\n🎉 Runner completed!")

if __name__ == "__main__":
    main()
