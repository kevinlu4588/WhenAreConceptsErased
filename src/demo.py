#!/usr/bin/env python3
# ================================================================
# 🧠 Test script for running all probes and evaluation
# ================================================================
# Available Probes:
# - StandardPromptProbe: Basic prompt-based image generation
# - NoiseBasedProbe: Noise-based generation (with/without classifier guidance)
# - DiffusionCompletionProbe: Diffusion completion from partial images
# - InterferenceProbe: Tests interference between concepts
# - InpaintingProbe: Inpainting-based generation
# - TextualInversionProbe: Textual inversion-based generation
# ================================================================

import os
from pathlib import Path
import torch
import yaml
from probes.standard_prompt_probe import StandardPromptProbe
from probes.noise_based_probe import NoiseBasedProbe
from probes.diffusion_completion_probe import DiffusionCompletionProbe
from probes.interference_probe import InterferenceProbe
from probes.inpainting_probe import InpaintingProbe
from probes.textual_inversion_probe import TextualInversionProbe
from evaluator import Evaluator

# ============================================================
# 🔧 Configuration
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "data" / "results"

BASE_MODEL_DIR = "DiffusionConceptErasure"
CONFIG_PATH = "configs/default.yaml"  # Changed to use minimal config
CONCEPTS = ["church"]
MODELS = ["uce", "stereo", "rece"]
NUM_IMAGES = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 🧩 Utility helpers
# ============================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_base_model_images(concept: str, num_images: int):
    """Ensure base model results exist; if not, run StandardPromptProbe."""
    base_output_dir = RESULTS_DIR / "base_model" / concept / "standardpromptprobe"
    existing_images = list(base_output_dir.glob("*.png"))

    if len(existing_images) >= num_images:
        print(f"✅ Found {len(existing_images)} base images for '{concept}'")
        return

    print(f"⚠️ No base images found for '{concept}', running StandardPromptProbe on base model...")

    from probes.standard_prompt_probe import StandardPromptProbe
    config = load_config(CONFIG_PATH)
    config["score_type"] = "clip"

    base_model = "CompVis/stable-diffusion-v1-4"
    pipeline_path = base_model  # Hugging Face identifier works fine for StandardPromptProbe

    ensure_dir(base_output_dir)

    probe = StandardPromptProbe(
        pipeline_path=pipeline_path,
        erasing_type="base_model",
        concept=concept,
        num_images=num_images,
        device=DEVICE,
        config=config,
    )
    probe.output_dir = str(base_output_dir)
    probe.run(num_images=num_images, debug=True)
    print(f"✅ Generated {num_images} base images for '{concept}'")


def make_pipeline_path(model: str, concept: str):
    """Resolve model path like .../final_models/esdx_golf_ball"""
    return os.path.join(BASE_MODEL_DIR, f"{model}_{concept}")

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
# 🚀 Core Execution Logic
# ============================================================
def run_all_probes_for_concept_and_model(concept: str, model: str):
    print(f"\n{'='*70}")
    print(f"🎯 Running ALL PROBES for {model.upper()} on concept '{concept}'")
    print(f"{'='*70}")

    # Ensure base model images exist before other probes
    if model != "base_model":
        ensure_base_model_images(concept, NUM_IMAGES)

    pipeline_path = make_pipeline_path(model, concept)

    # Load config
    config = load_config(CONFIG_PATH)
    config["score_type"] = "clip"
    config["classifier_root"] = BASE_MODEL_DIR

    # Define all probes to run
    probes_to_run = [
        {
            "name": "StandardPromptProbe",
            "class": StandardPromptProbe,
            "output_dir": f"{RESULTS_DIR}/{model}/{concept}/standardpromptprobe",
            "kwargs": {}
        },
        {
            "name": "StandardPromptProbe",
            "class": StandardPromptProbe,
            "output_dir": f"{RESULTS_DIR}/{model}/{concept}/classifier_guidance",
            "kwargs": {"debug": True, "use_classifier_guidance": True}
        },
        {
            "name": "NoiseBasedProbe", 
            "class": NoiseBasedProbe,
            "output_dir": f"{RESULTS_DIR}/{model}/{concept}/noisebasedprobe",
            "kwargs": {"debug": True, "use_classifier_guidance": False}
        },
        {
            "name": "NoiseBasedProbe", 
            "class": NoiseBasedProbe,
            "output_dir": f"{RESULTS_DIR}/{model}/{concept}/classifier_guidance_noise_based",
            "kwargs": {"debug": True, "use_classifier_guidance": True}
        },
        {
            "name": "DiffusionCompletionProbe",
            "class": DiffusionCompletionProbe, 
            "output_dir": f"{RESULTS_DIR}/{model}/{concept}/diffusioncompletionprobe",
            "kwargs": {}
        },
        {
            "name": "InterferenceProbe",
            "class": InterferenceProbe,
            "output_dir": f"{RESULTS_DIR}/{model}/{concept}/interference", 
            "kwargs": {}
        },
        {
            "name": "InpaintingProbe",
            "class": InpaintingProbe,
            "output_dir": f"{RESULTS_DIR}/{model}/{concept}/inpaintingprobe",
            "kwargs": {}
        },
        {
            "name": "TextualInversionProbe", 
            "class": TextualInversionProbe,
            "output_dir": f"{RESULTS_DIR}/{model}/{concept}/textualinversionprobe",
            "kwargs": {}
        }
    ]

    # Run each probe
    for probe_info in probes_to_run:
        print(f"\n➡️ Running {probe_info['name']}: {concept} ({model})")
        
        # Create output directory
        ensure_dir(probe_info["output_dir"])
    
        # Initialize probe
        probe = probe_info["class"](
            pipeline_path=pipeline_path,
            erasing_type=model,
            concept=concept,
            num_images=NUM_IMAGES,
            device=DEVICE,
            config=config,
        )
        probe.output_dir = probe_info["output_dir"]
        
        # Run probe with any additional kwargs
        probe.run(num_images=NUM_IMAGES, **probe_info["kwargs"])
        print(f"✅ {probe_info['name']} completed successfully")

# ============================================================
# 🏁 Main
# ============================================================
if __name__ == "__main__":
    print(f"🧠 Running ALL probes for {len(CONCEPTS)} concepts × {len(MODELS)} models")
    base_model = "CompVis/stable-diffusion-v1-4"
    # Run all probes
    for concept in CONCEPTS:
        for model in MODELS:
            run_all_probes_for_concept_and_model(concept, model)

    print(f"\n✅ All probes finished!")
    
    # Run evaluator
    print(f"\n{'='*70}")
    print(f"📊 Running Evaluator on results in '{RESULTS_DIR}'")
    print(f"{'='*70}")
    
    try:
        evaluator = Evaluator(RESULTS_DIR)
        evaluator.evaluate()
        print("✅ Evaluation completed successfully!")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

    print("\n🎉 Test script completed!")