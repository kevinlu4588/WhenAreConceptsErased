#!/usr/bin/env python3
import os
from pathlib import Path
import torch
import yaml
from probes.standard_prompt_probe import StandardPromptProbe
from probes.noise_based_probe import NoiseBasedProbe


# ============================================================
# 🔧 Configuration
# ============================================================
BASE_MODEL_DIR = "/share/u/kevin/ErasingDiffusionModels/final_models"
CLASSIFIER_DIR = "/share/u/kevin/DiffusionConceptErasure/classifier_guidance/latent_classifiers"
RESULTS_DIR = "classifier_results"
CONFIG_PATH = "configs/default.yaml"

CONCEPTS = ["airliner", "garbage_truck", "golf_ball"]
MODELS = ["esdx", "esdu", "uce", "stereo", "ga", "rece"]
# MODELS = ["stereo", "ga", "rece"]
NUM_IMAGES = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 🧩 Utility helpers
# ============================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_pipeline_path(model: str, concept: str):
    """Resolve model path like .../final_models/esdx_golf_ball"""
    return os.path.join(BASE_MODEL_DIR, f"{model}_{concept}")

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
# 🚀 Core Execution Logic
# ============================================================
def run_probes_for_concept_and_model(concept: str, model: str):
    print(f"\n{'='*70}")
    print(f"🎯 Running {model.upper()} on concept '{concept}'")
    print(f"{'='*70}")

    pipeline_path = make_pipeline_path(model, concept)
    if not os.path.exists(pipeline_path):
        print(f"⚠️ Missing pipeline: {pipeline_path}, skipping.")
        return

    # Load config
    config = load_config(CONFIG_PATH)
    config["score_type"] = "clip"
    config["classifier_root"] = CLASSIFIER_DIR

    # ------------------------------------------------------------
    # 1️⃣ Standard Prompt Probe
    # ------------------------------------------------------------
    out_dir = f"{RESULTS_DIR}/{model}/{concept}/standardpromptprobe"
    ensure_dir(out_dir)
    print(f"➡️ StandardPromptProbe: {concept} ({model})")
    std_probe = StandardPromptProbe(
        pipeline_path=pipeline_path,
        erasing_type=model,
        concept=concept,
        num_images=NUM_IMAGES,
        device=DEVICE,
        config=config,
    )
    std_probe.output_dir = out_dir
    try:
        std_probe.run(num_images=NUM_IMAGES)
    except Exception as e:
        print(f"❌ StandardPromptProbe failed for {concept}-{model}: {e}")
    
    # ------------------------------------------------------------
    # 4️⃣ Standard Prompt Probe (with classifier guidance)
    # ------------------------------------------------------------
    out_dir = f"{RESULTS_DIR}/{model}/{concept}/standardpromptprobe_cls"
    ensure_dir(out_dir)
    print(f"➡️ StandardPromptProbe (with classifier): {concept} ({model})")
    std_cls_probe = StandardPromptProbe(
        pipeline_path=pipeline_path,
        erasing_type=model,
        concept=concept,
        num_images=NUM_IMAGES,
        device=DEVICE,
        config=config,
    )
    std_cls_probe.output_dir = out_dir
    try:
        std_cls_probe.run(num_images=NUM_IMAGES, debug=False, use_classifier_guidance=True)
    except Exception as e:
        print(f"❌ StandardPromptProbe (cls) failed for {concept}-{model}: {e}")

    # # ------------------------------------------------------------
    # # 2️⃣ Noise-Based Probe (no classifier guidance)
    # # ------------------------------------------------------------
    out_dir = f"{RESULTS_DIR}/{model}/{concept}/noisebasedprobe_nocls"
    ensure_dir(out_dir)
    print(f"➡️ NoiseBasedProbe (no classifier): {concept} ({model})")
    nb_probe = NoiseBasedProbe(
        pipeline_path=pipeline_path,
        erasing_type=model,
        concept=concept,
        num_images=NUM_IMAGES,
        device=DEVICE,
        config=config,
    )
    nb_probe.output_dir = out_dir
    try:
        nb_probe.run(num_images=NUM_IMAGES, debug=False, use_cls_guidance=False)
    except Exception as e:
        print(f"❌ NoiseBasedProbe (no cls) failed for {concept}-{model}: {e}")

    # ------------------------------------------------------------
    # 3️⃣ Noise-Based Probe (with classifier guidance)
    # ------------------------------------------------------------
    out_dir = f"{RESULTS_DIR}/{model}/{concept}/noisebasedprobe_cls"
    ensure_dir(out_dir)
    print(f"➡️ NoiseBasedProbe (with classifier): {concept} ({model})")
    nb_cls_probe = NoiseBasedProbe(
        pipeline_path=pipeline_path,
        erasing_type=model,
        concept=concept,
        num_images=NUM_IMAGES,
        device=DEVICE,
        config=config,
    )
    nb_cls_probe.output_dir = out_dir
    nb_cls_probe.run(num_images=NUM_IMAGES, debug=False, use_cls_guidance=True)


# ============================================================
# 🏁 Main
# ============================================================
if __name__ == "__main__":
    print(f"🧠 Running probes for {len(CONCEPTS)} concepts × {len(MODELS)} models")

    for concept in CONCEPTS:
        for model in MODELS:
            run_probes_for_concept_and_model(concept, model)

    print("\n✅ All probes finished successfully!")
