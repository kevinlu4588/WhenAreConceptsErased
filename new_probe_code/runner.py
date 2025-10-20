import sys
import torch

# Add module path

from noise_based_probe import NoiseBasedProbe
from standard_prompt_probe import StandardPromptProbe
from inpainting_probe import InpaintProbe
from textual_inversion_probe import TextualInversionProbe
from config import ProbeConfig  # make sure this defines noise_based_probe_values & variance_scales
#/share/u/kevin/ErasingDiffusionModels/final_models
#/share/u/kevin/ErasingDiffusionModels/final_models/esda_church
if __name__ == "__main__":
    config = ProbeConfig()
    config.score_type = "CLIP"

    probe = TextualInversionProbe(
        pipeline_path="/share/u/kevin/ErasingDiffusionModels/final_models/esdx_van_gogh",
        # pipeline_path="CompVis/stable-diffusion-v1-4",
        erasing_type="esdx",
        concept="van_gogh",
        num_images=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        config=config,
    )

    print("🚀 Running NoiseProbe (debug mode ON)...")
    probe.run(num_images=1)
