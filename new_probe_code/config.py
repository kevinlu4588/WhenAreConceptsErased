class ProbeConfig:
    def __init__(self):
        # match your provided values
        self.noise_based_probe_values = [1.0, 1.17, 1.34, 1.51, 1.68, 1.85]
        self.variance_scales = [1.00, 1.02, 1.03, 1.04]
        self.score_type = "CLIP"  # or "classification"
        self.base_images_path = "/share/u/kevin/DiffusionConceptErasure/new_probe_code/results/base_model/"
        self.mask_size = (256, 256)  # width, height in pixels