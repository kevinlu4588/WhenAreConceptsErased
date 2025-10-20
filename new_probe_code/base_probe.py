import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from dataclasses import dataclass

# Optional model imports
from transformers import CLIPModel, CLIPProcessor
import torchvision.models as models
import torchvision.transforms as T


@dataclass
class ProbeArgs:
    pipeline_path: str
    erasing_type: str
    concept: str
    num_images: int = 10
    device: str = "cuda"
    score_type: str = "CLIP"


class BaseProbe:
    def __init__(self, pipeline_path, erasing_type, concept, num_images=10, device="cuda", config=None, probe_name=None):
        self.pipeline_path = pipeline_path
        self.erasing_type = erasing_type
        self.concept = concept
        self.num_images = num_images
        self.device = device
        self.config = config

        # Derive probe name automatically from subclass unless given
        self.probe_name = probe_name or self.__class__.__name__.lower()


        # --- Load the diffusion pipeline ---
        print(f"🔹 Loading pipeline from {pipeline_path}")
        self.pipe = StableDiffusionPipeline.from_pretrained(pipeline_path, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.safety_checker = None  # disable NSFW filter

        # --- Prompt and output paths ---
        self.prompt_csv = f"/share/u/kevin/DiffusionConceptErasure/final_data/prompts/{concept}.csv"
        self.output_dir = os.path.join("results", erasing_type, concept, self.probe_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Scoring setup (CLIP or classification) ---
        score_type = getattr(self.config, "score_type", "CLIP").lower() if self.config else "clip"

        if score_type == "clip":
            print("🔸 Loading CLIP model for scoring...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        elif score_type == "classification":
            print("🔸 Loading ResNet-50 classifier for scoring...")
            self.classifier = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            print(f"⚠️ Unknown score type '{score_type}', no scoring model loaded.")


    # ----------------------------------------------------------
    @classmethod
    def from_args(cls, args):
        """
        Factory method to create a probe instance from argparse or Typer args.
        Also constructs a ProbeConfig automatically if not supplied.
        """
        config = getattr(args, "config", None)
        if config is None:
            from config import ProbeConfig
            config = ProbeConfig()
            if hasattr(args, "score_type"):
                config.score_type = args.score_type

        return cls(
            pipeline_path=args.pipeline_path,
            erasing_type=args.erasing_type,
            concept=args.concept,
            num_images=getattr(args, "num_images", 10),
            device=getattr(args, "device", "cuda"),
            config=config,
        )

    # ----------------------------------------------------------
    def _load_prompts(self, num_images=None):
        """Loads (prompt, seed) pairs and truncates to num_images if specified."""
        if not os.path.exists(self.prompt_csv):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_csv}")
        df = pd.read_csv(self.prompt_csv)
        n = num_images or self.num_images
        df = df.sample(min(n, len(df)), random_state=42)
        return list(df.itertuples(index=False, name=None))  # (prompt, seed)

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    def save_image(self, image, name, subfolder=None):
        """
        Save an image to the probe's results directory.

        Args:
            image (PIL.Image): Image to save.
            name (str): Filename.
            subfolder (str, optional): Save under a subdirectory (e.g. 'debug').
        """
        base_dir = self.output_dir
        if subfolder:
            base_dir = os.path.join(base_dir, subfolder)
        os.makedirs(base_dir, exist_ok=True)

        out_path = os.path.join(base_dir, name)
        image.save(out_path)
        print("image saved to", out_path)
        return out_path

    # ----------------------------------------------------------
    def run(self, num_images=None):
        """Abstract run method (to be overridden)."""
        raise NotImplementedError("Subclasses must implement run(num_images)")

    # ----------------------------------------------------------
    def _clip_score(self, image: Image.Image, prompt: str):
        """Compute CLIP similarity between prompt and image."""
        if not hasattr(self, "clip_model") or not hasattr(self, "clip_processor"):
            raise RuntimeError("CLIP model not initialized. Set config.score_type='CLIP'.")
        inputs = self.clip_processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.clip_model(**inputs).logits_per_image
        return logits.item()

    def _classifier_score(self, image: Image.Image):
        """Compute simple ResNet-50 classification confidence."""
        if not hasattr(self, "classifier") or not hasattr(self, "transform"):
            raise RuntimeError("Classifier not initialized. Set config.score_type='classification'.")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(tensor)
        return logits.max().item()  # simple heuristic
