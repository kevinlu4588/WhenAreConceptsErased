import os
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from dataclasses import dataclass

# Optional model imports
from transformers import CLIPModel, CLIPProcessor
from torchvision.models import resnet50, ResNet50_Weights


@dataclass
class ProbeArgs:
    pipeline_path: str
    erasing_type: str
    concept: str
    num_images: int = 10
    device: str = "cuda"
    score_type: str = "CLIP"


from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch, os
from transformers import CLIPModel, CLIPProcessor
from torchvision.models import resnet50, ResNet50_Weights

class BaseProbe:
    def __init__(self, pipeline_path=None, unet_path=None, erasing_type=None, concept=None,
                 num_images=10, device="cuda", config=None, probe_name=None):
        self.pipeline_path = pipeline_path
        self.unet_path = unet_path
        self.erasing_type = erasing_type
        self.concept = concept
        self.num_images = num_images
        self.device = device
        self.config = config
        self.probe_name = probe_name or self.__class__.__name__.lower()

        # ============================================================
        # 🚀 Load base pipeline
        # ============================================================
        if self.pipeline_path:
            print(f"📦 Loading pipeline from {self.pipeline_path}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.pipeline_path, torch_dtype=torch.float16
            ).to(device)
        elif self.unet_path:
            print(f"🧩 Loading UNet from {self.unet_path} into SD1.4 base")
            base_model = "CompVis/stable-diffusion-v1-4"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                base_model, torch_dtype=torch.float16
            ).to(device)

            unet = UNet2DConditionModel.from_pretrained(
                self.unet_path, torch_dtype=torch.float16
            ).to(device)
            self.pipe.unet = unet
        else:
            raise ValueError("Either `pipeline_path` or `unet_path` must be provided.")

        self.pipe.safety_checker = None

        # ============================================================
        # 🧮 Setup scoring
        # ============================================================
        score_type = getattr(self.config, "score_type", "clip").lower() if self.config else "clip"

        if score_type == "clip":
            print("🔸 Loading CLIP model for scoring...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif score_type == "classification":
            print("🔸 Loading ResNet-50 classifier for scoring...")
            weights = ResNet50_Weights.DEFAULT
            self.classifier = resnet50(weights=weights).to(device).eval()
            self.classifier_weights = weights
            self.transform = weights.transforms()
        else:
            print(f"⚠️ Unknown score type '{score_type}', no scoring model loaded.")

        # ============================================================
        # 📁 Setup directories
        # ============================================================
        self.output_dir = os.path.join("results", erasing_type, concept, self.probe_name)
        os.makedirs(self.output_dir, exist_ok=True)


    # ------------------------------------------------------------------
    # 🧠 Initialization helpers
    # ------------------------------------------------------------------
    def _init_clip(self):
        """Initialize CLIP model + processor for image-text scoring."""
        print("🔸 Loading CLIP model for scoring...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

    def _init_classifier(self):
        """Initialize pretrained ResNet-50 classifier for ImageNet objects."""
        print("🔸 Loading ResNet-50 classifier for scoring...")
        self.classifier_weights = ResNet50_Weights.DEFAULT
        self.classifier = resnet50(weights=self.classifier_weights).to(self.device)
        self.classifier.eval()
        self.transform = self.classifier_weights.transforms()

    # ------------------------------------------------------------------
    @classmethod
    def from_args(cls, args):
        """Factory to create a probe from CLI args."""
        config = getattr(args, "config", None)
        if config is None:
            config = {"score_type": getattr(args, "score_type", "clip")}
        return cls(
            pipeline_path=args.pipeline_path,
            erasing_type=args.erasing_type,
            concept=args.concept,
            num_images=getattr(args, "num_images", 10),
            device=getattr(args, "device", "cuda"),
            config=config,
        )

    # ------------------------------------------------------------------
    def _load_prompts(self, num_images=None):
        """Loads (prompt, seed) pairs and truncates to num_images if specified."""
        if not os.path.exists(self.prompt_csv):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_csv}")
        df = pd.read_csv(self.prompt_csv)
        n = num_images or self.num_images
        df = df.sample(min(n, len(df)), random_state=42)
        return list(df.itertuples(index=False, name=None))  # (prompt, seed)

    # ------------------------------------------------------------------
    def save_image(self, image, name, subfolder=None):
        """Save an image to the probe's results directory."""
        base_dir = self.output_dir if not subfolder else os.path.join(self.output_dir, subfolder)
        os.makedirs(base_dir, exist_ok=True)
        out_path = os.path.join(base_dir, name)
        image.save(out_path)
        print(f"🖼️ Image saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    def run(self, num_images=None):
        raise NotImplementedError("Subclasses must implement run(num_images)")

    # ------------------------------------------------------------------
    def _clip_score(self, image: Image.Image, prompt: str):
        """Compute CLIP similarity between prompt and image."""
        if not hasattr(self, "clip_model"):
            raise RuntimeError("CLIP model not initialized. Call _init_clip().")

        inputs = self.clip_processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.clip_model(**inputs).logits_per_image
        return logits.item()

    # ------------------------------------------------------------------
    def _classifier_score(self, image: Image.Image):
        """
        Compute ResNet-50 confidence that the image belongs to this probe's concept.
        Returns the softmax probability for the concept class.
        """
        if not hasattr(self, "classifier"):
            raise RuntimeError("Classifier not initialized. Call _init_classifier().")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = logits.softmax(dim=1)

        # Match concept to ImageNet categories
        categories = [c.lower() for c in self.classifier_weights.meta["categories"]]
        concept_clean = self.concept.replace("_", " ").lower()
        if self.concept == "english_springer_spaniel":
            concept_clean = "english springer"
        match_idx = next((i for i, c in enumerate(categories) if concept_clean in c), None)
        if match_idx is None:
            return 0.0  # concept not found in ImageNet

        return probs[0, match_idx].item()
