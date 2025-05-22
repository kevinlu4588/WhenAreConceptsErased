
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, AutoPipelineForInpainting
from PIL import Image, ImageDraw
import numpy as np

def load_model_inpaint_pipeline(model_folder_path, foundation_path):
        """
        Load an enhanced Stable Diffusion inpainting pipeline from a folder containing model components.

        Args:
            model_folder_path (str): Path to the folder containing the fine-tuned model components.
            foundation_path (str): Path to the foundation model for pre-trained components.

        Returns:
            StableDiffusionInpaintPipeline: The inpainting pipeline with updated weights.
        """

        # Step 1: Load the fine-tuned pipeline directly
        fine_tuned_pipeline = StableDiffusionPipeline.from_pretrained(
            model_folder_path,
            torch_dtype=torch.float16
        )
        fine_tuned_pipeline.to("cuda")

        # Access the fine-tuned UNet and text encoder directly from the pipeline
        fine_tuned_unet = fine_tuned_pipeline.unet
        fine_tuned_text_encoder = fine_tuned_pipeline.text_encoder

        # Optional: You can combine this with components from the foundation pipeline
        foundation_pipeline = AutoPipelineForInpainting.from_pretrained(
            foundation_path,
            torch_dtype=torch.float16
        )
        foundation_pipeline.to("cuda")

        # Example: Replace the UNet in the foundation pipeline with the fine-tuned UNet
        foundation_pipeline.unet = fine_tuned_unet
        foundation_pipeline.text_encoder = fine_tuned_text_encoder

        # Disable safety checker if required
        foundation_pipeline.safety_checker = None

        return foundation_pipeline
def draw_mask_outline(image, mask, outline_color=(255, 0, 0), fill_color=(255, 255, 255)):
    """Draws a red outline around the mask and optionally fills it with white."""
    mask_array = np.array(mask)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    
    white_coords = np.where(mask_array == 255)
    if len(white_coords[0]) == 0:
        return image  # No mask detected
    
    y_min, y_max = white_coords[0].min(), white_coords[0].max()
    x_min, x_max = white_coords[1].min(), white_coords[1].max()
    
    # Fill the masked region with white
    draw.rectangle([x_min, y_min, x_max, y_max], fill=fill_color, outline=outline_color, width=3)
    return image

def generate_inpainting_images_with_mask_overlay(mask_path, base_image_path, pipeline, prompt, seeds):
    base_image = Image.open(base_image_path).convert("RGB")
    image_size = 512
    base_image = base_image.resize((image_size, image_size), Image.LANCZOS)
    mask = Image.open(mask_path).convert("L")

    base_with_mask = draw_mask_outline(base_image, mask)

    generated_images = []

    for seed in seeds:
        generator = torch.manual_seed(seed)
        
        inpainted_image = pipeline(prompt=prompt, image=base_image, mask_image=mask, generator=generator).images[0]
        inpainted_with_outline = draw_mask_outline(inpainted_image, mask, outline_color=(255, 0, 0), fill_color=None)
        
        generated_images.append(inpainted_with_outline)

    return generated_images, base_with_mask

def display_images(images, base_with_mask):
    fig, axes = plt.subplots(2, len(images), figsize=(15, 6))
    for i, inpainted_with_outline in enumerate(images):
        axes[0, i].imshow(base_with_mask)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Seed {i}")

        axes[1, i].imshow(inpainted_with_outline)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

def calculate_clip_score(images, prompt, mask_path, clip_model, clip_processor):
    mask = Image.open(mask_path).convert("L")  # Load mask as grayscale
    mask_array = np.array(mask)
    white_coords = np.where(mask_array == 255)
    y_min, y_max = white_coords[0].min(), white_coords[0].max()
    x_min, x_max = white_coords[1].min(), white_coords[1].max()

    cropped_images = []
    for image in images:
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_images.append(cropped_image)
    inputs = clip_processor(text=[prompt], images=cropped_images, return_tensors="pt", padding=True).to("cuda")
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    best_image_idx = logits_per_image.argmax().item()
    return images[best_image_idx], logits_per_image.max()
