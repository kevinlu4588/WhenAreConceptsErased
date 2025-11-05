import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def get_best_image(images, prompt):
    """
    Given a list of images and a prompt, returns the image with the highest CLIP score.

    Args:
        images (list): List of PIL Image objects or paths to image files.
        prompt (str): The text prompt to compare against the images.

    Returns:
        PIL.Image: Image with the highest CLIP score.
    """
    # Load CLIP model and processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Ensure all images are PIL images
    pil_images = [Image.open(img) if isinstance(img, str) else img for img in images]

    # Encode inputs
    inputs = processor(text=[prompt], images=pil_images, return_tensors="pt", padding=True).to(device)

    # Calculate CLIP scores
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image

    # Get the best image index
    best_idx = logits_per_image.argmax().item()

    return pil_images[best_idx]


