from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
    
import PIL
import requests
import torch
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from diffusers import StableDiffusionInpaintPipeline

def inpaint(init_image, mask_coord_list, model):
        
    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    # Load and resize the initial and mask images
    init_image = PIL.Image.open("starry night.webp").resize((512, 512))
    mask_image = PIL.Image.open("mask.png").resize((512, 512))

    # Load the inpainting model
    model_id_or_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        VAN_GOGH_MODEL, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # Define the prompt and generate the inpainted image
    prompt = "starry night"
    final_image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

    # Mask area coordinates (from your earlier mask creation step)
    mask_coords = [300, 50, 500, 250]  # x1, y1, x2, y2

    # Plot the images with the outlined box
    plt.figure(figsize=(15, 5))

    # Initial image
    plt.subplot(1, 3, 1)
    plt.imshow(init_image)
    plt.gca().add_patch(Rectangle(
        (mask_coords[0], mask_coords[1]),  # Bottom-left corner
        mask_coords[2] - mask_coords[0],  # Width
        mask_coords[3] - mask_coords[1],  # Height
        linewidth=2, edgecolor='red', facecolor='none'  # Outline style
    ))
    plt.axis("off")
    plt.title("Initial Image with Mask Outline")

    # Mask image
    plt.subplot(1, 3, 2)
    plt.imshow(mask_image, cmap='gray')
    plt.axis("off")
    plt.title("Mask Image")

    # Final image
    plt.subplot(1, 3, 3)
    plt.imshow(final_image)
    plt.gca().add_patch(Rectangle(
        (mask_coords[0], mask_coords[1]),
        mask_coords[2] - mask_coords[0],
        mask_coords[3] - mask_coords[1],
        linewidth=2, edgecolor='red', facecolor='none'
    ))
    plt.axis("off")
    plt.title("Final Image with Mask Outline")

    plt.tight_layout()
    plt.show()

