import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

# Load the Stable Diffusion pipeline
model_name = "CompVis/stable-diffusion-v1-4"  # Replace with a model fine-tuned for dogs if available
pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")  # Use GPU for faster generation

# Output directory for images
output_dir = Path("springer_spaniel_images")
output_dir.mkdir(exist_ok=True)

# Text prompt for English Springer Spaniels
prompt = "a realistic photo of an English Springer Spaniel"

# Number of images to generate
num_images = 40

# Generate images
for i in range(num_images):
    print(f"Generating image {i+1}/{num_images}...")
    image = pipeline(prompt).images[0]  # Generate an image from the prompt
    output_path = output_dir / f"springer_spaniel_{i+1}.png"
    image.save(output_path)  # Save the image
    print(f"Saved image to {output_path}")

print(f"Successfully generated {num_images} images of English Springer Spaniels in {output_dir}.")
