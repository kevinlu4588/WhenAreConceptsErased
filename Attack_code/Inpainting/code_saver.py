import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionInpaintPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt


def load_esd_model_inpaint_pipeline(model_path, foundation_path):
    # Step 1: Load the pre-trained components from the original model
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(foundation_path, torch_dtype=torch.float16)
    pipeline.to("cuda")

    # Step 2: Load the `.pt` file that contains your custom model weights
    checkpoint = torch.load(model_path)
    
    # Step 3: Load the VAE with the updated weights if it's in the checkpoint
    vae = AutoencoderKL.from_pretrained(foundation_path, subfolder="vae")
    if "vae" in checkpoint:
        vae.load_state_dict(checkpoint["vae"])
    pipeline.vae = vae

    # Step 4: Load the UNet with the updated weights if it's in the checkpoint
    unet = UNet2DConditionModel.from_pretrained(foundation_path, subfolder="unet")
    if "unet" in checkpoint:
        unet.load_state_dict(checkpoint["unet"])
    pipeline.unet = unet

    # Step 5: Load the text encoder with the updated weights if it's in the checkpoint
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    if "text_encoder" in checkpoint:
        text_encoder.load_state_dict(checkpoint["text_encoder"])
    pipeline.text_encoder = text_encoder

    # Step 6: Load the scheduler if it has trainable parameters that were fine-tuned
    scheduler = DDIMScheduler.from_pretrained(foundation_path, subfolder="scheduler")
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    pipeline.scheduler = scheduler
    pipeline.to("cuda")
    pipeline.safety_checker = None

    return pipeline


def load_esd_model_to_image2image_pipeline(model_path):
    # Step 1: Load the pre-trained components from the original model
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipeline.to("cuda")

    # Step 2: Load the `.pt` file that contains your custom model weights
    checkpoint = torch.load(model_path)
    
    # Step 3: Load the VAE with the updated weights if it's in the checkpoint
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    if "vae" in checkpoint:
        vae.load_state_dict(checkpoint["vae"])
    pipeline.vae = vae

    # Step 4: Load the UNet with the updated weights if it's in the checkpoint
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    if "unet" in checkpoint:
        unet.load_state_dict(checkpoint["unet"])
    pipeline.unet = unet

    # Step 5: Load the text encoder with the updated weights if it's in the checkpoint
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    if "text_encoder" in checkpoint:
        text_encoder.load_state_dict(checkpoint["text_encoder"])
    pipeline.text_encoder = text_encoder

    # Step 6: Load the scheduler if it has trainable parameters that were fine-tuned
    scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    pipeline.scheduler = scheduler
    pipeline.to("cuda")
    pipeline.safety_checker = None

    return pipeline
def load_esd_model_to_image2image_pipeline(model_path):
    # Step 1: Load the pre-trained components from the original model
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipeline.to("cuda")

    # Step 2: Load the `.pt` file that contains your custom model weights
    checkpoint = torch.load(model_path)
    
    # Step 3: Load the VAE with the updated weights if it's in the checkpoint
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    if "vae" in checkpoint:
        vae.load_state_dict(checkpoint["vae"])
    pipeline.vae = vae

    # Step 4: Load the UNet with the updated weights if it's in the checkpoint
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    if "unet" in checkpoint:
        unet.load_state_dict(checkpoint["unet"])
    pipeline.unet = unet

    # Step 5: Load the text encoder with the updated weights if it's in the checkpoint
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    if "text_encoder" in checkpoint:
        text_encoder.load_state_dict(checkpoint["text_encoder"])
    pipeline.text_encoder = text_encoder

    # Step 6: Load the scheduler if it has trainable parameters that were fine-tuned
    scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    pipeline.scheduler = scheduler
    pipeline.to("cuda")
    pipeline.safety_checker = None

    return pipeline


def load_esd_model_sd_pipeline(model_path):
    # Step 1: Load the pre-trained components from the original model
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipeline.to("cuda")

    # Step 2: Load the `.pt` file that contains your custom model weights
    checkpoint = torch.load(model_path)
    
    # Step 3: Load the VAE with the updated weights if it's in the checkpoint
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    if "vae" in checkpoint:
        vae.load_state_dict(checkpoint["vae"])
    pipeline.vae = vae

    # Step 4: Load the UNet with the updated weights if it's in the checkpoint
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    if "unet" in checkpoint:
        unet.load_state_dict(checkpoint["unet"])
    pipeline.unet = unet

    # Step 5: Load the text encoder with the updated weights if it's in the checkpoint
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    if "text_encoder" in checkpoint:
        text_encoder.load_state_dict(checkpoint["text_encoder"])
    pipeline.text_encoder = text_encoder

    # Step 6: Load the scheduler if it has trainable parameters that were fine-tuned
    scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    pipeline.scheduler = scheduler
    pipeline.to("cuda")
    pipeline.safety_checker = None

    return pipeline

def test_noise_strength_for_reconstruction(pipeline, prompt, noisy_image, strength_values, guidance_scale):
    generated_images = []
    for strength in strength_values:
        print(f"Running pipeline with strength: {strength}")
        images = pipeline(prompt=prompt, image=noisy_image, strength=strength, guidance_scale=7.5).images
        generated_images.append((strength, images[0]))

    # Display the generated images
    fig, axes = plt.subplots(1, len(generated_images), figsize=(20, 5))
    for ax, (strength, img) in zip(axes, generated_images):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Strength: {strength:.2f}")

    plt.tight_layout()
    plt.show()