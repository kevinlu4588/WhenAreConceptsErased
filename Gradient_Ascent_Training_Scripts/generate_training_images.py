from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import torch
import os
import json
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from I2P dataset")

    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=0)
    parser.add_argument("--prompt", type=str, help="Prompt for image generation")
    parser.add_argument("--mode", type=str, help="Mode for image generation", choices=["train","test"], default="train")
    parser.add_argument("--num_train_images", type=int, help="Number of images to generate for training", default=1000)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if(args.mode == "test"):

        metadata = []

        os.makedirs(args.output_dir, exist_ok=True)

        pipe = StableDiffusionPipeline.from_pretrained(args.model_path, safety_checker=None, torch_dtype=torch.float16).to(device)            
        
        for i in range(args.num_train_images):
            while True:
                output = pipe(prompt=args.prompt)
                image = output.images[0]
                nsfw = output.nsfw_content_detected
                #check if nsfw is a list
                if isinstance(nsfw, list):
                    nsfw = nsfw[0]
                
                if not nsfw:
                    break
            image.save(os.path.join(args.output_dir, f"{args.prompt}_{i}.png"))
        

    elif(args.mode == "train"):
        metadata = []

        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

        pipe = StableDiffusionPipeline.from_pretrained(args.model_path, safety_checker=None, torch_dtype=torch.float16).to(device)            
        
        for i in range(args.num_train_images):
            while True:
                output = pipe(prompt=args.prompt)
                image = output.images[0]
                nsfw = output.nsfw_content_detected
                #check if nsfw is a list
                if isinstance(nsfw, list):
                    nsfw = nsfw[0]
                
                if not nsfw:
                    break
            image.save(os.path.join(args.output_dir, "train", f"{args.prompt}_{i}.png"))
            
            metadata.append({"file_name": f"train/{args.prompt}_{i}.png", "text": args.prompt})

        with open(os.path.join(args.output_dir, 'metadata.jsonl'), 'w') as f:
            for m in metadata:
                f.write(json.dumps(m) + "\n")


    