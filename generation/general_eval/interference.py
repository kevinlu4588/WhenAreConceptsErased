import os
import argparse
import pandas as pd
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

def main(models_path, methods, concepts, prompts_path, results_path, device, num_images):

    os.makedirs(results_path, exist_ok=True)

    for method in methods:
        for concept in concepts:
            model_path = os.path.join(models_path, f"{method}_{concept}")
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}, skipping.")
                continue

            print(f"Loading model: {method} for concept: {concept}")
            pipeline = StableDiffusionPipeline.from_pretrained(model_path).to(device)
            pipeline.safety_checker = None

            save_folder = os.path.join(results_path, method, concept, "interference")
            os.makedirs(save_folder, exist_ok=True)

            for other_concept in concepts:
                if other_concept == concept:
                    continue

                prompt_file = os.path.join(prompts_path, f"{other_concept}.csv")
                if not os.path.exists(prompt_file):
                    print(f"Prompts file not found: {prompt_file}, skipping.")
                    continue

                df = pd.read_csv(prompt_file)
                prompts = df['prompt'].tolist()[:num_images]

                for idx, prompt in enumerate(prompts):
                    print(f"Generating image for prompt: '{prompt}' (other concept: {other_concept})")
                    image = pipeline(prompt).images[0]
                    image_name = f"{other_concept}_prompt_{idx + 1}.png"
                    image.save(os.path.join(save_folder, image_name))

    print("Image generation and organization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
    parser.add_argument("--models_path", type=str, required=True, help="Path to the models directory.")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to the prompts directory.")
    parser.add_argument("--results_path", type=str, required=True, help="Path to save the generated images.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--concepts", nargs="+")
    parser.add_argument("--num_images", type=int, default=10)
    
    args = parser.parse_args()
    print(args.methods)
    main(args.models_path, args.methods, args.concepts, args.prompts_path, args.results_path, args.device, args.num_images)
