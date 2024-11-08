import pandas as pd
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler
class ExperimentImageSet:
    def __init__(self, stable_diffusion, eta_0_image, attack_images, original_interference_images = None, interference_images = None, prompt: str = None, interference_prompt1 = None, interference_prompt2 = None, seed: int = None):
        self.stable_diffusion: np.ndarray = stable_diffusion
        self.eta_0_image: np.ndarray = eta_0_image
        self.attack_images: np.ndarray = attack_images
        self.original_interference_images: np.ndarray=original_interference_images
        self.interference_images: np.ndarray = interference_images
        self.target_prompt = prompt
        self.seed = seed
        self.interference_prompt1 = interference_prompt1
        self.interference_prompt2 = interference_prompt2
        self.clip_scores = None

def pipeline_erased_gen(target_csv_path, target_prompt, target_model_path, etas, num_prompts):
    # Load the target and interference CSV files
    target_data = pd.read_csv(target_csv_path)

    torch.cuda.empty_cache()
    variance_scales = [1.0]  # Adjust variance scales as needed

    # Placeholder for the total images and experiment sets
    total_images = []
    total_experiment_sets = []
    ct = 0
    original_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    original_pipeline.scheduler = DDIMScheduler.from_config(original_pipeline.scheduler.config)
    original_pipeline.safety_checker = None  # Disable the NSFW checker
    original_pipeline = original_pipeline.to("cuda")
    pipeline = StableDiffusionPipeline.from_pretrained(target_model_path)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.safety_checker = None  # Disable the NSFW checker
    pipeline = pipeline.to("cuda")

    # Iterate through the target data along with interference data from the other two CSVs
    for index, row in target_data.head(num_prompts).iterrows():

        prompt = row['prompt']
        seed = int(row['evaluation_seed'])
        
        # Base stable diffusion image
        generator = torch.manual_seed(seed)

        
        stable_diffusion = original_pipeline(prompt, num_inference_steps=50, generator=generator, eta=0.0).images[0]
        stable_diffusion = np.array(stable_diffusion)  # Convert to np.ndarray
        total_images.append(stable_diffusion)

        # No attack image (eta=0, variance_scale=0)
        finetuned_no_attack = pipeline(prompt, num_inference_steps=50, generator=generator, eta=0.0).images[0]
        finetuned_no_attack = np.array(finetuned_no_attack)  # Convert to np.ndarray
        total_images.append(finetuned_no_attack)

        # Attack images with varying eta and variance scales
        attack_images = []
        for eta in etas:
            for variance_scale in variance_scales:
                attacked_image = pipeline(
                    prompt,
                    num_inference_steps=50,
                    generator=generator,
                    eta=eta,
                    variance_scale=variance_scale  # Assuming variance_scale is supported directly
                ).images[0]
                attacked_image = np.array(attacked_image)  # Convert to np.ndarray
                attack_images.append(attacked_image)
        attack_images = np.array(attack_images)  # Convert list to np.ndarray
        total_images.extend(attack_images)

        # Construct an experiment set with the images, including the interference images
        experiment_set = ExperimentImageSet(
            stable_diffusion=stable_diffusion,
            eta_0_image=finetuned_no_attack,
            attack_images=attack_images,
            original_interference_images= None,
            interference_images=None, 
            prompt=target_prompt,
            seed=seed,
            interference_prompt1=None,
            interference_prompt2=None
        )
        total_experiment_sets.append(experiment_set)

        ct += 1 + len(etas) * len(variance_scales)
        print(f"diffusion-count {ct} for prompt: {prompt}")

    # Convert total images to a NumPy array
    total_images = np.array(total_images)

    # Assuming fixed_images is needed as an array of final images
    fixed_images = [image for image in total_images]
    fixed_images = np.array(fixed_images)

    print("Image grid shape:", fixed_images.shape)

    return fixed_images, total_experiment_sets



def interference_gen(target_csv_path, interference_path1, interference_path2, target_model_path, etas, num_prompts):
    # Load the target and interference CSV files
    target_data = pd.read_csv(target_csv_path)
    interference_data1 = pd.read_csv(interference_path1)
    interference_data2 = pd.read_csv(interference_path2)

    torch.cuda.empty_cache()
    variance_scales = [1.0]  # Adjust variance scales as needed

    # Placeholder for the total images and experiment sets
    total_images = []
    total_experiment_sets = []
    ct = 0
    original_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    original_pipeline.scheduler = DDIMScheduler.from_config(original_pipeline.scheduler.config)
    original_pipeline.safety_checker = None  # Disable the NSFW checker
    original_pipeline = original_pipeline.to("cuda")
    pipeline = StableDiffusionPipeline.from_pretrained(target_model_path)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.safety_checker = None  # Disable the NSFW checker
    pipeline = pipeline.to("cuda")

    # Iterate through the target data along with interference data from the other two CSVs
    for (index, row), (index1, row1), (index2, row2) in zip(
            target_data.head(num_prompts).iterrows(),
            interference_data1.head(num_prompts).iterrows(),
            interference_data2.head(num_prompts).iterrows()
        ):

        prompt = row['prompt']
        seed = int(row['evaluation_seed'])
        
        interference_prompt1 = row1['prompt']
        interference_seed1 = int(row1['evaluation_seed'])
        
        interference_prompt2 = row2['prompt']
        interference_seed2 = int(row2['evaluation_seed'])
        
        # Base stable diffusion image
        generator = torch.manual_seed(seed)

        
        stable_diffusion = original_pipeline(prompt, num_inference_steps=50, generator=generator, eta=0.0).images[0]
        stable_diffusion = np.array(stable_diffusion)  # Convert to np.ndarray
        total_images.append(stable_diffusion)

        # No attack image (eta=0, variance_scale=0)
        finetuned_no_attack = pipeline(prompt, num_inference_steps=50, generator=generator, eta=0.0).images[0]
        finetuned_no_attack = np.array(finetuned_no_attack)  # Convert to np.ndarray
        total_images.append(finetuned_no_attack)

        # Attack images with varying eta and variance scales
        attack_images = []
        for eta in etas:
            for variance_scale in variance_scales:
                attacked_image = pipeline(
                    prompt,
                    num_inference_steps=50,
                    generator=generator,
                    eta=eta,
                    variance_scale=variance_scale  # Assuming variance_scale is supported directly
                ).images[0]
                attacked_image = np.array(attacked_image)  # Convert to np.ndarray
                attack_images.append(attacked_image)
        attack_images = np.array(attack_images)  # Convert list to np.ndarray
        total_images.extend(attack_images)

        
        # Generate interference images using prompts and seeds from the interference CSVs
        generator1 = torch.manual_seed(interference_seed1)
        original_interference_image1 = pipeline(
            interference_prompt1,
            num_inference_steps=50,
            generator=generator1,
            eta=0.0,  # No attack
            variance_scale=0.0  # No variance
        ).images[0]

        original_interference_image1 = np.array(original_interference_image1)
        total_images.append(original_interference_image1)

        interference_image1 = pipeline(
            interference_prompt1,
            num_inference_steps=50,
            generator=generator1,
            eta=0.0,  # No attack
            variance_scale=0.0  # No variance
        ).images[0]
        interference_image1 = np.array(interference_image1)  # Convert to np.ndarray
        total_images.append(interference_image1)

        generator2 = torch.manual_seed(interference_seed2)
        original_interference_image2 = pipeline(
            interference_prompt2,
            num_inference_steps=50,
            generator=generator2,
            eta=0.0,  # No attack
            variance_scale=0.0  # No variance
        ).images[0]
        original_interference_image2 = np.array(original_interference_image2)  # Convert to np.ndarray
        total_images.append(original_interference_image2)

        interference_image2 = pipeline(
            interference_prompt2,
            num_inference_steps=50,
            generator=generator2,
            eta=0.0,  # No attack
            variance_scale=0.0  # No variance
        ).images[0]
        interference_image2 = np.array(interference_image2)  # Convert to np.ndarray
        total_images.append(interference_image2)

        # Construct an experiment set with the images, including the interference images
        experiment_set = ExperimentImageSet(
            stable_diffusion=stable_diffusion,
            eta_0_image=finetuned_no_attack,
            attack_images=attack_images,
            original_interference_images=[original_interference_image1, original_interference_image2],
            interference_images=[interference_image1, interference_image2],  # Adding interference images
            prompt="art in the style of Van Gogh",
            seed=seed,
            interference_prompt1="art in the style of Picasso",
            interference_prompt2="art in the style of Andy Warhol"
        )
        total_experiment_sets.append(experiment_set)

        ct += 1 + len(etas) * len(variance_scales)
        print(f"diffusion-count {ct} for prompt: {prompt}")

    # Convert total images to a NumPy array
    total_images = np.array(total_images)

    # Assuming fixed_images is needed as an array of final images
    fixed_images = [image for image in total_images]
    fixed_images = np.array(fixed_images)

    print("Image grid shape:", fixed_images.shape)

    return fixed_images, total_experiment_sets
