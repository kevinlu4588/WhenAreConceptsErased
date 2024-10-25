from PIL import Image
import requests
import os, glob
import pandas as pd
import numpy as np
import re
from transformers import CLIPProcessor, CLIPModel
import importlib
import torch
# Make changes to esd_diffusers.py file here
from eta_diffusion import FineTunedModel, StableDiffuser

class ExperimentImageSet:
    def __init__(self, stable_diffusion, eta_0_image, attack_images, interference_images = None, prompt: str = None, interference_prompt1 = None, interference_prompt2 = None, seed: int = None):
        self.stable_diffusion: np.ndarray = stable_diffusion
        self.eta_0_image: np.ndarray = eta_0_image
        self.attack_images: np.ndarray = attack_images
        self.interference_images: np.ndarray = interference_images
        self.target_prompt = prompt
        self.seed = seed
        self.interference_prompt1 = interference_prompt1
        self.interference_prompt2 = interference_prompt2

def erased_gen(target_csv_path, target_model_path, train_method, etas, num_prompts):
    # Load the CSV file
    target_data = pd.read_csv(target_csv_path)

    torch.cuda.empty_cache()
    variance_scales = [1.0]

    # Placeholder for the total images and experiment sets
    total_images = []
    total_experiment_sets = []
    ct = 0

    # Initialize the diffuser and finetuner models
    state_dict = torch.load(target_model_path)
    diffuser = StableDiffuser(scheduler='DDIM').to('cuda')
    finetuner = FineTunedModel(diffuser, train_method=train_method)
    finetuner.load_state_dict(state_dict)

    # Iterate through the target data
    for index, row in target_data.head(num_prompts).iterrows():
        prompt = row['prompt']
        seed = int(row['evaluation_seed'])  # Assuming 'evaluation_seed' contains the seed values
        
        # Base stable diffusion image
        stable_diffusion, images_steps, decoded_latents, latents, noise_preds, output_steps = diffuser(
            prompt, 
            n_steps=50, 
            generator=torch.manual_seed(seed), 
            eta=0.0, 
            variance_scale=0.0
        )
        total_images.append(stable_diffusion)

        # Finetuned no attack image
        with finetuner:
            finetuned_no_attack, images_steps, decoded_latents, latents, noise_preds, output_steps = diffuser(
                prompt, 
                n_steps=50, 
                generator=torch.manual_seed(seed), 
                eta=0.0, 
                variance_scale=0.0
            )
            total_images.append(finetuned_no_attack)

            attack_images = []
            for eta in etas:
                for variance_scale in variance_scales:
                    eta_image, images_steps, decoded_latents, latents, noise_preds, output_steps = diffuser(
                        prompt, 
                        n_steps=50, 
                        generator=torch.manual_seed(seed), 
                        eta=eta, 
                        variance_scale=variance_scale
                    )
                    attack_images.append(eta_image)
            total_images.extend(attack_images)

            # Construct an experiment set with the images
            experiment_set = ExperimentImageSet(
                stable_diffusion=stable_diffusion,
                eta_0_image=finetuned_no_attack,
                attack_images=np.array(attack_images),
                interference_images=None,  # Assuming no interference images in this case
                prompt=prompt,
                seed=seed,
                interference_prompt1 = None,
                interference_prompt2 = None
            )
            total_experiment_sets.append(experiment_set)

            ct += 1 + len(etas)
            print(f"diffusion-count {ct} for prompt: {prompt}")

    # Convert total images to a NumPy array
    total_images = np.array(total_images)

    # Assuming fixed_images is needed as an array of final images
    fixed_images = []
    for image in total_images:
        fixed_images.append(image[0][49])

    # Convert fixed_images to NumPy array
    fixed_images = np.array(fixed_images)

    print("Image grid shape:", fixed_images.shape)

    return fixed_images, total_experiment_sets

    from transformers import CLIPModel, CLIPProcessor
import torch
import numpy as np

from transformers import CLIPModel, CLIPProcessor
import torch
import numpy as np

def process_images(model, processor, prompt: str, images: list):
    """Processes images and returns CLIP scores."""
    images = np.array(images)
    images = images.squeeze()
    print(images.shape)
    images = [image[49] for image in images]
    inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return [clip_score.item() for clip_score in outputs.logits_per_image]

def calculate_experiment_scores(experiment, model, processor):
    """Calculates CLIP scores for each image set in the experiment."""
    targeted_images = [experiment.stable_diffusion, experiment.eta_0_image]
    targeted_images.extend(experiment.attack_images)
    clip_scores = process_images(model, processor, experiment.target_prompt, targeted_images)
    
    scores = {
        'SD': clip_scores[0],  # Stable diffusion image score
        'ETA_0': clip_scores[1],  # ETA_0 image score
        'ATTACK': max(clip_scores[2:]),  # Best attack image score
    }

    if experiment.interference_images:
        interference_images = experiment.interference_images
        interference_images = np.array(interference_images)
        interference_images = interference_images.squeeze()
        interference_images = [interference_image[49] for interference_image in interference_images]
        inputs = processor(text=[experiment.interference_prompt1], images=interference_images[0], return_tensors="pt", padding=True)
        outputs =  model(**inputs)
        interference_1 = outputs.logits_per_image.item()
        
        inputs = processor(text=[experiment.interference_prompt2], images=interference_images[1], return_tensors="pt", padding=True)
        outputs =  model(**inputs)
        interference_2 = outputs.logits_per_image.item()
        scores['INTERFERENCE1'] = interference_1  # Assuming first interference score is used
        scores['INTERFERENCE2'] = interference_2  # Assuming first interference score is used

    return scores

def get_clip_scores(experiment_sets: list['ExperimentImageSet']):
    """Processes a list of experiments and returns mean CLIP scores."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    total_clip_scores = {'SD': 0, 'ETA_0': 0, 'ATTACK': 0, 'INTERFERENCE1': 0, 'INTERFERENCE2' : 0}
    experiment_count = len(experiment_sets)

    for experiment in experiment_sets:
        experiment_scores = calculate_experiment_scores(experiment, model, processor)
        for key in total_clip_scores:
            total_clip_scores[key] += experiment_scores.get(key, 0)

    # Calculate mean scores
    mean_clip_scores = {key: score / experiment_count for key, score in total_clip_scores.items()}
    return mean_clip_scores

def get_simple_clip_scores(images_list, prompts):
    """
    Processes a list of images and prompts and returns the mean CLIP score for each prompt-image pair.

    Args:
        images_list (list of lists): List of image sets where each sublist contains images for one prompt.
        prompts (list of str): List of prompts corresponding to each image set.
    
    Returns:
        mean_clip_score (float): Mean CLIP score across all image-prompt pairs.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    total_score = 0.0
    total_images = 0
    full_clip_set = []
    for images, prompt in zip(images_list, prompts):
        inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True)  # Indentation fixed here
        outputs = model(**inputs)
        clip_scores = [clip_score.item() for clip_score in outputs.logits_per_image]
        full_clip_set.extend(np.round(clip_scores, 2))
        
    # Calculate mean score
    return full_clip_set

    import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def show_image_grid_with_scores(img_files, subtitles=None, clip_scores=None, num_rows=3, num_cols=4, fig_size=(15, 10)):
    """
    Displays a grid of images with subtitles and optional CLIP scores.

    Args:
        img_files (list of np.ndarray): List of images to display.
        subtitles (list of str): List of labels for the images.
        clip_scores (list of float): List of CLIP scores for the images.
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.
        fig_size (tuple): Size of the figure.
    """
    # Create a grid to display the images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    if not subtitles and clip_scores:
        subtitles = ['SD', 'Finetuned', 'ETA', "ETA", "ETA", 'eta']*(len(clip_scores)//6)
    else:
        subtitles = ['SD', 'Finetuned', 'ETA', "ETA", "ETA", 'eta']
    # Plot each image in the grid row-wise
    for i, ax in enumerate(axes.flatten()):
        img_index = i  # row-major order
        if img_index < len(img_files):
            img = img_files[img_index]
            ax.imshow(img)
            
            # Construct subtitle with label and optional CLIP score
            if subtitles and img_index < len(subtitles):
                subtitle = subtitles[img_index]
                if clip_scores and img_index < len(clip_scores):
                    subtitle += f" CLIP: {clip_scores[img_index]:.3f}"
                ax.set_title(subtitle, fontsize=14)
                
        ax.axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()

# Example usage
# erased_images = [image1, image2, image3, ...]  # Replace with actual images
# subtitles = ["Original", "Finetuner no attack", "Eta Attack", ...]  # Replace with actual subtitles
# clip_scores = [0.85, 0.92, 0.75, ...]  # Replace with actual CLIP scores
# show_image_grid_with_scores(erased_images, subtitles=subtitles, clip_scores=clip_scores, num_rows=2, num_cols=6)

def interference_gen(target_csv_path, interference_path1, interference_path2, target_model_path, train_method, etas, num_prompts):
    # Load the target and interference CSV files
    target_data = pd.read_csv(target_csv_path)
    interference_data1 = pd.read_csv(interference_path1)
    interference_data2 = pd.read_csv(interference_path2)

    torch.cuda.empty_cache()
    variance_scales = [1.0]

    # Placeholder for the total images and experiment sets
    total_images = []
    total_experiment_sets = []
    ct = 0

    # Initialize the diffuser and finetuner models
    state_dict = torch.load(target_model_path)
    diffuser = StableDiffuser(scheduler='DDIM').to('cuda')
    finetuner = FineTunedModel(diffuser, train_method=train_method)
    finetuner.load_state_dict(state_dict)

    # Iterate through the target data along with interference data from the other two CSVs
    for (index, row), (index1, row1), (index2, row2) in zip(
            target_data.head(num_prompts).iterrows(),
            interference_data1.head(num_prompts).iterrows(),
            interference_data2.head(num_prompts).iterrows()
        ):

        prompt = row['prompt']
        seed = int(row['evaluation_seed'])  # Assuming 'evaluation_seed' contains the seed values
        
        interference_prompt1 = row1['prompt']
        interference_seed1 = int(row1['evaluation_seed'])
        
        interference_prompt2 = row2['prompt']
        interference_seed2 = int(row2['evaluation_seed'])
        
        # Base stable diffusion image
        stable_diffusion, images_steps, decoded_latents, latents, noise_preds, output_steps = diffuser(
            prompt, 
            n_steps=50, 
            generator=torch.manual_seed(seed), 
            eta=0.0, 
            variance_scale=0.0
        )
        total_images.append(stable_diffusion)

        # Finetuned no attack image
        with finetuner:
            finetuned_no_attack, images_steps, decoded_latents, latents, noise_preds, output_steps = diffuser(
                prompt, 
                n_steps=50, 
                generator=torch.manual_seed(seed), 
                eta=0.0, 
                variance_scale=0.0
            )
            total_images.append(finetuned_no_attack)

            attack_images = []
            for eta in etas:
                for variance_scale in variance_scales:
                    eta_image, images_steps, decoded_latents, latents, noise_preds, output_steps = diffuser(
                        prompt, 
                        n_steps=50, 
                        generator=torch.manual_seed(seed), 
                        eta=eta, 
                        variance_scale=variance_scale
                    )
                    attack_images.append(eta_image)
            total_images.extend(attack_images)

                        # Generate interference images using prompts and seeds from the interference CSVs
            interference_image1, images_steps, decoded_latents, latents, noise_preds, output_steps = diffuser(
                interference_prompt1,
                n_steps=50,
                generator=torch.manual_seed(interference_seed1),
                eta=0.0,  # No attack (eta = 0)
                variance_scale=0.0  # No attack variance
            )
            total_images.append(interference_image1)

            interference_image2, images_steps, decoded_latents, latents, noise_preds, output_steps = diffuser(
                interference_prompt2,
                n_steps=50,
                generator=torch.manual_seed(interference_seed2),
                eta=0.0,  # No attack (eta = 0)
                variance_scale=0.0  # No attack variance
            )
            total_images.append(interference_image2)

            # Construct an experiment set with the images, including the interference images
            experiment_set = ExperimentImageSet(
                stable_diffusion=stable_diffusion,
                eta_0_image=finetuned_no_attack,
                attack_images=np.array(attack_images),
                interference_images=[interference_image1, interference_image2],  # Adding interference images
                prompt=prompt,
                seed=seed,
                interference_prompt1=interference_prompt1,
                interference_prompt2=interference_prompt2
            )
            total_experiment_sets.append(experiment_set)

            ct += 1 + len(etas)
            print(f"diffusion-count {ct} for prompt: {prompt}")

    # Convert total images to a NumPy array
    total_images = np.array(total_images)

    # Assuming fixed_images is needed as an array of final images
    fixed_images = []
    for image in total_images:
        fixed_images.append(image[0][49])

    # Convert fixed_images to NumPy array
    fixed_images = np.array(fixed_images)

    print("Image grid shape:", fixed_images.shape)

    return fixed_images, total_experiment_sets
