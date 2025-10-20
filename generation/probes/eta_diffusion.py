from PIL import Image
from matplotlib import pyplot as plt
import textwrap
import argparse
import torch
import copy
import os
import re
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from eta_diffusers.src.diffusers.schedulers.eta_ddim_scheduler import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

def show_image_grid(img_files, num_rows=3, num_cols=4, fig_size=(15, 10)):
    # Create a grid to display the images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)

    # Plot each image in the grid row-wise
    for i, ax in enumerate(axes.flatten()):
        img_index = i  # row-major order
        if img_index < len(img_files):
            img = img_files[img_index]
            ax.imshow(img)
        ax.axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()

# Example usage
# img_files = [image1, image2, image3, ...]  # Replace with actual images
# show_image_grid(img_files)

def to_gif(images, path):

    images[0].save(path, save_all=True,
                   append_images=images[1:], loop=0, duration=len(images) * 20)

def figure_to_image(figure):

    figure.set_dpi(300)

    figure.canvas.draw()

    return Image.frombytes('RGB', figure.canvas.get_width_height(), figure.canvas.tostring_rgb())

def image_grid(images, outpath=None, column_titles=None, row_titles=None):

    n_rows = len(images)
    n_cols = len(images[0])

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols, n_rows), squeeze=False)

    for row, _images in enumerate(images):

        for column, image in enumerate(_images):
            ax = axs[row][column]
            ax.imshow(image)
            if column_titles and row == 0:
                ax.set_title(textwrap.fill(
                    column_titles[column], width=12), fontsize='x-small')
            if row_titles and column == 0:
                ax.set_ylabel(row_titles[row], rotation=0, fontsize='x-small', labelpad=1.6 * len(row_titles[row]))
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)

    if outpath is not None:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout(pad=0)
        image = figure_to_image(plt.gcf())
        plt.close()
        return image

def get_module(module, module_name):

    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 0:
        return module
    else:
        module = getattr(module, module_name[0])
        return get_module(module, module_name[1:])

def set_module(module, module_name, new_module):

    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 1:
        return setattr(module, module_name[0], new_module)
    else:
        module = getattr(module, module_name[0])
        return set_module(module, module_name[1:], new_module)

def freeze(module):

    for parameter in module.parameters():

        parameter.requires_grad = False

def unfreeze(module):

    for parameter in module.parameters():

        parameter.requires_grad = True

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

class StableDiffuser(torch.nn.Module):

    def __init__(self,
                scheduler='DDIM'
        ):
        print('code changed')

        super().__init__()

        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae")
        print(self.vae.config.scaling_factor )
        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14")
        
        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet")
        
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="feature_extractor")
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="safety_checker")

        if scheduler == 'LMS':
            self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        elif scheduler == 'DDIM':
            self.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        elif scheduler == 'DDPM':
            self.scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")    

        self.eval()

    def get_noise(self, batch_size, img_size, generator=None):

        param = list(self.parameters())[0]

        return torch.randn(
            (batch_size, self.unet.in_channels, img_size // 8, img_size // 8),
            generator=generator).type(param.dtype).to(param.device)

    def add_noise(self, latents, noise, step):

        return self.scheduler.add_noise(latents, noise, torch.tensor([self.scheduler.timesteps[step]]))

    def text_tokenize(self, prompts):
        tokens = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        print("prompts", prompts)
        print("tokens", tokens)
        return tokens

    def text_detokenize(self, tokens):

        return [self.tokenizer.decode(token) for token in tokens if token != self.tokenizer.vocab_size - 1]

    def text_encode(self, tokens):

        return self.text_encoder(tokens.input_ids.to(self.unet.device))[0]

    def decode(self, latents):
        print(self.vae.config.scaling_factor)
        print(latents)
        print(1 / self.vae.config.scaling_factor * latents)
        print(self.vae.decode(1 / self.vae.config.scaling_factor * latents))
        print(self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample)
        return self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample
    
    def save_model_checkpoint(self, filepath):
        torch.save({
            'vae': self.vae.state_dict(),
            'unet': self.unet.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'scheduler_config': self.scheduler.config  # Save the scheduler configuration
        }, filepath)

    def encode(self, tensors):

        return self.vae.encode(tensors).latent_dist.mode() * 0.18215

    def to_image(self, image):

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        print(image)
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def set_scheduler_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.unet.device)

    def get_initial_latents(self, n_imgs, img_size, n_prompts, generator=None):

        noise = self.get_noise(n_imgs, img_size, generator=generator).repeat(n_prompts, 1, 1, 1)

        latents = noise * self.scheduler.init_noise_sigma

        return latents

    def get_noise(self, batch_size, img_size, generator=None):

        param = list(self.parameters())[0]

        return torch.randn(
            (batch_size, self.unet.in_channels, img_size // 8, img_size // 8),
            generator=generator).type(param.dtype).to(param.device)

    def add_noise(self, latents, noise, step):

        return self.scheduler.add_noise(latents, noise, torch.tensor([self.scheduler.timesteps[step]]))

    def text_tokenize(self, prompts):

        return self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

    def text_detokenize(self, tokens):

        return [self.tokenizer.decode(token) for token in tokens if token != self.tokenizer.vocab_size - 1]

    def text_encode(self, tokens):

        return self.text_encoder(tokens.input_ids.to(self.unet.device))[0]

    def decode(self, latents):

        return self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample

    def encode(self, tensors):

        return self.vae.encode(tensors).latent_dist.mode() * 0.18215

    def to_image(self, image):

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def set_scheduler_timesteps(self, n_steps):
        self.scheduler.set_timesteps(n_steps, device=self.unet.device)

    def get_initial_latents(self, n_imgs, img_size, n_prompts, generator=None):

        noise = self.get_noise(n_imgs, img_size, generator=generator).repeat(n_prompts, 1, 1, 1)

        latents = noise * self.scheduler.init_noise_sigma

        return latents

    def get_text_embeddings(self, prompts, n_imgs):

        text_tokens = self.text_tokenize(prompts)

        text_embeddings = self.text_encode(text_tokens)

        unconditional_tokens = self.text_tokenize([""] * len(prompts))

        unconditional_embeddings = self.text_encode(unconditional_tokens)

        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings]).repeat_interleave(n_imgs, dim=0)

        return text_embeddings

    def predict_noise(self,
             iteration,
             latents,
             text_embeddings,
             guidance_scale=7.5
             ):

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latents = torch.cat([latents] * 2)
        latents = self.scheduler.scale_model_input(
            latents, self.scheduler.timesteps[iteration])

        # predict the noise residual
        noise_prediction = self.unet(
            latents, self.scheduler.timesteps[iteration], encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_prediction_uncond, noise_prediction_text = noise_prediction.chunk(2)
        noise_prediction = noise_prediction_uncond + guidance_scale * \
            (noise_prediction_text - noise_prediction_uncond)

        return noise_prediction
    # New method for inpainting
    @torch.no_grad()
    def inpaint(self, img_tensor, mask_tensor, prompts, img_size=512, n_steps=50, n_imgs=1, **kwargs):
        assert 0 <= n_steps <= 1000

        self.set_scheduler_timesteps(n_steps)
        latents = self.get_initial_latents(n_imgs, img_size, len(prompts))
        
        # Prepare the mask
        masked_latents = latents.clone()
        masked_latents[mask_tensor == 1] = 0  # Set masked areas to 0 or noise

        text_embeddings = self.get_text_embeddings(prompts, n_imgs)

        latents_steps, trace_steps, noise_preds, output_steps = self.diffusion(
            masked_latents, text_embeddings, end_iteration=n_steps, **kwargs
        )

        images_steps = [self.to_image(latents) for latents in latents_steps]
        return images_steps

    

    @torch.no_grad()
    def diffusion(self,
                  latents,
                  text_embeddings,
                  end_iteration=1000,
                  start_iteration=0,
                  return_steps=True,#This was False before
                  pred_x0=False,
                  trace_args=None,                  
                  show_progress=True,
                  eta=0.0,
                  variance_scale = 0.0,
                  **kwargs):

        latents_steps = []
        trace_steps = []
        noise_preds = []
        output_steps = []
        trace = None
        def sigmoid_decay(start_step, end_step, current_step, initial_value=1.0, final_value=0.0):
            # Parameters for the sigmoid
            x_0 = (start_step + end_step) / 2
            k = 12 / (end_step - start_step)  # Adjust steepness

            # Compute sigmoid value
            sigmoid_value = 1 / (1 + np.exp(-k * (current_step - x_0)))

            # Scale to desired range
            return final_value + (initial_value - final_value) * (1 - sigmoid_value)
        for iteration in tqdm(range(start_iteration, end_iteration), disable=not show_progress):

            if trace_args:

                trace = TraceDict(self, **trace_args)
            
            noise_pred = self.predict_noise(
                iteration, 
                latents, 
                text_embeddings,
                **kwargs)
            # compute the previous noisy sample x_t -> x_t-1
            # print("Noise_Pred")
            # print(noise_pred)
            noise_preds.append(noise_pred)
            decay_eta = sigmoid_decay(start_iteration, end_iteration, iteration, initial_value=eta, final_value=0.0)
            output = self.scheduler.step(noise_pred, self.scheduler.timesteps[iteration], latents, eta=decay_eta, variance_scale=variance_scale)
            # print("Output")
            # print(output)
            output_steps.append(output)
            if trace_args:

                trace.close()

                trace_steps.append(trace)

            latents = output.prev_sample
            # print("latents")
            # print(latents)
            if return_steps or iteration == end_iteration - 1:

                output = output.pred_original_sample if pred_x0 else latents

                if return_steps:
                    latents_steps.append(output.cpu())
                else:
                    latents_steps.append(output)

        return latents_steps, trace_steps, noise_preds, output_steps

    @torch.no_grad()
    def __call__(self,
                 prompts,
                 img_size=512,
                 n_steps=50,
                 n_imgs=1,
                 end_iteration=None,
                 generator=None,
                 eta=0.0,
                 variance_scale = 0.0,
                 **kwargs):
        assert 0 <= n_steps <= 1000

        if not isinstance(prompts, list):
            prompts = [prompts]

        self.set_scheduler_timesteps(n_steps)
        latents = self.get_initial_latents(n_imgs, img_size, len(prompts), generator=generator)
        text_embeddings = self.get_text_embeddings(prompts, n_imgs)
        end_iteration = end_iteration or n_steps

        latents_steps, trace_steps, noise_preds, output_steps = self.diffusion(latents, text_embeddings, end_iteration=end_iteration, eta=eta, variance_scale=variance_scale, **kwargs)
        returned_latents = latents_steps
        latents_steps = [self.decode(latents.to(self.unet.device)) for latents in latents_steps]
        images_steps = [self.to_image(latents) for latents in latents_steps]

        np_latents = np.array([latents.cpu().numpy() for latents in latents_steps])
        
        print("latents_steps shape: ", np_latents.shape)

        # for i in range(len(images_steps)):
        #     self.safety_checker = self.safety_checker.float()
        #     safety_checker_input = self.feature_extractor(images_steps[i], return_tensors="pt").to(latents_steps[0].device)
        #     image, has_nsfw_concept = self.safety_checker(
        #         images=latents_steps[i].float().cpu().numpy(), clip_input=safety_checker_input.pixel_values.float()
        #     )

        #     images_steps[i][0] = self.to_image(torch.from_numpy(image))[0]
        
        np_images_steps = np.array(images_steps)

        final_steps = list(zip(*images_steps))


        # '*' unpacks the images_steps

        if trace_steps:
            return images_steps, trace_steps

        return final_steps
   
class FineTunedModel(torch.nn.Module):

    def __init__(self,
                 model,
                 train_method,
                 ):

        super().__init__()

        self.model = model
        self.ft_modules = {}
        self.orig_modules = {}

        freeze(self.model)

        for module_name, module in model.named_modules():
            if 'unet' not in module_name:
                continue
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if train_method == 'xattn':
                        if 'attn2' not in module_name:
                            continue
                elif train_method == 'xattn-strict':
                    if 'attn2' not in module_name or 'to_q' not in module_name or 'to_k' not in module_name:
                        continue
                elif train_method == 'noxattn':
                    if 'attn2' in module_name:
                        continue 
                elif train_method == 'noxattn_strong':
                    if 'attn2' in module_name:
                        continue 
                elif train_method == 'selfattn':
                    if 'attn1' not in module_name:
                        continue
                elif train_method == "allattn":
                    module_name = module_name
                elif train_method == "allattn_strong":
                    module_name = module_name
                elif train_method == "allattn_break":
                    module_name = module_name
                elif train_method == "noxattn_break":
                    module_name = module_name
                else:
                    raise NotImplementedError(
                        f"train_method: {train_method} is not implemented."
                    )
                ft_module = copy.deepcopy(module)
                    
                self.orig_modules[module_name] = module
                self.ft_modules[module_name] = ft_module

                unfreeze(ft_module)

        self.ft_modules_list = torch.nn.ModuleList(self.ft_modules.values())
        self.orig_modules_list = torch.nn.ModuleList(self.orig_modules.values())

        
    @classmethod
    def from_checkpoint(cls, model, checkpoint, train_method):

        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)

        modules = [f"{key}$" for key in list(checkpoint.keys())]

        ftm = FineTunedModel(model, train_method=train_method)
        ftm.load_state_dict(checkpoint)

        return ftm

        
    def __enter__(self):

        for key, ft_module in self.ft_modules.items():
            set_module(self.model, key, ft_module)

    def __exit__(self, exc_type, exc_value, tb):

        for key, module in self.orig_modules.items():
            set_module(self.model, key, module)

    def parameters(self):

        parameters = []

        for ft_module in self.ft_modules.values():

            parameters.extend(list(ft_module.parameters()))

        return parameters

    def state_dict(self):

        state_dict = {key: module.state_dict() for key, module in self.ft_modules.items()}

        return state_dict

    def load_state_dict(self, state_dict):

        for key, sd in state_dict.items():
            
            self.ft_modules[key].load_state_dict(sd)
def train(erase_concept, erase_from, train_method, iterations, negative_guidance, lr, save_path):
  
    nsteps = 50

    diffuser = StableDiffuser(scheduler='DDIM').to('cuda')
    diffuser.train()

    finetuner = FineTunedModel(diffuser, train_method=train_method)

    optimizer = torch.optim.Adam(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))
    erase_concept = erase_concept.split(',')
    erase_concept = [a.strip() for a in erase_concept]
    
    erase_from = erase_from.split(',')
    erase_from = [a.strip() for a in erase_from]
    
    
    if len(erase_from)!=len(erase_concept):
        if len(erase_from) == 1:
            c = erase_from[0]
            erase_from = [c for _ in erase_concept]
        else:
            print(erase_from, erase_concept)
            raise Exception("Erase from concepts length need to match erase concepts length")
            
    erase_concept_ = []
    for e, f in zip(erase_concept, erase_from):
        erase_concept_.append([e,f])
    
    
    
    erase_concept = erase_concept_
    
    
    
    print(erase_concept)

#     del diffuser.vae
#     del diffuser.text_encoder
#     del diffuser.tokenizer

    torch.cuda.empty_cache()

    for i in pbar:
        with torch.no_grad():
            index = np.random.choice(len(erase_concept), 1, replace=False)[0]
            erase_concept_sampled = erase_concept[index]
            
            
            neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
            positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]],n_imgs=1)
            target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]],n_imgs=1)
        

            diffuser.set_scheduler_timesteps(nsteps)

            optimizer.zero_grad()

            iteration = torch.randint(1, nsteps - 1, (1,)).item()

            latents = diffuser.get_initial_latents(1, 512, 1)

            with finetuner:

                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=3, 
                    show_progress=False
                )

            diffuser.set_scheduler_timesteps(1000)

            iteration = int(iteration / nsteps * 1000)
            
            positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
            neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
            target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
            if erase_concept_sampled[0] == erase_concept_sampled[1]:
                target_latents = neutral_latents.clone().detach()
        with finetuner:
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        

        loss = criteria(negative_latents, target_latents - (negative_guidance*(positive_latents - neutral_latents))) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        
        loss.backward()
        optimizer.step()

    torch.save(finetuner.state_dict(), save_path)

    del diffuser, loss, optimizer, finetuner, negative_latents, neutral_latents, positive_latents, latents_steps, latents

    torch.cuda.empty_cache()

if __name__ == '__main__':
    model_path='ESD_Models/car_noxattn_200.pt'
    state_dict = torch.load(model_path)
    diffuser = StableDiffuser(scheduler='DDIM').to('cuda')
    finetuner = FineTunedModel(diffuser, train_method='noxattn')

    finetuner.load_state_dict(state_dict)

    #generation loop
    all_images = []
    with finetuner:
        images = diffuser('image of a car', n_steps=50, generator=torch.manual_seed(2440), eta=1.0)
    plt.imshow(images[0][0])