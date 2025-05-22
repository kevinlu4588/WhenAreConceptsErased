
# Concept Erasure in Diffusion Models

  

This repository accompanies the paper **"When are Concepts Really Erased in Diffusion Models?"**, which investigates concept erasure techniques in diffusion models via multiple different perspectives outside of the traditional optimized inputs (prompt/raw text embedding).

  

## Overview

This project provides the following key components:

  

1.  **Demo Notebooks**

-  **`inpainting.ipynb`** — Demonstrates the use of inpainting to probe for erased concepts.

-  **`noise-based.ipynb`** — Showcases the Noise-Based probe, which adds noise to the diffusion trajectory to bypass concept erasure defenses.

-  **`diffusion-completion.ipynb`** — Shows how diffusion completion is done, as well as the difference in erased model behavior when conditioned on latents from the original model.
  

2.  **Training Scripts**

- Scripts for training **gradient ascent models**, a powerful approach for erasing concepts in diffusion models.

  

## Dependencies

To install the required dependencies, run:

```bash

pip  install  -r  requirements.txt
```
  
## Usage

### Running  the  Demos

  
Open  either  inpainting.ipynb,  noise-based.ipynb, or diffusion-completion.ipynb  in  Jupyter  Notebook.

Follow  the  instructions  in  each  notebook  to  run  the  attacks  and  visualize  the  results.

  

### Training  the  Gradient  Ascent  Models

To  train  a  model  with  gradient  ascent  for  concept  erasure,  run:

  

```bash
./train_ga_model.sh

```
  
With the appropriate hyperparameters & concepts

Noise-Based probe Details

The Noise-based probe is a training-free method that modifies the diffusion trajectory by adding controlled noise at each denoising step:

x̃<sub>t−1</sub> = (x̃<sub>t</sub> − αϵ<sub>D</sub>) + ηϵ

Where:


αϵ<sub>D</sub> = Standard diffusion step

η = Noise scaling factor


The attack explores broader regions of the latent space, exposing erased concepts that persist within the model's knowledge.

Paper Reference

If you use this code in your research, please cite:

When are Concepts Really Erased in Diffusion Models? Anonymous Authors

Contact

For questions or issues, please open a GitHub issue or reach out directly.