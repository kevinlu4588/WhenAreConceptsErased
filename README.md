
# Concept Erasure in Diffusion Models

  

This repository accompanies the paper **"Where Do Erased Concepts Go in Diffusion Models?"**, which investigates concept erasure techniques in diffusion models and introduces the **Noising Attack**, a training-free inference-time attack.

  

## Overview

This project provides the following key components:

  

1.  **Demo Notebooks**

-  **`inpainting_attack_demo.ipynb`** — Demonstrates the inpainting attack to probe erased concepts.

-  **`noising_attack_demo.ipynb`** — Showcases the Noising Attack, which adds noise to the diffusion trajectory to bypass concept erasure defenses.

  

2.  **Training Scripts**

- Scripts for training **gradient ascent models**, a powerful approach for erasing concepts in diffusion models.

  

## Dependencies

To install the required dependencies, run:

```bash

pip  install  -r  requirements.txt
```
  
## Usage

### Running  the  Demos

  
Open  either  inpainting_attack.ipynb  or  noising_attack.ipynb  in  Jupyter  Notebook.

Follow  the  instructions  in  each  notebook  to  run  the  attacks  and  visualize  the  results.

  

### Training  the  Gradient  Ascent  Models

To  train  a  model  with  gradient  ascent  for  concept  erasure,  run:

  

```bash
./train_ga_model.sh

```
  
With the appropriate hyperparameters & concepts

Noising Attack Details

The Noising Attack is a training-free method that modifies the diffusion trajectory by adding controlled noise at each denoising step:

x̃<sub>t−1</sub> = (x̃<sub>t</sub> − αϵ<sub>D</sub>) + ηϵ

Where:


αϵ<sub>D</sub> = Standard diffusion step

η = Noise scaling factor


The attack explores broader regions of the latent space, exposing erased concepts that persist within the model's knowledge.

Paper Reference

If you use this code in your research, please cite:

Where Do Erased Concepts Go in Diffusion Models? Anonymous Authors

Contact

For questions or issues, please open a GitHub issue or reach out directly.