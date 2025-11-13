# When Are Concepts Erased From Diffusion Models? (NeurIPS 2025)

[**Project website**](http://unerasing.baulab.info/) | [**Paper on arXiv**](https://arxiv.org/abs/2505.17013) | [**Finetuned model and classifier weights**](https://huggingface.co/DiffusionConceptErasure)

![Figure 1](images/Figure1.png)

## Overview

In concept erasure, a model is modified to selectively prevent it from generating a target concept. Despite the rapid development of new methods, it remains unclear how thoroughly these approaches remove the target concept from the model.

To assess whether a concept has been truly erased from the model, we introduce **a comprehensive suite of independent probing techniques**: supplying visual context, modifying the diffusion trajectory, applying classifier guidance, and analyzing the model's alternative generations that emerge in place of the erased concept. Our results shed light on the value of exploring concept erasure robustness outside of adversarial text inputs, and emphasize the importance of comprehensive evaluations for erasure in diffusion models.

## Environment Setup

Create and activate the provided Conda environment:

```bash
git clone https://github.com/kevinlu4588/WhenAreConceptsErased.git
cd WhenAreConceptsErased
pip install -r requirements.txt
```

## Running the Demo

Navigate to the `src` directory and run the demo script:

```bash
cd src
python demo.py
```

This will:
1. Run all available probes on the configured model(s)
2. Save generated images under `data/results/`
3. Automatically compute evaluation metrics (CLIP similarity and classification accuracy)

## Running Probes on Your Model

To run the probes on your own model:

```bash
cd src
python runner.py --concept <your_concept> --pipeline_path <path_to_your_model>
```

For example:
```bash
python runner.py --concept airliner --pipeline_path DiffusionConceptErasure/esdx_airliner
```

This will run all probes by default. You can also specify individual probes:
```bash
python runner.py --concept airliner --pipeline_path <model_path> --probes standardpromptprobe noisebasedprobe
```

## Key Notebooks

We provide several Jupyter notebooks that demonstrate our probing techniques and evaluation pipeline:

### Core Probe Implementations

- **[Noise-based Probing](probe_notebooks/noise-based.ipynb)**: Walkthrough showing how we manipulate diffusion trajectories to reveal latent concept knowledge in erased models

- **[Classifier Guidance](probe_notebooks/classifier_guidance.ipynb)**: Demonstration of applying classifier guidance to steer erased models back toward generating the target concept

### Results & Evaluation

- **[Demo Results Visualization](probe_notebooks/eval.ipynb)**: Visualization of probe demo results, including CLIP similarity scores, classification accuracies, and side-by-side comparisons across different erasure methods.

## Training new latent classifiers

**Quick start**:

  ```bash
  cd classifier_guidance
  
  python e2e_concept_classifier.py "church, church building" "airliner"\
    --epochs 70 --batch-size 8 --output-dir "./my_classifiers"
  ```

## Probe Execution Times for Demo

Running the probes on an NVIDIA A6000 GPU, typical execution times for a single concept/model pair are:

| Probe | Time per Image | Total Time (30 prompts) |
|-------|---------------|------------------------|
| **Standard Prompt** | 2 seconds | 1 minute |
| **Inpainting** | 2 seconds | 1 minute |
| **Diffusion Completion** | 2 seconds | 1 minute |
| **Noise-based** | 2 seconds × 24 samples | 24 minutes |
| **Classifier Guidance** | 2 seconds × 24 samples | 24 minutes |
| **Noise-based + Classifier** | 2 seconds × 24 samples | 24 minutes |
| **Textual Inversion** | - | 60 minutes (training time per concept model pair)|

## 📖 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{lu2025concepts,
  title={When Are Concepts Erased From Diffusion Models?},
  author={Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, and Niv Cohen},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---

## 🔗 Related Work

Our work builds upon a growing body of research on concept erasure and targeted model editing, including  

- **[Erased Stable Diffusion (ESD)](https://arxiv.org/abs/2303.07326)** — model finetuning for concept removal  
- **[Universal Concept Editing (UCE)](https://arxiv.org/abs/2307.00756)** — lightweight cross attention projection
- **[TaskVectors](https://arxiv.org/abs/2302.00658)** — linear task steering in model weight space  
- **[STEREO](https://arxiv.org/abs/2402.04362)** — ESD + Textual Inversion loop
- **[RECE](https://arxiv.org/abs/2403.13862)** — UCE + additional embedding projection  
- **[UnlearnDiffAtk](https://arxiv.org/abs/2403.08598)** — adversarial prompt optimization

We thank the authors of these methods for laying the groundwork for this research.


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

For questions about the code or paper, please open an issue or contact [lu.kev@northeastern.edu].

---