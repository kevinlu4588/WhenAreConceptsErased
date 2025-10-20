# When Are Concepts Erased From Diffusion Models?

This repository provides the official implementation of **"When Are Concepts Erased From Diffusion Models?"** (https://arxiv.org/abs/2505.17013) accepted at NeurIPS 2025.

## The paper

Most evaluation methods for concept erasure involve optimizing adversarial prompts. What if we explored a framework focused on other types of inputs? We explore evaluations in the form of:
1. Overloading Gaussian noise in the scheduler
2. Context injection via inpainting or starting the diffusion process from an intermediate step
3. Classifier guidance

And compare the model behavior against traditional probes (standard prompt, textual inversion, unlearndiffatk)
---

## Environment Setup

Create and activate the provided Conda environment:

```bash
conda env create -f erasing_env.yaml
conda activate erasing_env
```

## ⚡ Quick Start Example

Run a full probe and evaluation in one command:

```bash
./run_single_model.sh
```

This will:
1. Load an SD1.4 model with "airliner" erased via esd-x
2. Run all available probes
3. Save generated images under `results/`
4. Automatically compute CLIP and classifier-based evaluation metrics

## 💻 Using Your Own Models (Direct Python Usage)

You can run `runner.py` directly instead of the shell script.

### If your model directory contains a complete pipeline (e.g., UNet, VAE, text encoder):

```bash
python runner.py \
  --concept airliner \
  --erasing_type esdx \
  --probes noisebasedprobe \
  --num_images 10 \
  --device cuda \
  --config configs/default.yaml \
  --pipeline_path kevinlu4588/airliner
```

### Or a path to a custom UNET checkpoint:

```bash
python runner.py \
  --concept airliner \
  --erasing_type esdx \
  --probes noisebasedprobe \
  --num_images 10 \
  --device cuda \
  --config configs/default.yaml \
  --unet_path /path/to/your/custom_unet
```

## 📊 Running Evaluator

```bash
python evaluator.py
```
---

## 🧪 Available Probes

Our framework includes multiple probing techniques to test concept erasure:

- **Noise-based Probe**: Tests model robustness to trajectory perturbations
- **Inpainting Probe**: Evaluates concept regeneration in masked regions
- **Textual Inversion Probe**: Assesses embedding-level concept understanding
- **Diffusion Compeletion**
- **Classifier Guidance**

## 📈 Evaluation Metrics

The evaluation framework computes:
- **CLIP Similarity**: Measures semantic similarity between generated images and target concepts
- **Classification Accuracy**: Uses pre-trained classifiers to detect presence of erased concepts

---

## 📖 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{lu2025concepts,
  title={When Are Concepts Erased From Diffusion Models?},
  author={Lu, Kevin and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---

## 🔗 Related Work

This research builds upon several concept erasure methods. If you use our evaluation framework with these methods, please also cite the original papers:

### ESD (Erased Stable Diffusion)
```bibtex
@inproceedings{gandikota2023erasing,
  title={Erasing Concepts from Diffusion Models},
  author={Gandikota, Rohit and Materzynska, Joanna and Fiotto-Kaufman, Jaden and Bau, David},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2426--2436},
  year={2023}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

We thank the authors of the original concept erasure methods for making their code publicly available. This work was supported by [acknowledgment details].

## 📧 Contact

For questions about the code or paper, please open an issue or contact [lu.kev@northeastern.edu].

---