[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
![Python Version](https://img.shields.io/badge/python-3.8-blue) 
![PyTorch Version](https://img.shields.io/badge/pytorch-2.0.1-blue)  [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2410.11247)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR--2025-green)](https://openreview.net/forum?id=yIlyHJdYV3&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)) 
[![Hugging Face](https://img.shields.io/badge/Hugging--Face-Model-yellow?logo=huggingface&logoColor=yellow)](https://huggingface.co/papers/2410.11247)

# A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Translations | ICLR 2025


In subsurface imaging, learning the mapping from velocity maps to seismic waveforms (forward problem) and waveforms to velocity (inverse problem) is important for several
applications. While traditional techniques for solving forward and inverse problems are computationally prohibitive, there is a growing interest to leverage recent advances in deep learning to learn the mapping between velocity maps and seismic waveform images directly from data. Despite the variety of architectures explored in previous works, several open questions still remain unanswered such as the effect of latent space sizes, the importance of manifold learning, the complexity of translation models, and the value of jointly solving forward and inverse problems. We propose a unified framework to systematically characterize prior research in this area termed the Generalized Forward-Inverse (GFI) framework, building on the assumption of manifolds and latent space translations. We show that GFI encompasses previous works in deep learsning for subsurface imaging, which can be viewed as specific instantiations of GFI. We also propose two new model architectures within the framework of GFI: Latent UNet and Invertible X-Net, leveraging the power of U-Nets for domain translation and the ability of IU-Nets to simultaneously learn forward and inverse translations, respectively. We show that our proposed models achieve state-of-the-art (SOTA) performance for forward and inverse problems on a wide range of synthetic benchmark datasets, and also investigate their zero-shot effectiveness on two real-world datasets.


**Paper Link: [OpenReview](https://openreview.net/forum?id=yIlyHJdYV3)** 

**Paper Arxiv Link: [https://arxiv.org/abs/2310.09441](https://arxiv.org/abs/2410.11247)** (For updated results and information)

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Environment](#setting-up-the-environment)
- [Training Instructions](#training-instructions)
  - [General Notes](#general-notes)
  - [Model Training](#model-training)
    - [UNet (Forward)](#unet-large-347m-small-178m)
    - [UNet (Inverse)](#unet-inverse)
    - [Invertible X-Net Training](#invertible-x-net-training)
- [Evaluation Instructions](#evaluation-instructions)
  - [Forward Model Evaluation (Zero-Shot)](#forward-model-evaluation-zero-shot)
  - [Inverse Model Evaluation (Zero-Shot)](#inverse-model-evaluation-zero-shot)
  - [Inveritble X-Net Evaluation (Zero-Shot)](#inveritble-x-net-evaluation-zero-shot)
- [Visualization and Plotting](#visualization-and-plotting)
- [Dataset and Baselines](#dataset-and-baselines)
- [Bibtex](#bibtex)

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git (if cloning the repository)

### Setting Up the Environment

Follow these steps to set up the Python environment using the provided `.yaml` file:

1. **Clone the repository (if you haven't already):**

    ```bash
    git clone https://github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI.git
    cd your-repo-name
    ```

2. **Create the Conda environment:**

    Ensure you are in the directory containing the `environment.yaml` file, then run:

    ```bash
    conda env create -f environment.yaml
    ```

    This command will create a new Conda environment with all the necessary dependencies specified in the `environment.yaml` file.

3. **Activate the environment:**

    After the environment is created, activate it with:

    ```bash
    conda activate your-environment-name
    ```

    Replace `your-environment-name` with the actual name specified in the `environment.yaml` file, or simply use the default name provided by Conda.

4. **Verify the environment setup:**

    To ensure everything is installed correctly, you can list all the installed packages:

    ```bash
    conda list
    ```

## Training Instructions

Training scripts are located in the `src/` directory. Sample configurations can be found in the `scripts/` folder.

### General Notes

- `-ds` specifies the dataset name. Options: `flatvel-a`, `curvevel-a`, `curvefault-a`, `style-a`, and `*-b` variants.
- `-t` and `-v` specify paths to training and validation splits.
- `--mask_factor` allows training on dataset subsets (values from `0.0` to `0.9`)
- `--latent-dim` (optional) sets latent space size for Invertible X-Net
- Models include: `UNetForwardModel`, `UNetInverseModel`, `IUnetForwardModel`, `IUnetInverseModel`, `Invertible_XNet`, `WaveformNet`, `InversionNet`

---

### Model Training

#### UNet (Large: ~34.7M, Small: ~17.8M)
```bash
python -u src/train_forward.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetForwardModel_33M/ \
  --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 \
  -t train_test_splits/flatvel_a_train.txt -v train_test_splits/flatvel_a_val.txt \
  --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2
```

#### UNet Inverse
```
python -u src/train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetInverseModel_33M/ \
  --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 \
  --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2
```

### Invertible X-Net Training

To train the **Invertible X-Net** model on the `flatvel-a` dataset without cycle-consistency loss:

```bash
python -u src/train_invertible_x_net.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/Invertible_XNet/ \
  --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 \
  --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 \
  --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0
```
Additional Notes:
-  To enable cycle-consistency loss, set --lambda_cycle_vel and --lambda_cycle_amp to non-zero values (e.g., 1.0).
-  To use a warmup schedule for cycle loss, add --warmup_cycle_epochs <num_epochs>.
-  To modify latent space dimension, add --latent-dim <8|16|32|64|70>.
-  For dataset variants (e.g., curvevel-a, style-b), change -ds, output path -o, and update -t / -v for train/test splits accordingly.



## Evaluation Instructions

Evaluation scripts are located in the `evaluate/` directory. These support zero-shot evaluation for forward and inverse models.  
**Important:** Before running evaluation, update the `base_path` variable in the evaluation scripts to point to your saved model directories.

---

### Forward Model Evaluation (Zero-Shot)

```bash
python -u evaluate/evaluate_zero_shot_forward_models.py \
  --model_save_name "UNetForwardModel_33M" --model_type "UNetForwardModel"
```
Supported model types:
- UNetForwardModel
- UNetForwardModel with --unet_repeat_blocks 1 (for smaller variant)
- WaveformNet
- IUnetForwardModel
- FNO (if applicable)


### Inverse Model Evaluation (Zero-Shot)
```
python -u evaluate/evaluate_zero_shot_inverse_models.py \
  --model_save_name "UNetInverseModel_33M" --model_type "UNetInverseModel"
```
Supported model types:
- UNetInverseModel
- UNetInverseModel with --unet_repeat_blocks 1 (for smaller variant)
- IUnetInverseModel
- InversionNet
  
### Inveritble X-Net Evaluation (Zero-Shot)
```
python -u evaluate/evaluate_zero_shot_inverse_models.py \
  --model_save_name "Invertible_XNet" --model_type "IUNET"
```


### Visualization and Plotting

All scripts for generating visualizations and plots are located in the `evaluate/plot_codes/` directory.  
This includes utilities for plotting predictions, errors, waveform overlays, and comparison figures for both forward and inverse models.


### Dataset and Baselines

Refer to the OpenFWI GitHub repo to download the datasets and to access the baseline codes. https://github.com/lanl/OpenFWI.git


### Bibtex
```
@inproceedings{guptaunified,
  title={A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Translations},
  author={Gupta, Naveen and Sawhney, Medha and Daw, Arka and Lin, Youzuo and Karpatne, Anuj},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```
```
@article{gupta2024unified,
  title={A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Translations},
  author={Gupta, Naveen and Sawhney, Medha and Daw, Arka and Lin, Youzuo and Karpatne, Anuj},
  journal={arXiv preprint arXiv:2410.11247},
  year={2024}
}
```
