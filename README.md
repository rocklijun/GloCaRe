<div align="center">

# GloCaRe

### Causality-Inspired Global Decorrelation for Semi-Supervised Medical Image Segmentation

**A DINOv2-based semi-supervised segmentation framework with global decorrelation-guided reweighting**

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DINOv2](https://img.shields.io/badge/Backbone-DINOv2-blue)](https://github.com/facebookresearch/dinov2)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Code-Partial%20Release-orange)](#release-status)

[Overview](#overview) · [Method](#method) · [Datasets](#datasets) · [Getting Started](#getting-started) · [Evaluation](#evaluation) · [Citation](#citation)

</div>

## Overview

**GloCaRe** is a causality-inspired framework for **semi-supervised medical image segmentation**. 
## Datasets

| Dataset | Imaging modality | Segmentation task | Classes in provided configuration |
| --- | --- | --- | --- |
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | Cardiac MRI | Cardiac structures | 4 (including background) |
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | Gastrointestinal endoscopy | Polyp | 2 (including background) |
| [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/) | Gastrointestinal endoscopy | Polyp | 2 (including background) |
| PVP-ICC100 | Portal venous phase CT | ICC lesion | 2 (including background) |

**Data availability:** Public datasets should be obtained from their respective providers and used in accordance with their licenses and access conditions. **PVP-ICC100 is a private research dataset and is not distributed in this repository.**

The current dataset loaders expect **HDF5 files** containing datasets named `image` and `label`. File locations are read from the corresponding text files under `splits/` and resolved relative to the dataset root. Raw NIfTI or PNG datasets therefore need to be converted to the expected HDF5 layout before use.

A minimal example is:

```text
data/
└── ACDC/
    └── [paths referenced by splits/acdc/test.txt]
```

Each referenced HDF5 file should provide the relevant `image` and `label` arrays. For ACDC/PVP-ICC100, the implemented loader reads grayscale image/label arrays; for Kvasir-SEG/CVC-ClinicDB, the loader reads a color image and a segmentation mask. Match preprocessing, pixel ranges, mask encoding, and sample ordering to the associated loader and split files.

## Repository Structure

```text
GloCaRe/
├── configs/
│   ├── acdc.yaml
│   ├── cvc.yaml
│   ├── kvasir.yaml
│   └── pvpicc100.yaml
├── dataset/
│   ├── acdc.py
│   ├── semicd.py
│   └── transform.py
├── model/
│   ├── backbone/           # DINOv2-related components
│   └── util/
├── ops/
│   └── config.py           # CLI argument definitions
├── splits/
│   ├── acdc/
│   ├── cvc/
│   ├── kvasir/
│   └── pvpicc100/
├── util/                   # Helpers and losses
├── test.py                 # Evaluation entry point
├── train.py                 # Train entry point
├── utils_calc_metrics.py   # Medical segmentation metrics/visualization
├── LICENSE
└── README.md
```

The tree above reflects the files available on the default branch when this README was prepared. It does not imply that the full training pipeline has been released.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rocklijun/GloCaRe.git
cd GloCaRe
```

### 2. Prepare a Python environment

A CUDA-capable GPU and a compatible PyTorch build are required by the existing evaluation entry point. An environment specification and tested dependency versions have **not** yet been published in this repository.

```bash
conda create -n glocare python=3.10 -y
conda activate glocare
```

Install a PyTorch/CUDA combination appropriate for your platform by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The currently visible source also imports packages such as `pyyaml`, `numpy`, `scipy`, `h5py`, `pillow`, `torchvision`, `tensorboard`, and `SimpleITK`. The exact dependency list must be finalized with the complete release; the commands here are **not a validated environment reproduction recipe**.

### 3. Prepare the dataset

Download a supported public dataset, preprocess it into the HDF5 layout used by the loaders, and check the relevant `splits/<dataset>/` text files. Dataset paths configured in the provided YAML files are machine-specific examples and should be overridden with your local paths.

### 4. Check configuration

Configuration files are provided for:

```text
configs/acdc.yaml
configs/kvasir.yaml
configs/cvc.yaml
configs/pvpicc100.yaml
```

The current YAML files use `dinov2_base`, `crop_size: 252`, and dataset-specific class counts. Options such as `flag_level`, `lambda_ind`, and `apply_reweighting` appear in the provided configurations. **Do not infer the final paper's complete training setup from these configuration files alone**, especially while the training entry point is absent.

## Evaluation

The current `test.py` accepts `--config`, `--checkpoint`, `--mode`, `--data-root`, `labeled_id_path` and `--save_vis_dir` among other arguments. An example of its intended command-line interface is:

```bash
python test.py \
  --config configs/acdc.yaml \
  --data-root /path/to/ACDC \
  --checkpoint /path/to/compatible_checkpoint.pth \
  --mode test \
  --labeled-id-path "${labeled_id_path}" \
  --save_vis_dir ./outputs/acdc
```

**Important:** This is an interface example, **not an executable quick start for the repository in its current state**. The evaluation script imports `model.semseg.dpt.DPT`, which is not currently present. It also expects a checkpoint with a `model` state-dictionary key. After all required files are released, checkpoint compatibility and dataset preprocessing should be verified before using the command.

Available evaluation modes in `test.py` are `val`, `valtest`, and `test`, subject to the corresponding split files existing for the dataset. For example, the current endoscopy split directories do not include `valtest.txt`; use `val` or `test` there.

### Metrics

| Metric | Interpretation | Preferred direction |
| --- | --- | --- |
| Dice similarity coefficient (DSC) | Region overlap | Higher ↑ |
| Jaccard index | Intersection over union | Higher ↑ |
| HD95 | 95th-percentile boundary distance | Lower ↓ |
| ASD | Average surface distance | Lower ↓ |

Results should be compared using consistent preprocessing, aggregation rules, spacing conventions, and evaluation splits. Physical-distance values for HD95 and ASD depend on the supplied image spacing and metric implementation.

## Release Status

As of **November 2026**, the default branch contains dataset loaders, DINOv2-related backbone components, configurations, split definitions, metric utilities, and a test entry point. However, the following materials are not currently available as a complete runnable release:

- [ ] Full GloCaRe segmentation architecture referenced by `test.py` (`model/semseg/dpt.py`)
- [ ] Full training script(s) and reproducible training commands
- [ ] Tested dependency specification (`requirements.txt` or `environment.yml`)
- [ ] Compatible pretrained weights or example checkpoints
- [ ] Verified end-to-end execution instructions and dataset-conversion scripts

This list is provided to make the current reproducibility status transparent. Repository updates may extend the released components.

## Citation

If you use GloCaRe in your research, please cite the associated paper once its final bibliographic information is available. **A verified title, author list, venue, and DOI/BibTeX record have not yet been added here**; citation metadata will be updated following publication.

Until then, please reference the project repository:

```text
GloCaRe: Global Causality-Inspired Reweighting for Semi-Supervised Medical Image Segmentation.
https://github.com/rocklijun/GloCaRe
```

## Acknowledgements

This project builds on research in semi-supervised semantic segmentation, foundation-model representations, and causality-inspired stable learning. In particular, the implementation references concepts and components associated with [DINOv2](https://github.com/facebookresearch/dinov2) and [UniMatch V2](https://github.com/LiheYoung/UniMatch-V2). Please consult their original repositories and papers when reusing related methods or code.

## License

This repository is distributed under the [Apache License 2.0](LICENSE). Third-party dependencies, model weights, and datasets may be subject to separate licenses and usage restrictions.

## Contact

For questions, suggestions, or reproducibility issues, please use [GitHub Issues](https://github.com/rocklijun/GloCaRe/issues).
