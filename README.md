# Self-Supervised Spectral–Temporal Denoising for Dynamic DMI

This repository provides a flexible deep-learning framework for denoising complex-valued dynamic DMI / MRSI data using self-supervised learning (Noise2Self).

The framework is designed for multi-dimensional spectroscopic imaging data.  
Users can freely choose which axes of the data are processed by the network, enabling denoising across spatial, spectral, or temporal dimensions without requiring ground-truth data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data format](#data-format)
- [Training](#training)
- [Inference](#inference)
- [Notebooks](#notebooks)

## Features

- **Flexible axis selection**  
  Any combination of 2–3 axes of the input data can be treated as network dimensions.  
  This allows training spatial, spectral, temporal, or mixed spatio–spectral-temporal models.

- **Optional channel folding**  
  One additional axis can be mapped to the channel dimension, allowing the network to process up to four dimensions simultaneously.

- **2D and 3D U-Net architectures**  
  The model is automatically configured as a 2D or 3D U-Net depending on the selected axes.

- **Self-supervised denoising**  
  Noise2Self-style masking enables training directly on noisy measurements without clean reference data.

- **Configurable experiments via YAML**  
  Training parameters, masking strategy, and axis selection are defined in a simple configuration file.

- **Reproducible environment**  
  Docker environment, automated tests, and CI ensure reproducibility.

## Installation

The project uses a **Docker environment** to ensure a reproducible setup with all required dependencies.

Make sure **Git** and **Docker** are installed on your system.

### 1. Clone repository

```bash
git clone https://github.com/EdgarFischer/DMI_Denoising.git
cd DMI_Denoising
```
### 2. Build Docker container

```bash
bash docker/build_docker.sh
```

### 3. Launch container

```
bash docker/launch_docker.sh
```

The repository will be mounted inside the container under:

/workspace/DMI_Denoising

### 4. Verify installation (Quick sanity check)

To verify that the installation works correctly, start a shell inside the container:

```bash
docker exec -it mrjo bash
```
Then run the sanity check:

```bash
python3 DMI_Denoising/scripts/sanity_check.py
```

This script generates a small synthetic dataset and launches a short training run using the standard training pipeline.
The console output should display training and validation losses, confirming that the full pipeline is working correctly.

## Data format

Dynamic DMI data are expected in the form

(x, y, z, t, T)

where

- x, y, z denote spatial dimensions  
- t denotes the spectral dimension (FID)  
- T denotes the repetition / dynamic dimension  

The last dimension is optional, so the data may also be 4D.

Input data can be provided either as

- a NumPy file `data.npy` containing a complex array with shape `(x, y, z, t, T)`
- a MATLAB file `CombinedCSI.mat`, where the data are stored in `csi.Data`

Internally, the pipeline standardizes the input format for consistent downstream processing.

## Training

Training is launched via

```bash
python3 scripts/train.py
```

or (using nohup so the process continues running in the background)

```bash
bash scripts/train.sh
```

Training parameters are defined in

configs/train.yaml

The configuration file specifies e.g.

- datasets used for training and validation
- which axes of the data are processed by the network (`image_axes`)
- optional channel folding (`channel_axis`)
- the masking strategy used for Noise2Self training
- model architecture and optimization parameters

See `configs/train.yaml` for a fully documented example configuration.

During training, the configuration file and the relevant source code are automatically copied to the experiment directory to ensure full reproducibility.

## Inference

An example inference script is provided in

scripts/infer.sh

```bash
bash scripts/infer.sh
```
to apply a trained model to new data.

The script loads a trained checkpoint and applies the denoising network to the specified input dataset.
Input and output paths as well as model parameters can be configured inside the script.

For reproducibility, inference can be run using the same train.yaml configuration that was saved automatically in the model directory during training.
This ensures that the model is applied with the same axis configuration and preprocessing settings that were used during training.

The denoising pipeline can also be called directly from Python (e.g. in notebooks) using the provided inference API.

## Notebooks

Example notebooks for visualizing denoising results are provided in the notebooks/ directory.

The notebook EvaluateModelBeforeFitting demonstrates how to directly compare noisy and denoised data from the network output.

EmpiricalNoiseCorrelations can be used to estimate noise correlations empirically for any 4D or 5D MRSI dataset along each of its axes. This is important for identifying which dimensions are suitable for self-supervised training.

Additional notebooks reproduce figures from the associated research workflow. These require external metabolite quantification software, which is not included in this repository.