# Latent Space Interpolation of Doodles

Can a VAE learn a latent representation of simple hand-drawn images? Using cat and clock doodles from Google's Quick, Draw! dataset, we will train a VAE in PyTorch to reconstruct images, generate new samples, and investigate how images change as we interpolate between points in the latent space.

The project focuses on implementing and understanding the model rather than relying on a pretrained network.

## Project Goals

- Implement a Variational Autoencoder from scratch in PyTorch.
- Explore how the model encodes and reconstructs doodles.
- Generate new images by sampling from the learned latent space.
- Interpolate between cat and clock representations and examine whether the transitions are meaningful.
- Evaluate reconstruction quality and monitor the reconstruction and KL-divergence components of the loss.

## Dataset

We use the cat and clock categories from Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset). The bitmap files contain 28 × 28 grayscale images.

The raw files are downloaded locally and excluded from version control. The preprocessing notebook creates a balanced subset, normalizes pixel values to the range [0, 1], and divides the data into training, validation, and test sets.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/samanthajuteram/latent-space-interpolation-of-doodles.git
cd latent-space-interpolation-of-doodles
```


### 2. Set up the environment

Create and activate a virtual environment, or use an existing Python environment with the required packages installed.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Download the dataset

Open `notebooks/01_data_loading.ipynb` and run its cells. The notebook downloads the cat and clock bitmap files if they are not already present and saves them under `data/raw/`.

Expected files:

- `data/raw/cat.npy`
- `data/raw/clock.npy`

You only need to download these files once unless you delete them or intentionally refresh the dataset.

### 4. Explore and preprocess the data

Open `notebooks/02_data_preprocessing_exploration.ipynb` and run the cells in order.

The notebook examines sample images and pixel range, selects a balanced subset of cats and clocks, normalizes and reshapes the images.

The processed dataset is saved to:

`data/processed/quickdrae_dataset.npz`

The default configuration uses up to 5,000 images per category.

### 5. Train and evaluate the model

*Subject to change*

After preprocessing, the scripts in `src/` will use the saved dataset to train and evaluate the VAE.

- `data.py` loads the processed arrays and prepares the PyTorch data loaders.
- `vae.py` defines the encoder, latent sampling procedure, and decoder.
- `train.py` runs model training and tracks the loss components.
- `evaluate.py` evaluates reconstructions and generates samples and latent-space interpolation visualizations.


## Method

Fill in later 

## Team

Sophia and Sam

## Acknowledgments

Dataset: [Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)  
Framework: [PyTorch](https://pytorch.org/)