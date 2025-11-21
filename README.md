
# CountsDiff

A diffusion model on the natural numbers for generation and imputation of count-based data.


## Installation

First, make sure you have Python 3.11 and pip installed. Then, in the root directory, run:
```bash

# Install the package
pip install -e .

```

The `data/` directory is intentionally empty in this repository to keep the artifact size manageable.

To download the full dataset, checkpoint, and experimental results (i.e. hyperparameter sweeps) (~20GB compressed, ~50GB uncompressed), run:

```bash
bash scripts/download_data.sh

## File Structure

```
README.md
configs/          # Various training configuration files
data/dnadiff/           # Data files and checkpoints
  checkpoints/   # Pretrained model checkpoints
  evals/        # Image evaluation results, from running evaluation scripts in scripts/
  random_masks/  # Precomputed random masks for imputation experiments
  filtered_hca_data.hdf5 # Preprocessed fetus scRNA-seq data
  filtered_heart_data.hdf5 # Preprocessed heart scRNA-seq data
figs/          # Figures and visualizations
src/            # Source code for CountsDiff
scripts/       # Hyperparameter sweep and evaluation scripts
images_guided.ipynb # Jupyter notebook for guided generation examples on image data
remasking_images.ipynb # Jupyter notebook for generating samples at different attrition rate.
simulated_data.ipynb # Jupyter notebook for simulated experiments and some figure reproduction
requirements.txt # Python dependencies
setup.py       # Setup script for package installation


```

### 2. Training

Train models using the CLI. For example, to train on cifar10 on GPU 0, run

```bash
countsdiff train --config configs/cifar10.yaml
```

To resume training from a checkpoint, use the continue argument:

```bash
countsdiff continue --run-id <RUN_ID> 
```




