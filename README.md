
# CountsDiff

A diffusion model on the natural numbers for generation and imputation of count-based data.


## Installation

First, make sure you have Python 3.11 and pip installed. Then, in the root directory, run:
```bash
# Install the package
pip install -e .
```

The `data/` directory is intentionally empty in this repository to keep the artifact size manageable.

To download the full dataset, checkpoints, and experimental results (i.e. hyperparameter sweeps; ~20GB compressed, ~50GB uncompressed) from the project's [Zenodo](https://zenodo.org) record, run:

```bash
bash scripts/download_data.sh
```
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
attrition_images.ipynb # Jupyter notebook for generating samples at different attrition rates.
simulated_data.ipynb # Jupyter notebook for simulated experiments and some figure reproduction
requirements.txt # Python dependencies
setup.py       # Setup script for package installation


```

## Training

Train models using the CLI. For example, to train on cifar10 on GPU 0, run

```bash
countsdiff train --config configs/cifar10.yaml
```

To resume training from a checkpoint, use the continue argument:

```bash
countsdiff continue --run-id <RUN_ID> 
```

## Experiment tracking (Weights & Biases)

CountsDiff uses [Weights & Biases](https://wandb.ai) for experiment tracking and run
resolution. The published runs live in the public W&B project
`anonymous2-icml/countsdiff-icml`, so loading a trained model by run id — used by
`countsdiff generate`, `countsdiff continue`, and the evaluation scripts — works
without an API key.

To log your own training runs, set your W&B API key:

```bash
export WANDB_API_KEY=<your-wandb-key>
```

then set `wandb: {enabled: true}` in your config (logging is `false` by default). To
train without W&B, leave it disabled or set `WANDB_MODE=offline`. The entity and
project used for logging are defined in `src/countsdiff/utils/tracking.py`.

## Baselines

Imputation baselines live under `baselines/` and are run through
`scripts/generate_multiple_imputation.py` via `--baseline-model`: `magic`, `gain`,
`hivae`, `scidpm`, `remdm`, `forestdiff`, `scgpt_scratch` / `scgpt_pretrained`, and
`xtrimogene`. scGPT and xTrimoGene are vendored as code only — download their
pretrained weights from the respective upstream model zoos and point the wrappers at
them.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{soatto2026countsdiff,
  title     = {{CountsDiff}: A Diffusion Model on the Natural Numbers for Generation and Imputation of Count-Based Data},
  author    = {Soatto, Renzo G. and Hoel, Anders and Ren, Greycen and Alam, Shorna and Bates, Stephen and Daskalakis, Nikolaos P. and Uhler, Caroline and Skoularidou, Maria},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {arXiv:2604.03779},
}
```




