scGPT Baseline

This baseline is run through a dedicated conda environment so the main repo
does not need to install scGPT directly.

This wrapper uses vanilla PyTorch attention with `use_fast_transformer=False`,
so `flash-attn` is intentionally not required.
It also loads `scgpt/model/model.py` directly instead of importing the full
`scgpt` package, which avoids optional tokenizer and plotting dependencies
such as `torchtext` and `PIL`.

Expected pretrained model directory layout:

- `args.json`
- `best_model.pt`
- `vocab.json`

Our own trained checkpoints use the same core files, plus `metadata.json`.

Environment setup:

```bash
conda env create -f baselines/scGPT/environment.yml
```

Example usage from the repo root:

```bash
python scripts/generate_multiple_imputation.py \
  --baseline_model scgpt_scratch \
  --data_type heart \
  --mask_type MCAR \
  --dropout 0.4 \
  --num_imputations 5
```

Pretrained fine-tuning example:

```bash
python scripts/generate_multiple_imputation.py \
  --baseline_model scgpt_pretrained \
  --scgpt-pretrained-dir /path/to/scgpt_model_dir \
  --data_type heart \
  --mask_type MCAR \
  --dropout 0.4 \
  --num_imputations 5
```

Separate training and imputation:

```bash
conda run -n scgpt-baseline python baselines/scGPT/train_scgpt.py \
  --data-file data/dnadiff/filtered_heart_data.hdf5 \
  --checkpoint-dir data/dnadiff/scgpt_checkpoints/heart_pretrained_seed1337 \
  --init-mode pretrained \
  --pretrained-dir /path/to/scgpt_model_dir \
  --cond-key batch \
  --cond-key cell_type \
  --cond-key gender \
  --cond-key age
```

```bash
conda run -n scgpt-baseline python baselines/scGPT/impute_scgpt.py \
  --data-file data/dnadiff/filtered_heart_data.hdf5 \
  --mask-file data/dnadiff/random_masks/MCAR_masks/heart_dropout_0.4.npy \
  --checkpoint-dir data/dnadiff/scgpt_checkpoints/heart_pretrained_seed1337 \
  --output-path /tmp/scgpt_heart.npy
```

Reuse a trained checkpoint from the top-level baseline entrypoint:

```bash
python scripts/generate_multiple_imputation.py \
  --baseline_model scgpt_pretrained \
  --scgpt-checkpoint-dir data/dnadiff/scgpt_checkpoints/heart_pretrained_seed1337 \
  --data_type heart \
  --mask_type MCAR \
  --dropout 0.4 \
  --num_imputations 1
```

One-shot train + impute runs can also save a reusable checkpoint:

```bash
conda run -n scgpt-baseline python baselines/scGPT/run_scgpt_imputation.py \
  --data-file data/dnadiff/filtered_heart_data.hdf5 \
  --mask-file data/dnadiff/random_masks/MCAR_masks/heart_dropout_0.4.npy \
  --output-path /tmp/scgpt_heart.npy \
  --init-mode scratch \
  --cond-key batch \
  --cond-key cell_type \
  --cond-key gender \
  --cond-key age \
  --save-checkpoint-dir data/dnadiff/scgpt_checkpoints/heart_scratch_seed1337
```

Implementation notes:

- Conditioning uses a single categorical ID built from the full metadata tuple
  for each cell.
- Input normalization is leak-free: the retained library size is computed after
  masking the target genes, and both inputs and targets use that same masked
  profile normalization.
- Predictions are made in log-normalized space and inverted back to counts
  using the masked-profile library size.
- Direct `train_scgpt.py` and `impute_scgpt.py` runs show native `tqdm`
  progress bars. The top-level baseline entrypoint uses its own wrapper-based
  progress bar instead.
