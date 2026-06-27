# xTrimoGene Baseline

This directory contains a scratch xTrimoGene-style baseline for the DNADiff imputation benchmarks.

Design notes:
- Uses the xTrimoGene-3M architecture shape.
- Uses the xTrimoGene masked-expression reconstruction objective.
- Applies leak-free masked-profile normalization before model input construction.
- Supports reusable checkpoints via `train_xtrimogene.py` and `impute_xtrimogene.py`.

One-time setup:
```bash
conda env create -f baselines/xTrimoGene/environment.yml
```
