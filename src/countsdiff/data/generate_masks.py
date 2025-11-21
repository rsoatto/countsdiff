import numpy as np
import math
import numpy as np

def _resolve_rate(rate, i):
    try:
        return float(rate[i])
    except (TypeError, IndexError):
        return float(rate)

def MNAR_mask_high(dataset, rate, sl_window, valid_mask, rng=None):
    """
    Mark as missing (True) n_missing = floor(rate * n_valid_in_col) entries per column,
    sampling from the *highest* values with a slack window of `sl_window`.
    Only entries with valid_mask==True are eligible.
    """
    if rng is None:
        rng = np.random.default_rng()
    dataset = np.asarray(dataset)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    assert dataset.shape == valid_mask.shape, "dataset and valid_mask must have same shape"

    n_rows, n_cols = dataset.shape
    mask = np.zeros_like(valid_mask, dtype=bool)

    for i in range(n_cols):
        col_valid_idx = np.flatnonzero(valid_mask[:, i])
        if col_valid_idx.size == 0:
            continue

        col_rate = _resolve_rate(rate, i)
        n_missing = int(np.floor(col_rate * col_valid_idx.size))
        if n_missing <= 0:
            continue

        col_vals = dataset[col_valid_idx, i]
        k = min(n_missing + int(sl_window), col_valid_idx.size)

        # top-k among valid entries
        part = np.argpartition(-col_vals, kth=k-1)[:k]
        candidates = col_valid_idx[part]

        # choose exactly n_missing (k >= n_missing by construction)
        chosen = rng.choice(candidates, size=n_missing, replace=False)
        mask[chosen, i] = True

    return mask


def MNAR_mask_low(dataset, rate, sl_window, valid_mask, rng=None):
    """
    Mark as missing (True) n_missing = floor(rate * n_valid_in_col) entries per column,
    sampling from the *lowest* values with a slack window of `sl_window`.
    Only entries with valid_mask==True are eligible.
    """
    if rng is None:
        rng = np.random.default_rng()
    dataset = np.asarray(dataset)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    assert dataset.shape == valid_mask.shape, "dataset and valid_mask must have same shape"

    n_rows, n_cols = dataset.shape
    mask = np.zeros_like(valid_mask, dtype=bool)

    for i in range(n_cols):
        col_valid_idx = np.flatnonzero(valid_mask[:, i])
        if col_valid_idx.size == 0:
            continue

        col_rate = _resolve_rate(rate, i)
        n_missing = int(np.floor(col_rate * col_valid_idx.size))
        if n_missing <= 0:
            continue

        col_vals = dataset[col_valid_idx, i]
        k = min(n_missing + int(sl_window), col_valid_idx.size)

        # bottom-k among valid entries
        part = np.argpartition(col_vals, kth=k-1)[:k]
        candidates = col_valid_idx[part]

        chosen = rng.choice(candidates, size=n_missing, replace=False)
        mask[chosen, i] = True

    return mask

def MAR_mask_10(shape):
    """
    MAR mask 
    """
    mask = np.zeros(shape)
    c_40= np.round(shape[1] * 0.4).astype(int) # 40% of columns
    c_25 = np.round(shape[1] * 0.25).astype(int) # 25% of columns
    c_10 = np.round(shape[1] * 0.1).astype(int) # 10% of columns

    r_5 = np.round(shape[0] * 0.05).astype(int)
    r_20 = r_5 + np.round(shape[0] * 0.2).astype(int)
    r_30 = r_20  + np.round(shape[0] * 0.3).astype(int)

    mask[0:r_5, :c_40 ] = 1 # remove top 5% rows and 40% columns, 2% of total
    print(mask.mean())
    mask[(r_5+1):r_20, :c_25] = 1 # remove next 20% rows and 25% columns, 5% of total
    print(mask.mean())
    mask[(r_20+1): r_30, :c_10] = 1 # remove next 30% rows and 10% columns, 3% of total, overall 10%
    print(mask.mean())
    np.random.shuffle(mask) # shuffle rows; column structure remains the same

    return mask.astype(bool)


def MAR_mask_25(shape):
    mask = np.zeros(shape)
    c_60= np.round(shape[1] * 0.6).astype(int)
    c_50 = np.round(shape[1] * 0.5).astype(int)
    c_40 = np.round(shape[1] * 0.4).astype(int)
    c_30 = np.round(shape[1] * 0.3).astype(int)

    r_5 = np.round(shape[0] * 0.05).astype(int)
    r_10 = r_5 +  np.round(shape[0] * 0.1).astype(int)
    r_20 = r_10 + np.round(shape[0] * 0.2).astype(int)
    r_30 = r_20 + np.round(shape[0] * 0.3).astype(int)

    mask[0:r_5, :c_60] = 1 # 3% of total
    mask[(r_5 +1) :r_10, :c_50] = 1 # 5% of total
    mask[(r_10 +1): r_20, :c_40] = 1 # 8% of total
    mask[(r_20 +1): r_30, :c_30] = 1 # 9% of total, total 25%
    np.random.shuffle(mask)
    return mask.astype(bool)

