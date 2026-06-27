"""
Evaluation metrics for CountsDiff models
"""

import numpy as np
import torch
from scipy import stats
from typing import Union, Tuple
from scipy import linalg
from torchmetrics import Metric
import scvi
import pandas as pd
import anndata
import itertools
from typing import List, Dict, Optional, Union
from scipy.sparse import issparse, spmatrix
import warnings
from collections import defaultdict


def _compute_mmd_from_kernel_matrices(
    XX: np.ndarray,
    YY: np.ndarray,
    XY: np.ndarray,
    u_stat: bool = False,
) -> float:
    m, n = XX.shape[0], YY.shape[0]

    if u_stat:
        mmd = (XX.sum() - np.diag(XX).sum()) / (m * (m - 1))
        mmd += (YY.sum() - np.diag(YY).sum()) / (n * (n - 1))
    else:
        mmd = XX.sum() / (m * m)       # include diagonal
        mmd += YY.sum() / (n * n)      # include diagonal
    mmd -= 2 * XY.sum() / (m * n)

    return float(mmd)  # MMD should be non-negative


def _get_mmd_heuristic_bandwidth(
    X: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> float:
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    half_size = X.shape[0] // 2
    if half_size == 0:
        return 1.0

    X_first_half = X[:half_size]
    X_second_half = X[half_size:2 * half_size]
    med_sq_dist = torch.median(torch.cdist(X_first_half, X_second_half, p=2) ** 2)
    return med_sq_dist.item() + eps


def _rbf_kernel_block_sums(
    X: torch.Tensor,
    Y: torch.Tensor,
    gammas: torch.Tensor,
    u_stat: bool = False,
    block_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RBF kernel matrix sums in blocks without materializing full N×N matrices.

    For each gamma, computes sum(K) and (if u_stat) sum(diag(K)) where
    K_ij = exp(-gamma * ||X_i - Y_j||^2).  Exploits symmetry when X is Y.

    Returns:
        totals:    [G] float64 tensor of kernel matrix sums
        diag_sums: [G] float64 tensor of diagonal sums (meaningful only when X is Y)
    """
    G = len(gammas)
    totals = torch.zeros(G, dtype=torch.float64, device=X.device)
    diag_sums = torch.zeros(G, dtype=torch.float64, device=X.device)
    symmetric = X is Y

    M, N = X.shape[0], Y.shape[0]
    for i in range(0, M, block_size):
        Xi = X[i : i + block_size]
        j_start = i if symmetric else 0
        for j in range(j_start, N, block_size):
            Yj = Y[j : j + block_size]
            sq_dists = torch.cdist(Xi, Yj).pow_(2)

            is_diag_block = symmetric and (i == j)
            multiplier = 1.0 if is_diag_block else (2.0 if symmetric else 1.0)

            for g in range(G):
                K = torch.exp(-gammas[g] * sq_dists)
                totals[g] += K.sum().to(torch.float64) * multiplier
                if u_stat and is_diag_block:
                    diag_sums[g] += K.diag().sum().to(torch.float64)

    return totals, diag_sums


def calculate_mmd(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    kernel: str = 'rbf',
    gamma: Union[float, List[float], Tuple[float, ...], np.ndarray] = 1.0,
    u_stat: bool = False,
    block_size: int = 2048,
) -> float:
    """
    Calculate Maximum Mean Discrepancy (MMD) between two distributions.

    Uses block-wise GPU-accelerated kernel evaluation to avoid materializing
    full N×N matrices (memory O(block_size²) instead of O(N²)).

    Args:
        X: First sample set
        Y: Second sample set
        kernel: Kernel type ('rbf', 'linear')
        gamma: RBF kernel parameter(s). If multiple values are provided,
            MMD is computed for each and then averaged.
        u_stat: Use unbiased U-statistic estimator.
        block_size: Block size for blocked kernel computation.

    Returns:
        MMD value
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)

    X = X.float()
    Y = Y.float()

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if Y.ndim > 2:
        Y = Y.reshape(Y.shape[0], -1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    Y = Y.to(device)

    gamma_values = np.atleast_1d(np.asarray(gamma, dtype=np.float64))
    if gamma_values.size == 0:
        raise ValueError("gamma must contain at least one value")
    if np.any(gamma_values <= 0):
        raise ValueError("gamma must be positive")

    if kernel == 'rbf':
        gammas_t = torch.from_numpy(gamma_values).to(device)
        m, n = X.shape[0], Y.shape[0]

        xx_sums, xx_diag = _rbf_kernel_block_sums(X, X, gammas_t, u_stat=u_stat, block_size=block_size)
        yy_sums, yy_diag = _rbf_kernel_block_sums(Y, Y, gammas_t, u_stat=u_stat, block_size=block_size)
        xy_sums, _ = _rbf_kernel_block_sums(X, Y, gammas_t, u_stat=False, block_size=block_size)

        mmd_values = torch.zeros(len(gammas_t), dtype=torch.float64)
        for g in range(len(gammas_t)):
            if u_stat:
                mmd = (xx_sums[g] - xx_diag[g]) / (m * (m - 1))
                mmd += (yy_sums[g] - yy_diag[g]) / (n * (n - 1))
            else:
                mmd = xx_sums[g] / (m * m)
                mmd += yy_sums[g] / (n * n)
            mmd -= 2.0 * xy_sums[g] / (m * n)
            mmd_values[g] = mmd

        return float(mmd_values.mean())

    elif kernel == 'linear':
        XX = (X @ X.T).cpu().numpy()
        YY = (Y @ Y.T).cpu().numpy()
        XY = (X @ Y.T).cpu().numpy()
        return _compute_mmd_from_kernel_matrices(XX, YY, XY, u_stat=u_stat)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def calculate_jsd(
    P: Union[np.ndarray, torch.Tensor],
    Q: Union[np.ndarray, torch.Tensor],
    bins: int = 50
) -> float:
    """
    Calculate Jensen-Shannon Divergence between two distributions
    
    Args:
        P: First distribution samples
        Q: Second distribution samples
        bins: Number of bins for histogram estimation
        
    Returns:
        JSD value
    """
    # Convert to numpy if needed
    if isinstance(P, torch.Tensor):
        P = P.cpu().numpy()
    if isinstance(Q, torch.Tensor):
        Q = Q.cpu().numpy()
    
    # Flatten arrays
    P = P.flatten()
    Q = Q.flatten()
    
    # Determine range for histograms
    min_val = min(P.min(), Q.min())
    max_val = max(P.max(), Q.max())
    
    # Create histograms
    p_hist, _ = np.histogram(P, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(Q, bins=bins, range=(min_val, max_val), density=True)
    
    # Normalize to probabilities
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    
    # Calculate M = (P + Q) / 2
    m_hist = (p_hist + q_hist) / 2
    
    # Calculate KL divergences
    kl_pm = stats.entropy(p_hist, m_hist)
    kl_qm = stats.entropy(q_hist, m_hist)
    
    # Jensen-Shannon divergence
    jsd = (kl_pm + kl_qm) / 2
    
    return jsd


def calculate_wasserstein_distance(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate 1-Wasserstein distance between two 1D distributions
    
    Args:
        X: First distribution samples
        Y: Second distribution samples
        
    Returns:
        Wasserstein distance
    """
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    
    # Flatten arrays
    X = X.flatten()
    Y = Y.flatten()
    
    return stats.wasserstein_distance(X, Y)


def calculate_energy_distance(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calculate energy distance between two 1D distributions.

    Args:
        X: First distribution samples
        Y: Second distribution samples

    Returns:
        Energy distance
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()

    return float(stats.energy_distance(X, Y))


def calculate_basic_stats(data: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    Calculate basic statistics for generated data
    
    Args:
        data: Data array
        
    Returns:
        Dictionary of statistics
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    data = data.flatten()
    
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75))
    }

class scFID(Metric):
    """
    Custom metric class for calculating scFID following torchmetrics FID. 

    This class loads a pre-trained scvi-tools model once upon initialization to accept 
    any number of categorical covariates for conditional signal. 
    The metric state stores only low-dimensional embeddings. 

    Attributes:
        full_state_update (bool): A torchmetrics attribute set to False.
    """
    full_state_update: bool = False

    def __init__(
        self,
        gene_names: List[str],
        categorical_covariates: Dict[str, List[str]],
        feature_model_path: str
    ):
        """
        Initializes the scFID metric.

        Args:
            gene_names (List[str]): A list of the gene names from the dataset.

            categorical_covariates (Dict[str, List[str]]): A dictionary where keys
                are the names of categorical covariates (e.g., 'cell_type', 'batch')
                and values are the complete lists of all possible categories for each.
            feature_model_path (str): The file path to the directory containing the
                saved scvi-tools model.
        """
        super().__init__()

        # Eager load through dummy anndata object:
        var_df = pd.DataFrame(index=gene_names)
        all_dummy_obs = []
        if categorical_covariates is None:
            categorical_covariates = {}
            categorical_covariates['batch'] = ['0']

        elif 'batch' not in categorical_covariates.keys():
            categorical_covariates['batch'] = ['0']

        for cov_name, categories in categorical_covariates.items():
            for category in categories:
                dummy_obs_entry = {k: v[0] for k, v in categorical_covariates.items()}
                dummy_obs_entry[cov_name] = category
                all_dummy_obs.append(dummy_obs_entry)

        dummy_obs_df = pd.DataFrame(all_dummy_obs).drop_duplicates().reset_index(drop=True)

        self.categorical_covariates = categorical_covariates # store covariates types

        # Match dtypes with possible categories
        for col, all_categories in categorical_covariates.items():
            cat_dtype = pd.CategoricalDtype(categories=all_categories)
            dummy_obs_df[col] = dummy_obs_df[col].astype(cat_dtype)

        # Create dummy AnnData object
        num_dummy_cells = len(dummy_obs_df)
        dummy_adata = anndata.AnnData(
            X=np.zeros((num_dummy_cells, len(gene_names)), dtype=np.float32),
            obs=dummy_obs_df,
            var=var_df
        )

        # Load the model
        scvi.model.SCVI.prepare_query_anndata(dummy_adata, feature_model_path)
        self.model = scvi.model.SCVI.load_query_data(dummy_adata, feature_model_path)
        self.model.is_trained_ = True
        self.gene_names = var_df

        # Add metric states
        self.add_state("real_latents", default=[], dist_reduce_fx="cat")
        self.add_state("fake_latents", default=[], dist_reduce_fx="cat")

    def update(
        self,
        counts: Union[np.ndarray, spmatrix],
        covariates_data: Dict[str, List],
        real: bool
    ):
        """
        Updates the metric state with a new batch of data.

        This method computes the latent representation for the given batch of
        counts and stores it in the appropriate state list.

        Args:
            counts (Union[np.ndarray, "scipy.sparse.spmatrix"]): The count matrix
                for the batch of cells (cells x genes).
            covariates_data (Dict[str, List]): A dictionary containing the covariate
                labels for the current batch. Keys must match those provided in
                the constructor.
            real (bool): If True, the data is treated as real; otherwise, it's
                treated as fake/generated.
        """
        if issparse(counts):
            counts = counts.toarray()
        num_cells = counts.shape[0]

        if covariates_data is None:
            obs_df = pd.DataFrame(index=np.arange(num_cells))
        elif isinstance(covariates_data, pd.DataFrame):
            obs_df = covariates_data.copy()
        else:
            obs_df = pd.DataFrame(covariates_data)

        if len(obs_df) != num_cells:
            raise ValueError(
                f"Covariates rows ({len(obs_df)}) must match counts rows ({num_cells})."
            )

        for col, all_categories in self.categorical_covariates.items():
            if not all_categories:
                continue
            default_category = all_categories[0]
            if col not in obs_df.columns:
                obs_df[col] = [default_category] * num_cells
            else:
                obs_df[col] = obs_df[col].fillna(default_category)

            cat_dtype = pd.CategoricalDtype(categories=all_categories)
            obs_df[col] = obs_df[col].astype(cat_dtype)
        
        adata_batch = anndata.AnnData(X=counts, obs=obs_df, var=self.gene_names)

        scvi.model.SCVI.prepare_query_anndata(adata_batch, self.model)
        latents = self.model.get_latent_representation(adata=adata_batch)

        latents_tensor = torch.from_numpy(latents).to(self.device)
        if real:
            self.real_latents.append(latents_tensor)
        else:
            self.fake_latents.append(latents_tensor)

    def compute(self) -> torch.Tensor:
        """
        Computes the final scFID score from the stored latent embeddings.

        Returns:
            torch.Tensor: A tensor containing the final FID score.
        
        Raises:
            ValueError: If the metric has not been updated with both real and
                fake data.
        """
        if not self.real_latents or not self.fake_latents:
            raise ValueError("scFID metric must be updated with both real and fake data before computing.")

        real_latents_all = torch.cat(self.real_latents, dim=0).cpu().numpy()
        fake_latents_all = torch.cat(self.fake_latents, dim=0).cpu().numpy()

        fid_score = self._calculate_fid(real_latents_all, fake_latents_all)
        return torch.tensor(fid_score, device=self.device)

    def _calculate_fid(self, real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
        """
        Helper function to calculate the FID score from feature arrays.
        
        Args:
            real_feats (np.ndarray): Latent embeddings for the real data.
            fake_feats (np.ndarray): Latent embeddings for the fake data.

        Returns:
            float: The calculated FID score.
        """
        mu_real, sigma_real = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
        mu_fake, sigma_fake = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

        sum_sq_diff = np.sum((mu_real - mu_fake)**2.0)

        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = sum_sq_diff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
        return float(fid)
    





def r2_per_sample_masked_torch(y: torch.Tensor,
                               yhat: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
    """
    Row-wise R^2 over masked entries.
    y, yhat, mask: [N, D] (mask is bool; True = include)
    Returns: r2 [N] with NaN for rows having <2 masked points or zero variance.
    """
    # ensure float
    y = y.float()
    yhat = yhat.float()
    m = mask.bool()
    mf = m.float()

    # number of evaluated entries per row
    k = mf.sum(dim=1)  # [N]
    # row means over masked entries
    y_sum = (y * mf).sum(dim=1)                     # [N]
    y_bar = y_sum / k.clamp_min(1.0)                # avoid div-by-zero
    y_bar = y_bar.unsqueeze(1)                      # [N,1] for broadcasting

    # SSE and SST restricted to mask
    ss_res = (((y - yhat) ** 2) * mf).sum(dim=1)    # [N]
    ss_tot = (((y - y_bar) ** 2) * mf).sum(dim=1)   # [N]

    r2 = 1.0 - ss_res / ss_tot.clamp_min(0.0)       # safe, will patch below

    # invalid where <2 points or zero variance (ss_tot == 0)
    invalid = (k < 2) | (ss_tot == 0)
    r2 = r2.masked_fill(invalid, float('nan'))
    return r2

@torch.no_grad()
def sliced_wasserstein_torch(real: torch.Tensor, gen: torch.Tensor, num_projections=1000) -> float:
    """
    Computes Sliced Wasserstein Distance (SWD) to approximate Joint Wasserstein.
    Projects high-dim data to random 1D lines -> Sorts -> Computes L2 distance.
    """

    # Check dimensions
    if real.shape[1] != gen.shape[1]:
        raise ValueError("Shape mismatch between real and gen data")

    real = real.float()
    gen = gen.float()

    dim = real.shape[1]

    # 1. Generate random projections (normalized)
    projections = torch.randn(num_projections, dim, device=real.device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)
    
    # 2. Project data [N, num_projections]
    real_proj = real @ projections.T
    gen_proj = gen @ projections.T
    
    # 3. Sort projections (SWD relies on sorting 1D distributions)
    real_proj_sorted, _ = torch.sort(real_proj, dim=0)
    gen_proj_sorted, _ = torch.sort(gen_proj, dim=0)
    
    # 4. Compute L2 distance averaged over projections
    diff = real_proj_sorted - gen_proj_sorted
    wd = torch.mean(torch.pow(diff, 2)) # MSE
    return torch.sqrt(wd).item()        # RMSE


@torch.no_grad()
def _rankdata_average_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Tie-aware ranks (average method), 1..n, for 1D tensor x (float).
    Pure PyTorch implementation.
    """
    n = x.numel()
    if n == 0:
        return x.new_empty((0,))
    # sort and keep original positions
    vals, order = torch.sort(x)               # [n]
    ranks = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)  # 1..n

    # find group boundaries where value changes
    # starts includes 0; ends includes n
    neq = torch.ones(n, dtype=torch.bool, device=x.device)
    neq[1:] = vals[1:] != vals[:-1]
    group_starts = torch.nonzero(neq, as_tuple=False).flatten()
    group_ends = torch.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:]
    group_ends[-1] = n

    # assign average rank to each tie group
    avg_ranks_sorted = torch.empty_like(ranks)
    for s, e in zip(group_starts.tolist(), group_ends.tolist()):
        # positions are 1-indexed ranks[s:e]
        avg = ranks[s:e].mean()
        avg_ranks_sorted[s:e] = avg

    # scatter back to original order
    avg_ranks = torch.empty_like(avg_ranks_sorted)
    avg_ranks[order] = avg_ranks_sorted
    return avg_ranks


@torch.no_grad()
def spearman_per_sample_masked_torch(y: torch.Tensor,
                                     yhat: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
    """
    Row-wise Spearman ρ over masked entries (True = include).
    y, yhat, mask: [N, D]
    Returns: [N] Spearman per row, NaN for rows with <2 points or zero variance.
    """
    assert y.shape == yhat.shape == mask.shape
    y = y.float()
    yhat = yhat.float()
    mask = mask.bool()

    N, D = y.shape
    rho = y.new_full((N,), float('nan'))

    for i in range(N):
        idx = mask[i]
        k = int(idx.sum().item())
        if k < 2:
            continue

        yi = y[i, idx]
        zi = yhat[i, idx]

        # ranks with average ties
        ry = _rankdata_average_torch(yi)
        rz = _rankdata_average_torch(zi)

        # Pearson on ranks
        ryc = ry - ry.mean()
        rzc = rz - rz.mean()
        ry_norm = ryc.norm()
        rz_norm = rzc.norm()
        if ry_norm == 0:
            continue
        if rz_norm == 0:
            rho[i] = 0.0
            continue
        rho[i] = (ryc @ rzc) / (ry_norm * rz_norm)

    return rho


@torch.no_grad()
def pearson_per_sample_masked_torch(y: torch.Tensor,
                                    yhat: torch.Tensor,
                                    mask: torch.Tensor) -> torch.Tensor:
    """
    Row-wise Pearson r over masked entries (True = include).
    y, yhat, mask: [N, D]
    Returns: [N] Pearson per row, NaN for rows with <2 points or zero variance.
    """
    assert y.shape == yhat.shape == mask.shape
    y = y.float()
    yhat = yhat.float()
    mask = mask.bool()

    N, _ = y.shape
    rho = y.new_full((N,), float('nan'))

    for i in range(N):
        idx = mask[i]
        k = int(idx.sum().item())
        if k < 2:
            continue

        yi = y[i, idx]
        zi = yhat[i, idx]

        y_centered = yi - yi.mean()
        z_centered = zi - zi.mean()
        denom = y_centered.norm() * z_centered.norm()
        if denom == 0:
            continue
        rho[i] = (y_centered @ z_centered) / denom

    return rho


def _compute_one_pass_eval_metrics(
    scfid_metric: Optional[scFID],
    imputed_data,
    raw_data,
    impute_mask,
    covariates_dict=None,
    include_distribution_metrics: bool = True,
    round_distribution_metrics: bool = False,
):
    if isinstance(imputed_data, np.ndarray):
        imputed_data = torch.from_numpy(imputed_data)
    if isinstance(impute_mask, np.ndarray):
        impute_mask = torch.from_numpy(impute_mask)
    if isinstance(raw_data, np.ndarray):
        raw_data = torch.from_numpy(raw_data)
    imputed_vals = torch.masked_select(imputed_data, impute_mask)
    actual_vals = torch.masked_select(raw_data, impute_mask)
    if imputed_vals.numel() == 0:
        warnings.warn("No values selected by impute_mask in this sample. Returning NaNs.")
        return {
            "r2": float("nan"), 
            "rmse": float("nan"), 
            "mae": float("nan"),
            "raw_bias": float("nan"), 
            "spearman_corr": float("nan"),
            "pearson_corr": float("nan"),
            "energy_distance": float("nan"),
            "scfid": float("nan")
        }
    raw_bias = torch.mean(imputed_vals - actual_vals).item()
    mae = torch.mean(torch.abs(imputed_vals-actual_vals)).item()
    rmse = torch.sqrt(torch.mean((imputed_vals - actual_vals) ** 2)).item()
    dist_imputed_data = imputed_data
    dist_imputed_vals = imputed_vals
    if round_distribution_metrics:
        dist_imputed_data = torch.round(imputed_data).clamp_min(0)
        dist_imputed_vals = torch.masked_select(dist_imputed_data, impute_mask)

    energy_distance = calculate_energy_distance(actual_vals, dist_imputed_vals)

    r2 = r2_per_sample_masked_torch(raw_data, imputed_data, impute_mask)  # [N]
    r2 = torch.nanmean(r2)  # average over samples, ignoring NaNs

    fake_covars = covariates_dict
    if covariates_dict is not None:
        n_imputed = imputed_data.shape[0]
        if n_imputed != raw_data.shape[0]:
            if isinstance(covariates_dict, pd.DataFrame):
                fake_covars = covariates_dict.iloc[:n_imputed].copy()
            else:
                fake_covars = {}
                for k, v in covariates_dict.items():
                    fake_covars[k] = v[:n_imputed]

    spearman_corr = imputed_data.new_tensor(float("nan"))
    pearson_corr = imputed_data.new_tensor(float("nan"))
    if imputed_vals.numel() > 1:
        spearman_corr = spearman_per_sample_masked_torch(imputed_data, raw_data, impute_mask)
        spearman_corr = torch.nanmean(spearman_corr)  # average over samples, ignoring NaNs
        pearson_corr = pearson_per_sample_masked_torch(imputed_data, raw_data, impute_mask)
        pearson_corr = torch.nanmean(pearson_corr)  # average over samples, ignoring NaNs

    raw_data_np = raw_data.detach().cpu().numpy()
    imputed_data_np = dist_imputed_data.detach().cpu().numpy()

    mmd = float("nan")
    swd = float("nan")
    if include_distribution_metrics:
        base_bandwidth = _get_mmd_heuristic_bandwidth(raw_data)
        bandwidth_scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float64)
        bandwidths = base_bandwidth * bandwidth_scales
        gammas = 1.0 / (2.0 * bandwidths)
        print(
            "Using MMD base bandwidth="
            f"{base_bandwidth} and averaging over bandwidths={bandwidths.tolist()}"
        )
        mmd = calculate_mmd(raw_data_np, imputed_data_np, gamma=gammas)
        swd = sliced_wasserstein_torch(raw_data, dist_imputed_data)

    scfid = float("nan")
    if scfid_metric is not None:
        scfid_metric.update(raw_data_np, covariates_dict, True)
        scfid_metric.update(imputed_data_np, fake_covars, False)
        scfid = scfid_metric.compute().item()
        scfid_metric.reset()
    results = {
        "r2": r2.detach().cpu().item(),
        "rmse": rmse,
        "mae": mae,
        "raw_bias": raw_bias,
        "spearman_corr": spearman_corr.detach().cpu().item(),
        "pearson_corr": pearson_corr.detach().cpu().item(),
        "energy_distance": energy_distance,
        "scfid": scfid,
        'mmd': mmd,
        'swd': swd
    }
    return results

def compute_resampled_eval(
    scfid_metric,
    imputed_data,
    raw_data,
    impute_mask,
    covariates_dict=None,
    n_samples=50000,
    n_resamples=10,
    round_distribution_metrics: bool = False,
):
    total_size = raw_data.shape[0]
    if n_samples > total_size:
        raise ValueError(f"n_samples ({n_samples}) cannot be larger than the total dataset size ({total_size}).")

    iteration_results = defaultdict(list)

    print(f"starting resampling eval: {n_resamples} iterations of {n_samples} samples...")
    for i in range(n_resamples):
        sample_indices = torch.randint(0, total_size, (n_samples, ), device = 'cpu')
        imputed_subset = imputed_data[sample_indices]
        raw_subset = raw_data[sample_indices]
        mask_subset = impute_mask[sample_indices]
        
        covariates_subset = None
        if covariates_dict:
            covariates_subset = {key: np.array(val)[sample_indices.numpy()].tolist() for key, val in covariates_dict.items()}

        single_run_metrics = _compute_one_pass_eval_metrics(
            scfid_metric,
            imputed_subset,
            raw_subset,
            mask_subset,
            covariates_subset,
            round_distribution_metrics=round_distribution_metrics,
        )

        for key, value in single_run_metrics.items():
            iteration_results[key].append(value)
        
        print(f"iteration {i+1}/{n_resamples} complete")

    return iteration_results
