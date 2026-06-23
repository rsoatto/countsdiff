"""
HI-VAE in PyTorch — supports both Poisson (count) and Gaussian (real) likelihoods.

Architecture:
  - Encoder:  q(s | x^obs) via Gumbel-softmax, q(z | s, x^obs) via reparameterisation
  - Prior:    p(z | s) — Gaussian with learned mean, unit variance (GMM)
  - Decoder:  z -> y -> per-gene output parameter(s) -> likelihood
  - Poisson mode: log(x) input normalisation, softplus rate, +1/-1 shift handled by caller
  - Gaussian mode: standardised input, predicted mean + log-variance

Reference: Nazabal et al., "Handling Incomplete Heterogeneous Data using VAEs", 2020.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class HIVAE(nn.Module):
    def __init__(self, input_dim: int, z_dim: int = 64, s_dim: int = 10,
                 y_dim: int = 1, likelihood: str = 'poisson'):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        self.likelihood = likelihood
        y_total = input_dim * y_dim
        self.y_total = y_total

        # ── Encoder ────────────────────────────────────────────────────────────
        self.s_enc = nn.Linear(input_dim, s_dim)
        self.z_mean_enc   = nn.Linear(input_dim + s_dim, z_dim)
        self.z_logvar_enc = nn.Linear(input_dim + s_dim, z_dim)

        # ── Prior ───────────────────────────────────────────────────────────────
        self.z_prior_mean = nn.Linear(s_dim, z_dim)

        # ── Decoder ─────────────────────────────────────────────────────────────
        self.y_dec = nn.Linear(z_dim, y_total)
        self.theta_yw = nn.Parameter(torch.empty(input_dim, y_dim))
        self.theta_sw = nn.Parameter(torch.empty(input_dim, s_dim))

        if likelihood == 'gaussian':
            self.logvar_yw = nn.Parameter(torch.empty(input_dim, y_dim))
            self.logvar_sw = nn.Parameter(torch.empty(input_dim, s_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.theta_yw, std=0.05)
        nn.init.normal_(self.theta_sw, std=0.05)
        if self.likelihood == 'gaussian':
            nn.init.normal_(self.logvar_yw, std=0.05)
            nn.init.normal_(self.logvar_sw, std=0.05)

    # ── Forward components ───────────────────────────────────────────────────────

    def _normalise_input(self, x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        if self.likelihood == 'poisson':
            return torch.log(x.clamp(min=1e-6)) * obs_mask
        else:
            return x * obs_mask

    def encode(self, x_norm: torch.Tensor, tau: float):
        log_pi   = F.log_softmax(self.s_enc(x_norm), dim=-1)
        s        = F.gumbel_softmax(log_pi, tau=tau, hard=False)
        xz       = torch.cat([x_norm, s], dim=-1)
        mu_z     = self.z_mean_enc(xz)
        logvar_z = self.z_logvar_enc(xz).clamp(-15.0, 15.0)
        z        = mu_z + (logvar_z / 2).exp() * torch.randn_like(mu_z)
        return s, z, (log_pi, mu_z, logvar_z)

    def decode(self, s: torch.Tensor, z: torch.Tensor):
        mu_pz = self.z_prior_mean(s)
        y = self.y_dec(z).view(-1, self.input_dim, self.y_dim)
        theta = (y * self.theta_yw).sum(-1) + s @ self.theta_sw.T

        if self.likelihood == 'poisson':
            lam = F.softplus(theta).clamp(min=1e-6)
            return lam, mu_pz
        else:
            log_var = (y * self.logvar_yw).sum(-1) + s @ self.logvar_sw.T
            log_var = log_var.clamp(-15.0, 15.0)
            return (theta, log_var), mu_pz

    # ── ELBO ────────────────────────────────────────────────────────────────────

    def elbo(self, x: torch.Tensor, obs_mask: torch.Tensor, tau: float, tau2: float = 0.0):
        x_norm = self._normalise_input(x, obs_mask)
        s, z, (log_pi, mu_z, logvar_z) = self.encode(x_norm, tau)
        output, mu_pz = self.decode(s, z)

        if self.likelihood == 'poisson':
            lam = output
            log_p_x = (x * torch.log(lam) - lam - torch.lgamma(x + 1)) * obs_mask
        else:
            mean, log_var = output
            log_p_x = (-0.5 * (np.log(2 * np.pi) + log_var
                        + (x - mean) ** 2 / log_var.exp())) * obs_mask

        log_p_x = log_p_x.sum(1)
        KL_z = 0.5 * (logvar_z.exp() + (mu_pz - mu_z) ** 2 - logvar_z - 1).sum(1)
        pi   = log_pi.exp()
        KL_s = (pi * log_pi).sum(1) + np.log(self.s_dim)

        ELBO = (log_p_x - KL_z - KL_s).mean()
        return ELBO, {
            'log_p_x': log_p_x.mean().item(),
            'KL_z':    KL_z.mean().item(),
            'KL_s':    KL_s.mean().item(),
        }

    # ── Inference ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def impute(self, x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        x_norm = self._normalise_input(x, obs_mask)
        log_pi = F.log_softmax(self.s_enc(x_norm), dim=-1)
        s      = F.one_hot(log_pi.argmax(-1), num_classes=self.s_dim).float()
        xz   = torch.cat([x_norm, s], dim=-1)
        mu_z = self.z_mean_enc(xz)
        output, _ = self.decode(s, mu_z)
        if self.likelihood == 'poisson':
            return output
        else:
            mean, _ = output
            return mean


# ── Training entry point ─────────────────────────────────────────────────────────

def run_hivae(
    data_x:    np.ndarray,
    obs_mask:  np.ndarray,
    epochs:    int   = 100,
    batch_size: int  = 256,
    z_dim:     int   = 64,
    s_dim:     int   = 10,
    y_dim:     int   = 1,
    lr:        float = 1e-3,
    device:    str   = 'cpu',
    likelihood: str  = 'poisson',
) -> np.ndarray:
    N, D = data_x.shape
    dev  = torch.device(device)

    obs_bool = obs_mask.astype(bool)

    if likelihood == 'gaussian':
        obs_float = np.where(obs_bool, data_x, np.nan)
        gene_mean = np.nanmean(obs_float, axis=0)
        gene_std  = np.nanstd(obs_float, axis=0)
        gene_std[gene_std < 1e-6] = 1.0
        data_norm = (data_x - gene_mean) / gene_std
        data_norm[~obs_bool] = 0.0
        x_t = torch.tensor(data_norm, dtype=torch.float32, device=dev)
    else:
        x_t = torch.tensor(data_x, dtype=torch.float32, device=dev)

    mask_t = torch.tensor(obs_mask.astype(float), dtype=torch.float32, device=dev)

    model     = HIVAE(D, z_dim=z_dim, s_dim=s_dim, y_dim=y_dim, likelihood=likelihood).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Training ────────────────────────────────────────────────────────────────
    model.train()
    for epoch in tqdm(range(epochs), desc='HI-VAE training'):
        tau  = max(1.0 - 0.01 * epoch, 1e-3)
        tau2 = min(0.001 * epoch, 1.0)

        perm      = torch.randperm(N, device=dev)
        x_shuf    = x_t[perm]
        m_shuf    = mask_t[perm]
        n_batches = max(N // batch_size, 1)

        total_elbo = 0.0
        for i in range(n_batches):
            xb = x_shuf[i * batch_size: (i + 1) * batch_size]
            mb = m_shuf[i * batch_size: (i + 1) * batch_size]

            optimizer.zero_grad()
            elbo, _ = model.elbo(xb, mb, tau, tau2)
            (-elbo).backward()
            optimizer.step()
            total_elbo += elbo.item()

        if epoch % 10 == 0:
            tqdm.write(f'  epoch {epoch:4d}  ELBO: {total_elbo / n_batches:.3f}  tau: {tau:.3f}')

    # ── Inference ───────────────────────────────────────────────────────────────
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = x_t[i: i + batch_size]
            mb = mask_t[i: i + batch_size]
            parts.append(model.impute(xb, mb))

    imputed = torch.cat(parts, dim=0).cpu().numpy()

    if likelihood == 'gaussian':
        imputed = imputed * gene_std + gene_mean

    imputed[obs_bool] = data_x[obs_bool]
    return imputed
