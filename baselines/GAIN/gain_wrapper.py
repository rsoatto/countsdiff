import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, m):
        inp = torch.cat([x, m], dim=1)
        h1 = self.relu(self.fc1(inp))
        h2 = self.relu(self.fc2(h1))
        return self.fc3(h2)


class Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, h):
        inp = torch.cat([x, h], dim=1)
        h1 = self.relu(self.fc1(inp))
        h2 = self.relu(self.fc2(h1))
        return torch.sigmoid(self.fc3(h2))


def _normalization(data):
    _, dim = data.shape
    norm_data = data.copy()
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    for i in range(dim):
        min_val[i] = np.nanmin(norm_data[:, i])
        norm_data[:, i] -= min_val[i]
        max_val[i] = np.nanmax(norm_data[:, i])
        norm_data[:, i] /= (max_val[i] + 1e-6)
    return norm_data, {"min_val": min_val, "max_val": max_val}


def _renormalization(norm_data, params):
    _, dim = norm_data.shape
    out = norm_data.copy()
    for i in range(dim):
        out[:, i] = out[:, i] * (params["max_val"][i] + 1e-6) + params["min_val"][i]
    return out


class GAIN:
    def __init__(self, dim, alpha=100, hint_rate=0.9, batch_size=128,
                 iterations=10000, device="cpu"):
        self.dim = dim
        self.alpha = alpha
        self.hint_rate = hint_rate
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device

        self.G = Generator(dim).to(device)
        self.D = Discriminator(dim).to(device)
        self.opt_G = torch.optim.Adam(self.G.parameters())
        self.opt_D = torch.optim.Adam(self.D.parameters())

    def train_and_impute(self, data_x):
        """Train GAIN on data_x (NaN = missing) and return imputed array in count space.

        GAIN is transductive: it trains directly on the data it imputes.
        Returns rounded, non-negative imputed counts.
        """
        data_m = 1 - np.isnan(data_x).astype(np.float32)
        no, dim = data_x.shape

        norm_data, norm_params = _normalization(data_x)
        norm_data_x = np.nan_to_num(norm_data, nan=0.0).astype(np.float32)

        eps = 1e-8
        self.G.train()
        self.D.train()

        for it in range(self.iterations):
            batch_idx = np.random.choice(no, min(self.batch_size, no), replace=False)
            X_mb = norm_data_x[batch_idx]
            M_mb = data_m[batch_idx]
            Z_mb = np.random.uniform(0, 0.01, (len(batch_idx), dim)).astype(np.float32)
            H_mb = M_mb * (np.random.uniform(0, 1, (len(batch_idx), dim)) < self.hint_rate).astype(np.float32)
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            X_t = torch.from_numpy(X_mb).to(self.device)
            M_t = torch.from_numpy(M_mb).to(self.device)
            H_t = torch.from_numpy(H_mb).to(self.device)

            # Update discriminator
            g_out = self.G(X_t, M_t)
            hat_x = X_t * M_t + g_out.detach() * (1 - M_t)
            d_prob = self.D(hat_x, H_t)
            d_loss = -torch.mean(
                M_t * torch.log(d_prob + eps) + (1 - M_t) * torch.log(1 - d_prob + eps)
            )
            self.opt_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            # Update generator
            g_out = self.G(X_t, M_t)
            hat_x = X_t * M_t + g_out * (1 - M_t)
            d_prob = self.D(hat_x, H_t)
            g_adv = -torch.mean((1 - M_t) * torch.log(d_prob + eps))
            mse = torch.mean((M_t * X_t - M_t * g_out) ** 2) / torch.mean(M_t)
            g_loss = g_adv + self.alpha * mse
            self.opt_G.zero_grad()
            g_loss.backward()
            self.opt_G.step()

        # Impute full dataset
        self.G.eval()
        Z_full = np.random.uniform(0, 0.01, (no, dim)).astype(np.float32)
        X_full = data_m * norm_data_x + (1 - data_m) * Z_full

        with torch.no_grad():
            chunk = 4096
            g_parts = []
            for i in range(0, no, chunk):
                x_t = torch.from_numpy(X_full[i:i + chunk]).to(self.device)
                m_t = torch.from_numpy(data_m[i:i + chunk]).to(self.device)
                g_parts.append(self.G(x_t, m_t).cpu().numpy())
            g_full = np.concatenate(g_parts, axis=0)

        imputed = data_m * norm_data_x + (1 - data_m) * g_full
        imputed = _renormalization(imputed, norm_params)
        imputed = np.clip(imputed, 0, None)
        return imputed
