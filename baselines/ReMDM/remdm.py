import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from diffusers.models.embeddings import TimestepEmbedding, Timesteps

# Optional tqdm for progress bars
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # fallback no-op if tqdm is unavailable
        return iterable


def cosine_alpha(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t * torch.pi / 2) ** 2


def build_rescaled_sigma_schedule(observation_times: Union[np.ndarray, torch.Tensor], alpha_scheduler, eta_rescale: float = 0.005, device=None) -> List[torch.Tensor]:
    if isinstance(observation_times, np.ndarray):
        ts = torch.from_numpy(observation_times.astype(np.float32))
    else:
        ts = observation_times.float().clone()
    if device is not None:
        ts = ts.to(device)
    T = ts.shape[0]
    sigmas: List[torch.Tensor] = []
    for i in range(T - 1):
        t = ts[i]
        s = ts[i + 1]
        a_t = alpha_scheduler(t)
        a_s = alpha_scheduler(s)
        if not torch.is_tensor(a_t):
            a_t = torch.tensor(a_t, device=ts.device, dtype=torch.float32)
        if not torch.is_tensor(a_s):
            a_s = torch.tensor(a_s, device=ts.device, dtype=torch.float32)
        sigma_max = torch.clamp((1.0 - a_s) / (a_t + 1e-12), min=0.0, max=1.0)
        sigmas.append(torch.clamp(sigma_max * float(eta_rescale), max=1.0))
    sigmas.append(torch.tensor(0.0, device=ts.device))
    return sigmas


def corrupt_ids(x_tokens: torch.Tensor, t: torch.Tensor, mask_token: int, *, current_t: Optional[torch.Tensor] = None) -> torch.Tensor:
    b, l = x_tokens.shape
    # Ensure t is broadcastable to batch size b
    if t.dim() == 0 or t.numel() == 1:
        t = t.reshape(1).expand(b)
    else:
        t = t.view(-1)
        if t.numel() != b:
            # Fallback: use the scalar value (first element) expanded to batch
            t = t.reshape(-1)[0].expand(b)

    if current_t is not None:
        ct = current_t
        if ct.dim() == 0 or ct.numel() == 1:
            ct = ct.reshape(1).expand(b)
        else:
            ct = ct.view(-1)
            if ct.numel() != b:
                ct = ct.reshape(-1)[0].expand(b)
        keep_p = (cosine_alpha(t) / (cosine_alpha(ct) + 1e-12)).view(b, 1)
    else:
        keep_p = cosine_alpha(t).view(b, 1)

    keep = (torch.rand_like(x_tokens.float()) < keep_p).long()
    was_mask = (x_tokens == mask_token).long()
    # masks persist; otherwise apply additional masking
    new_tokens = torch.where((keep == 1) & (was_mask == 0), x_tokens, torch.full_like(x_tokens, mask_token))
    return new_tokens


class ReMDMAttention(nn.Module):
    def __init__(self, num_genes: int, num_classes: int, num_layers: int, embed_dim: int, num_heads: int, dropout: float, all_num_classes: Iterable[int]):
        super().__init__()
        self.num_genes = num_genes
        self.num_classes = num_classes  # real classes (exclude mask)
        self.vocab_size = num_classes + 1  # include mask

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=False, dim_feedforward=embed_dim * 4)
            for _ in range(num_layers)
        ])
        self.token_embeddings = nn.Embedding(self.vocab_size, embed_dim)
        self.gene_embeddings = nn.Embedding(num_genes, embed_dim)
        self.output_proj = nn.Linear(embed_dim, num_classes)

        # Label embedders
        self.embedders = []
        for i, size in enumerate(all_num_classes):
            if size is not None and size > 0:
                embedder = nn.Embedding(size, embed_dim)
                setattr(self, f"label_embedder_{i}", embedder)
                self.embedders.append((f"label_embedder_{i}", embedder))
        self.all_num_classes = all_num_classes

        self.time_proj = Timesteps(num_channels=embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(embed_dim, embed_dim)
        self.register_buffer("gene_idx", torch.arange(num_genes))

    def forward(self, x_tokens: torch.Tensor, timesteps: torch.Tensor, class_labels: Union[Iterable[torch.Tensor], None], valid_mask: Optional[torch.Tensor] = None, uncond_mask: Optional[torch.Tensor] = None):
        b, l = x_tokens.shape
        assert l == self.num_genes

        # conditioning tokens
        emb_list = []
        t_emb = self.time_embedding(self.time_proj(timesteps)).unsqueeze(1)
        emb_list.append(t_emb)
        for i, (name, embedder) in enumerate(self.embedders):
            label = class_labels[i] if class_labels is not None and i < len(class_labels) else None
            if label is not None:
                lbl = embedder(label).unsqueeze(1)
                if uncond_mask is not None:
                    lbl[uncond_mask] = 0.
                emb_list.append(lbl)

        # sequence embeddings
        tok = self.token_embeddings(x_tokens)
        gene = self.gene_embeddings(self.gene_idx).unsqueeze(0)
        seq = tok + gene

        x = torch.cat(emb_list + [seq], dim=1)

        if valid_mask is not None:
            n_cond = len(emb_list)
            cond_mask = torch.ones(b, n_cond, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cond_mask, valid_mask], dim=1)
            attn_mask = ~full_mask
        else:
            attn_mask = None

        for layer in self.layers:
            x = layer(x.transpose(0, 1), src_key_padding_mask=attn_mask).transpose(0, 1)

        n_cond = len(emb_list)
        logits = self.output_proj(x[:, n_cond:])  # (B, L, num_classes)
        return logits


class ReMDM(nn.Module):
    def __init__(self, num_genes: int, num_classes: int, all_num_classes: Iterable[int], num_layers: int = 3, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.1, p_uncond: float = 0.2, device: str = 'cuda',
                 n_steps: int = 1000,
                 guidance_scale: float = 1.0,
                 batch_size: int = 512,
                 sigma_method: str = 'rescaled',
                 sigma_kwargs: Optional[Dict[str, Any]] = {'eta_rescale': 0.005},
                 repaint_num_iters: int = 1,
                 repaint_jump: int = 1):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = int(num_classes)
        self.mask_token = int(num_classes)  # dedicated mask token id
        self.model = ReMDMAttention(num_genes=num_genes, num_classes=num_classes, num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, all_num_classes=all_num_classes).to(self.device)
        self.p_uncond = float(p_uncond)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        self.step = 0
        # In-memory metrics that will be serialized in checkpoints
        self.metrics: Dict[str, List[Dict[str, Union[int, float]]]] = {
            'train': [],
            'scfid': [],
        }
        self.hparams: Dict[str, Union[int, float, str, List[int]]] = {
            'num_genes': int(num_genes),
            'num_classes': int(num_classes),
            'all_num_classes': list(all_num_classes) if isinstance(all_num_classes, (list, tuple)) else list(all_num_classes),
            'num_layers': int(num_layers),
            'embed_dim': int(embed_dim),
            'num_heads': int(num_heads),
            'dropout': float(dropout),
            'p_uncond': float(p_uncond),
            'device': str(self.device),
        }
        # Imputation parameters
        self.n_steps = int(n_steps)
        self.guidance_scale = float(guidance_scale)
        self.batch_size = int(batch_size)
        self.sigma_method = str(sigma_method)
        self.sigma_kwargs = sigma_kwargs if sigma_kwargs is not None else {}
        self.repaint_num_iters = int(repaint_num_iters)
        self.repaint_jump = int(repaint_jump)
        # scFID evaluator placeholder (initialized on first use)
        self.scfid_evaluator = None

    def _labels_to_device(self, labels):
        if labels is None:
            return None
        if isinstance(labels, (list, tuple)):
            return [x.to(self.device).long() for x in labels]
        return labels.to(self.device).long()

    def fit(
        self,
        train_loader,
        epochs: int = 1,
        checkpoint_path: Optional[str] = None,
        save_every: Optional[int] = None,
        keep_intermediate: bool = False,
        resume: bool = False,
        *,
        verbose: bool = False,
        log_every: int = 100,
        eval_every: Optional[int] = None,
        scfid_train_dataset=None,
        scfid_val_dataset=None,
        scfid_feature_model_path: Optional[str] = None,
        scfid_num_samples: int = 1000,
        scfid_n_steps: int = 50,
        scfid_guidance_scale: float = 1.0,
        scfid_eta_rescale: float = 0.005,
        scfid_repaint_num_iters: int = 1,
        scfid_repaint_jump: int = 1,
    ):
        self.train()
        if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
            try:
                self.load_checkpoint(checkpoint_path, load_optimizer=True)
            except Exception:
                pass
        for epoch in range(epochs):
            iterator = tqdm(
                train_loader,
                desc=f"ReMDM | Epoch {epoch+1}/{epochs}",
                unit="batch",
                leave=False,
            ) if verbose else train_loader
            for counts, labels, missing_mask in iterator:
                
                counts = counts.to(self.device)
                labels = [lbl.to(self.device).long() for lbl in labels]
                valid_mask = (~missing_mask).to(self.device).bool()

                # map to token ids
                x0 = counts.round().clamp(min=0).clamp(max=self.num_classes - 1).long()

                t = torch.rand(x0.shape[0], device=self.device)
                x_t = corrupt_ids(x0, t, mask_token=self.mask_token)

                self.optimizer.zero_grad()
                uncond_mask = (torch.rand(x0.shape[0], device=self.device) < self.p_uncond).bool()

                logits = self.model(x_t, t, class_labels=labels, valid_mask=valid_mask, uncond_mask=uncond_mask)
                # loss on valid positions only
                vm = valid_mask
                if vm.sum() == 0:
                    continue
                loss = F.cross_entropy(logits[vm], x0[vm])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.step += 1
                # Record train loss on log interval
                if (self.step % int(max(1, log_every)) == 0):
                    self.metrics.setdefault('train', []).append({
                        'step': int(self.step),
                        'loss_ce': float(loss.item()),
                    })
                if verbose and (self.step % int(max(1, log_every)) == 0):
                    print(f"[ReMDM] step {self.step} | train CE: {float(loss.item()):.4f}")
                if (
                    verbose
                    and eval_every is not None
                    and scfid_train_dataset is not None
                    and scfid_val_dataset is not None
                    and scfid_feature_model_path is not None
                    and (self.step % int(max(1, eval_every)) == 0)
                ):
                    fid = self.scfid_evaluate(
                        train_dataset=scfid_train_dataset,
                        val_dataset=scfid_val_dataset,
                        feature_model_path=scfid_feature_model_path,
                        num_samples=scfid_num_samples,
                        n_steps=scfid_n_steps,
                        guidance_scale=scfid_guidance_scale,
                        eta_rescale=scfid_eta_rescale,
                        repaint_num_iters=scfid_repaint_num_iters,
                        repaint_jump=scfid_repaint_jump,
                    )
                    # Record scFID metric
                    self.metrics.setdefault('scfid', []).append({
                        'step': int(self.step),
                        'scfid': float(fid),
                        'num_samples': int(scfid_num_samples),
                        'n_steps': int(scfid_n_steps),
                        'guidance_scale': float(scfid_guidance_scale),
                        'eta_rescale': float(scfid_eta_rescale),
                        'repaint_num_iters': int(scfid_repaint_num_iters),
                        'repaint_jump': int(scfid_repaint_jump),
                    })
                    print(f"[ReMDM] step {self.step} | scFID: {fid:.4f}")
                if checkpoint_path is not None and save_every is not None and self.step % int(save_every) == 0:
                    # Always update the main/latest checkpoint
                    self.save_checkpoint(checkpoint_path + 'latest.pth')
                    # Optionally keep an intermediate copy with step suffix
                    if keep_intermediate:
                        inter_path = f"{checkpoint_path}_step{self.step:010d}.pth"
                        self.save_checkpoint(inter_path)
            if checkpoint_path is not None and (save_every is None):
                # End-of-epoch save
                self.save_checkpoint(checkpoint_path)
                if keep_intermediate:
                    base, ext = os.path.splitext(checkpoint_path)
                    inter_path = f"{base}_epoch{epoch+1:04d}_step{self.step:010d}{ext or '.pth'}"
                    self.save_checkpoint(inter_path)

    @torch.no_grad()
    def impute(self, counts, impute_mask: torch.Tensor, labels=None, valid_mask: Optional[torch.Tensor] = None, n_steps: int = 100, guidance_scale: float = 1.0, sigma_method: str = 'rescaled', eta_rescale: float = 0.01, repaint_num_iters: int = 1, repaint_jump: int = 1, batch_size = 512) -> np.ndarray:
        self.eval()
        device = self.device
        if isinstance(counts, np.ndarray):
            counts = torch.from_numpy(counts)
        counts = counts.to(device).float()
        impute_mask = impute_mask.to(device).bool()
        valid_mask = torch.ones_like(impute_mask, dtype=torch.bool, device=device) if valid_mask is None else valid_mask.to(device).bool()
        labels = self._labels_to_device(labels)

        # Simple batching: process in chunks to save memory
        if batch_size is not None and counts.shape[0] > int(batch_size):
            outs: List[np.ndarray] = []
            B = counts.shape[0]
            for start in range(0, B, int(batch_size)):
                end = min(start + int(batch_size), B)
                counts_b = counts[start:end]
                impute_mask_b = impute_mask[start:end]
                valid_mask_b = valid_mask[start:end] if valid_mask is not None else None
                if labels is None:
                    labels_b = None
                elif isinstance(labels, list):
                    labels_b = [lbl[start:end] for lbl in labels]
                else:
                    labels_b = labels[start:end]
                out_b = self.impute(
                    counts=counts_b,
                    impute_mask=impute_mask_b,
                    labels=labels_b,
                    valid_mask=valid_mask_b,
                    n_steps=n_steps,
                    guidance_scale=guidance_scale,
                    sigma_method=sigma_method,
                    eta_rescale=eta_rescale,
                    repaint_num_iters=repaint_num_iters,
                    repaint_jump=repaint_jump,
                    batch_size=None,
                )
                outs.append(out_b if isinstance(out_b, np.ndarray) else out_b.detach().cpu().numpy())
            return np.concatenate(outs, axis=0)

        x0 = counts.round().clamp(min=0).clamp(max=self.num_classes - 1).long()
        # initialize state: masked unknowns, knowns forward-corrupted at t=1
        state = x0.clone()
        state[impute_mask] = self.mask_token
        if valid_mask is not None:
            state[~valid_mask] = self.mask_token

        # Same remask as countsdiff
        obs_times = torch.from_numpy(np.linspace(1, 0, n_steps).astype(np.float32)).to(device)
        if sigma_method == 'rescaled':
            sigmas = build_rescaled_sigma_schedule(obs_times, cosine_alpha, eta_rescale=eta_rescale, device=device)
        else:
            sigmas = [0.0 for _ in range(len(obs_times))]

        ts = obs_times[:-1]
        ss = obs_times[1:]
        num_steps = len(ts)
        num_groups = int(np.ceil(num_steps / repaint_jump))

        for step_group in range(num_groups):
            start = step_group * repaint_jump
            end = min(start + repaint_jump, num_steps)
            first_t = ts[start]
            current_s = ss[start]

            for it in range(repaint_num_iters):
                for j in range(start, end):
                    t = torch.full((state.shape[0],), ts[j], device=device)
                    s = torch.full((state.shape[0],), ss[j], device=device)
                    # guidance via discrete-CFG (log-prob interpolation)
                    full_uncond_mask = torch.ones(state.shape[0], dtype=torch.bool, device=device)
                    logits_u = self.model(state, t, class_labels=labels, valid_mask=valid_mask, uncond_mask=full_uncond_mask)
                    logits_c = self.model(state, t, class_labels=labels, valid_mask=valid_mask, uncond_mask=~full_uncond_mask)
                    logp_u = F.log_softmax(logits_u, dim=-1)
                    logp_c = F.log_softmax(logits_c, dim=-1)
                    logp = guidance_scale * logp_c + (1.0 - guidance_scale) * logp_u
                    probs = torch.exp(logp)  # (B, L, C)

                    # sample only for currently masked unknown positions
                    masked_unknown = (state == self.mask_token) & impute_mask & valid_mask
                    if masked_unknown.any():
                        B, L, C = probs.shape
                        # multinomial expects 2D; flatten masked positions
                        idxs = masked_unknown.nonzero(as_tuple=False)
                        p_sel = probs[idxs[:, 0], idxs[:, 1]]  # (N, C)
                        samples = torch.multinomial(p_sel, num_samples=1).squeeze(1)
                        state[idxs[:, 0], idxs[:, 1]] = samples

                    # remask a fraction sigma of unknown positions to allow refinement
                    sigma = sigmas[j]
                    if not torch.is_tensor(sigma):
                        sigma = torch.tensor(float(sigma), device=device)
                    if sigma.item() > 0:
                        remask = (torch.rand_like(state.float()) < sigma).bool() & impute_mask
                        state[remask] = self.mask_token

                    # project known region to forward-corrupted level s
                    proj_known = corrupt_ids(x0, torch.tensor(s, device=device), mask_token=self.mask_token)
                    state[~impute_mask] = proj_known[~impute_mask]
                    current_s = s

                if it < repaint_num_iters - 1:
                    # forward re-noise (s -> first_t) for all; then project known region at t
                    state = corrupt_ids(state, first_t, mask_token=self.mask_token, current_t=current_s)
                    proj_known_t = corrupt_ids(x0, first_t, mask_token=self.mask_token)
                    state[~impute_mask] = proj_known_t[~impute_mask]

        # final clean-up: any remaining masks in unknowns -> sample once from p(x0|state, t=0)
        leftover = (state == self.mask_token) & impute_mask & valid_mask
        if leftover.any():
            t0 = torch.zeros(state.shape[0], device=device)
            full_uncond_mask = torch.ones(state.shape[0], dtype=torch.bool, device=device)
            logits_u = self.model(state, t0, class_labels=labels, valid_mask=valid_mask, uncond_mask=full_uncond_mask)
            logits_c = self.model(state, t0, class_labels=labels, valid_mask=valid_mask, uncond_mask=~full_uncond_mask)
            logp = guidance_scale * F.log_softmax(logits_c, -1) + (1.0 - guidance_scale) * F.log_softmax(logits_u, -1)
            probs = torch.exp(logp)
            idxs = leftover.nonzero(as_tuple=False)
            p_sel = probs[idxs[:, 0], idxs[:, 1]]
            samples = torch.multinomial(p_sel, num_samples=1).squeeze(1)
            state[idxs[:, 0], idxs[:, 1]] = samples

        out = state.clone()
        out[out == self.mask_token] = 0  # just in case
        return out.float().detach().cpu().numpy()

    def update_hyperparameters(self, n_steps: Optional[int] = None, device: Optional[str] = None, remasking_prob: Optional[float] = None, guidance_scale: Optional[float] = None, batch_size: Optional[int] = None, sigma_method: Optional[str] = None, sigma_kwargs: Optional[Dict[str, Any]] = None, sigma_per_token: Optional[torch.Tensor] = None, repaint_num_iters: Optional[int] = None, repaint_jump: Optional[int] = None):
            if n_steps is not None:
                self.n_steps = n_steps
                print(f"Updated n_steps to {n_steps}")
            if device is not None:
                self.device = device
                print(f"Updated device to {device}")
            if remasking_prob is not None:
                self.remasking_prob = remasking_prob
                print(f"Updated remasking_prob to {remasking_prob}")
            if guidance_scale is not None:
                self.guidance_scale = guidance_scale
                print(f"Updated guidance_scale to {guidance_scale}")
            if batch_size is not None:
                self.batch_size = batch_size
                print(f"Updated batch_size to {batch_size}")
            if sigma_method is not None:
                self.sigma_method = sigma_method
                print(f"Updated sigma_method to {sigma_method}")
            if sigma_kwargs is not None:
                self.sigma_kwargs = sigma_kwargs
                print(f"Updated sigma_kwargs to {sigma_kwargs}")
            if sigma_per_token is not None:
                self.sigma_per_token = sigma_per_token
                print(f"Updated sigma_per_token to {sigma_per_token}")
            if repaint_num_iters is not None:
                self.repaint_num_iters = repaint_num_iters
                print(f"Updated repaint_num_iters to {repaint_num_iters}")
            if repaint_jump is not None:
                self.repaint_jump = repaint_jump
                print(f"Updated repaint_jump to {repaint_jump}")
                
    def impute_data(self, counts, impute_mask: torch.Tensor, labels=None, valid_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        return self.impute(
            counts=counts,
            impute_mask=impute_mask,
            labels=labels,
            valid_mask=valid_mask,
            n_steps=self.n_steps,
            guidance_scale=self.guidance_scale,
            sigma_method=self.sigma_method,
            eta_rescale=self.sigma_kwargs.get('eta_rescale', 0.005),
            repaint_num_iters=self.repaint_num_iters,
            repaint_jump=self.repaint_jump,
            batch_size=self.batch_size,
        )

    @torch.no_grad()
    def generate_samples(
        self,
        dataset,
        num_samples: int,
        n_steps: int = 50,
        guidance_scale: float = 1.0,
        labels: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        valid_mask: Optional[torch.Tensor] = None,
        sigma_method: str = 'rescaled',
        eta_rescale: float = 0.005,
        repaint_num_iters: int = 1,
        repaint_jump: int = 1,
        batch_size: Optional[int] = 256,
    ) -> np.ndarray:
        """
        Generate samples by imputing with everything masked out.
        If labels are None, draw them from the provided dataset like countsdiffGenerator.
        """
        device = self.device
        # Resolve labels from dataset if not provided
        if labels is None or valid_mask is None:
            from torch.utils.data import DataLoader
            if len(dataset) >= num_samples:
                dl = DataLoader(dataset, batch_size=num_samples, shuffle=True)
                batch = next(iter(dl))
                batch_labels = batch[1]
                missingness_mask = batch[2]
                if valid_mask is None:
                    valid_mask = ~missingness_mask.to(device).bool()
                if labels is None: 
                    if isinstance(batch_labels, list):
                        labels = [lbl.to(device) for lbl in batch_labels]
                    else:
                        labels = batch_labels.to(device)
            else:
                # bootstrap indices to reach num_samples
                dl = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
                batch = next(iter(dl))
                batch_labels = batch[1]
                import numpy as _np
                boot_idx = _np.random.choice(len(dataset), size=num_samples, replace=True)
                boot_idx = torch.from_numpy(boot_idx).long()
                if labels is None:
                    if isinstance(batch_labels, list):
                        labels = [lbl[boot_idx].to(device) for lbl in batch_labels]
                    else:
                        labels = batch_labels[boot_idx].to(device)
                if valid_mask is None:
                    missingness_mask = batch[2]
                    valid_mask = ~missingness_mask.to(device).bool()
                    valid_mask = valid_mask[boot_idx]
        else:
            # move provided labels to device
            if isinstance(labels, list):
                labels = [lbl.to(device) for lbl in labels]
            elif isinstance(labels, torch.Tensor):
                labels = labels.to(device)
        

        # Build empty counts and masks, optionally in batches
        G = self.model.num_genes
        if batch_size is None or num_samples <= int(batch_size):
            counts = torch.zeros((num_samples, G), device=device, dtype=torch.float32)
            impute_mask = torch.ones_like(counts, dtype=torch.bool)
            return self.impute(
                counts=counts,
                impute_mask=impute_mask,
                labels=labels,
                valid_mask=valid_mask,
                n_steps=n_steps,
                guidance_scale=guidance_scale,
                sigma_method=sigma_method,
                eta_rescale=eta_rescale,
                repaint_num_iters=repaint_num_iters,
                repaint_jump=repaint_jump,
                batch_size=batch_size,
            )
        else:
            outs: List[np.ndarray] = []
            for start in range(0, int(num_samples), int(batch_size)):
                end = min(start + int(batch_size), int(num_samples))
                bs = end - start
                counts_b = torch.zeros((bs, G), device=device, dtype=torch.float32)
                impute_mask_b = torch.ones_like(counts_b, dtype=torch.bool)
                if labels is None:
                    labels_b = None
                elif isinstance(labels, list):
                    labels_b = [lbl[start:end] for lbl in labels]
                else:
                    labels_b = labels[start:end]
                valid_mask_b = None if valid_mask is None else valid_mask[start:end]
                out_b = self.impute(
                    counts=counts_b,
                    impute_mask=impute_mask_b,
                    labels=labels_b,
                    valid_mask=valid_mask_b,
                    n_steps=n_steps,
                    guidance_scale=guidance_scale,
                    sigma_method=sigma_method,
                    eta_rescale=eta_rescale,
                    repaint_num_iters=repaint_num_iters,
                    repaint_jump=repaint_jump,
                    batch_size=None,
                )
                outs.append(out_b if isinstance(out_b, np.ndarray) else out_b.detach().cpu().numpy())
            return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def scfid_evaluate(
        self,
        train_dataset,
        val_dataset,
        feature_model_path: str,
        num_samples: int = 1000,
        n_steps: int = 50,
        guidance_scale: float = 1.0,
        eta_rescale: float = 0.005,
        repaint_num_iters: int = 1,
        repaint_jump: int = 1,
    ) -> float:
        """
        Compute scFID for ReMDM by sampling conditioned on labels from val_dataset
        and comparing to a batch of real train counts.
        """
        from torch.utils.data import DataLoader
        from countsdiff.utils.metrics import scFID

        device = self.device
        # 1) Prepare labels for generated samples (from val dataset)
        if len(val_dataset) >= num_samples:
            dl_val = DataLoader(val_dataset, batch_size=num_samples, shuffle=True)
            batch_val = next(iter(dl_val))
            val_labels = batch_val[1]
            if isinstance(val_labels, list):
                val_labels = [lbl.to(device) for lbl in val_labels]
            else:
                val_labels = val_labels.to(device)
        else:
            dl_val = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
            batch_val = next(iter(dl_val))
            val_labels_raw = batch_val[1]
            import numpy as _np
            boot_idx = _np.random.choice(len(val_dataset), size=num_samples, replace=True)
            boot_idx = torch.from_numpy(boot_idx).long()
            if isinstance(val_labels_raw, list):
                val_labels = [lbl[boot_idx].to(device) for lbl in val_labels_raw]
            else:
                val_labels = val_labels_raw[boot_idx].to(device)

        # 2) Generate samples (all imputed)
        gen = self.generate_samples(
            dataset=val_dataset,
            num_samples=num_samples,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            labels=val_labels,
            sigma_method='rescaled',
            eta_rescale=eta_rescale,
            repaint_num_iters=repaint_num_iters,
            repaint_jump=repaint_jump,
            batch_size=128,
        )
        gen = torch.from_numpy(gen) if not isinstance(gen, torch.Tensor) else gen

        # 3) Real train batch
        dl_train = DataLoader(
            train_dataset,
            batch_size=min(num_samples, len(train_dataset)),
            shuffle=False,
        )
        train_batch = next(iter(dl_train))
        train_counts = train_batch[0]
        train_labels = train_batch[1]

        # 4) Build scFID metric
        gene_names = getattr(val_dataset, 'gene_names', None)
        categorical_covariates = val_dataset.get_obs_dict(unique=True)
        metric = scFID(gene_names=gene_names, categorical_covariates=categorical_covariates, feature_model_path=feature_model_path)
        metric = metric.to(device).eval()
        metric.reset()

        # 5) Build covariate DataFrames using the training dataset mapping (like trainer)
        train_cov_df = train_dataset.build_covariate_df(train_labels)
        val_cov_df = train_dataset.build_covariate_df(val_labels)

        # 6) Update and compute
        metric.update(train_counts.cpu().numpy(), train_cov_df, True)
        metric.update(gen.detach().cpu().numpy(), val_cov_df, False)
        fid_score = float(metric.compute().item())
        metric.reset()
        return fid_score

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        payload = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'hparams': self.hparams,
            'mask_token': int(self.mask_token),
            'step': int(self.step),
            'metrics': self.metrics,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str, *, load_optimizer: bool = False, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'], strict=strict)
        if load_optimizer and 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass
        self.mask_token = int(ckpt.get('mask_token', self.mask_token))
        self.step = int(ckpt.get('step', 0))
        self.metrics = ckpt.get('metrics', {'train': [], 'scfid': []})

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cuda'):
        ckpt = torch.load(path, map_location=device if torch.cuda.is_available() else 'cpu')
        hp = ckpt['hparams']
        model = cls(
            num_genes=int(hp['num_genes']),
            num_classes=int(hp['num_classes']),
            all_num_classes=list(hp['all_num_classes']),
            num_layers=int(hp['num_layers']),
            embed_dim=int(hp['embed_dim']),
            num_heads=int(hp['num_heads']),
            dropout=float(hp['dropout']),
            p_uncond=float(hp.get('p_uncond', 0.1)),
            device=device,
        )
        model.model.load_state_dict(ckpt['state_dict'], strict=True)
        model.mask_token = int(ckpt.get('mask_token', model.mask_token))
        model.step = int(ckpt.get('step', 0))
        model.metrics = ckpt.get('metrics', {'train': [], 'scfid': []})
        try:
            model.optimizer.load_state_dict(ckpt['optimizer'])
        except Exception:
            pass
        return model
