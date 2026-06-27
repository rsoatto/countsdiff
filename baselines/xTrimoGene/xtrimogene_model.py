from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn


MASK_TOKEN_VALUE = -1.0
PAD_TOKEN_VALUE = 103.0


def _softmax_kernel(
    data: torch.Tensor,
    *,
    projection_matrix: torch.Tensor,
    is_query: bool,
    eps: float = 1e-4,
) -> torch.Tensor:
    bsz, n_heads, _, dim_head = data.shape
    data_normalizer = dim_head ** -0.25
    ratio = projection_matrix.shape[0] ** -0.5

    projection = projection_matrix.to(device=data.device, dtype=data.dtype)
    projection = projection.unsqueeze(0).unsqueeze(0).expand(bsz, n_heads, -1, -1)
    data_dash = torch.einsum("bhnd,bhmd->bhnm", data_normalizer * data, projection)

    diag_data = (data.square().sum(dim=-1, keepdim=True) / 2.0) * (data_normalizer ** 2)
    if is_query:
        data_dash = data_dash - diag_data - data_dash.max(dim=-1, keepdim=True).values
    else:
        data_dash = data_dash - diag_data - data_dash.amax(dim=(-1, -2), keepdim=True)
    return ratio * (torch.exp(data_dash) + eps)


def _linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    k_cumsum = k.sum(dim=-2)
    d_inv = 1.0 / torch.einsum("bhnd,bhd->bhn", q, k_cumsum.type_as(q)).clamp(min=1e-6)
    context = torch.einsum("bhnd,bhne->bhde", k, v)
    return torch.einsum("bhde,bhnd,bhn->bhne", context, q, d_inv)


class FastAttention(nn.Module):
    def __init__(self, dim_head: int, nb_features: int | None = None) -> None:
        super().__init__()
        nb_features = nb_features or max(1, int(dim_head * math.log(max(dim_head, 2))))
        self.dim_head = dim_head
        self.nb_features = nb_features
        projection_matrix = self._create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

    def _create_projection(self) -> torch.Tensor:
        blocks = []
        n_full = self.nb_features // self.dim_head
        for _ in range(n_full):
            q, _ = torch.linalg.qr(torch.randn(self.dim_head, self.dim_head), mode="reduced")
            blocks.append(q.t())
        remainder = self.nb_features - n_full * self.dim_head
        if remainder > 0:
            q, _ = torch.linalg.qr(torch.randn(self.dim_head, self.dim_head), mode="reduced")
            blocks.append(q.t()[:remainder])
        if blocks:
            matrix = torch.cat(blocks, dim=0)
        else:
            matrix = torch.randn(self.nb_features, self.dim_head)
        multiplier = torch.randn(self.nb_features, self.dim_head).norm(dim=1)
        return torch.diag(multiplier) @ matrix

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = _softmax_kernel(q, projection_matrix=self.projection_matrix, is_query=True)
        k = _softmax_kernel(k, projection_matrix=self.projection_matrix, is_query=False)
        return _linear_attention(q, k, v)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int = 64,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.fast_attention = FastAttention(dim_head)
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, *, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.to_q(x).view(batch, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(batch, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(batch, seq_len, self.heads, self.dim_head).transpose(1, 2)

        if padding_mask is not None:
            keep_mask = (~padding_mask).unsqueeze(1).unsqueeze(-1)
            q = q.masked_fill(~keep_mask, 0.0)
            k = k.masked_fill(~keep_mask, 0.0)
            v = v.masked_fill(~keep_mask, 0.0)

        out = self.fast_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.heads * self.dim_head)
        return self.dropout(self.to_out(out))


class PreLayerNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class PerformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = PreLayerNorm(
            dim,
            SelfAttention(dim, heads=heads, dim_head=dim_head, attn_dropout=attn_dropout),
        )
        self.ff = PreLayerNorm(dim, FeedForward(dim, mult=ff_mult, dropout=ff_dropout))

    def forward(self, x: torch.Tensor, *, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(x, padding_mask=padding_mask)
        x = x + self.ff(x)
        return x


class PerformerModule(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = 64,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList(
            [
                PerformerLayer(
                    dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f"sequence length {x.shape[1]} exceeds max_seq_len={self.max_seq_len}")
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return self.norm(x)


class PytorchTransformerModule(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        ff_mult: int = 4,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=dim * ff_mult,
                    batch_first=True,
                    norm_first=norm_first,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f"sequence length {x.shape[1]} exceeds max_seq_len={self.max_seq_len}")
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        *,
        bin_num: int,
        bin_alpha: float,
        mask_token_id: float | None = None,
        pad_token_id: float | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        self.register_buffer("bin_num_idx", torch.arange(self.bin_num, dtype=torch.long))

    def forward(self, x: torch.Tensor, output_weight: int = 0) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        values = x.squeeze(-1)
        x_mask_idx = (values == self.mask_token_id).nonzero(as_tuple=False)
        x_pad_idx = (values == self.pad_token_id).nonzero(as_tuple=False)

        x = self.mlp(x)
        x = self.leaky_relu(x)
        x_crosslayer = self.mlp2(x)
        x = self.bin_alpha * x + x_crosslayer
        weight = self.softmax(x)

        token_emb = self.emb(self.bin_num_idx.to(device=x.device))
        x = torch.matmul(weight, token_emb)

        zero_idx = torch.zeros((), dtype=torch.long, device=x.device)
        if x_mask_idx.numel() > 0:
            mask_token_emb = self.emb_mask(zero_idx).to(dtype=x.dtype)
            x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = mask_token_emb
        if x_pad_idx.numel() > 0:
            pad_token_emb = self.emb_pad(zero_idx).to(dtype=x.dtype)
            x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = pad_token_emb

        if output_weight:
            return x, weight
        return x


class MaeAutobin(nn.Module):
    def __init__(
        self,
        *,
        num_tokens: int,
        max_seq_len: int,
        embed_dim: int,
        decoder_embed_dim: int,
        bin_alpha: float = 1.0,
        bin_num: int = 100,
        pad_token_id: float | None = None,
        mask_token_id: float | None = None,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        self.token_emb = AutoDiscretizationEmbedding2(
            embed_dim,
            max_seq_len,
            bin_num=bin_num,
            bin_alpha=bin_alpha,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
        )
        self.pos_emb = nn.Embedding(max_seq_len + 1, embed_dim)
        self.encoder: nn.Module | None = None
        self.decoder: nn.Module | None = None
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        padding_label: torch.Tensor,
        encoder_position_gene_ids: torch.Tensor,
        encoder_labels: torch.Tensor,
        decoder_data: torch.Tensor,
        mask_gene_name: bool,
        mask_labels: torch.Tensor | None,
        decoder_position_gene_ids: torch.Tensor,
        decoder_data_padding_labels: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[1] > self.max_seq_len:
            raise ValueError(f"sequence length {x.shape[1]} exceeds max_seq_len={self.max_seq_len}")
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Encoder and decoder must be attached before use.")

        x = self.token_emb(x.unsqueeze(2), output_weight=0)
        x = x + self.pos_emb(encoder_position_gene_ids)
        x = self.encoder(x, padding_mask=padding_label)

        decoder_data = self.token_emb(decoder_data.unsqueeze(2))
        decoder_data = decoder_data + self.pos_emb(decoder_position_gene_ids)
        if mask_gene_name:
            raise NotImplementedError("Gene-name masking is not implemented for this baseline.")

        batch_idx, gene_idx = encoder_labels.nonzero(as_tuple=True)
        decoder_data[batch_idx, gene_idx] = x[~padding_label].to(decoder_data.dtype)

        decoder_data = self.decoder_embed(decoder_data)
        x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)
        x = self.norm(x)
        x = self.to_final(x)
        return x.squeeze(2)


def gather_data(data: torch.Tensor, labels: torch.Tensor, pad_token_id: float) -> Tuple[torch.Tensor, torch.Tensor]:
    value_nums = labels.sum(1)
    max_num = int(value_nums.max().item()) if value_nums.numel() > 0 else 0
    max_num = max(max_num, 1)

    fake_data = torch.full((data.shape[0], max_num), pad_token_id, device=data.device, dtype=data.dtype)
    data = torch.hstack([data, fake_data])

    fake_label = torch.ones((labels.shape[0], max_num), device=labels.device, dtype=torch.float32)
    none_labels = ~labels
    scored = labels.float()
    scored[none_labels] = -float("inf")

    tmp_data = torch.arange(labels.shape[1], 0, -1, device=labels.device, dtype=torch.float32) * 20000
    scored = scored + tmp_data
    scored = torch.hstack([scored, fake_label])

    gather_idx = scored.topk(max_num, dim=1).indices
    new_data = torch.gather(data, 1, gather_idx)
    padding_labels = new_data == pad_token_id
    return new_data, padding_labels


def get_encoder_decoder_data(
    data: torch.Tensor,
    data_raw: torch.Tensor,
    config: Dict[str, object],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor]:
    decoder_data = data.clone().detach()
    decoder_data_padding = torch.full_like(data, False, dtype=torch.bool, device=data.device)

    encoder_data_labels = data_raw > 0
    encoder_data, encoder_data_padding = gather_data(decoder_data, encoder_data_labels, float(config["pad_token_id"]))
    new_data_raw = data_raw
    data_gene_ids = torch.arange(data.shape[1], device=data.device, dtype=torch.long).repeat(data.shape[0], 1)
    encoder_position_gene_ids, _ = gather_data(data_gene_ids, encoder_data_labels, int(config["seq_len"]))
    decoder_position_gene_ids = data_gene_ids
    data_mask_labels = None

    encoder_position_gene_ids[encoder_data_padding] = int(config["seq_len"])
    decoder_position_gene_ids[decoder_data_padding] = int(config["seq_len"])

    return (
        encoder_data,
        encoder_position_gene_ids,
        encoder_data_padding,
        encoder_data_labels,
        decoder_data,
        decoder_data_padding,
        new_data_raw,
        data_mask_labels,
        decoder_position_gene_ids,
    )


def build_xtrimogene_3m_config(num_genes: int) -> Dict[str, object]:
    return {
        "model": "mae_autobin",
        "n_class": 1,
        "seq_len": int(num_genes),
        "bin_alpha": 1.0,
        "bin_num": 100,
        "pad_token_id": float(PAD_TOKEN_VALUE),
        "mask_token_id": float(MASK_TOKEN_VALUE),
        "encoder": {
            "module_type": "performer",
            "hidden_dim": 128,
            "depth": 4,
            "heads": 2,
            "dim_head": 64,
            "ff_mult": 4,
            "ff_dropout": 0.0,
            "attn_dropout": 0.0,
        },
        "decoder": {
            "module_type": "transformer",
            "hidden_dim": 128,
            "depth": 2,
            "heads": 2,
            "ff_mult": 4,
        },
    }


def build_model_from_config(config: Dict[str, object]) -> MaeAutobin:
    encoder_cfg = dict(config["encoder"])
    decoder_cfg = dict(config["decoder"])

    model = MaeAutobin(
        num_tokens=int(config["n_class"]),
        max_seq_len=int(config["seq_len"]),
        embed_dim=int(encoder_cfg["hidden_dim"]),
        decoder_embed_dim=int(decoder_cfg["hidden_dim"]),
        bin_alpha=float(config["bin_alpha"]),
        bin_num=int(config["bin_num"]),
        pad_token_id=float(config["pad_token_id"]),
        mask_token_id=float(config["mask_token_id"]),
    )
    model.encoder = PerformerModule(
        max_seq_len=int(config["seq_len"]),
        dim=int(encoder_cfg["hidden_dim"]),
        depth=int(encoder_cfg["depth"]),
        heads=int(encoder_cfg["heads"]),
        dim_head=int(encoder_cfg.get("dim_head", 64)),
        ff_mult=int(encoder_cfg.get("ff_mult", 4)),
        ff_dropout=float(encoder_cfg.get("ff_dropout", 0.0)),
        attn_dropout=float(encoder_cfg.get("attn_dropout", 0.0)),
    )
    model.decoder = PytorchTransformerModule(
        max_seq_len=int(config["seq_len"]),
        dim=int(decoder_cfg["hidden_dim"]),
        depth=int(decoder_cfg["depth"]),
        heads=int(decoder_cfg["heads"]),
        ff_mult=int(decoder_cfg.get("ff_mult", 4)),
    )
    return model


def build_xtrimogene_3m_model(num_genes: int) -> Tuple[MaeAutobin, Dict[str, object]]:
    config = build_xtrimogene_3m_config(num_genes)
    model = build_model_from_config(config)
    return model, config
