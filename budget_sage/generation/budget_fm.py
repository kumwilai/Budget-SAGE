"""BudgetFMGenerator — a budget-conditioned flow-matching generator for
text-to-sign production.

The generator extends the existing UPC-anchored Ham2PoseV2 + flow-matching
backbone (Sec.~6 of the current draft, `outputs/ham2pose_v55_b1_upc_anchor`)
with three new components that let a *single* checkpoint serve the whole
budget-vs-provenance operating curve:

  1. A FiLM-modulated budget-and-time conditioning that injects the user-set
     scalar b ∈ [0,1] alongside the existing flow time τ ∈ [0,1] into every
     transformer block.
  2. A short retrieval-context encoder that embeds the top-1 reranker
     exemplar's UPC sequence u_ret into H_ret, plus per-block cross-attention
     gated by σ(W[b, log b, log(1-b)] + b_α). At b=0 the gate is trained to
     vanish (independent generation); at b=1 it dominates (retrieval-style).
  3. A budget-mixed flow-matching target  p*(b) = (1-b) y_indep + b y_ret
     that supervises the same θ to interpolate between the two endpoints
     in pose space along the flow trajectory.

The generator is intentionally drop-in compatible with the inference API of
the existing UPC-FM (`generate(upc, T_pose, n_steps)`); the new entry point
takes an extra `b` scalar and an optional `u_ret` UPC sequence.

This module is the architecture surface only. The training loop, losses, and
operating-curve evaluation live in:

  - budget_sage/generation/losses.py
  - budget_sage/generation/retrieval_context.py
  - scripts/train_budget_fm_phoenix.py
  - scripts/eval_budget_fm_curve.py

See `reports/budget_conditioned_fm_spec.md` for the full design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# UPC token specials — must match the existing CSL UPC-FM convention so we
# can warm-start from outputs/csl_upc_fm_generator/best.pt or the Phoenix
# v55 checkpoint without re-keying the embedding table.
# ----------------------------------------------------------------------
PAD, BOS, EOS = 0, 1, 2
UPC_OFFSET = 3
DEFAULT_NUM_UPC_VOCAB = 515  # K=512 + 3 specials


# ----------------------------------------------------------------------
# 1. Sinusoidal positional encoding
# ----------------------------------------------------------------------
def sinusoidal_pe(L: int, D: int) -> torch.Tensor:
    pe = torch.zeros(L, D)
    pos = torch.arange(0, L).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ----------------------------------------------------------------------
# 2. Budget-and-time FiLM head
# ----------------------------------------------------------------------
class BudgetTimeFiLM(nn.Module):
    """Per-layer (γ, β) generator that conditions a hidden state on the
    flow-matching time τ and the budget b."""

    def __init__(self, d_model: int, n_layers: int, hidden: int = 128):
        super().__init__()
        in_dim = 6  # τ, b, sin(2πb), cos(2πb), τ², b·τ
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * d_model * n_layers),
        )
        self.n_layers = n_layers
        self.d_model = d_model

    def forward(self, tau: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """tau, b: (B,) tensors in [0, 1]. Returns (B, n_layers, 2, d_model)."""
        feats = torch.stack(
            [tau, b, torch.sin(2 * math.pi * b), torch.cos(2 * math.pi * b),
             tau * tau, b * tau], dim=-1)
        out = self.mlp(feats)
        return out.view(-1, self.n_layers, 2, self.d_model)


def apply_film(x: torch.Tensor, film_pair: torch.Tensor) -> torch.Tensor:
    """x: (B, T, D); film_pair: (B, 2, D). γ(x) + β, with γ=1+δγ around 0."""
    gamma = 1.0 + film_pair[:, 0].unsqueeze(1)   # (B,1,D)
    beta  = film_pair[:, 1].unsqueeze(1)
    return gamma * x + beta


# ----------------------------------------------------------------------
# 3. Retrieval-context encoder + per-block gate
# ----------------------------------------------------------------------
class RetrievalContextEncoder(nn.Module):
    """Embeds the top-1 reranker exemplar's UPC sequence into H_ret. Two
    pre-norm transformer encoder layers; shares the UPC embedding with the
    main anchor encoder via the constructor's `tok_embed` argument."""

    def __init__(self, tok_embed: nn.Embedding, d_model: int = 256,
                 n_heads: int = 4, ff: int = 1024, n_layers: int = 2,
                 dropout: float = 0.1, max_T: int = 128):
        super().__init__()
        self.tok_embed = tok_embed   # shared with anchor encoder
        self.register_buffer("pe", sinusoidal_pe(max_T, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, u_ret: torch.Tensor) -> torch.Tensor:
        """u_ret: (B, L_r) padded UPC tokens. Returns H_ret: (B, L_r, D),
        plus a key-padding mask."""
        B, L = u_ret.shape
        x = self.tok_embed(u_ret) + self.pe[:L].unsqueeze(0)
        kpm = u_ret == PAD
        return self.transformer(x, src_key_padding_mask=kpm), kpm


class BudgetGate(nn.Module):
    """Scalar gate α(b) = σ(W[b, log b, log(1-b)] + b_α). Initialised so the
    gate is *near-linear* across [0,1] (not saturated), so gradients near
    the endpoints are non-vanishing. Concretely, with the default init the
    pre-activation moves over [-3, +3] across b ∈ [0, 1], giving
    α(0) ≈ 0.05, α(1) ≈ 0.95."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)
        # Init: pre-activation = 6·b − 3 (note: log-features get tiny weight).
        # At b=0:  6·0 + 0.05·log(eps) − 0.05·log(1) − 3 ≈ −3 + 0.05·(−9.2)
        #       ≈ −3.46 → σ ≈ 0.030  (close to spec'd 0.05)
        # At b=1:  6·1 + 0.05·log(1) − 0.05·log(eps) − 3 ≈ +3 + 0.46
        #       ≈ +3.46 → σ ≈ 0.970  (close to spec'd 0.95)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor([[6.0, 0.05, -0.05]]))
            self.lin.bias.fill_(-3.0)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """b: (B,) or (B,T) in [0,1]. Returns α: (B, 1, 1) or (B, T, 1).

        The per-frame form is used when the trainer hands us a structure-aware
        budget vector (Phase 2C) — one budget per output frame so the model can
        learn to retrieve hand-stable plateaus and generate stroke peaks.
        """
        eps = 1e-4
        feats = torch.stack([b,
                              torch.log(b.clamp_min(eps)),
                              torch.log((1 - b).clamp_min(eps))], dim=-1)
        out = torch.sigmoid(self.lin(feats))
        if b.dim() == 1:
            return out.view(-1, 1, 1)
        if b.dim() == 2:
            return out                                     # (B, T, 1)
        raise ValueError(f"BudgetGate expects b of dim 1 or 2, got {b.dim()}")


# ----------------------------------------------------------------------
# 4. The main Budget-FM denoiser block (one transformer layer)
# ----------------------------------------------------------------------
class BudgetFMBlock(nn.Module):
    """One FM-decoder block: pre-norm self-attention, FiLM(τ,b), cross-attn
    to anchor (existing), gated cross-attn to retrieval context (new), FFN.
    """

    def __init__(self, d_model: int = 384, n_heads: int = 6, ff: int = 1536,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.x_attn_anc = nn.MultiheadAttention(d_model, n_heads,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.x_attn_ret = nn.MultiheadAttention(d_model, n_heads,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.norm4 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                H_anc: torch.Tensor, H_anc_kpm: torch.Tensor,
                H_ret: Optional[torch.Tensor],
                H_ret_kpm: Optional[torch.Tensor],
                film_pair: torch.Tensor,
                ret_alpha: torch.Tensor) -> torch.Tensor:
        # 1. self-attention
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        # 2. FiLM(τ, b) before cross-attention
        x = apply_film(x, film_pair)
        # 3. cross-attention to anchor
        h = self.norm2(x)
        a, _ = self.x_attn_anc(h, H_anc, H_anc, key_padding_mask=H_anc_kpm,
                                need_weights=False)
        x = x + self.drop(a)
        # 4. cross-attention to retrieval context (gated by α(b)).
        #
        # Bug guard: if every key for some sample is masked out
        # (i.e. H_ret_kpm row all-True), PyTorch MultiheadAttention silently
        # returns NaN for that row. We detect that case and zero out the
        # corresponding rows' contribution.
        if H_ret is not None:
            h = self.norm3(x)
            if H_ret_kpm is not None:
                # all_pad_row[b] = True iff every key in sample b is padding
                all_pad_row = H_ret_kpm.all(dim=-1)        # (B,)
                if all_pad_row.any():
                    # Make a single sentinel key visible for those rows so
                    # attention is well-defined; we zero out their gate after.
                    H_ret_kpm = H_ret_kpm.clone()
                    H_ret_kpm[all_pad_row, 0] = False
            a, _ = self.x_attn_ret(h, H_ret, H_ret, key_padding_mask=H_ret_kpm,
                                    need_weights=False)
            if H_ret_kpm is not None and 'all_pad_row' in dir() and all_pad_row.any():
                a = a.clone()
                a[all_pad_row] = 0.0
            x = x + self.drop(ret_alpha * a)
        # 5. FFN
        h = self.norm4(x)
        x = x + self.drop(self.ffn(h))
        return x


# ----------------------------------------------------------------------
# 5. The full BudgetFMGenerator
# ----------------------------------------------------------------------
@dataclass
class BudgetFMConfig:
    num_upc: int = DEFAULT_NUM_UPC_VOCAB
    pose_dim: int = 201
    d_model: int = 384
    n_heads: int = 6
    ff: int = 1536
    n_enc: int = 4               # anchor encoder layers
    n_dec: int = 6               # FM denoiser layers
    n_ret_enc: int = 2           # retrieval context encoder layers
    dropout: float = 0.1
    max_T_upc: int = 128
    max_T_pose: int = 256
    max_T_ret: int = 128


class BudgetFMGenerator(nn.Module):
    """Budget-conditioned flow-matching generator. See
    `reports/budget_conditioned_fm_spec.md` Sec. 2 for the design."""

    def __init__(self, cfg: BudgetFMConfig = BudgetFMConfig()):
        super().__init__()
        self.cfg = cfg
        # Token embedding (shared with retrieval encoder)
        self.tok_embed = nn.Embedding(cfg.num_upc, cfg.d_model, padding_idx=PAD)
        self.register_buffer("pe_anc", sinusoidal_pe(cfg.max_T_upc, cfg.d_model))
        self.register_buffer("pe_pose", sinusoidal_pe(cfg.max_T_pose, cfg.d_model))
        # Anchor encoder (UPC sequence → H_anc)
        anc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.anc_encoder = nn.TransformerEncoder(anc_layer, num_layers=cfg.n_enc)
        # Retrieval-context encoder (UPC sequence → H_ret)
        self.ret_encoder = RetrievalContextEncoder(
            self.tok_embed, d_model=cfg.d_model, n_heads=cfg.n_heads,
            ff=cfg.ff, n_layers=cfg.n_ret_enc, dropout=cfg.dropout,
            max_T=cfg.max_T_ret)
        # FM denoiser stack
        self.dec_blocks = nn.ModuleList([
            BudgetFMBlock(d_model=cfg.d_model, n_heads=cfg.n_heads,
                           ff=cfg.ff, dropout=cfg.dropout)
            for _ in range(cfg.n_dec)
        ])
        # Pose I/O projections
        self.pose_in_proj  = nn.Linear(cfg.pose_dim, cfg.d_model)
        self.pose_out_proj = nn.Linear(cfg.d_model, cfg.pose_dim)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        # Conditioning heads
        self.film = BudgetTimeFiLM(cfg.d_model, n_layers=cfg.n_dec)
        self.gate = BudgetGate()
        # Per-frame budget injection: small MLP from [b(t), log b(t), log(1-b(t))]
        # to d_model, added to the pose embedding before the decoder stack so
        # the network sees b(t) locally. Initialised to near-zero so a scalar
        # warm-start checkpoint produces approximately the same forward pass.
        self.b_frame_proj = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, cfg.d_model),
        )
        with torch.no_grad():
            self.b_frame_proj[-1].weight.zero_()
            self.b_frame_proj[-1].bias.zero_()

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------
    def encode_anc(self, u_anc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """u_anc: (B, L_a) padded UPC tokens. Returns H_anc, key-pad-mask."""
        B, L = u_anc.shape
        x = self.tok_embed(u_anc) + self.pe_anc[:L].unsqueeze(0)
        kpm = u_anc == PAD
        return self.anc_encoder(x, src_key_padding_mask=kpm), kpm

    def encode_ret(self, u_ret: Optional[torch.Tensor]
                    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if u_ret is None:
            return None, None
        return self.ret_encoder(u_ret)

    # ------------------------------------------------------------------
    # Forward (single FM denoiser pass)
    # ------------------------------------------------------------------
    def forward(self, x_tau: torch.Tensor, tau: torch.Tensor, b: torch.Tensor,
                H_anc: torch.Tensor, H_anc_kpm: torch.Tensor,
                H_ret: Optional[torch.Tensor] = None,
                H_ret_kpm: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict the conditional velocity v_θ(x_τ, τ, H_anc, H_ret, b).

        Args:
            x_tau     : (B, T, pose_dim) noised pose latent.
            tau       : (B,) flow time in [0, 1].
            b         : (B,) scalar budget OR (B, T) per-frame budget in [0, 1].
            H_anc     : (B, L_a, d_model) anchor representation.
            H_anc_kpm : (B, L_a) key-padding mask.
            H_ret     : optional (B, L_r, d_model) retrieval context.
            H_ret_kpm : optional (B, L_r) key-padding mask.
        Returns:
            v_pred    : (B, T, pose_dim) predicted velocity field.
        """
        B, T, _ = x_tau.shape
        h = self.pose_in_proj(x_tau) + self.pe_pose[:T].unsqueeze(0)
        # Promote scalar b to per-frame for unified downstream handling.
        if b.dim() == 1:
            b_frame = b.unsqueeze(1).expand(B, T)         # (B, T)
        elif b.dim() == 2:
            if b.size(1) != T:
                raise ValueError(
                    f"per-frame b has T={b.size(1)} but pose has T={T}")
            b_frame = b
        else:
            raise ValueError(f"b must be (B,) or (B,T), got {tuple(b.shape)}")
        b_bar = b_frame.mean(dim=-1)                      # (B,) for FiLM
        # Per-frame additive injection: encodes b(t), log b(t), log(1-b(t))
        eps = 1e-4
        b_feats = torch.stack([
            b_frame,
            torch.log(b_frame.clamp_min(eps)),
            torch.log((1.0 - b_frame).clamp_min(eps)),
        ], dim=-1)                                        # (B, T, 3)
        h = h + self.b_frame_proj(b_feats)                # zero-init residual
        film_grid = self.film(tau, b_bar)                 # (B, n_dec, 2, D)
        ret_alpha = self.gate(b_frame)                    # (B, T, 1)
        for ell, block in enumerate(self.dec_blocks):
            h = block(
                h,
                H_anc=H_anc, H_anc_kpm=H_anc_kpm,
                H_ret=H_ret, H_ret_kpm=H_ret_kpm,
                film_pair=film_grid[:, ell],
                ret_alpha=ret_alpha,
            )
        return self.pose_out_proj(self.out_norm(h))

    # ------------------------------------------------------------------
    # Inference: 10-step Euler ODE
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, u_anc: torch.Tensor, b, T_pose: int,
                  u_ret: Optional[torch.Tensor] = None,
                  n_steps: int = 10) -> torch.Tensor:
        """A single forward integration.

        Args:
            u_anc : (1, L_a) padded UPC tokens.
            b     : scalar in [0,1] OR (1, T_pose) per-frame budget tensor.
            T_pose: target span (overridden if b is a per-frame tensor).
            u_ret : optional retrieval UPC sequence.
        Returns:
            (1, T_pose, pose_dim).
        """
        device = u_anc.device
        H_anc, H_anc_kpm = self.encode_anc(u_anc)
        H_ret, H_ret_kpm = self.encode_ret(u_ret)
        if isinstance(b, torch.Tensor):
            b_t = b.to(device)
            if b_t.dim() == 1 and b_t.numel() == T_pose:    # per-frame, no batch
                b_t = b_t.unsqueeze(0)                      # (1, T)
            if b_t.dim() == 2:
                T_pose = b_t.size(1)
        else:
            b_t = torch.full((1,), float(b), device=device)
        x = torch.randn(1, T_pose, self.cfg.pose_dim, device=device) * 0.5
        d_tau = 1.0 / max(1, n_steps)
        for k in range(n_steps):
            tau = torch.full((1,), (k + 0.5) * d_tau, device=device)
            v   = self.forward(x, tau, b_t,
                                H_anc=H_anc, H_anc_kpm=H_anc_kpm,
                                H_ret=H_ret, H_ret_kpm=H_ret_kpm)
            x = x + d_tau * v
        return x


__all__ = [
    "BudgetFMConfig", "BudgetFMGenerator",
    "BudgetTimeFiLM", "RetrievalContextEncoder", "BudgetGate", "BudgetFMBlock",
    "PAD", "BOS", "EOS", "UPC_OFFSET", "DEFAULT_NUM_UPC_VOCAB",
]
