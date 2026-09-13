"""Fallback-conditioned flow refiner for PHOENIX PT-201 poses.

This module implements the generative idea we want to test after observing
that an unconditional UPC-FM branch is not strong enough by itself.  Instead
of sampling a pose from pure noise, the model starts from a real fallback pose
such as PG-RAST++ and learns a conditional flow from that fallback trajectory
to the ground-truth pose:

    x_0 = y_fb + sigma eps,    x_tau = (1 - tau) x_0 + tau y_gt,
    v*(x_tau, tau) = y_gt - x_0.

The fallback pose is not merely an inference initializer.  It is an explicit
conditioning input, so the model can preserve recognisable timing and hand
structure while correcting systematic fallback errors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from budget_sage.generation.budget_fm import (
    DEFAULT_NUM_UPC_VOCAB,
    PAD,
    sinusoidal_pe,
)


@dataclass
class FallbackRefinerConfig:
    pose_dim: int = 201
    num_upc: int = DEFAULT_NUM_UPC_VOCAB
    d_model: int = 256
    n_heads: int = 4
    ff: int = 1024
    n_pose_layers: int = 4
    n_upc_layers: int = 2
    dropout: float = 0.05
    max_T_pose: int = 128
    max_T_upc: int = 128


def _time_features(tau: torch.Tensor) -> torch.Tensor:
    tau = tau.clamp(0.0, 1.0)
    return torch.stack(
        [
            tau,
            tau * tau,
            torch.sin(2.0 * math.pi * tau),
            torch.cos(2.0 * math.pi * tau),
        ],
        dim=-1,
    )


class ResidualConditionBlock(nn.Module):
    """Transformer block with fallback and UPC cross-attention.

    The block keeps the architecture intentionally compact.  The fallback
    stream is frame-aligned with the noised pose, while the UPC stream gives
    sentence-level symbolic context.  This is enough for a first controlled
    experiment and can be scaled later without changing the training target.
    """

    def __init__(self, d_model: int, n_heads: int, ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.fb_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.upc_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        h_fb: torch.Tensor,
        h_upc: torch.Tensor,
        upc_kpm: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.drop(a)

        h = self.norm2(x)
        a, _ = self.fb_attn(h, h_fb, h_fb, need_weights=False)
        x = x + self.drop(a)

        h = self.norm3(x)
        a, _ = self.upc_attn(h, h_upc, h_upc, key_padding_mask=upc_kpm, need_weights=False)
        x = x + self.drop(a)

        h = self.norm4(x)
        return x + self.drop(self.ffn(h))


class FallbackConditionedRefiner(nn.Module):
    """Predicts the velocity from a noised fallback pose to a target pose."""

    def __init__(self, cfg: FallbackRefinerConfig = FallbackRefinerConfig()):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.num_upc, cfg.d_model, padding_idx=PAD)
        self.register_buffer("pose_pe", sinusoidal_pe(cfg.max_T_pose, cfg.d_model))
        self.register_buffer("upc_pe", sinusoidal_pe(cfg.max_T_upc, cfg.d_model))

        upc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.upc_encoder = nn.TransformerEncoder(upc_layer, num_layers=cfg.n_upc_layers)

        self.x_proj = nn.Linear(cfg.pose_dim, cfg.d_model)
        self.fb_proj = nn.Linear(cfg.pose_dim, cfg.d_model)
        self.delta_proj = nn.Linear(cfg.pose_dim, cfg.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(4, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualConditionBlock(cfg.d_model, cfg.n_heads, cfg.ff, cfg.dropout)
                for _ in range(cfg.n_pose_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.pose_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode_upc(self, upc_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = upc_ids.shape
        if L > self.cfg.max_T_upc:
            raise ValueError(f"UPC length {L} exceeds max_T_upc={self.cfg.max_T_upc}")
        kpm = upc_ids == PAD
        h = self.tok_embed(upc_ids) + self.upc_pe[:L].unsqueeze(0)
        return self.upc_encoder(h, src_key_padding_mask=kpm), kpm

    def forward(
        self,
        x_tau: torch.Tensor,
        fallback_pose: torch.Tensor,
        tau: torch.Tensor,
        upc_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x_tau.shape
        if D != self.cfg.pose_dim:
            raise ValueError(f"pose dim {D} != configured {self.cfg.pose_dim}")
        if T > self.cfg.max_T_pose:
            raise ValueError(f"pose length {T} exceeds max_T_pose={self.cfg.max_T_pose}")
        if fallback_pose.shape != x_tau.shape:
            raise ValueError("fallback_pose must have the same shape as x_tau")

        h_upc, upc_kpm = self.encode_upc(upc_ids)
        h_fb = self.fb_proj(fallback_pose) + self.pose_pe[:T].unsqueeze(0)
        h = (
            self.x_proj(x_tau)
            + self.delta_proj(x_tau - fallback_pose)
            + self.pose_pe[:T].unsqueeze(0)
            + self.time_mlp(_time_features(tau)).unsqueeze(1)
        )
        for block in self.blocks:
            h = block(h, h_fb, h_upc, upc_kpm)
        return self.out(self.out_norm(h))

    @torch.no_grad()
    def refine(
        self,
        fallback_pose: torch.Tensor,
        upc_ids: torch.Tensor,
        n_steps: int = 12,
        noise_scale: float = 0.05,
    ) -> torch.Tensor:
        """Integrate the learned flow from a fallback pose to a refined pose."""
        if fallback_pose.dim() != 3:
            raise ValueError("fallback_pose must be (B,T,D)")
        x = fallback_pose
        if noise_scale > 0:
            x = x + torch.randn_like(x) * float(noise_scale)
        dt = 1.0 / float(max(1, n_steps))
        for k in range(max(1, n_steps)):
            tau = torch.full(
                (x.shape[0],),
                (k + 0.5) * dt,
                dtype=x.dtype,
                device=x.device,
            )
            x = x + dt * self.forward(x, fallback_pose, tau, upc_ids)
        return x


def make_fallback_flow_batch(
    y_gt: torch.Tensor,
    y_fb: torch.Tensor,
    noise_scale: float = 0.05,
    tau_min: float = 0.0,
    tau_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one fallback-conditioned flow-matching training batch."""
    if y_gt.shape != y_fb.shape:
        raise ValueError("y_gt and y_fb must have the same shape")
    B = y_gt.shape[0]
    tau = torch.empty(B, device=y_gt.device, dtype=y_gt.dtype).uniform_(tau_min, tau_max)
    x0 = y_fb
    if noise_scale > 0:
        x0 = x0 + torch.randn_like(y_fb) * float(noise_scale)
    x_tau = (1.0 - tau[:, None, None]) * x0 + tau[:, None, None] * y_gt
    target = y_gt - x0
    return x_tau, target, tau


def endpoint_losses(pred: torch.Tensor, target: torch.Tensor, y_gt: torch.Tensor, y_fb: torch.Tensor) -> dict:
    """Return diagnostic losses for logs."""
    fm = F.mse_loss(pred, target)
    fb_mse = F.mse_loss(y_fb, y_gt)
    residual_mse = F.mse_loss(y_fb + pred, y_gt)
    return {"fm": fm, "fallback_mse": fb_mse.detach(), "one_step_mse": residual_mse.detach()}


__all__ = [
    "FallbackRefinerConfig",
    "FallbackConditionedRefiner",
    "make_fallback_flow_batch",
    "endpoint_losses",
]
