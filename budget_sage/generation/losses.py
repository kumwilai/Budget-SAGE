"""Losses for Budget-Conditioned Flow Matching.

Core terms (Sec. 3 of `reports/budget_conditioned_fm_spec.md`):

    L = L_FM(b) + λ_rr · L_RR + λ_ac · (1 − b) · L_AC
        + λ_kin · L_KIN + λ_pref · (1 − b) · L_PREF

L_FM(b)   — flow-matching MSE on the budget-mixed velocity field.
L_RR      — recognizer-readability CTC loss; OPTIONAL (default off for the
            first training run; provide a frozen recognizer to enable).
L_AC      — anti-copy: minus the resampled-MJE distance to a random
            subsample of the train pose pool, weighted by (1−b) so it is
            active at low budgets and inactive at b=1.
L_KIN     — articulation-preserving kinematic profile matching.  It matches
            speed, acceleration, jerk, and hand-spread statistics to the
            current budget target instead of simply smoothing motion.
L_PREF    — counterfactual preference margin: the proxy output must be closer
            to the feasible source-budget target than to the restricted
            source-heavy endpoint.

The FM target is the budget-mixed pose
    p*(b)  =  (1−b) · y_indep  +  b · y_ret
where y_indep is the GT pose for the input clip (supervises the b=0
endpoint) and y_ret is the retrieved exemplar pose (supervises the b=1
endpoint). Both are linearly time-resampled to the same span length T
*before* mixing, so the mixture is well-defined frame-by-frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Pose alignment helper (linear time-resampling)
# ---------------------------------------------------------------------------

def resample_T(p: torch.Tensor, T: int) -> torch.Tensor:
    """p: (B, T_src, D). Returns (B, T, D) linearly time-resampled."""
    B, T_src, D = p.shape
    if T_src == T:
        return p
    if T_src < 2:
        return p.expand(B, T, D)
    idx = torch.linspace(0, T_src - 1, T, device=p.device)
    i0 = idx.floor().long().clamp(0, T_src - 1)
    i1 = (i0 + 1).clamp(0, T_src - 1)
    w = (idx - i0.float()).unsqueeze(0).unsqueeze(-1)
    return (1.0 - w) * p[:, i0] + w * p[:, i1]


# ---------------------------------------------------------------------------
# 2. Flow-matching loss with budget-mixed target
# ---------------------------------------------------------------------------

def fm_loss(model_v_pred: torch.Tensor,
            x0: torch.Tensor,
            p_star: torch.Tensor) -> torch.Tensor:
    """v_θ(x_τ, τ, …) is supervised on the displacement (p* − x0).

    Args:
        model_v_pred: (B, T, D) predicted velocity field.
        x0          : (B, T, D) Gaussian noise sample.
        p_star      : (B, T, D) budget-mixed target pose.
    Returns:
        Scalar MSE loss.
    """
    target_field = p_star - x0
    return F.mse_loss(model_v_pred, target_field)


def budget_mixed_target(y_indep: torch.Tensor,
                         y_ret: torch.Tensor,
                         b: torch.Tensor,
                         T: int) -> torch.Tensor:
    """Build p*(b) = (1−b) y_indep + b y_ret, both pre-aligned to length T.

    b: (B,) scalar OR (B, T) per-frame, in [0, 1].
    y_indep, y_ret: (B, T_*, D) (any T_*; we resample)."""
    yi = resample_T(y_indep, T)
    yr = resample_T(y_ret, T)
    if b.dim() == 1:
        bb = b.view(-1, 1, 1)                              # (B, 1, 1)
    elif b.dim() == 2:
        if b.size(1) != T:
            raise ValueError(
                f"per-frame b has T={b.size(1)} but target T={T}")
        bb = b.unsqueeze(-1)                               # (B, T, 1)
    else:
        raise ValueError(f"b must be (B,) or (B,T), got {tuple(b.shape)}")
    return (1.0 - bb) * yi + bb * yr


# ---------------------------------------------------------------------------
# 3. Anti-copy loss
# ---------------------------------------------------------------------------

def fast_pose_distance_batch(p_pred: torch.Tensor,
                              train_pool: torch.Tensor) -> torch.Tensor:
    """Resampled mean joint distance from each predicted clip to every train
    clip in the subsample pool.

    Args:
        p_pred    : (B, T, D) predicted poses.
        train_pool: (P, T, D) train pose subsample (already resampled to T).
    Returns:
        (B, P) tensor of root-mean-square pose distances.
    """
    diff = p_pred.unsqueeze(1) - train_pool.unsqueeze(0)   # (B, P, T, D)
    return diff.pow(2).mean(dim=(-1, -2)).sqrt()


def anti_copy_loss(p_pred: torch.Tensor,
                    train_pool: torch.Tensor,
                    weights: Optional[torch.Tensor] = None,
                    scale: float = 80.0,
                    saturate_at: float = 2.0) -> torch.Tensor:
    """L_AC = − Σ_i  w_i · min(d_i / scale, saturate_at) / Σ_i w_i.

    The minus sign turns this into "maximise distance to nearest train clip".
    `scale` (pixel units) sets the unit of nominal MJE; `saturate_at`
    bounds how far the loss can pull the generator from the train pool, so
    L_AC cannot dominate L_FM by emitting absurdly far-from-data poses.

    `weights` is an optional per-sample weighting tensor (B,); use this to
    apply the (1−b) per-sample weighting from the spec. If `weights` is
    None, the loss is uniformly weighted.
    """
    d = fast_pose_distance_batch(p_pred, train_pool)        # (B, P)
    nn_d = d.min(dim=-1).values                              # (B,)
    saturated = torch.clamp(nn_d / scale, max=saturate_at)   # (B,)
    if weights is None:
        return -saturated.mean()
    w = weights.to(saturated)
    eps = 1e-6
    return -(w * saturated).sum() / w.sum().clamp_min(eps)


# ---------------------------------------------------------------------------
# 4. Kinematic profile matching
# ---------------------------------------------------------------------------

def _finite_diff(x: torch.Tensor, order: int) -> torch.Tensor:
    out = x
    for _ in range(order):
        if out.size(1) < 2:
            return out.new_zeros(out.size(0), 1, out.size(2))
        out = out[:, 1:] - out[:, :-1]
    return out


def _rms_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean(dim=(1, 2)).clamp_min(1e-12).sqrt()


def _safe_log_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.log(pred.clamp_min(1e-6)), torch.log(target.clamp_min(1e-6)))


def kinematic_profile_loss(pred_pose: torch.Tensor,
                           target_pose: torch.Tensor,
                           hand_start_dim: int = 75,
                           hand_only: bool = True,
                           w_speed: float = 1.0,
                           w_accel: float = 0.5,
                           w_jerk: float = 0.5,
                           w_spread: float = 0.25) -> torch.Tensor:
    """Match phase-space motion statistics to the current budget target.

    The loss is intentionally *profile matching*, not smoothing.  Minimising
    jerk alone can produce under-articulated signing; here low jerk is penalised
    when the target trajectory has sharper transitions.  For PHOENIX PT-201 the
    manual hand channels start after 25 body joints (25*3 = 75 dims).
    """

    if hand_only and pred_pose.size(-1) > hand_start_dim:
        pred = pred_pose[..., hand_start_dim:]
        target = target_pose[..., hand_start_dim:]
    else:
        pred = pred_pose
        target = target_pose

    total = pred.new_zeros(())
    if w_speed:
        total = total + float(w_speed) * _safe_log_mse(
            _rms_per_sample(_finite_diff(pred, 1)),
            _rms_per_sample(_finite_diff(target, 1)),
        )
    if w_accel:
        total = total + float(w_accel) * _safe_log_mse(
            _rms_per_sample(_finite_diff(pred, 2)),
            _rms_per_sample(_finite_diff(target, 2)),
        )
    if w_jerk:
        total = total + float(w_jerk) * _safe_log_mse(
            _rms_per_sample(_finite_diff(pred, 3)),
            _rms_per_sample(_finite_diff(target, 3)),
        )
    if w_spread:
        pred_spread = pred.std(dim=1).mean(dim=1).clamp_min(1e-6)
        target_spread = target.std(dim=1).mean(dim=1).clamp_min(1e-6)
        total = total + float(w_spread) * _safe_log_mse(pred_spread, target_spread)
    return total


# ---------------------------------------------------------------------------
# 5. Counterfactual preference loss
# ---------------------------------------------------------------------------

def counterfactual_preference_loss(pred_pose: torch.Tensor,
                                   positive_pose: torch.Tensor,
                                   negative_pose: torch.Tensor,
                                   margin: float = 0.05,
                                   weights: Optional[torch.Tensor] = None,
                                   hand_start_dim: int = 75,
                                   hand_only: bool = True) -> torch.Tensor:
    """Prefer the feasible source-budget target over a counterfactual source.

    For low-source or source-dropped training, the positive target is the
    independent/generated endpoint and the negative target is the retrieval
    exemplar.  For high-replay budgets this naturally weakens through the
    caller-provided weights.  This converts the flow model from ordinary
    interpolation into a source-replacement model: the generated proxy must be
    closer to the feasible target than to the restricted source-heavy target by
    a margin.
    """
    if hand_only and pred_pose.size(-1) > hand_start_dim:
        pred = pred_pose[..., hand_start_dim:]
        pos = positive_pose[..., hand_start_dim:]
        neg = negative_pose[..., hand_start_dim:]
    else:
        pred, pos, neg = pred_pose, positive_pose, negative_pose

    d_pos = _rms_per_sample(pred - pos)
    d_neg = _rms_per_sample(pred - neg)
    loss = F.relu(d_pos - d_neg + float(margin))
    if weights is not None:
        w = weights.to(loss).flatten()
        return (w * loss).sum() / w.sum().clamp_min(1e-6)
    return loss.mean()


# ---------------------------------------------------------------------------
# 6. Recognizer-readability loss (OPTIONAL — wrap any frozen CTC head)
# ---------------------------------------------------------------------------

class RecognizerReadabilityHead(nn.Module):
    """Adapter around a frozen pose→gloss-CTC recognizer.

    Subclasses or callers must implement `.ctc_log_probs(pose) -> (B, T, V)`
    and provide a gloss tokenizer that produces (B, U) integer tensors plus
    valid lengths.

    This module exposes a single `loss(pred_pose, gloss_ids, gloss_lengths)`
    method that returns the CTC negative log-likelihood. The gradient flows
    through `pred_pose`; the recognizer's own parameters are frozen.
    """

    def __init__(self, recognizer_fn: callable, blank_id: int = 0):
        super().__init__()
        self.recognizer_fn = recognizer_fn
        self.blank_id = blank_id

    def forward(self, pred_pose: torch.Tensor,
                gloss_ids: torch.Tensor,
                gloss_lengths: torch.Tensor) -> torch.Tensor:
        log_probs = self.recognizer_fn(pred_pose)            # (B, T, V)
        T = log_probs.size(1)
        input_lengths = torch.full((log_probs.size(0),), T,
                                    dtype=torch.long, device=log_probs.device)
        return F.ctc_loss(log_probs.transpose(0, 1), gloss_ids,
                           input_lengths, gloss_lengths,
                           blank=self.blank_id, zero_infinity=True,
                           reduction='mean')


# ---------------------------------------------------------------------------
# 6. Composite loss config
# ---------------------------------------------------------------------------

@dataclass
class BudgetFMLossConfig:
    lambda_rr: float = 0.0          # 0.0 disables RR
    lambda_ac: float = 0.05
    lambda_kin: float = 0.0
    lambda_pref: float = 0.0
    ac_subsample: int = 64
    ac_scale: float = 80.0
    ac_saturate_at: float = 2.0     # bounds the maximum reward for "far from train"
    kin_hand_only: bool = True
    kin_hand_start_dim: int = 75
    kin_speed: float = 1.0
    kin_accel: float = 0.5
    kin_jerk: float = 0.5
    kin_spread: float = 0.25
    pref_margin: float = 0.05
    pref_hand_only: bool = True


def composite_loss(model_v_pred: torch.Tensor,
                    x0: torch.Tensor,
                    p_star: torch.Tensor,
                    pred_pose_proxy: Optional[torch.Tensor],
                    train_subsample: Optional[torch.Tensor],
                    b: torch.Tensor,
                    p_negative: Optional[torch.Tensor] = None,
                    rr_head: Optional[RecognizerReadabilityHead] = None,
                    gloss_ids: Optional[torch.Tensor] = None,
                    gloss_lengths: Optional[torch.Tensor] = None,
                    cfg: BudgetFMLossConfig = BudgetFMLossConfig()
                    ) -> dict[str, torch.Tensor]:
    """Compute the full budget-FM loss and its component breakdown.

    Args:
        model_v_pred  : (B, T, D) velocity output of BudgetFMGenerator.
        x0, p_star    : (B, T, D) noise sample and budget-mixed target.
        pred_pose_proxy: (B, T, D) one-step Euler proxy x_τ + (1−τ)·v_pred.
        train_subsample: (P, T, D) per-batch random subsample of train poses
                         (already resampled to T) for the anti-copy loss.
        b             : (B,) budget scalars in [0, 1].
        rr_head       : optional RecognizerReadabilityHead for L_RR.
        gloss_ids,gloss_lengths: optional gloss reference for L_RR.
        cfg           : BudgetFMLossConfig.
    Returns:
        A dict with `total`, `fm`, `ac`, `rr`, `kin`, `pref`
        (zero when disabled).
    """
    fm = fm_loss(model_v_pred, x0, p_star)

    if cfg.lambda_ac > 0 and pred_pose_proxy is not None and train_subsample is not None:
        # Per-sample (1-b) weighting baked into anti_copy_loss so the spec
        # "L_AC vanishes at b=1, full strength at b=0" holds per-sample.
        # Anti-copy is a *clip-level* property, so when b is per-frame we
        # collapse to its mean before weighting.
        b_clip = b if b.dim() == 1 else b.mean(dim=-1)
        ac = anti_copy_loss(pred_pose_proxy, train_subsample,
                             weights=(1.0 - b_clip),
                             scale=cfg.ac_scale,
                             saturate_at=cfg.ac_saturate_at)
    else:
        ac = torch.zeros((), device=fm.device, dtype=fm.dtype)

    if cfg.lambda_rr > 0 and rr_head is not None and pred_pose_proxy is not None:
        rr = rr_head(pred_pose_proxy, gloss_ids, gloss_lengths)
    else:
        rr = torch.zeros((), device=fm.device, dtype=fm.dtype)

    if cfg.lambda_kin > 0 and pred_pose_proxy is not None:
        kin = kinematic_profile_loss(
            pred_pose_proxy,
            p_star,
            hand_start_dim=cfg.kin_hand_start_dim,
            hand_only=cfg.kin_hand_only,
            w_speed=cfg.kin_speed,
            w_accel=cfg.kin_accel,
            w_jerk=cfg.kin_jerk,
            w_spread=cfg.kin_spread,
        )
    else:
        kin = torch.zeros((), device=fm.device, dtype=fm.dtype)

    if cfg.lambda_pref > 0 and pred_pose_proxy is not None and p_negative is not None:
        b_clip = b if b.dim() == 1 else b.mean(dim=-1)
        pref = counterfactual_preference_loss(
            pred_pose_proxy,
            p_star,
            p_negative,
            margin=cfg.pref_margin,
            weights=(1.0 - b_clip),
            hand_start_dim=cfg.kin_hand_start_dim,
            hand_only=cfg.pref_hand_only,
        )
    else:
        pref = torch.zeros((), device=fm.device, dtype=fm.dtype)

    total = (fm + cfg.lambda_rr * rr + cfg.lambda_ac * ac
             + cfg.lambda_kin * kin + cfg.lambda_pref * pref)
    return {"total": total, "fm": fm, "ac": ac, "rr": rr,
            "kin": kin, "pref": pref}


__all__ = [
    "resample_T",
    "fm_loss", "budget_mixed_target",
    "fast_pose_distance_batch", "anti_copy_loss",
    "kinematic_profile_loss", "counterfactual_preference_loss",
    "RecognizerReadabilityHead",
    "BudgetFMLossConfig", "composite_loss",
]
