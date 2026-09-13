"""Detector v2 and embedding v2 for the keyed pose watermark.

Built alongside scripts/pose_watermark.py, which is NOT modified: the July
scheme stays byte-reproducible. This module supplies

  * detector_v2 -- one detector serving the July banks (chip length L=2,
    rectangular, uniform amplitude) and the v2 banks (L=4, Hann-COLA hands,
    per-channel amplitude). Blind hypothesis search over effective chip length
    and phase, dead-channel exclusion, and a search-multiplicity-absorbing null
    built from M wrong keys.
  * embed_v2 -- the v2 embedding.

Layout (scripts/train_sign_jepa_ae_fullclip.py lines 42-55):
  body 0:8, hands 8:50, face 50:178, of 178 joints x 3 axes = 534 channels.
"""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import numpy as np
import torch

from scripts.pose_watermark import _highpass, _payload_bits

N_JOINTS, N_AXES = 178, 3
N_CHAN = N_JOINTS * N_AXES
BODY, HAND, FACE = (0, 8), (8, 50), (50, 178)

# --- hypothesis grid (frozen in the preregistration) -------------------------
F_GRID = np.geomspace(0.4, 2.5, 63)          # 3% geometric, 63 values
PHASE_FRACS = (0.0, 0.25, 0.5, 0.75)         # 4 phases -> 252 hypotheses
N_BLOCKS = 16
N_NULL_KEYS = 32
Z_KEY_THRESHOLD = 8.0
SIG_CLAMP = 1e-5


# --------------------------------------------------------------- chip stream
def chip_base(key: str, sid: str, n: int) -> np.ndarray:
    """Antipodal chip stream [n, 534] from SHA-256(key||sid).

    Drawn row-major, so base[:m] is the identical prefix for any n >= m; this
    is what makes the blind chip-index search valid on a shortened stream.
    """
    seed = int.from_bytes(hashlib.sha256(f"{key}||{sid}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 2, size=(n, N_JOINTS, N_AXES)).astype(np.float32) * 2.0 - 1.0
    return b.reshape(n, N_CHAN)


def null_key(key: str, i: int) -> str:
    return hashlib.sha256(f"{key}||null||{i}".encode()).hexdigest()


# ------------------------------------------------------------- hypothesis plan
@lru_cache(maxsize=512)
def _plan(T: int, L: float):
    """Per-f groups of (C, boundaries[4,C+1], blk[4,C], nb[4,16]); cached by (T,L)."""
    groups = []
    cmax = 0
    for f in F_GRID:
        Lp = float(f) * L
        Cs, bnds, blks, nbs = [], [], [], []
        for pf in PHASE_FRACS:
            phi = pf * Lp
            chip_of_frame = np.floor((np.arange(T) + phi) / Lp).astype(np.int64)
            C = int(chip_of_frame[-1]) + 1
            Cs.append(C)
            # boundary b_i = first frame of chip i (chip indices are monotone)
            b = np.searchsorted(chip_of_frame, np.arange(C + 1), side="left")
            b[-1] = T
            bnds.append(b)
            blk = np.minimum((np.arange(C) * N_BLOCKS) // max(1, C), N_BLOCKS - 1)
            blks.append(blk)
            nbs.append(np.bincount(blk[chip_of_frame], minlength=N_BLOCKS).astype(np.float64))
        Cg = max(Cs)
        cmax = max(cmax, Cg)
        B = np.zeros((4, Cg + 1), dtype=np.int64)
        K = np.zeros((4, Cg), dtype=np.int64)
        for p in range(4):
            B[p, :Cs[p] + 1] = bnds[p]
            B[p, Cs[p] + 1:] = T
            K[p, :Cs[p]] = blks[p]
        groups.append((Cg, torch.from_numpy(B), torch.from_numpy(K),
                       torch.from_numpy(np.stack(nbs))))
    return groups, cmax


# ------------------------------------------------------------------- detector
def _front_end(pose: torch.Tensor):
    x = pose.float()
    T = x.shape[0]
    r = _highpass(x).reshape(T, N_CHAN)
    sig = r.std(dim=0)
    # dead-channel rule: >50% of the high-pass residual samples exactly zero,
    # or the channel's std hits the sig.clamp_min floor (degenerate constant
    # residual). Neither fires on unquantised float32 motion.
    dead = ((r == 0).float().mean(dim=0) > 0.5) | (sig <= SIG_CLAMP)
    w = torch.where(dead, torch.zeros_like(sig), 1.0 / sig.clamp_min(SIG_CLAMP) ** 2)
    return r * w, float(w.sum()), int(dead.sum())


def _stat_all_keys(Rw: torch.Tensor, var_frame: float, T: int, L: float,
                   bases: torch.Tensor):
    """Max-over-hypotheses folded statistic for every key in `bases`.

    bases: [K, Cmax, 534]. Returns (zmax [K], best_bits [K,16], best_h [K]).
    """
    K = bases.shape[0]
    cum = torch.cat([torch.zeros(1, N_CHAN), Rw.cumsum(0)], dim=0)
    groups, _ = _plan(T, L)
    best = torch.full((K,), -float("inf"))
    best_bits = torch.zeros((K, N_BLOCKS), dtype=torch.int64)
    best_h = torch.zeros((K,), dtype=torch.int64)
    hid = 0
    Bt = bases.permute(1, 2, 0).contiguous()      # [Cmax, 534, K]
    for (Cg, bnd, blk, nb) in groups:
        A = cum[bnd[:, 1:Cg + 1]] - cum[bnd[:, :Cg]]      # [4, Cg, 534]
        c = torch.bmm(A.permute(1, 0, 2).contiguous(), Bt[:Cg])   # [Cg,4,K]
        # per-phase block accumulation (phases have their own chip->block map)
        cb = torch.zeros(N_BLOCKS, 4, K)
        for p in range(4):
            cb[:, p, :].index_add_(0, blk[p], c[:, p, :])
        vb = var_frame * nb                                  # [4,16]
        mu0 = (math.sqrt(2.0 / math.pi) * torch.sqrt(vb)).sum(dim=1)          # [4]
        var0 = ((1.0 - 2.0 / math.pi) * vb).sum(dim=1)                        # [4]
        z = (cb.abs().sum(dim=0) - mu0[:, None]) / (torch.sqrt(var0)[:, None] + 1e-9)  # [4,K]
        for p in range(4):
            upd = z[p] > best
            if bool(upd.any()):
                best = torch.where(upd, z[p], best)
                bits = (cb[:, p, :] > 0).long().T                             # [K,16]
                best_bits = torch.where(upd[:, None], bits, best_bits)
                best_h = torch.where(upd, torch.full_like(best_h, hid + p), best_h)
        hid += 4
    return best, best_bits, best_h


def detector_v2(pose: torch.Tensor, key: str, sid: str, L: float,
                base_cache: dict | None = None, want_analytic: bool = False) -> dict:
    """Blind detection. Returns z_key (primary), the max statistic, payload bits."""
    T = int(pose.shape[0])
    Rw, var_frame, n_dead = _front_end(pose)
    _, cmax = _plan(T, L)
    keys = [key] + [null_key(key, i) for i in range(N_NULL_KEYS)]
    bases = _get_bases(keys, sid, cmax, base_cache)
    zmax, bits, bh = _stat_all_keys(Rw, var_frame, T, L, bases)
    s_true = float(zmax[0])
    s_null = zmax[1:].numpy().astype(np.float64)
    mu, sd = float(s_null.mean()), float(s_null.std(ddof=1))
    out = {
        "z_key": float((s_true - mu) / max(sd, 1e-9)),
        "s_max": s_true,
        "null_mean": mu, "null_std": sd, "null_max": float(s_null.max()),
        "bits": bits[0].tolist(), "best_hyp": int(bh[0]), "n_dead": n_dead,
    }
    if want_analytic:
        out["z_analytic_identity"] = _analytic_identity(Rw, var_frame, T, L, sid, key)
    return out


def _analytic_identity(Rw, var_frame, T, L, sid, key) -> float:
    """July's analytic z with the dead-channel fix, identity hypothesis only."""
    chip = (np.arange(T) / L).astype(np.int64)
    C = int(chip[-1]) + 1
    s = torch.from_numpy(chip_base(key, sid, C)[chip])
    pf = (Rw * s).sum(dim=1)
    block_of = np.minimum((np.arange(T) * N_BLOCKS) // max(1, T), N_BLOCKS - 1)
    a = m = v = 0.0
    for b in range(N_BLOCKS):
        msk = torch.from_numpy(block_of == b)
        nf = float(msk.sum())
        if nf == 0:
            continue
        a += abs(float(pf[msk].sum()))
        vb = var_frame * nf
        m += math.sqrt(2.0 / math.pi) * math.sqrt(vb)
        v += (1.0 - 2.0 / math.pi) * vb
    return float((a - m) / (math.sqrt(v) + 1e-9))


def _get_bases(keys: list[str], sid: str, n: int, cache: dict | None) -> torch.Tensor:
    ck = (keys[0], sid)
    if cache is not None and ck in cache and cache[ck].shape[1] >= n:
        return cache[ck][:, :n, :]
    n_alloc = max(n, 64)
    arr = torch.from_numpy(np.stack([chip_base(k, sid, n_alloc) for k in keys]))
    if cache is not None:
        if len(cache) > 6:
            cache.clear()
        cache[ck] = arr
    return arr[:, :n, :]


# ------------------------------------------------------------------ embedding
def hann_cola(L: int) -> np.ndarray:
    """2L-long periodic Hann; hop L gives exact constant overlap-add."""
    n = np.arange(2 * L)
    return (np.sin(np.pi * n / (2 * L)) ** 2).astype(np.float32)


def channel_amplitude(pose: torch.Tensor, alpha: float) -> torch.Tensor:
    """a = alpha * sigma_channel, with alpha_face = alpha, alpha_hand = alpha_body = alpha/2."""
    sig = _highpass(pose.float()).std(dim=0)                 # [178,3]
    scale = torch.full((N_JOINTS, 1), 0.5)
    scale[FACE[0]:FACE[1]] = 1.0
    return alpha * sig * scale


def embed_v2(pose: torch.Tensor, key: str, sid: str, payload_hex: str,
             alpha: float, L: int = 4, n_blocks: int = N_BLOCKS) -> torch.Tensor:
    x = pose.clone().float()
    T = x.shape[0]
    C = int(math.ceil(T / L))
    base = chip_base(key, sid, C + 1).reshape(C + 1, N_JOINTS, N_AXES)
    bits = _payload_bits(payload_hex, n_blocks)
    blk = np.minimum((np.arange(C + 1) * n_blocks) // max(1, C), n_blocks - 1)
    chip = base * bits[blk][:, None, None]                   # [C+1,178,3]
    t = np.arange(T)
    # rectangular chips (body + face)
    s_rect = chip[np.minimum(t // L, C)]                     # [T,178,3]
    # Hann-COLA chips on the same grid (hands): two overlapping contributions
    w = hann_cola(L)
    i0 = (t + L // 2) // L
    n0 = t - i0 * L + L // 2
    s_h = np.zeros_like(s_rect)
    for i, n in ((i0, n0), (i0 - 1, n0 + L)):
        ok = (i >= 0) & (i <= C)
        s_h[ok] += chip[np.clip(i, 0, C)][ok] * w[n][ok, None, None]
    s = torch.from_numpy(s_rect)
    s[:, HAND[0]:HAND[1]] = torch.from_numpy(s_h[:, HAND[0]:HAND[1]])
    return x + channel_amplitude(x, alpha).unsqueeze(0) * s
