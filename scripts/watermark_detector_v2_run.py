"""Phase 1 (dev pilot) and Phase 2 (test, once) for pose-watermark detector v2.

Preregistration: outputs/watermark_detector_v2_20260911/
preregistration_detector_v2_20260911_frozen.md
sha256 c204848a8918b246687fc0750ba169077726b6ae77680c191db6af2fd3361b9b

scripts/pose_watermark.py is imported unmodified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path("/home/kumwilai/research/signgen-t2m")
sys.path.insert(0, str(ROOT))

from scripts.pose_watermark import _highpass, _resample, as_pose, embed, payload_hex_for  # noqa: E402
from scripts.pose_watermark_v2 import (  # noqa: E402
    N_BLOCKS, Z_KEY_THRESHOLD, _get_bases, _plan, detector_v2, embed_v2, null_key,
)
from scripts.watermark_generative_route_20260911 import (  # noqa: E402
    dev_key, key_id, load_bank, motion_ratios, sha256_file, true_bits,
)
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import beta  # noqa: E402

OUT = ROOT / "outputs/watermark_detector_v2_20260911"
PREREG = "c204848a8918b246687fc0750ba169077726b6ae77680c191db6af2fd3361b9b"
DEV_BANK = ROOT / "outputs/revision/generative_route_restore_20260911/dev80_mt5gloss.pt"
TEST_BANK = ROOT / "outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt"
JULY_WM_25 = ROOT / "outputs/watermark_generative_route_20260911/test641_mt5gloss_wm_eps2.5e-4.pt"
JULY_WM_1E3 = ROOT / "outputs/watermark_generative_route_20260911/test641_mt5gloss_wm.pt"
GT = {"dev": ROOT / ("external/SLRTP-Sign-Production-Evaluation/pretrained/"
                     "SLRTP-Sign-Production-Evaluation-Data/data/dev.pt"),
      "test": ROOT / ("external/SLRTP-Sign-Production-Evaluation/pretrained/"
                      "SLRTP-Sign-Production-Evaluation-Data/data/test.pt")}


# ------------------------------------------------------------------ transforms
def _rng(name: str, sid: str) -> np.random.Generator:
    h = hashlib.sha256(f"0||{name}||{sid}".encode()).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))


def movavg(p: torch.Tensor, k: int) -> torch.Tensor:
    return p - _highpass(p, k=k)


def face_drop(p: torch.Tensor) -> torch.Tensor:
    q = p.clone()
    q[:, 50:178] = 0.0
    return q


def transform_list() -> dict[str, Callable[[torch.Tensor, str], torch.Tensor]]:
    t: dict[str, Callable] = {"identity": lambda p, s: p}
    for g in (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2):
        t[f"noise_{g:g}"] = (lambda g_: lambda p, s: p + torch.from_numpy(
            _rng(f"noise_{g_:g}", s).normal(0, g_, size=tuple(p.shape)).astype(np.float32)))(g)
    for k in (3, 5, 7, 9, 11, 15):
        t[f"smooth_k{k}"] = (lambda k_: lambda p, s: movavg(p, k_))(k)
    for f in (0.5, 0.75, 0.9, 1.1, 1.25, 1.5):
        t[f"blind_resample_{f:g}x"] = (lambda f_: lambda p, s: _resample(
            p, max(2, int(round(p.shape[0] * f_)))))(f)
    for f in (0.75, 1.25):
        t[f"certguided_resample_{f:g}x"] = (lambda f_: lambda p, s: _resample(
            _resample(p, max(2, int(round(p.shape[0] * f_)))), p.shape[0]))(f)
    for q in (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2):
        t[f"quantize_{q:g}"] = (lambda q_: lambda p, s: torch.round(p / q_) * q_)(q)
    t["evaluator_decimate2"] = lambda p, s: p[::2].contiguous()
    t["decimate2_odd"] = lambda p, s: p[1::2].contiguous()
    t["face_drop"] = lambda p, s: face_drop(p)
    return t


# ------------------------------------------------------------------- statistics
def cp_lower(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    if k >= n:
        return float(0.025 ** (1.0 / n))
    if k == 0:
        return 0.0
    return float(beta.ppf(0.025, k, n - k + 1))


def cp_upper(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    if k == 0:
        return float(1.0 - 0.025 ** (1.0 / n))
    if k >= n:
        return 1.0
    return float(beta.ppf(0.975, k + 1, n - k))


def empty_blocks(T: int, L: float, hyp: int) -> np.ndarray:
    groups, _ = _plan(T, L)
    g, p = hyp // 4, hyp % 4
    return (groups[g][3][p].numpy() == 0)


# ------------------------------------------------------------------ embedding
def make_marked(bank: dict[str, torch.Tensor], cfg: dict, key: str,
                bank_name: str) -> dict[str, torch.Tensor]:
    out = {}
    for sid, p in bank.items():
        ph = payload_hex_for({"sid": sid, "route": "generated", "bank": bank_name})
        if cfg["family"] == "tier1":
            out[sid] = embed(p, key, sid, ph, eps=cfg["eps"])
        else:
            out[sid] = embed_v2(p, key, sid, ph, alpha=cfg["alpha"], L=cfg["L"])
    return out


# ----------------------------------------------------------------- main sweep
def run_sweep(marked: dict[str, torch.Tensor], src: dict[str, torch.Tensor],
              gt_raw: dict, key: str, L: float, bank_name: str,
              tag: str, transforms: dict) -> dict[str, Any]:
    ids = sorted(marked.keys())
    n = len(ids)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    assert all(perm[i] != i for i in range(n))
    wrong_key = key + "-wrong"

    tnames = list(transforms.keys())
    Z = {p: {t: np.full(n, np.nan) for t in tnames} for p in ("P", "N1", "N2", "N3", "N4")}
    BER = {t: np.full(n, np.nan) for t in tnames}
    ZAN = {t: np.full(n, np.nan) for t in tnames}
    NDEAD = {t: np.zeros(n) for t in tnames}
    ERR = {p: {t: 0 for t in tnames} for p in ("P", "N1", "N2", "N3", "N4")}
    TW: dict[str, dict[str, torch.Tensor]] = {t: {} for t in tnames}

    t0 = time.time()
    for i, sid in enumerate(ids):
        cache: dict = {}
        Tmax = int(math.ceil(marked[sid].shape[0] * 1.5)) + 4
        gtT = int(math.ceil(as_pose(gt_raw[sid]).shape[0] * 1.5)) + 4
        _, cm = _plan(max(Tmax, gtT), L)
        other = ids[int(perm[i])]
        for k_, s_ in ((key, sid), (wrong_key, sid), (key, other)):
            _get_bases([k_] + [null_key(k_, j) for j in range(32)], s_, cm + 2, cache)
        tb = true_bits(sid, bank_name)
        for tname, fn in transforms.items():
            want_an = (tname == "identity")
            try:
                pw = fn(marked[sid], sid)
                TW[tname][sid] = pw
                d = detector_v2(pw, key, sid, L, base_cache=cache, want_analytic=want_an)
                Z["P"][tname][i] = d["z_key"]
                NDEAD[tname][i] = d["n_dead"]
                if want_an:
                    ZAN[tname][i] = d["z_analytic_identity"]
                got = np.asarray(d["bits"][:len(tb)])
                emp = empty_blocks(int(pw.shape[0]), L, d["best_hyp"])[:len(tb)]
                BER[tname][i] = float(((got != tb) | emp).mean())
            except Exception:
                ERR["P"][tname] += 1
                Z["P"][tname][i] = -np.inf
                BER[tname][i] = 1.0
                TW[tname][sid] = marked[sid]
            for pool, pose_src, k_, s_ in (
                ("N1", src[sid], key, sid),
                ("N2", marked[sid], wrong_key, sid),
                ("N3", marked[sid], key, other),
                ("N4", as_pose(gt_raw[sid]), key, sid),
            ):
                try:
                    pp = fn(pose_src, s_)
                    Z[pool][tname][i] = detector_v2(pp, k_, s_, L, base_cache=cache)["z_key"]
                except Exception:
                    ERR[pool][tname] += 1
                    Z[pool][tname][i] = np.inf
        if (i + 1) % 20 == 0:
            print(f"[{tag}] {i+1}/{n} {time.time()-t0:.0f}s", flush=True)

    rec: dict[str, Any] = {"tag": tag, "n": n, "L": L, "conditions": {}}
    for tname in tnames:
        zp, z1 = Z["P"][tname], Z["N1"][tname]
        k8 = int((zp > Z_KEY_THRESHOLD).sum())
        finite = zp[np.isfinite(zp)]
        row = {
            "n_pos": n,
            "tpr_at_zkey8": k8 / n,
            "tpr_cp95_lower": cp_lower(k8, n),
            "z_pos_median": float(np.median(zp[np.isfinite(zp)])) if len(finite) else float("nan"),
            "z_pos_min": float(np.min(zp)), "z_pos_max": float(np.max(zp[np.isfinite(zp)])) if len(finite) else float("nan"),
            "payload_ber_mean": float(np.mean(BER[tname])),
            "n_dead_channels_median": float(np.median(NDEAD[tname])),
            "n_dead_channels_max": float(np.max(NDEAD[tname])),
            "detector_errors_pos": ERR["P"][tname],
        }
        if tname == "identity":
            row["z_analytic_identity_median"] = float(np.nanmedian(ZAN[tname]))
            row["z_analytic_identity_min"] = float(np.nanmin(ZAN[tname]))
        for pool in ("N1", "N2", "N3", "N4"):
            zn = Z[pool][tname]
            kf = int((zn > Z_KEY_THRESHOLD).sum())
            row[f"{pool}_fpr_at_zkey8"] = kf / n
            row[f"{pool}_fpr_cp95_upper"] = cp_upper(kf, n)
            row[f"{pool}_z_max"] = float(np.max(zn))
            row[f"{pool}_z_median"] = float(np.median(zn[np.isfinite(zn)])) if np.isfinite(zn).any() else float("inf")
            row[f"{pool}_detector_errors"] = ERR[pool][tname]
        fp, fn_ = np.isfinite(zp), np.isfinite(z1)
        if fp.all() and fn_.all():
            y = np.r_[np.ones(n), np.zeros(n)]
            row["auc_vs_N1"] = float(roc_auc_score(y, np.r_[zp, z1]))
        else:
            sp = np.where(np.isfinite(zp), zp, -1e30)
            sn = np.where(np.isfinite(z1), z1, 1e30)
            row["auc_vs_N1"] = float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[sp, sn]))
        row["motion_vs_gt"] = motion_ratios(TW[tname], gt_raw, ids)
        rec["conditions"][tname] = row
        print(f"[{tag}] {tname:26s} TPR@z8={row['tpr_at_zkey8']:.3f} zmed={row['z_pos_median']:.1f} "
              f"zmin={row['z_pos_min']:.1f} N2max={row['N2_z_max']:.2f} BER={row['payload_ber_mean']:.4f} "
              f"jerk={row['motion_vs_gt']['hand_jerk_ratio']:.4f}", flush=True)
    rec["seconds"] = time.time() - t0
    rec["binding"] = {
        "n": n,
        "match_own_id_at_zkey8": int((Z["P"]["identity"] > Z_KEY_THRESHOLD).sum()),
        "transfer_other_id_at_zkey8": int((Z["N3"]["identity"] > Z_KEY_THRESHOLD).sum()),
        "z_own_median": float(np.median(Z["P"]["identity"])),
        "z_own_min": float(np.min(Z["P"]["identity"])),
        "z_other_max": float(np.max(Z["N3"]["identity"])),
    }
    rec["z_raw"] = {p: {t: Z[p][t].tolist() for t in tnames} for p in Z}
    rec["ids"] = ids
    return rec


CONFIGS = [
    {"name": "tier1_eps2.5e-4", "family": "tier1", "eps": 2.5e-4, "L": 2},
    {"name": "tier2_alpha0.35", "family": "v2", "alpha": 0.35, "L": 4},
    {"name": "tier2_alpha0.5", "family": "v2", "alpha": 0.5, "L": 4},
    {"name": "tier2_alpha0.7", "family": "v2", "alpha": 0.7, "L": 4},
]


def phase1(args) -> int:
    key = dev_key()
    src = load_bank(DEV_BANK)
    gt_raw = torch.load(GT["dev"], map_location="cpu", weights_only=False)
    ids = sorted(src.keys())
    missing = [s for s in ids if s not in gt_raw]
    assert not missing, f"dev ids missing from GT: {missing[:3]}"
    transforms = transform_list()
    base = {"stage": "phase1_dev_pilot", "preregistration_sha256": PREREG,
            "interpreter": sys.executable, "key_id_sha256_16": key_id(key),
            "z_key_threshold": Z_KEY_THRESHOLD, "n_clips": len(ids),
            "dev_bank": {"path": str(DEV_BANK), "sha256": sha256_file(DEV_BANK)},
            "transforms": list(transforms.keys()), "configs": {}}
    base["unmarked_motion_vs_gt"] = motion_ratios(src, gt_raw, ids)
    print("[phase1] dev80 unmarked baseline:", json.dumps(base["unmarked_motion_vs_gt"]), flush=True)
    for cfg in CONFIGS:
        if args.only and cfg["name"] not in args.only:
            continue
        marked = make_marked(src, cfg, key, "dev80_mt5gloss.pt")
        dmax = max(float((marked[s] - src[s]).abs().max()) for s in ids)
        dmean = float(np.mean([float((marked[s] - src[s]).abs().mean()) for s in ids]))
        rec = run_sweep(marked, src, gt_raw, key, cfg["L"], "dev80_mt5gloss.pt",
                        cfg["name"], transforms)
        rec["config"] = cfg
        rec["max_abs_pose_delta"] = dmax
        rec["mean_abs_pose_delta"] = dmean
        base["configs"][cfg["name"]] = rec
        (OUT / f"phase1_{cfg['name']}.json").write_text(json.dumps(rec, indent=2))
        del marked
    (OUT / "phase1_summary.json").write_text(json.dumps(
        {k: v for k, v in base.items() if k != "configs"} |
        {"configs": {k: {kk: vv for kk, vv in v.items() if kk not in ("z_raw", "ids")}
                     for k, v in base["configs"].items()}}, indent=2))
    print("wrote", OUT / "phase1_summary.json")
    return 0


# --------------------------------------------------------------- phase 2
from scripts.train_sign_jepa_ae_fullclip import hand_jerk, region_speed  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402


def motion_terms(p: torch.Tensor, r: torch.Tensor) -> dict[str, float]:
    """Per-clip terms of motion_ratios, identical arithmetic, streamed."""
    p = p.float(); r = r.float()
    if p.shape[0] != r.shape[0]:
        p = resize_seq(p.reshape(p.shape[0], -1), r.shape[0]).reshape(r.shape[0], 178, 3)
    pb, rb = p[None], r[None]
    a, b = region_speed(pb), region_speed(rb)
    return {"hand_pred": a["hand"], "hand_real": b["hand"], "body_pred": a["body"],
            "body_real": b["body"], "face_pred": a["face"], "face_real": b["face"],
            "jerk_pred": hand_jerk(pb), "jerk_real": hand_jerk(rb),
            "std_pred": float(pb[:, :, 8:50].std()), "std_real": float(rb[:, :, 8:50].std())}


def motion_from_accum(acc: dict[str, list], n: int) -> dict[str, float]:
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    rt = lambda a, b: float(m[a] / max(m[b], 1e-9))
    return {"clips": n, "hand_speed_ratio": rt("hand_pred", "hand_real"),
            "body_speed_ratio": rt("body_pred", "body_real"),
            "face_speed_ratio": rt("face_pred", "face_real"),
            "hand_jerk_ratio": rt("jerk_pred", "jerk_real"),
            "hand_posestd_ratio": rt("std_pred", "std_real")}


def run_sweep_streaming(marked, src, gt_raw, key, L, bank_name, tag, transforms):
    """Same as run_sweep but accumulates motion terms instead of storing tensors."""
    ids = sorted(marked.keys()); n = len(ids)
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n; perm[i], perm[j] = perm[j], perm[i]
    assert all(perm[i] != i for i in range(n))
    wrong_key = key + "-wrong"
    tnames = list(transforms.keys())
    Z = {p: {t: np.full(n, np.nan) for t in tnames} for p in ("P", "N1", "N2", "N3", "N4")}
    BER = {t: np.full(n, np.nan) for t in tnames}
    ZAN = np.full(n, np.nan); NDEAD = {t: np.zeros(n) for t in tnames}
    ERR = {p: {t: 0 for t in tnames} for p in ("P", "N1", "N2", "N3", "N4")}
    MACC = {t: {k: [] for k in ("hand_pred", "hand_real", "body_pred", "body_real",
                                "face_pred", "face_real", "jerk_pred", "jerk_real",
                                "std_pred", "std_real")} for t in tnames}
    t0 = time.time()
    for i, sid in enumerate(ids):
        cache: dict = {}
        gp = as_pose(gt_raw[sid])
        Tm = max(int(math.ceil(marked[sid].shape[0] * 1.5)), int(math.ceil(gp.shape[0] * 1.5))) + 4
        _, cm = _plan(Tm, L)
        other = ids[int(perm[i])]
        for k_, s_ in ((key, sid), (wrong_key, sid), (key, other)):
            _get_bases([k_] + [null_key(k_, j) for j in range(32)], s_, cm + 2, cache)
        tb = true_bits(sid, bank_name)
        for tname, fn in transforms.items():
            want_an = (tname == "identity")
            try:
                pw = fn(marked[sid], sid)
                for k, v in motion_terms(pw, gp).items():
                    MACC[tname][k].append(v)
                d = detector_v2(pw, key, sid, L, base_cache=cache, want_analytic=want_an)
                Z["P"][tname][i] = d["z_key"]; NDEAD[tname][i] = d["n_dead"]
                if want_an:
                    ZAN[i] = d["z_analytic_identity"]
                got = np.asarray(d["bits"][:len(tb)])
                emp = empty_blocks(int(pw.shape[0]), L, d["best_hyp"])[:len(tb)]
                BER[tname][i] = float(((got != tb) | emp).mean())
            except Exception:
                ERR["P"][tname] += 1; Z["P"][tname][i] = -np.inf; BER[tname][i] = 1.0
                for k, v in motion_terms(marked[sid], gp).items():
                    MACC[tname][k].append(v)
            for pool, pose_src, k_, s_ in (("N1", src[sid], key, sid),
                                           ("N2", marked[sid], wrong_key, sid),
                                           ("N3", marked[sid], key, other),
                                           ("N4", gp, key, sid)):
                try:
                    Z[pool][tname][i] = detector_v2(fn(pose_src, s_), k_, s_, L,
                                                    base_cache=cache)["z_key"]
                except Exception:
                    ERR[pool][tname] += 1; Z[pool][tname][i] = np.inf
        if (i + 1) % 50 == 0:
            print(f"[{tag}] {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    rec: dict[str, Any] = {"tag": tag, "n": n, "L": L, "conditions": {}}
    for tname in tnames:
        zp, z1 = Z["P"][tname], Z["N1"][tname]
        k8 = int((zp > Z_KEY_THRESHOLD).sum()); fin = zp[np.isfinite(zp)]
        row = {"n_pos": n, "tpr_at_zkey8": k8 / n, "tpr_cp95_lower": cp_lower(k8, n),
               "z_pos_median": float(np.median(fin)) if len(fin) else float("nan"),
               "z_pos_min": float(np.min(zp)),
               "z_pos_max": float(np.max(fin)) if len(fin) else float("nan"),
               "payload_ber_mean": float(np.mean(BER[tname])),
               "n_dead_channels_median": float(np.median(NDEAD[tname])),
               "n_dead_channels_max": float(np.max(NDEAD[tname])),
               "detector_errors_pos": ERR["P"][tname]}
        if tname == "identity":
            row["z_analytic_identity_median"] = float(np.nanmedian(ZAN))
            row["z_analytic_identity_min"] = float(np.nanmin(ZAN))
        for pool in ("N1", "N2", "N3", "N4"):
            zn = Z[pool][tname]; kf = int((zn > Z_KEY_THRESHOLD).sum())
            row[f"{pool}_fpr_at_zkey8"] = kf / n
            row[f"{pool}_fpr_cp95_upper"] = cp_upper(kf, n)
            row[f"{pool}_z_max"] = float(np.max(zn))
            row[f"{pool}_z_median"] = (float(np.median(zn[np.isfinite(zn)]))
                                       if np.isfinite(zn).any() else float("inf"))
            row[f"{pool}_detector_errors"] = ERR[pool][tname]
        sp = np.where(np.isfinite(zp), zp, -1e30); sn = np.where(np.isfinite(z1), z1, 1e30)
        row["auc_vs_N1"] = float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[sp, sn]))
        row["motion_vs_gt"] = motion_from_accum(MACC[tname], n)
        rec["conditions"][tname] = row
        print(f"[{tag}] {tname:26s} TPR@z8={row['tpr_at_zkey8']:.4f} zmed={row['z_pos_median']:.1f} "
              f"zmin={row['z_pos_min']:.1f} N2max={row['N2_z_max']:.2f} BER={row['payload_ber_mean']:.4f} "
              f"jerk={row['motion_vs_gt']['hand_jerk_ratio']:.4f}", flush=True)
    rec["seconds"] = time.time() - t0
    rec["binding"] = {"n": n,
                      "match_own_id_at_zkey8": int((Z["P"]["identity"] > Z_KEY_THRESHOLD).sum()),
                      "transfer_other_id_at_zkey8": int((Z["N3"]["identity"] > Z_KEY_THRESHOLD).sum()),
                      "z_own_median": float(np.median(Z["P"]["identity"])),
                      "z_own_min": float(np.min(Z["P"]["identity"])),
                      "z_other_max": float(np.max(Z["N3"]["identity"]))}
    rec["z_raw"] = {p: {t: Z[p][t].tolist() for t in tnames} for p in Z}
    rec["ids"] = ids
    return rec


def phase2(args) -> int:
    key = dev_key()
    src = load_bank(TEST_BANK)
    gt_raw = torch.load(GT["test"], map_location="cpu", weights_only=False)
    transforms = transform_list()
    banks = {"selected_tier1_eps2.5e-4": JULY_WM_25, "july_eps1e-3": JULY_WM_1E3}
    for name, path in banks.items():
        if args.only and name not in args.only:
            continue
        marked = load_bank(path)
        rec = run_sweep_streaming(marked, src, gt_raw, key, 2, "test641_mt5gloss.pt",
                                  name, transforms)
        rec.update({"stage": "phase2_test", "preregistration_sha256": PREREG,
                    "interpreter": sys.executable, "key_id_sha256_16": key_id(key),
                    "z_key_threshold": Z_KEY_THRESHOLD,
                    "marked_bank": {"path": str(path), "sha256": sha256_file(path)},
                    "source_bank": {"path": str(TEST_BANK), "sha256": sha256_file(TEST_BANK)},
                    "transforms": list(transforms.keys())})
        (OUT / f"phase2_{name}.json").write_text(json.dumps(rec, indent=2))
        print("wrote", OUT / f"phase2_{name}.json", flush=True)
        del marked
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["phase1", "phase2"])
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"phase1": phase1, "phase2": phase2}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
