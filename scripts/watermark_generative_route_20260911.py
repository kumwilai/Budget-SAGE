"""Keyed pose watermark on the restored leak-free generative route (test-641).

Preregistration: outputs/watermark_generative_route_20260911/
preregistration_watermark_20260911_frozen.md
sha256 065bdc4701a77ea0871cb15cbc1b41c3660cf0e0359545ced6b00ba5fcfcf3e1

The watermark scheme itself is imported unmodified from scripts/pose_watermark.py
(July's implementation).  This driver only (a) marks a 100%-generated bank, which
July's driver could not do because it keyed off a generated/retrieved route split,
(b) supplies the four negative pools that a 100%-generated bank forces us to
define explicitly, and (c) reports motion damage alongside every detection number
in the failure-boundary sweep.

Stages: embed | detect | boundary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path("/home/kumwilai/research/signgen-t2m")
sys.path.insert(0, str(ROOT))

from scripts.pose_watermark import (  # noqa: E402
    _highpass, _pattern, _payload_bits, _resample, as_pose, detect, embed,
    payload_hex_for,
)
from scripts.train_sign_jepa_ae_fullclip import hand_jerk, region_speed  # noqa: E402
from scripts.train_sign_jepa_slrtp178 import resize_seq  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import beta  # noqa: E402

OUT = ROOT / "outputs/watermark_generative_route_20260911"
SRC_BANK = ROOT / "outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt"
GT_BANK = (ROOT / "external/SLRTP-Sign-Production-Evaluation/pretrained/"
           "SLRTP-Sign-Production-Evaluation-Data/data/test.pt")
Z_THRESHOLD = 8.0  # preregistered, fixed before any z on this bank was seen


def dev_key() -> str:
    """July's development key, read from its source default.

    Deliberately not restated as a literal here or in any output file; only the
    identifier sha256(key)[:16] is ever recorded.
    """
    src = (ROOT / "scripts/pose_watermark.py").read_text()
    m = re.search(r'"--key",\s*default="([^"]+)"', src)
    if m is None:
        raise RuntimeError("could not read --key default from scripts/pose_watermark.py")
    return m.group(1)


def key_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank(p: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(p, map_location="cpu", weights_only=False)
    return {str(k): as_pose(v).contiguous() for k, v in raw.items()}


def cert_row(sid: str, bank_name: str) -> dict[str, str]:
    return {"sid": sid, "route": "generated", "bank": bank_name}


# ---------------------------------------------------------------- embed stage
def stage_embed(args: argparse.Namespace) -> int:
    key = dev_key()
    bank = load_bank(SRC_BANK)
    bank_name = args.bank_name
    t0 = time.time()
    wm: dict[str, torch.Tensor] = {}
    payloads: dict[str, str] = {}
    for sid, pose in bank.items():
        payloads[sid] = payload_hex_for(cert_row(sid, bank_name))
        wm[sid] = embed(pose, key, sid, payloads[sid], eps=args.eps)
    out_pt = OUT / args.out_name
    torch.save(wm, out_pt)
    # perturbation magnitude, all 641 clips (no subsampling)
    dmax = max(float((wm[s] - bank[s]).abs().max()) for s in bank)
    dmean = float(np.mean([float((wm[s] - bank[s]).abs().mean()) for s in bank]))
    rec = {
        "stage": "embed",
        "preregistration_sha256": args.prereg_sha,
        "interpreter": sys.executable,
        "eps": args.eps,
        "n_blocks": 16,
        "frame_block": 2,
        "key_id_sha256_16": key_id(key),
        "key_source": "scripts/pose_watermark.py --key default (development key, not secret)",
        "n_clips_marked": len(wm),
        "n_clips_total": len(bank),
        "source_bank": {"path": str(SRC_BANK), "sha256": sha256_file(SRC_BANK)},
        "marked_bank": {"path": str(out_pt), "sha256": sha256_file(out_pt)},
        "payload_bank_field": bank_name,
        "max_abs_pose_delta": dmax,
        "mean_abs_pose_delta": dmean,
        "payloads": payloads,
        "seconds": time.time() - t0,
    }
    (OUT / f"embed_record_eps{args.eps:g}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if k != "payloads"}, indent=2))
    return 0


# ------------------------------------------------------------- detect helpers
def z_of(pose: torch.Tensor, key: str, sid: str) -> tuple[float, list[int]]:
    d = detect(pose, key, sid)
    return float(d["z"]), list(d["bits"])


def true_bits(sid: str, bank_name: str) -> np.ndarray:
    return ((_payload_bits(payload_hex_for(cert_row(sid, bank_name))) + 1) / 2).astype(int)


def cp_lower(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    if k >= n:
        return float(0.025 ** (1.0 / n))
    if k == 0:
        return 0.0
    return float(beta.ppf(0.025, k, n - k + 1))


def auc_and_tpr(zpos: np.ndarray, zneg: np.ndarray) -> dict[str, Any]:
    y = np.r_[np.ones_like(zpos), np.zeros_like(zneg)]
    s = np.r_[zpos, zneg]
    auc = float(roc_auc_score(y, s)) if len(zpos) and len(zneg) else float("nan")
    thr = float(zneg.max()) if len(zneg) else 0.0
    k8 = int((zpos > Z_THRESHOLD).sum())
    return {
        "n_pos": int(len(zpos)), "n_neg": int(len(zneg)),
        "auc": auc,
        "tpr_at_zero_fpr": float((zpos > thr).mean()) if len(zpos) else float("nan"),
        "zero_fpr_threshold_z": thr,
        "tpr_at_z8": float(k8 / len(zpos)) if len(zpos) else float("nan"),
        "tpr_at_z8_cp95_lower": cp_lower(k8, len(zpos)),
        "fpr_at_z8": float((zneg > Z_THRESHOLD).mean()) if len(zneg) else float("nan"),
        "z_pos_median": float(np.median(zpos)) if len(zpos) else float("nan"),
        "z_pos_min": float(np.min(zpos)) if len(zpos) else float("nan"),
        "z_neg_median": float(np.median(zneg)) if len(zneg) else float("nan"),
        "z_neg_max": float(np.max(zneg)) if len(zneg) else float("nan"),
    }


def stage_detect(args: argparse.Namespace) -> int:
    key = dev_key()
    src = load_bank(SRC_BANK)
    wm = load_bank(OUT / args.out_name)
    gt_raw = torch.load(GT_BANK, map_location="cpu", weights_only=False)
    ids = list(wm.keys())
    n = len(ids)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    for i in range(n):  # derangement: no clip keeps its own id
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    assert all(perm[i] != i for i in range(n))

    z_pos, ber, z_n1, z_n2, z_n3, z_n4 = [], [], [], [], [], []
    errors = {"pos": 0, "n1": 0, "n2": 0, "n3": 0, "n4": 0}
    t0 = time.time()
    for i, sid in enumerate(ids):
        try:
            z, bits = z_of(wm[sid], key, sid)
            tb = true_bits(sid, args.bank_name)
            got = np.asarray(bits[:len(tb)])
            z_pos.append(z); ber.append(float((got != tb).mean()))
        except Exception:
            errors["pos"] += 1; z_pos.append(float("-inf")); ber.append(1.0)
        for pool, pose, k_, s_ in (
            ("n1", src[sid], key, sid),
            ("n2", wm[sid], key + "-wrong", sid),
            ("n3", wm[sid], key, ids[int(perm[i])]),
        ):
            try:
                zz, _ = z_of(pose, k_, s_)
            except Exception:
                errors[pool] += 1; zz = float("inf")  # failure counts against us
            {"n1": z_n1, "n2": z_n2, "n3": z_n3}[pool].append(zz)
        try:
            zz, _ = z_of(as_pose(gt_raw[sid]), key, sid)
        except Exception:
            errors["n4"] += 1; zz = float("inf")
        z_n4.append(zz)
        if (i + 1) % 200 == 0:
            print(f"[detect] {i+1}/{n} {time.time()-t0:.0f}s", flush=True)

    zp = np.asarray(z_pos)
    pools = {"N1_unmarked_source": np.asarray(z_n1), "N2_wrong_key": np.asarray(z_n2),
             "N3_wrong_clip_id": np.asarray(z_n3), "N4_real_motion_gt": np.asarray(z_n4)}
    rec: dict[str, Any] = {
        "stage": "detect", "condition": "clean",
        "preregistration_sha256": args.prereg_sha,
        "interpreter": sys.executable,
        "eps": args.eps, "z_threshold": Z_THRESHOLD,
        "key_id_sha256_16": key_id(key),
        "marked_bank": {"path": str(OUT / args.out_name),
                        "sha256": sha256_file(OUT / args.out_name)},
        "n_positives": n, "detector_errors": errors,
        "payload_ber_mean": float(np.mean(ber)), "payload_ber_max": float(np.max(ber)),
        "payload_exact_recovery": int(sum(1 for b in ber if b == 0.0)),
        "z_pos": {"median": float(np.median(zp)), "min": float(np.min(zp)),
                  "max": float(np.max(zp)), "p05": float(np.percentile(zp, 5))},
        "pools": {},
        "binding": {
            "n": n,
            "binding_match_own_id_at_z8": int((zp > Z_THRESHOLD).sum()),
            "transfer_detections_other_id_at_z8": int((pools["N3_wrong_clip_id"] > Z_THRESHOLD).sum()),
            "z_own_median": float(np.median(zp)),
            "z_own_min": float(np.min(zp)),
            "z_other_median": float(np.median(pools["N3_wrong_clip_id"])),
            "z_other_max": float(np.max(pools["N3_wrong_clip_id"])),
        },
        "seconds": time.time() - t0,
    }
    for name, zn in pools.items():
        rec["pools"][name] = auc_and_tpr(zp, zn)
        rec["pools"][name]["z_neg_p99"] = float(np.percentile(zn, 99))
    (OUT / f"detect_record_eps{args.eps:g}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if k != "payloads"}, indent=2))
    return 0


# ------------------------------------------------------------ boundary stage
def movavg(p: torch.Tensor, k: int) -> torch.Tensor:
    return p - _highpass(p, k=k)


def build_transforms(rng: np.random.Generator) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    t: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {"identity": lambda p: p}
    for s in (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2):
        t[f"noise_{s:g}"] = (lambda s_: lambda p: p + torch.from_numpy(
            rng.normal(0, s_, size=tuple(p.shape)).astype(np.float32)))(s)
    for k in (3, 5, 7, 9, 11, 15):
        t[f"smooth_k{k}"] = (lambda k_: lambda p: movavg(p, k_))(k)
    for f in (0.5, 0.75, 0.9, 1.1, 1.25, 1.5):
        t[f"blind_resample_{f:g}x"] = (lambda f_: lambda p: _resample(
            p, max(2, int(round(p.shape[0] * f_)))))(f)
    for f in (0.75, 1.25):
        t[f"certguided_resample_{f:g}x"] = (lambda f_: lambda p: _resample(
            _resample(p, max(2, int(round(p.shape[0] * f_)))), p.shape[0]))(f)
    for q in (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2):
        t[f"quantize_{q:g}"] = (lambda q_: lambda p: torch.round(p / q_) * q_)(q)
    t["evaluator_decimate2"] = lambda p: p[::2].contiguous()
    return t


def motion_ratios(pred: dict[str, torch.Tensor], gt_raw: dict[str, Any],
                  ids: list[str]) -> dict[str, float]:
    """Identical arithmetic to sign_jepa_source_locked_fusion.motion_eval."""
    acc = {f"{r}_{k}": [] for r in ("hand", "body", "face") for k in ("pred", "real")}
    jp, jr, sp_, sr_ = [], [], [], []
    for sid in ids:
        p = pred[sid].float()
        r = as_pose(gt_raw[sid]).float()
        if p.shape[0] != r.shape[0]:
            p = resize_seq(p.reshape(p.shape[0], -1), r.shape[0]).reshape(r.shape[0], 178, 3)
        pb, rb = p[None], r[None]
        a, b = region_speed(pb), region_speed(rb)
        for reg in ("hand", "body", "face"):
            acc[f"{reg}_pred"].append(a[reg]); acc[f"{reg}_real"].append(b[reg])
        jp.append(hand_jerk(pb)); jr.append(hand_jerk(rb))
        sp_.append(float(pb[:, :, 8:50].std())); sr_.append(float(rb[:, :, 8:50].std()))
    ratio = lambda x, y: float(np.mean(x) / max(float(np.mean(y)), 1e-9))
    return {
        "clips": len(ids),
        "hand_speed_ratio": ratio(acc["hand_pred"], acc["hand_real"]),
        "body_speed_ratio": ratio(acc["body_pred"], acc["body_real"]),
        "face_speed_ratio": ratio(acc["face_pred"], acc["face_real"]),
        "hand_jerk_ratio": ratio(jp, jr),
        "hand_posestd_ratio": ratio(sp_, sr_),
    }


def stage_boundary(args: argparse.Namespace) -> int:
    key = dev_key()
    src = load_bank(SRC_BANK)
    wm = load_bank(OUT / args.out_name)
    gt_raw = torch.load(GT_BANK, map_location="cpu", weights_only=False)
    ids = [s for s in gt_raw.keys() if str(s) in wm]
    ids = [str(s) for s in ids]
    rng = np.random.default_rng(0)
    transforms = build_transforms(rng)
    rec: dict[str, Any] = {
        "stage": "boundary", "preregistration_sha256": args.prereg_sha,
        "interpreter": sys.executable, "eps": args.eps, "z_threshold": Z_THRESHOLD,
        "key_id_sha256_16": key_id(key), "seed": 0,
        "n_positives": len(wm), "n_negatives_pool_N1": len(src),
        "conditions": {},
    }
    for name, fn in transforms.items():
        t0 = time.time()
        zp, zn, ber = [], [], []
        errs = 0
        tw: dict[str, torch.Tensor] = {}
        for sid in wm:
            try:
                pw_ = fn(wm[sid])
                tw[sid] = pw_
                z, bits = z_of(pw_, key, sid)
                tb = true_bits(sid, args.bank_name)
                got = np.asarray(bits[:len(tb)])
                zp.append(z); ber.append(float((got != tb).mean()))
            except Exception:
                errs += 1; zp.append(float("-inf")); ber.append(1.0)
                tw[sid] = wm[sid]
            try:
                zz, _ = z_of(fn(src[sid]), key, sid)
            except Exception:
                zz = float("inf")
            zn.append(zz)
        zp_a, zn_a = np.asarray(zp), np.asarray(zn)
        row = auc_and_tpr(zp_a, zn_a)
        row["payload_ber_mean"] = float(np.mean(ber))
        row["detector_errors"] = errs
        row["motion_vs_gt"] = motion_ratios(tw, gt_raw, ids)
        row["mad_vs_marked"] = float(np.mean([
            float((tw[s] - wm[s]).abs().mean()) if tw[s].shape == wm[s].shape
            else float((resize_seq(tw[s].reshape(tw[s].shape[0], -1), wm[s].shape[0])
                        .reshape(wm[s].shape) - wm[s]).abs().mean())
            for s in wm]))
        row["seconds"] = time.time() - t0
        rec["conditions"][name] = row
        print(f"[bnd] {name:24s} AUC={row['auc']:.4f} TPR@z8={row['tpr_at_z8']:.3f} "
              f"BER={row['payload_ber_mean']:.4f} zpos_med={row['z_pos_median']:.1f} "
              f"hand_jerk={row['motion_vs_gt']['hand_jerk_ratio']:.3f} "
              f"hand_spd={row['motion_vs_gt']['hand_speed_ratio']:.3f} "
              f"({row['seconds']:.0f}s)", flush=True)
        (OUT / f"boundary_record_eps{args.eps:g}.json").write_text(json.dumps(rec, indent=2))
    print(f"wrote {OUT / f'boundary_record_eps{args.eps:g}.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["embed", "detect", "boundary"])
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--out_name", default="test641_mt5gloss_wm.pt")
    ap.add_argument("--bank_name", default="test641_mt5gloss.pt")
    ap.add_argument("--prereg_sha",
                    default="065bdc4701a77ea0871cb15cbc1b41c3660cf0e0359545ced6b00ba5fcfcf3e1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"embed": stage_embed, "detect": stage_detect, "boundary": stage_boundary}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
