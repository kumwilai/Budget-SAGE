"""Recognizer-free hand-motion audit vs ground truth (IEEE Access revision).

Computes, for a predicted-pose .pt against the official SLRTP GT split, the
hand kinematic ratios used in the paper:
  hand_speed_ratio   = median_clip(mean |1st diff| over hand joints 8:50) / GT
  hand_jerk_ratio    = median_clip(mean |3rd diff| over hand joints 8:50) / GT
  hand_posestd_ratio = median_clip(mean over joints of std_t(pos))          / GT
All use full xyz of joints 8:50. "jerk" is the third time-derivative magnitude,
matching the paper's definition.  Ratios are median(method)/median(GT) over the
clips present in BOTH the prediction file and the GT split.

Release note: the original script imports `as_pose` and `load_pose_map` from
`scripts.sign_jepa_motion_amplify`, whose own import chain pulls in
`scripts/sign_jepa_motion_texture_retrieval.py`, `scripts/train_sign_jepa_flow_generator.py`,
and `scripts/train_sign_jepa_slrtp178.py`. Those four files are not part of this
release, so `as_pose` (from `scripts/sign_jepa_motion_texture_retrieval.py`) and
`load_pose_map` (from `scripts/sign_jepa_motion_amplify.py`) are reproduced here
verbatim instead of imported. Nothing else about this script has changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

HAND = slice(8, 50)


def as_pose(x) -> torch.Tensor:
    if isinstance(x, dict):
        x = x.get("poses_3d", x.get("pose"))
    x = torch.as_tensor(x).float()
    if x.ndim == 2:
        x = x.reshape(x.shape[0], 178, 3)
    if x.ndim != 3 or x.shape[1:] != (178, 3):
        raise ValueError(f"bad pose shape {tuple(x.shape)}")
    return x


def load_pose_map(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = {}
    for sid, pose in data.items():
        try:
            out[str(sid)] = as_pose(pose).contiguous()
        except Exception:
            continue
    return out


def _hand_xyz(pose: torch.Tensor) -> torch.Tensor:
    return pose.reshape(pose.shape[0], 178, 3).float()[:, HAND]  # [T,42,3]


def hand_speed(pose: torch.Tensor) -> float:
    x = _hand_xyz(pose)
    if x.shape[0] < 2:
        return float("nan")
    v = x[1:] - x[:-1]
    return float(v.norm(dim=-1).mean())


def hand_jerk(pose: torch.Tensor) -> float:
    x = _hand_xyz(pose)
    if x.shape[0] < 4:
        return float("nan")
    v = x[1:] - x[:-1]
    a = v[1:] - v[:-1]
    j = a[1:] - a[:-1]
    return float(j.norm(dim=-1).mean())


def hand_posestd(pose: torch.Tensor) -> float:
    x = _hand_xyz(pose)
    if x.shape[0] < 2:
        return float("nan")
    return float(x.std(dim=0).mean())


def load_gt(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sid, v in raw.items():
        try:
            out[str(sid)] = as_pose(v).contiguous()
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred_pt", required=True)
    ap.add_argument("--gt_pt", required=True)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    pred = load_pose_map(ROOT / args.pred_pt)
    gt = load_gt(Path(args.gt_pt) if os.path.isabs(args.gt_pt) else ROOT / args.gt_pt)
    ids = [i for i in pred if i in gt]
    fns = {"hand_speed": hand_speed, "hand_jerk": hand_jerk, "hand_posestd": hand_posestd}
    res = {"n_clips": len(ids)}
    for name, fn in fns.items():
        pm = np.array([fn(pred[i]) for i in ids], dtype=np.float64)
        gm = np.array([fn(gt[i]) for i in ids], dtype=np.float64)
        pm, gm = pm[np.isfinite(pm) & np.isfinite(gm)], gm[np.isfinite(pm) & np.isfinite(gm)]
        res[f"{name}_pred_med"] = float(np.median(pm))
        res[f"{name}_gt_med"] = float(np.median(gm))
        res[f"{name}_ratio"] = float(np.median(pm) / np.median(gm))
    print(json.dumps(res, indent=2))
    if args.out_json:
        op = ROOT / args.out_json
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(res, indent=2))
        print(f"saved -> {op}", flush=True)


if __name__ == "__main__":
    main()
