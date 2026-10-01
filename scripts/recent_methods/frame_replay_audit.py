"""Frame-level verbatim-replay detector for the Table VIII replay-share column.

Ruling: outputs/baseline_retrain_design_2026-09-28/ustcmoe_detector_ruling.md (Fable 5.1, 2026-09-28).
The 16-frame, stride-4 sub-sequence detector (scripts/audit_subsequence_memorization.py) is blind to
verbatim runs shorter than 16 frames. This script is the frame-level detector (window 1, stride 1):
for every frame of a query bank it finds the per-dim RMS distance to the single nearest training frame.
It reuses the exact signature space of the old script -- import, not copy -- so the two detectors are
comparable: `window_signatures` (first 50 body+hand joints, xy, per-frame centering, standardized) and
`as_pose` (float32, [T,178,3]). The (win, stride) knobs generalize the old script's own window/stride
so the same code path reproduces its recorded numbers at (16, 4) and computes the new headline at (1, 1).

A frame is flagged when its distance (minimum over every window that covers it) is at or below the
threshold. Padded frames (all-zero pose, before the joint subset and centering) are reported separately
and excluded from the "non-padded" share. Nearest-neighbour search is GPU-chunked over the training bank
(chunk rows at a time), never over the query side, matching scripts/audit_subsequence_memorization.py
and scripts/recent_baselines/s2_post.py's existing frame_level().

Per bank the report holds: n_clips, n_frames, n_zero_frames, share_all_frames, share_nonpad_frames,
clips_flagged (clips with a below-threshold frame), clip_min_median/min/max (the old script's per-clip
statistic, at whatever (win, stride) was requested), a distance-quantile table over all frames and over
non-padded frames only, and a per_clip breakdown.

CLI:
  frame_replay_audit.py --pred NAME=PATH.pt [--pred NAME2=PATH2.pt ...] --win 1 --stride 1 \\
      --threshold 0.005 --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.audit_subsequence_memorization import as_pose, window_signatures  # noqa: E402

DEFAULT_TRAIN_PT = (ROOT / "external/SLRTP-Sign-Production-Evaluation/pretrained/"
                    "SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
QUANTILES = (0, 5, 25, 50, 75, 95, 100)


# ----------------------------------------------------------------------------- windowing
def window_starts(T: int, win: int, stride: int) -> list[int]:
    """Same start-position rule as window_signatures: covers [0, T) with the last window flush to T."""
    Tp = max(T, win)
    starts = list(range(0, Tp - win + 1, stride))
    if starts[-1] != Tp - win:
        starts.append(Tp - win)
    return starts


def zero_frames(pose: torch.Tensor) -> np.ndarray:
    """A padded frame is all-zero across every joint and coordinate of the full pose (before the
    50-joint xy subset window_signatures uses), matching s2_post.py's frame_level() convention."""
    T = pose.shape[0]
    return (pose.reshape(T, -1).abs().sum(dim=1) == 0).numpy()


# ----------------------------------------------------------------------------- bank + distances
def build_bank(train: dict[str, Any], win: int, stride: int) -> torch.Tensor:
    """[N_train_windows, win*100] float32 signature bank, built once and reused for every query bank."""
    sigs = [window_signatures(as_pose(meta), win, stride).half() for meta in train.values()]
    return torch.cat(sigs).float()


def nearest_distances(query_sigs: torch.Tensor, bank: torch.Tensor, bank_sq: torch.Tensor,
                      device: str, chunk: int) -> np.ndarray:
    """Per-dim RMS distance from each query window to its nearest bank window, chunked over the bank."""
    q = query_sigs.to(device)
    qn = (q * q).sum(dim=1, keepdim=True)
    best = torch.full((q.shape[0],), float("inf"), device=device)
    for s in range(0, bank.shape[0], chunk):
        b = bank[s:s + chunk].to(device)
        bn = bank_sq[s:s + chunk].to(device)
        d2 = (qn + bn.unsqueeze(0) - 2.0 * (q @ b.T)).clamp_min(0).min(dim=1).values
        best = torch.minimum(best, d2)
    dim = query_sigs.shape[1]
    return torch.sqrt((best / dim).clamp_min(0)).cpu().numpy()


def audit_clip(pose: torch.Tensor, bank: torch.Tensor, bank_sq: torch.Tensor,
              device: str, win: int, stride: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (frame_dist[T]: min distance of any window covering that frame, window_dist[n_win]:
    the raw per-window distances, whose min is the old script's per-clip minimum-window-distance)."""
    T = pose.shape[0]
    sig = window_signatures(pose, win, stride)
    starts = window_starts(T, win, stride)
    window_dist = nearest_distances(sig, bank, bank_sq, device, chunk)
    frame_dist = np.full(T, np.inf, dtype=np.float64)
    for st, d in zip(starts, window_dist):
        end = min(st + win, T)
        frame_dist[st:end] = np.minimum(frame_dist[st:end], d)
    return frame_dist, window_dist


def quantile_table(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {f"p{q}": float("nan") for q in QUANTILES}
    vals = np.percentile(arr, QUANTILES)
    return {f"p{q}": float(v) for q, v in zip(QUANTILES, vals)}


# ----------------------------------------------------------------------------- per-bank report
def audit_bank(pred: dict[str, Any], bank: torch.Tensor, bank_sq: torch.Tensor, device: str,
              win: int, stride: int, chunk: int, threshold: float, max_clips: int = 0) -> dict[str, Any]:
    ids = list(pred.keys())
    if max_clips:
        ids = ids[:max_clips]
    all_frame_dist, nonpad_frame_dist, clip_min = [], [], []
    n_frames = n_zero = n_flag_all = n_flag_nonpad = clips_flagged = 0
    per_clip: dict[str, Any] = {}
    for sid in ids:
        pose = as_pose(pred[sid])
        T = pose.shape[0]
        fdist, wdist = audit_clip(pose, bank, bank_sq, device, win, stride, chunk)
        zero = zero_frames(pose)
        flagged = fdist <= threshold
        n_frames += T
        n_zero += int(zero.sum())
        n_flag_all += int(flagged.sum())
        n_flag_nonpad += int((flagged & ~zero).sum())
        cmin = float(wdist.min())
        clip_min.append(cmin)
        if cmin <= threshold:
            clips_flagged += 1
        all_frame_dist.append(fdist)
        nonpad_frame_dist.append(fdist[~zero])
        per_clip[sid] = {"T": T, "n_zero": int(zero.sum()), "n_flagged": int(flagged.sum()),
                          "n_flagged_nonpad": int((flagged & ~zero).sum()), "clip_min": cmin,
                          "share": float(flagged.sum()) / T}
    all_arr = np.concatenate(all_frame_dist) if all_frame_dist else np.array([])
    nonpad_arr = np.concatenate(nonpad_frame_dist) if nonpad_frame_dist else np.array([])
    cm = np.array(clip_min)
    n_nonpad = n_frames - n_zero
    return {
        "n_clips": len(ids), "n_frames": n_frames, "n_zero_frames": n_zero,
        "n_flagged_all": n_flag_all, "n_flagged_nonpad": n_flag_nonpad,
        "share_all_frames": n_flag_all / max(1, n_frames),
        "share_nonpad_frames": n_flag_nonpad / max(1, n_nonpad),
        "clips_flagged": clips_flagged,
        "clip_min_min": float(cm.min()) if cm.size else float("nan"),
        "clip_min_median": float(np.median(cm)) if cm.size else float("nan"),
        "clip_min_max": float(cm.max()) if cm.size else float("nan"),
        "quantiles_all_frames": quantile_table(all_arr),
        "quantiles_nonpad_frames": quantile_table(nonpad_arr),
        "per_clip": per_clip,
    }


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_pt", default=str(DEFAULT_TRAIN_PT))
    ap.add_argument("--pred", action="append", default=[], required=True,
                    help="NAME=path.pt, repeatable; each is scored against the same training bank")
    ap.add_argument("--win", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.005)
    ap.add_argument("--chunk", type=int, default=65536, help="training-bank rows sent to the GPU per step")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_clips", type=int, default=0, help="cap query clips per bank (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    print(f"[bank] loading train.pt from {args.train_pt}", flush=True)
    train = torch.load(args.train_pt, map_location="cpu", weights_only=False)
    bank = build_bank(train, args.win, args.stride)
    del train
    print(f"[bank] {bank.shape[0]} windows x {bank.shape[1]} dims, win={args.win} stride={args.stride}", flush=True)
    bank_sq = (bank * bank).sum(dim=1)

    report: dict[str, Any] = {"win": args.win, "stride": args.stride, "threshold": args.threshold,
                              "train_pt": str(args.train_pt), "n_train_windows": int(bank.shape[0]),
                              "device": args.device, "banks": {}}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        print(f"[audit] {name} <- {path}", flush=True)
        pred = torch.load(path, map_location="cpu", weights_only=False)
        row = audit_bank(pred, bank, bank_sq, args.device, args.win, args.stride, args.chunk,
                         args.threshold, args.max_clips)
        row["pred_path"] = str(path)
        report["banks"][name] = row
        print(f"[audit] {name}: clips {row['clips_flagged']}/{row['n_clips']} <= {args.threshold}, "
              f"frames {row['n_flagged_all']}/{row['n_frames']} = {row['share_all_frames']:.4%}, "
              f"non-padded {row['share_nonpad_frames']:.4%}, clip-min median {row['clip_min_median']:.4f}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
