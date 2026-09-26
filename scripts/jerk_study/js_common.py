"""Shared helpers for the jerk study. Read-only on frozen artifacts."""
import json, math, sys, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch
ROOT = Path("/home/kumwilai/research/signgen-t2m"); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import scripts.exp_revision_motion_ratios as paper_metric  # xyz, official GT
OUT = ROOT / "outputs/jerk_study"; OUT.mkdir(exist_ok=True)
GT_PT = ROOT / "external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/test.pt"
BANK = ROOT / "outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt"
TRACE = ROOT / "outputs/sota_chase/phase43_phrase_lattice/trace_revision_clean_phrase_lattice_test.json"
LOCAL_PT = ROOT / "outputs/revision/revision_clean_phrase_lattice_test.pt"
PLANS = ROOT / "outputs/revision/clean_source_test_plans.json"
DONOR = ROOT / "outputs/revision/clean_source_test_donor_trace.json"
HAND_XY = []
for j in range(8, 50): HAND_XY += [3*j, 3*j+1]
HAND_XYZ = [3*j+c for j in range(8, 50) for c in range(3)]
GT_MED = {"hand_speed": 0.02902499958872795, "hand_jerk": 0.017476815730333328, "hand_posestd": 0.0649891048669815}
GT_MED_XY = None

def load_gt():
    return paper_metric.load_gt(GT_PT)

def xy_profile(pose_flat):  # lead's scale: xy only, joints 8..49; pose_flat (T,534)
    p = pose_flat.reshape(-1, 178, 3)[:, 8:50, :2].astype(np.float64)
    if p.shape[0] < 4: return None
    v = np.diff(p, axis=0); j = np.diff(np.diff(v, axis=0), axis=0)
    return float(np.linalg.norm(v, axis=2).mean()), float(np.linalg.norm(j, axis=2).mean())

def ratios(preds: dict, gt: dict, ids):
    """Paper metric (xyz vs official GT) plus xy-only version with the same GT."""
    fns = {"hand_speed": paper_metric.hand_speed, "hand_jerk": paper_metric.hand_jerk, "hand_posestd": paper_metric.hand_posestd}
    res = {"n": len(ids)}
    for name, fn in fns.items():
        pm = np.array([fn(torch.from_numpy(preds[i])) for i in ids]); gm = np.array([fn(gt[i]) for i in ids])
        ok = np.isfinite(pm) & np.isfinite(gm)
        res[name] = float(np.median(pm[ok]) / np.median(gm[ok]))
    # xy-only (lead's diagnostic scale) against xy GT medians and, for continuity, against the xyz GT constants
    ps = []; pj = []; gs = []; gj = []
    for i in ids:
        a = xy_profile(preds[i]); b = xy_profile(gt[i].reshape(gt[i].shape[0], -1).numpy())
        if a and b: ps.append(a[0]); pj.append(a[1]); gs.append(b[0]); gj.append(b[1])
    res["xy_speed_vs_xyGT"] = float(np.median(ps) / np.median(gs)); res["xy_jerk_vs_xyGT"] = float(np.median(pj) / np.median(gj))
    res["xy_speed_lead_scale"] = float(np.median(ps) / GT_MED["hand_speed"]); res["xy_jerk_lead_scale"] = float(np.median(pj) / GT_MED["hand_jerk"])
    res["joint_obj"] = abs(res["hand_jerk"] - 1) + abs(res["hand_speed"] - 1)
    return res

def frozen_blend(segments, blend_w=4):
    """Exact copy of the frozen assembler's concatenation + cosine blend (modifies copies)."""
    segs = [s.copy() for s in segments]
    if not segs: return np.zeros((4, 534), np.float32), []
    out = [segs[0]]; joins = []
    for cur in segs[1:]:
        prev = out[-1]
        w = min(blend_w, prev.shape[0] // 2, cur.shape[0] // 2)
        if w >= 1:
            ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, w))).astype(np.float32)[:, None]
            prev[-w:] = (1.0 - ramp) * prev[-w:] + ramp * cur[:w]
        joins.append(w)
        out.append(cur)
    return np.concatenate(out, 0).astype(np.float32), joins

def join_positions(segments):
    pos = []; t = 0
    for s in segments[:-1]:
        t += s.shape[0]; pos.append(t)
    return pos  # index of first frame of each new segment
