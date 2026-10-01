"""Score SLRTP-178 pose banks with the Sign-IDD authors' back-translator (the SLT that Sign-IDD and FGDM used for
their published PHOENIX test numbers). Brief: outputs/baseline_retrain_design_2026-09-28/their_evaluator_brief.md.

Evaluator: external/baselines/Sign-IDD-SLT-authors/sign_skels_model/best.ckpt run through slt_run.py (signjoey,
unmodified, memory shim of 2026-09-28). Decoding is PINNED to the R0 settings (recognition beam 10, translation beam
2, alpha -1, chosen on the authors' ground-truth dev skels in R0), and the dev split of every run is the authors'
ground-truth dev skels, so no bank's own outputs choose its decoding. References are the authors' own (the gloss and
text fields of the R0 ground-truth test pickle), for every bank.

Input side: every bank is [T, 178, 3] SLRTP at 25 fps, mapped by slrtp_pt_mapping.slrtp178_to_pt50 to [T, 50, 3],
flattened joint-major to [T, 150] (the PT-150 layout of build_pt50_data.py and of the authors' pickles), and written
as the gzip pickle list of dicts {name, signer, gloss, text, sign} that signjoey reads (signidd_repro.py gt_pickle).
Items are in the authors' PT test order restricted to the 641 keys the SLRTP test set shares with it (PT has 642).
ground_test = the run's own test file: signjoey only uses it for its FID number, which is not reported.

  build NAME   convert one bank (or 'gate0_slrtp_gt', the SLRTP ground truth) into inputs/NAME.test.gz + config
  slt NAME     run the authors' SLT on it (wrap in light.sh, or nolock_L.sh's L when another session holds the lock)
  diag         Gate 0 diagnostic: SLRTP ground truth after the mapping against the authors' PT-150 test skels
  gate0        Gate 0 verdict against R0 on the 641 common keys (also reproduces R0's recorded 11.93 / 71.94 first)
  score        every bank: BLEU-1, BLEU-4, WER (the SLT's gloss WER) + paired bootstrap (seed 30373, 10,000)
  checks       conversion checks 1-4 (key alignment, agreement with the authors' GT, round trip, unit test),
               which must all pass before the Gate 0 SLT run
  fitfix       fit the Gate 0 fix candidate fix1 on the dev split (see the fix1 block); then --fix fix1 on the others
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2_common as C  # noqa: E402
from slrtp_pt_mapping import slrtp178_to_pt50  # noqa: E402

ROOT = C.ROOT
RB = C.RB
OUT = RB / "their_evaluator"
REP = RB / "signidd/reproduction"
GT_DIR = REP / "slt_data/GroundTruth"
PT_GT_TEST = GT_DIR / "phoenix14t.skels.test.gz"
PT_GT_DEV = GT_DIR / "phoenix14t.skels.dev.gz"
R0_CFG = REP / "config/slt_R0_groundtruth.yaml"
R0_PREFIX = REP / "slt_R0_rerun_memshim/R0_groundtruth"
R0_RECORDED = {"bleu4": 11.93, "wer": 71.94, "dev_bleu4": 12.13, "dev_wer": 74.17, "n": 642}
SLT_DIR = ROOT / "external/baselines/FGDM-main/SLT-main"
SLT_CKPT = ROOT / "external/baselines/Sign-IDD-SLT-authors/sign_skels_model/best.ckpt"
SLT_CKPT_SHA = "d32bf93d1696502df7aaf852dd4a9d6b2f97f17785e1221062bdaa93e21a523d"
PINNED = {"recognition_beam_sizes": [10], "translation_beam_sizes": [2], "translation_beam_alphas": [-1]}
SUFFIX = {"gls": ".BW_010.{}.gls", "txt": ".BW_02.A_-1.{}.txt"}
SEED, N_BOOT = 30373, 10000
PUBLISHED = {"Sign-IDD": {"bleu4": 9.08, "wer": 76.66}, "FGDM": {"bleu4": 9.67, "wer": 70.70}}

BANKS = {
    "gate0_slrtp_gt": C.DATA / "test.pt",
    "ours_fixed": ROOT / "outputs/revision/clean_rerank_frame40_test.pt",
    "ours_learned": ROOT / "outputs/revision/clean_strict_cac_frame40_test.pt",
    "ours_allgen": ROOT / "outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt",
    "ustcmoe": RB / "ustc_moe/preds/ustcmoe_mt5_natlen_test.pt",
    "darslp": RB / "darslp/test_after_choice/preds/darslp_released_periodfree_test.pt",
    "signidd": RB / "signidd/gate5/preds/signidd_gate4best_test_seed11.pt",
}
OURS = ("ours_fixed", "ours_learned", "ours_allgen")
BASELINES = ("ustcmoe", "darslp", "signidd")


# ---------------------------------------------------------------- conversion (pure, tested)
def bare(name: str) -> str:
    return name.split("/", 1)[1] if "/" in name else name


def common_refs(pt_items: list, keys) -> list:
    """The authors' PT items whose key is in `keys`, in the authors' order."""
    ks = set(keys)
    return [x for x in pt_items if bare(x["name"]) in ks]


def pose_of(v) -> np.ndarray:
    if isinstance(v, dict):
        v = v["poses_3d"]
    return np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)


def to_slt_items(bank: dict, refs: list, mapper=slrtp178_to_pt50) -> list:
    """bank: key -> [T, 178, 3]; refs: the authors' items (name, signer, gloss, text) in the output order."""
    import torch
    out = []
    for r in refs:
        p = pose_of(bank[bare(r["name"])])
        T = p.shape[0]
        sign = mapper(p).reshape(T, 150).astype(np.float32)
        out.append({"name": r["name"], "signer": r["signer"], "gloss": r["gloss"], "text": r["text"],
                    "sign": torch.from_numpy(np.ascontiguousarray(sign))})
    return out


# ---------------------------------------------------------------- Gate 0 fix candidate "fix1"
# Added after the first Gate 0 FAIL, from gate0_diag.json and a 150 x 534 cross-correlation of the authors' PT test
# skels against the SLRTP test poses (frame counts equal on all 641 clips). Two deterministic differences were found:
#   - side labels: the authors' PT joints 2-4 and hand 29-49 track SLRTP 3-5 and hand 29-49, and PT 5-7 and hand 8-28
#     track SLRTP 0-2 and hand 8-28 (within-clip r up to 0.94 on y, 0.77 on x), i.e. slrtp178_to_pt50 is side-swapped
#     against the authors' data (its layout follows Sign-IDD helpers.py, which the authors' files do not);
#   - nose: the SLRTP face is centred per frame, so its landmark 82 carries no head position.
# fix1 = the side-swapped joint map, one global affine map (3x3 + offset) fitted by least squares on the DEV split only
# (SLRTP dev.pt against the authors' PT dev skels, frame-aligned clips, joints 1-49), then nose = mapped neck + the
# authors' dev-mean (nose - neck). No test data enters the fit. Applied identically to every bank if it passes.
PT_FROM_SLRTP_FIX1 = {2: 3, 3: 4, 4: 5, 5: 0, 6: 1, 7: 2, **{j: j for j in range(8, 50)}}
FIX1 = OUT / "fix1_params.npz"
IDENT = {"affine": np.vstack([np.eye(3), np.zeros((1, 3))]), "nose_offset": np.zeros(3)}


def map_fix1(pose178, params=IDENT) -> np.ndarray:
    """[T, 178, 3] -> [T, 50, 3] in the authors' PT coordinates (fix1)."""
    q = np.asarray(pose178, dtype=np.float64)
    out = np.zeros((q.shape[0], 50, 3))
    out[:, 1] = 0.5 * (q[:, 0] + q[:, 3])
    for p, s_ in PT_FROM_SLRTP_FIX1.items():
        out[:, p] = q[:, s_]
    M = np.asarray(params["affine"], dtype=np.float64)
    out = out @ M[:3] + M[3]
    out[:, 0] = out[:, 1] + np.asarray(params["nose_offset"], dtype=np.float64)
    return out.astype(np.float32)


def fit_fix1() -> dict:
    import torch
    pt = {bare(x["name"]): x["sign"].numpy().astype(np.float64).reshape(-1, 50, 3) for x in load_pt_gt(PT_GT_DEV)}
    dev = torch.load(C.DATA / "dev.pt", map_location="cpu", weights_only=False)
    P, Q, keys = [], [], []
    for k, v in dev.items():
        q = pose_of(v)
        if k in pt and q.shape[0] == pt[k].shape[0]:
            P.append(pt[k][:, 1:]); Q.append(map_fix1(q)[:, 1:].astype(np.float64)); keys.append(k)
    n_dev = len(dev)
    del dev
    P = np.concatenate(P).reshape(-1, 3); Q = np.concatenate(Q).reshape(-1, 3)
    Qh = np.concatenate([Q, np.ones((len(Q), 1))], 1)
    M, *_ = np.linalg.lstsq(Qh, P, rcond=None)
    res = P - Qh @ M
    allp = np.concatenate([pt[k] for k in keys])
    params = {"affine": M, "nose_offset": (allp[:, 0] - allp[:, 1]).mean(0)}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(FIX1, **params)
    rep = {"fit_split": "dev", "n_dev_slrtp": n_dev, "n_dev_authors": len(pt), "n_clips_frame_aligned": len(keys),
           "affine": M.round(5).tolist(), "nose_offset": params["nose_offset"].round(5).tolist(),
           "dev_fit_r2_per_axis": (1 - res.var(0) / P.var(0)).round(4).tolist(),
           "dev_fit_resid_rms_over_pt_std": float(np.sqrt((res ** 2).mean()) / P.std()), "sha256": C.sha256(FIX1)}
    (OUT / "fix1_fit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return params


def map_swap_neckcentred(pose178) -> np.ndarray:
    """Parameter-free normalization found by check 2: the authors' PT neck is exactly (0, 0, 0) in every frame (neck-
    centred per frame; shoulder width 0.3345 +- 0.002), the SLRTP neck is not (mean y 0.160, z 0.113). Side-swapped
    joint map, then the per-frame neck subtracted from every joint. The nose is left at the neck (it has no SLRTP
    source), so agreement for this map is quoted on joints 1-49."""
    out = map_fix1(pose178).astype(np.float64)
    out = out - out[:, 1:2]
    return out.astype(np.float32)


def mapper_for(fix):
    if not fix:
        return slrtp178_to_pt50
    z = np.load(FIX1)
    params = {k: z[k] for k in z.files}
    return lambda p: map_fix1(p, params)


def tag_of(name, fix):
    return name + (f"__{fix}" if fix else "")


def read_hyps(path) -> tuple[list, list]:
    names, hyps = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        n, h = line.split("|", 1)
        names.append(n); hyps.append(h)
    return names, hyps


# ---------------------------------------------------------------- the evaluator's own metrics
def slt_metrics():
    sys.path.insert(0, str(SLT_DIR))
    from signjoey.metrics import bleu, wer_list, wer_single  # noqa: E402
    from signjoey.phoenix_utils.phoenix_cleanup import clean_phoenix_2014_trans  # noqa: E402
    return bleu, wer_list, wer_single, clean_phoenix_2014_trans


def refs_as_slt(items: list, clean) -> tuple[list, list]:
    """Reference strings exactly as signjoey builds them (data.py fields + prediction.py)."""
    txt = [" ".join(x["text"].strip().lower().split()) for x in items]
    gls = [clean(" ".join(x["gloss"].strip().split())) for x in items]
    return txt, gls


def corpus(txt_hyp, txt_ref, gls_hyp, gls_ref):
    bleu, wer_list, _, _ = slt_metrics()
    b = bleu(references=txt_ref, hypotheses=txt_hyp)
    w = wer_list(references=gls_ref, hypotheses=gls_hyp)
    return {"bleu1": b["bleu1"], "bleu4": b["bleu4"], "wer": w["wer"], "del": w["del_rate"], "ins": w["ins_rate"],
            "sub": w["sub_rate"], "n": len(txt_hyp)}


def load_pt_gt(path=PT_GT_TEST) -> list:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def slrtp_keys() -> list:
    return C.TEST_KEYS.read_text().split()


# ---------------------------------------------------------------- build / slt
def cmd_build(a):
    import torch
    name = tag_of(a.name, a.fix)
    (OUT / "inputs").mkdir(parents=True, exist_ok=True); (OUT / "config").mkdir(exist_ok=True)
    pt = load_pt_gt()
    keys = slrtp_keys()
    refs = common_refs(pt, keys)
    bank = torch.load(BANKS[a.name], map_location="cpu", weights_only=False)
    assert set(bank) == set(keys), f"{name}: bank keys differ from the 641 pinned keys"
    items = to_slt_items(bank, refs, mapper_for(a.fix))
    del bank
    dst = OUT / f"inputs/{name}.test.gz"
    with gzip.open(dst, "wb") as f:
        pickle.dump(items, f)
    lens = [int(x["sign"].shape[0]) for x in items]
    rec = {"bank": str(BANKS[a.name]), "bank_sha256": C.sha256(BANKS[a.name]), "mapping": a.fix or "slrtp178_to_pt50",
           "fix_params_sha256": C.sha256(FIX1) if a.fix else None, "n": len(items), "n_pt_test": len(pt),
           "missing_from_slrtp": sorted(set(bare(x["name"]) for x in pt) - set(keys)),
           "mean_frames": float(np.mean(lens)), "feature_size": int(items[0]["sign"].shape[1]),
           "finite": bool(all(torch.isfinite(x["sign"]).all() for x in items)),
           "coord_absmax": float(max(x["sign"].abs().max() for x in items)),
           "order": "authors' PT test order restricted to the SLRTP keys",
           "names_sha256": hashlib.sha256("\n".join(x["name"] for x in items).encode()).hexdigest(),
           "input_sha256": C.sha256(dst)}
    (OUT / f"inputs/{name}.json").write_text(json.dumps(rec, indent=1))
    import yaml
    cfg = yaml.safe_load(R0_CFG.read_text())
    cfg["data"].update({"data_path": str(OUT) + "/", "train": str(PT_GT_DEV), "dev": str(PT_GT_DEV),
                        "ground_dev": str(PT_GT_DEV), "test": str(dst), "ground_test": str(dst)})
    cfg["testing"].update(PINNED)
    cfg["training"]["model_dir"] = str(OUT / f"runs/{name}")
    (OUT / f"config/{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps(rec, indent=1))
    ok = rec["n"] == 641 and rec["feature_size"] == 150 and rec["finite"]
    return 0 if ok else 1


def cmd_slt(a):
    name = tag_of(a.name, a.fix)
    assert C.sha256(SLT_CKPT) == SLT_CKPT_SHA
    (OUT / f"runs/{name}").mkdir(parents=True, exist_ok=True); (OUT / "logs").mkdir(exist_ok=True)
    cmd = [C.PY, str(HERE / "slt_run.py"), "test", str(OUT / f"config/{name}.yaml"), "--ckpt", str(SLT_CKPT),
           "--output_path", str(OUT / f"runs/{name}/{name}")]
    return C.run(cmd, log=OUT / f"logs/{name}.log")


# ---------------------------------------------------------------- Gate 0
def cmd_diag(a):
    """SLRTP ground truth after the mapping vs the authors' PT-150 test skels, on the common keys."""
    import torch
    pt = {bare(x["name"]): x["sign"].numpy().astype(np.float64) for x in load_pt_gt()}
    gt = torch.load(BANKS["gate0_slrtp_gt"], map_location="cpu", weights_only=False)
    keys = [k for k in slrtp_keys() if k in pt]
    dT, A, B = [], [], []
    mapper = mapper_for(a.fix)
    for k in keys:
        q = mapper(pose_of(gt[k])).reshape(-1, 150).astype(np.float64)
        p = pt[k]
        dT.append(q.shape[0] - p.shape[0])
        if q.shape[0] == p.shape[0]:
            A.append(p); B.append(q)
    del gt
    rep = {"n_common": len(keys), "n_equal_frames": len(A), "frames_diff_median": float(np.median(dT)),
           "frames_diff_abs_mean": float(np.mean(np.abs(dT))), "frames_ratio_slrtp_over_pt_median":
           float(np.median([(d + pt[k].shape[0]) / pt[k].shape[0] for d, k in zip(dT, keys)]))}
    allp = np.concatenate([pt[k] for k in keys]).reshape(-1, 50, 3)
    allq = np.concatenate(B).reshape(-1, 50, 3) if B else None
    rep["pt_axis_mean"] = allp.reshape(-1, 3).mean(0).round(4).tolist()
    rep["pt_axis_std"] = allp.reshape(-1, 3).std(0).round(4).tolist()
    rep["pt_shoulder_width_median"] = float(np.median(np.linalg.norm(allp[:, 2] - allp[:, 5], axis=-1)))
    if allq is not None:
        rep["slrtp_axis_mean"] = allq.reshape(-1, 3).mean(0).round(4).tolist()
        rep["slrtp_axis_std"] = allq.reshape(-1, 3).std(0).round(4).tolist()
        rep["slrtp_shoulder_width_median"] = float(np.median(np.linalg.norm(allq[:, 2] - allq[:, 5], axis=-1)))
        P, Q = np.concatenate(A).reshape(-1, 50, 3), allq
        # best global affine map pt ~ [q, 1] @ M, pooled over frames and joints (12 parameters)
        Qh = np.concatenate([Q.reshape(-1, 3), np.ones((Q.shape[0] * 50, 1))], 1)
        M, *_ = np.linalg.lstsq(Qh, P.reshape(-1, 3), rcond=None)
        res = P.reshape(-1, 3) - Qh @ M
        rep["affine_M"] = M.round(4).tolist()
        rep["affine_r2_per_axis"] = (1 - res.var(0) / P.reshape(-1, 3).var(0)).round(4).tolist()
        rep["affine_resid_rms_over_pt_std"] = float(np.sqrt((res ** 2).mean()) / P.reshape(-1, 3).std())
        # the same per joint group, and per-dimension correlation without any fit
        r = [np.corrcoef(P.reshape(-1, 150)[:, d], Q.reshape(-1, 150)[:, d])[0, 1] for d in range(150)]
        rep["per_dim_corr_median"] = float(np.nanmedian(r))
        rep["per_dim_corr_by_group"] = {g: float(np.nanmedian([r[j * 3 + c] for j in js for c in range(3)]))
                                        for g, js in (("nose_neck", (0, 1)), ("arms", range(2, 8)),
                                                      ("left_hand", range(8, 29)), ("right_hand", range(29, 50)))}
        rep["max_abs_diff_raw"] = float(np.abs(P - Q).max())
        rep["resid_rms_raw_over_pt_std"] = float(np.sqrt(((P - Q) ** 2).mean()) / P.std())
    OUT.mkdir(parents=True, exist_ok=True)
    rep["mapping"] = a.fix or "slrtp178_to_pt50"
    (OUT / f"gate0_diag{'__' + a.fix if a.fix else ''}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def run_outputs(prefix, split="test"):
    g = read_hyps(str(prefix) + SUFFIX["gls"].format(split))
    t = read_hyps(str(prefix) + SUFFIX["txt"].format(split))
    assert g[0] == t[0]
    return g[0], t[1], g[1]


def log_test_line(log):
    m = re.findall(r"\[TEST\].*?WER ([\d.]+).*?BLEU-4 ([\d.]+)\s*\(BLEU-1: ([\d.]+)", Path(log).read_text(), re.S)
    return tuple(map(float, m[-1])) if m else None


def cmd_gate0(a):
    _, _, _, clean = slt_metrics()
    pt = load_pt_gt()
    keys = set(slrtp_keys())
    # 1. reproduce R0's recorded numbers from its saved hypotheses with this scorer (all 642)
    names0, txt0, gls0 = run_outputs(R0_PREFIX)
    assert names0 == [x["name"] for x in pt]
    tref, gref = refs_as_slt(pt, clean)
    r0_full = corpus(txt0, tref, gls0, gref)
    rep = {"R0_recorded": R0_RECORDED, "R0_rescored_642": r0_full,
           "R0_repro_diff": {"bleu4": round(r0_full["bleu4"], 2) - R0_RECORDED["bleu4"],
                             "wer": round(r0_full["wer"], 2) - R0_RECORDED["wer"]}}
    repro_ok = abs(rep["R0_repro_diff"]["bleu4"]) < 1e-9 and abs(rep["R0_repro_diff"]["wer"]) < 1e-9
    C.record("their-evaluator R0 rescoring reproduces", "this scorer on R0's saved 642 hypotheses gives the recorded "
             "BLEU-4 11.93 / WER 71.94 (difference 0 at 2 dp)",
             f"BLEU-4 {r0_full['bleu4']:.4f}, WER {r0_full['wer']:.4f}; difference {rep['R0_repro_diff']}", repro_ok,
             OUT / "gate0.json")
    if not repro_ok:
        (OUT / "gate0.json").write_text(json.dumps(rep, indent=1)); return 2
    # 2. R0 restricted to the 641 common keys vs the Gate 0 run
    sel = [i for i, x in enumerate(pt) if bare(x["name"]) in keys]
    tag = tag_of("gate0_slrtp_gt", a.fix)
    names_g, txt_g, gls_g = run_outputs(OUT / f"runs/{tag}/{tag}")
    assert names_g == [names0[i] for i in sel], "Gate 0 run order differs from R0 on the common keys"
    items = [pt[i] for i in sel]
    tref, gref = refs_as_slt(items, clean)
    r0 = corpus([txt0[i] for i in sel], tref, [gls0[i] for i in sel], gref)
    g0 = corpus(txt_g, tref, gls_g, gref)
    same_t = [txt0[i] == h for i, h in zip(sel, txt_g)]
    same_g = [gls0[i] == h for i, h in zip(sel, gls_g)]
    both = float(np.mean([x and y for x, y in zip(same_t, same_g)]))
    logged = log_test_line(OUT / f"logs/{tag}.log")
    rep.update({"n": len(sel), "R0_641": r0, "gate0_641": g0, "identical_text": float(np.mean(same_t)),
                "identical_gloss": float(np.mean(same_g)), "identical_both": both,
                "delta_bleu4": g0["bleu4"] - r0["bleu4"], "delta_wer": g0["wer"] - r0["wer"],
                "gate0_log_test_line_wer_b4_b1": logged,
                "rule": "identical_both >= 0.99 and |delta BLEU-4| <= 0.2 and |delta WER| <= 0.5"})
    ok = both >= 0.99 and abs(rep["delta_bleu4"]) <= 0.2 and abs(rep["delta_wer"]) <= 0.5
    rep["pass"] = ok
    rep["mapping"] = a.fix or "slrtp178_to_pt50"
    diag = OUT / f"gate0_diag{'__' + a.fix if a.fix else ''}.json"
    if diag.exists():
        rep["diag"] = json.loads(diag.read_text())
    g0path = OUT / f"gate0{'__' + a.fix if a.fix else ''}.json"
    g0path.write_text(json.dumps(rep, indent=1))
    C.record(f"their-evaluator Gate 0 (SLRTP GT via {rep['mapping']} vs R0, 641 keys)",
             "identical hypotheses >= 99%, |dBLEU-4| <= 0.2, |dWER| <= 0.5 against R0 on the same 641 keys",
             f"identical text {rep['identical_text']:.3f}, gloss {rep['identical_gloss']:.3f}, both {both:.3f}; "
             f"BLEU-4 {g0['bleu4']:.2f} vs R0 {r0['bleu4']:.2f} (d {rep['delta_bleu4']:+.2f}); "
             f"WER {g0['wer']:.2f} vs R0 {r0['wer']:.2f} (d {rep['delta_wer']:+.2f})", ok, g0path)
    print(json.dumps({k: v for k, v in rep.items() if k != "diag"}, indent=1))
    return 0 if ok else 1


# ---------------------------------------------------------------- conversion checks (coordinator, 2026-09-29)
FGDM_PT = ROOT / "external/baselines/FGDM-main/Data/PHOENIX14T"
FILL_STATS = RB / "gate0/fill_stats.npz"
MANUAL = list(range(0, 6)) + list(range(8, 50))
TOL = 1e-4


def _agreement(pt: dict, gt: dict, keys: list, mapper) -> dict:
    """Frame-by-frame |converted SLRTP ground truth - the authors' PT ground truth| on the matched clips."""
    D, nT = [], 0
    for k in keys:
        q = mapper(pose_of(gt[k])).astype(np.float64)
        p = pt[k].reshape(q.shape[0] if pt[k].shape[0] == q.shape[0] else -1, q.shape[1], 3)
        if q.shape[0] != p.shape[0]:
            nT += 1; continue
        D.append(np.abs(q - p))
    D = np.concatenate(D)  # [frames, 50, 3]
    ax = "xyz"
    return {"n_clips_frame_count_mismatch": nT, "n_frames": int(D.shape[0]),
            "per_axis": {ax[c]: {"max": float(D[:, :, c].max()), "median": float(np.median(D[:, :, c]))} for c in range(3)},
            "per_joint": {str(j): {"max": float(D[:, j].max()), "median": float(np.median(D[:, j]))} for j in range(D.shape[1])},
            "overall_max": float(D.max()), "overall_median": float(np.median(D))}


def _best_match(pt: dict, gt: dict, keys: list) -> dict:
    """For each PT joint, the SLRTP manual joint (or the shoulder midpoint) whose x and y track it best within clips."""
    cand = {f"slrtp{j}": (lambda q, j=j: q[:, j]) for j in MANUAL}
    cand["shoulder_mid"] = lambda q: 0.5 * (q[:, 0] + q[:, 3])
    P, Q = [], {c: [] for c in cand}
    for k in keys:
        p = pt[k].reshape(-1, 50, 3); q = pose_of(gt[k]).astype(np.float64)
        if p.shape[0] != q.shape[0]:
            continue
        P.append(p[:, :, :2] - p[:, :, :2].mean(0))
        for c, f in cand.items():
            v = f(q)[:, :2]; Q[c].append(v - v.mean(0))
    P = np.concatenate(P); Q = {c: np.concatenate(v) for c, v in Q.items()}

    def r(a, b):
        return float(np.mean([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(2)]))
    out = {}
    for j in range(50):
        sc = {c: r(P[:, j], Q[c]) for c in cand}
        best = max(sc, key=sc.get)
        out[str(j)] = {"best": best, "r_xy": round(sc[best], 3)}
    return out


def cmd_checks(a):
    """Checks 1-4 of the coordinator's message of 2026-09-29, each recorded as a ledger line."""
    import torch
    from slrtp_pt_mapping import pt50_to_slrtp178, load_stats
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    pt_items = load_pt_gt()
    keys = slrtp_keys()
    gt = torch.load(BANKS["gate0_slrtp_gt"], map_location="cpu", weights_only=False)
    # 1. key alignment
    files = (FGDM_PT / "test.files").read_text().split()
    ftext = (FGDM_PT / "test.text").read_text().splitlines()
    fgloss = (FGDM_PT / "test.gloss").read_text().splitlines()
    ptn = [bare(x["name"]) for x in pt_items]
    matched = [k for k in ptn if k in set(keys)]
    unmatched_pt = [k for k in ptn if k not in set(keys)]
    unmatched_slrtp = [k for k in keys if k not in set(ptn)]
    inp = OUT / "inputs/gate0_slrtp_gt.test.gz"
    with gzip.open(inp, "rb") as f:
        inp_names = [x["name"] for x in pickle.load(f)]
    ref_order = [f"test/{k}" for k in files if k in set(keys)]
    fi = {k: i for i, k in enumerate(files)}
    rep["check1_keys"] = {
        "n_slrtp": len(keys), "n_pt": len(ptn), "n_matched_by_name": len(matched), "pt_without_slrtp": unmatched_pt,
        "slrtp_without_pt": unmatched_slrtp, "pt_pickle_order_equals_test_files": ptn == files,
        "input_order_equals_test_files_order_on_matched": inp_names == ref_order,
        "input_order_equals_pt_pickle_order_on_matched": inp_names == [f"test/{k}" for k in matched],
        "pickle_text_equals_test_text_file": sum(pt_items[i]["text"].strip() == ftext[fi[ptn[i]]].strip() for i in range(len(ptn))),
        "pickle_gloss_equals_test_gloss_file": sum(pt_items[i]["gloss"].strip() == fgloss[fi[ptn[i]]].strip() for i in range(len(ptn))),
        "slrtp_gloss_equals_pt_gloss": sum(gt[k]["gloss"].strip() == pt_items[ptn.index(k)]["gloss"].strip() for k in matched)}
    c1 = rep["check1_keys"]
    ok1 = (c1["n_matched_by_name"] == 641 and c1["slrtp_without_pt"] == [] and len(c1["pt_without_slrtp"]) == 1
           and c1["input_order_equals_test_files_order_on_matched"] and c1["pt_pickle_order_equals_test_files"]
           and c1["pickle_text_equals_test_text_file"] == len(ptn) and c1["pickle_gloss_equals_test_gloss_file"] == len(ptn))
    C.record("their-evaluator check 1 key alignment", "641 SLRTP keys matched to PT test by name; one PT key unmatched; "
             "SLT input order = order of the authors' test.files/.text/.gloss", f"matched {c1['n_matched_by_name']}/641; "
             f"PT key without SLRTP: {c1['pt_without_slrtp']}; input order = test.files order: "
             f"{c1['input_order_equals_test_files_order_on_matched']}; reference text/gloss = test.text/.gloss lines: "
             f"{c1['pickle_text_equals_test_text_file']}/{c1['pickle_gloss_equals_test_gloss_file']} of {len(ptn)}", ok1,
             OUT / "checks.json")
    # 2. numerical agreement with the authors' ground truth, frame by frame
    pt = {bare(x["name"]): x["sign"].numpy().astype(np.float64) for x in pt_items}
    ag = _agreement(pt, gt, matched, slrtp178_to_pt50)
    rep["check2_agreement_documented_mapping"] = ag
    rep["check2_best_matching_slrtp_joint_per_pt_joint"] = _best_match(pt, gt, matched)
    if FIX1.exists():
        rep["check2_agreement_fix1_for_information"] = _agreement(pt, gt, matched, mapper_for("fix1"))
    sn = _agreement({k: v.reshape(-1, 50, 3)[:, 1:].reshape(v.shape[0], -1) for k, v in pt.items()}, gt, matched,
                    lambda q: map_swap_neckcentred(q)[:, 1:])
    rep["check2_agreement_swap_neckcentred_joints1to49_for_information"] = sn
    ok2 = ag["n_clips_frame_count_mismatch"] == 0 and ag["overall_max"] <= TOL
    worst = sorted(ag["per_joint"].items(), key=lambda kv: -kv[1]["median"])[:3]
    bm = rep["check2_best_matching_slrtp_joint_per_pt_joint"]
    C.record("their-evaluator check 2 agreement with the authors' GT (slrtp178_to_pt50)",
             f"frame counts equal per clip; max |converted - authors' PT GT| <= {TOL} per coordinate and per joint",
             f"frame-count mismatches {ag['n_clips_frame_count_mismatch']}/641; max x/y/z "
             f"{ag['per_axis']['x']['max']:.3f}/{ag['per_axis']['y']['max']:.3f}/{ag['per_axis']['z']['max']:.3f}, "
             f"median x/y/z {ag['per_axis']['x']['median']:.3f}/{ag['per_axis']['y']['median']:.3f}/"
             f"{ag['per_axis']['z']['median']:.3f}; worst joints by median {[(j, round(v['median'], 3)) for j, v in worst]}; "
             f"best-tracking SLRTP joint for PT 2,3,4,5,6,7 = {[bm[str(j)]['best'] for j in range(2, 8)]}", ok2,
             OUT / "checks.json")
    # 3. round trip on the manual joints, all 641 clips
    stats = load_stats(str(FILL_STATS))
    rt = max(float(np.abs(pt50_to_slrtp178(slrtp178_to_pt50(pose_of(gt[k])), stats)[:, MANUAL]
                          - pose_of(gt[k])[:, MANUAL]).max()) for k in keys)
    rep["check3_round_trip_manual_max_abs"] = rt
    C.record("their-evaluator check 3 round trip", "pt50_to_slrtp178(slrtp178_to_pt50(x)) = x on manual joints 0-5, "
             "8-49, all 641 test clips", f"max abs error {rt:.1e}", rt == 0.0, OUT / "checks.json")
    del gt
    # 4. unit test (real clip included)
    r = subprocess.run([C.PY, str(HERE / "test_their_evaluator.py")], capture_output=True, text=True)
    rep["check4_unit_test"] = {"rc": r.returncode, "stdout": r.stdout.strip().splitlines(), "stderr_tail": r.stderr[-800:]}
    C.record("their-evaluator check 4 unit test", "shape [T,150], float32, first and last joint order, one known real-clip "
             "frame value, key order", f"rc {r.returncode}; {len(rep['check4_unit_test']['stdout'])} tests ok",
             r.returncode == 0, OUT / "checks.json")
    rep["all_pass"] = bool(ok1 and ok2 and rt == 0.0 and r.returncode == 0)
    (OUT / "checks.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"check1": c1, "check2_per_axis": ag["per_axis"], "check2_overall": [ag["overall_max"], ag["overall_median"]],
                      "check2_fix1_per_axis": rep.get("check2_agreement_fix1_for_information", {}).get("per_axis"),
                      "check2_swap_neckcentred_per_axis": sn["per_axis"],
                      "check3": rt, "check4_rc": r.returncode, "all_pass": rep["all_pass"]}, indent=1))
    return 0 if rep["all_pass"] else 1


# ---------------------------------------------------------------- score + bootstrap
def cmd_score(a):
    g0 = json.loads((OUT / f"gate0{'__' + a.fix if a.fix else ''}.json").read_text())
    if not g0.get("pass"):
        print("Gate 0 did not pass; no bank is scored", file=sys.stderr); return 3
    bleu, wer_list, wer_single, clean = slt_metrics()
    pt = load_pt_gt()
    keys = set(slrtp_keys())
    items = [x for x in pt if bare(x["name"]) in keys]
    tref, gref = refs_as_slt(items, clean)
    _, dev_t0, dev_g0 = run_outputs(R0_PREFIX, "dev")
    res, hyp_t, hyp_g = {}, {}, {}
    for name in OURS + BASELINES:
        tg = tag_of(name, a.fix)
        prefix = OUT / f"runs/{tg}/{tg}"
        names, t, g = run_outputs(prefix)
        assert names == [x["name"] for x in items], name
        _, dt, dg = run_outputs(prefix, "dev")
        m = corpus(t, tref, g, gref)
        logged = log_test_line(OUT / f"logs/{tg}.log")
        m["log_test_line_wer_b4_b1"] = logged
        m["log_agrees_2dp"] = bool(logged and abs(round(m["wer"], 2) - logged[0]) < 1e-9
                                   and abs(round(m["bleu4"], 2) - logged[1]) < 1e-9)
        m["dev_identical_to_R0"] = bool(dt == dev_t0 and dg == dev_g0)
        m["input"] = json.loads((OUT / f"inputs/{tg}.json").read_text())
        res[name], hyp_t[name], hyp_g[name] = m, t, g
    # paired bootstrap: scripts/bootstrap_strict_cac.paired_bootstrap on the saved hypotheses. BLEU on the text
    # hypotheses. WER on the gloss hypotheses, with the per-item WER statistic set to the evaluator's own
    # (signjoey wer_single: weighted-cost alignment, uint8 matrix, no punctuation stripping), so the bootstrap's
    # point WER is exactly the SLT's gloss WER. The jiwer default is kept below as a recorded comparison.
    sys.path.insert(0, str(ROOT))
    import scripts.bootstrap_strict_cac as BS
    pairs = tuple((f"{o}_vs_{b}", o, b) for o in OURS for b in BASELINES)
    bt = BS.paired_bootstrap(hyp_t, tref, pairs=pairs, n_boot=N_BOOT, seed=SEED)
    jiwer_point = {n: float(BS.metrics_from_stats(BS.sufficient_stats(hyp_g[n], gref))[0][2]) for n in hyp_g}

    def _slt_wer_stats(hyp, ref):
        r = wer_single(r=ref, h=hyp)
        return int(r["num_err"]), int(r["num_ref"])
    orig = BS.per_hyp_wer_stats
    BS.per_hyp_wer_stats = _slt_wer_stats
    try:
        bg = BS.paired_bootstrap(hyp_g, gref, pairs=pairs, n_boot=N_BOOT, seed=SEED)
    finally:
        BS.per_hyp_wer_stats = orig
    agree = {n: {"bleu4_bootstrap_minus_slt": bt["methods"][n]["bleu4"] - res[n]["bleu4"],
                 "wer_bootstrap_minus_slt": bg["methods"][n]["corpus_wer"] - res[n]["wer"],
                 "wer_jiwer_minus_slt": jiwer_point[n] - res[n]["wer"]} for n in res}
    agree_ok = all(abs(v["bleu4_bootstrap_minus_slt"]) < 1e-9 and abs(v["wer_bootstrap_minus_slt"]) < 1e-9
                   for v in agree.values())
    deltas = {}
    for p, o, b in pairs:
        cb, cg = bt["comparisons"][p], bg["comparisons"][p]
        deltas[p] = {"bleu4": {"point": cb["point_delta"]["bleu4"], **cb["bootstrap_delta"]["bleu4"]},
                     "bleu1": {"point": cb["point_delta"]["bleu1"], **cb["bootstrap_delta"]["bleu1"]},
                     "wer": {"point": cg["point_delta"]["corpus_wer"], **cg["bootstrap_delta"]["corpus_wer"]}}
    out = {"evaluator": {"ckpt": str(SLT_CKPT), "sha256": SLT_CKPT_SHA, "pinned": PINNED,
                         "dev": "authors' ground-truth dev skels (same for every run)", "references": "authors' test "
                         "gloss and text (R0 ground-truth pickle)", "n": len(items), "wer": "SLT gloss-recognition WER"},
           "published": PUBLISHED, "gate0": {k: g0[k] for k in ("identical_both", "delta_bleu4", "delta_wer", "pass")},
           "results": res, "bootstrap": {"seed": SEED, "n_boot": N_BOOT, "deltas": deltas,
                                         "orientation": "ours minus baseline; negative WER delta favours ours"},
           "agreement": agree, "agreement_ok": agree_ok,
           "input_side_note": "Sign-IDD and FGDM (published rows) took ground-truth glosses and ground-truth lengths; "
                              "our routes take German text only."}
    (OUT / "results.json").write_text(json.dumps(out, indent=1))
    for n in OURS + BASELINES:
        m = res[n]
        C.record(f"their-evaluator score {n}", "authors' SLT, pinned R0 decoding, 641 test keys; log line and dev "
                 "outputs identical to R0 dev", f"BLEU-1 {m['bleu1']:.2f}, BLEU-4 {m['bleu4']:.2f}, WER {m['wer']:.2f}",
                 m["log_agrees_2dp"] and m["dev_identical_to_R0"], OUT / "results.json")
    C.record("their-evaluator paired bootstrap (ours vs baselines)", f"seed {SEED}, {N_BOOT} resamples; bootstrap point "
             "values equal the SLT's BLEU-4 and gloss WER", "; ".join(
                 f"{p}: dB4 {d['bleu4']['point']:+.2f} [{d['bleu4']['ci95'][0]:+.2f},{d['bleu4']['ci95'][1]:+.2f}] "
                 f"dWER {d['wer']['point']:+.2f} [{d['wer']['ci95'][0]:+.2f},{d['wer']['ci95'][1]:+.2f}]"
                 for p, d in deltas.items()), agree_ok, OUT / "results.json")
    print(json.dumps({"results": {n: {k: res[n][k] for k in ("bleu1", "bleu4", "wer")} for n in res},
                      "agreement_ok": agree_ok}, indent=1))
    return 0 if agree_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "slt", "diag", "gate0", "score", "fitfix", "checks"))
    ap.add_argument("name", nargs="?", choices=tuple(BANKS))
    ap.add_argument("--fix", choices=("fix1",), default=None, help="Gate 0 fix candidate (see fit_fix1)")
    a = ap.parse_args()
    if a.cmd in ("build", "slt") and not a.name:
        ap.error("build and slt need a bank name")
    if a.cmd == "fitfix":
        fit_fix1(); sys.exit(0)
    sys.exit({"build": cmd_build, "slt": cmd_slt, "diag": cmd_diag, "gate0": cmd_gate0, "score": cmd_score,
              "checks": cmd_checks}[a.cmd](a))


if __name__ == "__main__":
    main()
