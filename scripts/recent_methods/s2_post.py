"""Post-score measurements for one 641-item bank, each reproducing a recorded value first.

  ratios    --pred P.pt --name N --outdir D --step S
            K ratios (scripts/exp_revision_motion_ratios.py: hand speed, jerk, variation) and the
            variance ratios (scripts/signbase_variance_ratio.py: duration, pose and motion variance,
            manual joints). The Sign-Base bank is re-run first and must reproduce its recorded values.
  audit     --pred P.pt --name N --outdir D --step S [--min_share_nonpad 0.95]
            scripts/audit_subsequence_memorization.py (16-frame windows, stride 4, threshold 0.02) on
            the bank plus the whole-clip replay positive control (text1nn_tfidf_test.pt, must hit
            641/641), then the frame-level replay share with the same window signatures and threshold:
            a frame is flagged when any window covering it lies within 0.02 of a training window.
            The per-clip minimum of the frame-level pass must agree with the official script.
  bootstrap --text_preds T.pt --name N --outdir D --step S [--no_overall]
            scripts/bootstrap_clean_route_comparisons.py (seed 30373, 10,000 paired resamples) of
            N against the fixed 40% route; the fixed bank must reproduce 15.1032 / 85.582 with sha
            49db5307... first. Then the same paired bootstrap inside the two strata fixed by the
            route before test (ledger mode: 192 whole-clip replays, 449 assembled requests).
exit 0 = pass or recorded, 1 = gate failed, 2 = reproduction or invariant failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_common as C  # noqa: E402

WIN, STRIDE, THR = 16, 4, 0.02


# ----------------------------------------------------------------------------- ratios
def cmd_ratios(a):
    wd = Path(a.outdir); wd.mkdir(parents=True, exist_ok=True)
    log = wd / f"ratios_{a.name}.log"
    test_pt = C.DATA / "test.pt"
    sb = wd / f"repro_signbase_K_ratios_before_{a.name}.json"
    if not a.dry:
        C.run([C.PY, C.ROOT / "scripts/exp_revision_motion_ratios.py", "--pred_pt", C.SIGNBASE_BANK, "--gt_pt", test_pt,
               "--out_json", sb], log=log)
    r = C.load_json(sb)
    dk = {k: r[k] - v for k, v in C.SIGNBASE_K.items()}
    if any(abs(v) > 1e-12 for v in dk.values()):
        C.record(a.step, "Sign-Base K ratios reproduce (difference 0)", json.dumps(dk), False, sb); return 2
    kj = wd / f"{a.name}_K_ratios.json"
    vj = wd / f"{a.name}_and_signbase_variance_ratio.json"
    if not a.dry:
        C.run([C.PY, C.ROOT / "scripts/exp_revision_motion_ratios.py", "--pred_pt", Path(a.pred).resolve(), "--gt_pt",
               test_pt, "--out_json", kj], log=log)
        C.run([C.PY, C.ROOT / "scripts/signbase_variance_ratio.py", "--pred", Path(a.pred).resolve(), C.SIGNBASE_BANK,
               "--out", vj], log=log)
    k = C.load_json(kj)
    v = C.load_json(vj)
    runs = {Path(x["pred_file"]["path"]).name: x for x in v["runs"]}
    sbv = runs[C.SIGNBASE_BANK.name]
    dv = {"pose_manual": sbv["pose_variance_ratio"]["mean_ratio_manual"] - C.SIGNBASE_VAR["pose_manual"],
          "motion_manual": sbv["motion_variance_ratio"]["mean_ratio_manual"] - C.SIGNBASE_VAR["motion_manual"]}
    if any(abs(x) > 1e-12 for x in dv.values()):
        C.record(a.step, "Sign-Base variance ratios reproduce (difference 0)", json.dumps(dv), False, vj); return 2
    me = runs[Path(a.pred).name]
    pv, mv = me["pose_variance_ratio"]["mean_ratio_manual"], me["motion_variance_ratio"]["mean_ratio_manual"]
    flag = "variation ratio below 0.3 = mode collapse" if k["hand_posestd_ratio"] < 0.3 else \
        ("variance ratio far above 1 = scale suspect" if pv > 2.0 else "within band")
    num = (f"speed {k['hand_speed_ratio']:.3f}, jerk {k['hand_jerk_ratio']:.3f}, variation {k['hand_posestd_ratio']:.3f}; "
           f"mean variance ratio (manual) pose {pv:.3f}, motion {mv:.3f}; duration {me['duration']['mean_ratio_pred_over_gt']:.3f} "
           f"({k['n_clips']}/641 clips; Sign-Base reproduced, difference 0); {flag}")
    C.record(a.step, "K ratios and variance ratios, Sign-Base reproduced first", num, "recorded", kj,
             extra={"K": k, "variance_json": str(vj)})
    return 0


# ----------------------------------------------------------------------------- audit
def window_starts(T):
    Tp = max(T, WIN)
    s = list(range(0, Tp - WIN + 1, STRIDE))
    if s[-1] != Tp - WIN:
        s.append(Tp - WIN)
    return s


def frame_level(pred_paths: dict, dry=False):
    """Per-window minimum distance to the training windows, frame flags, per-clip minimum."""
    import numpy as np
    import torch
    sys.path.insert(0, str(C.ROOT))
    from scripts.audit_subsequence_memorization import as_pose, window_signatures
    train = torch.load(C.DATA / "train.pt", map_location="cpu", weights_only=False)
    bank = torch.cat([window_signatures(as_pose(m), WIN, STRIDE).half() for m in train.values()]).float()
    del train
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bank = bank.to(dev)
    bn = (bank * bank).sum(1)
    out = {}
    for name, path in pred_paths.items():
        bankp = torch.load(path, map_location="cpu", weights_only=False)
        ids = list(bankp.keys())
        clip_min, n_frames, n_flag, n_zero, n_flag_nonzero = [], 0, 0, 0, 0
        per_clip = {}
        for sid in ids[: (5 if dry else None)]:
            pose = as_pose(bankp[sid])
            T = pose.shape[0]
            q = window_signatures(pose, WIN, STRIDE).to(dev)
            qn = (q * q).sum(1, keepdim=True)
            best = torch.full((q.shape[0],), float("inf"), device=dev)
            for s in range(0, bank.shape[0], 8192):
                b = bank[s:s + 8192]
                d2 = (qn + bn[s:s + 8192].unsqueeze(0) - 2.0 * (q @ b.T)).clamp_min(0).min(1).values
                best = torch.minimum(best, d2)
            dist = torch.sqrt(best / q.shape[1]).cpu().numpy()
            flags = np.zeros(max(T, WIN), dtype=bool)
            for st, d in zip(window_starts(T), dist):
                if d < THR:
                    flags[st:st + WIN] = True
            flags = flags[:T]
            zero = (pose.reshape(T, -1).abs().sum(1) == 0).numpy()
            clip_min.append(float(dist.min()))
            n_frames += T; n_flag += int(flags.sum()); n_zero += int(zero.sum()); n_flag_nonzero += int((flags & ~zero).sum())
            per_clip[sid] = {"T": T, "min": float(dist.min()), "flagged_frames": int(flags.sum()), "zero_frames": int(zero.sum())}
        arr = np.array(clip_min)
        out[name] = {"n": len(arr), "min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max()),
                     "near_copy_lt_0.02": int((arr < THR).sum()), "frames": n_frames, "flagged_frames": n_flag,
                     "zero_frames": n_zero, "share_all_frames": n_flag / max(1, n_frames),
                     "share_nonzero_frames": n_flag_nonzero / max(1, n_frames - n_zero), "per_clip": per_clip}
    return out


def cmd_audit(a):
    wd = Path(a.outdir); wd.mkdir(parents=True, exist_ok=True)
    log = wd / f"audit_{a.name}.log"
    oj = wd / f"{a.name}_subsequence_memorization_audit.json"
    if not a.dry:
        rc = C.run([C.PY, C.ROOT / "scripts/audit_subsequence_memorization.py", "--banned_json", "",
                    "--method", f"{a.name}={Path(a.pred).resolve()}",
                    "--method", "replay_1nn_positive_control=text1nn_tfidf_test.pt", "--out", oj], log=log)
        if rc:
            C.record(a.step, "audit script runs", f"exit {rc}", False, log); return 2
    off = C.load_json(oj)["methods"]
    pc = off["replay_1nn_positive_control"]
    if pc["near_copy_lt_0.02"] != pc["n"]:
        C.record(a.step, "detector positive control hits every replay clip", f"{pc['near_copy_lt_0.02']}/{pc['n']}", False, oj)
        return 2
    fl = frame_level({a.name: Path(a.pred).resolve()}, dry=a.dry)[a.name]
    fj = wd / f"{a.name}_frame_replay_share.json"
    fj.write_text(json.dumps({"definition": "frame flagged when any 16-frame window (stride 4) covering it has per-dim RMS "
                              "distance < 0.02 to a training window, signatures of scripts/audit_subsequence_memorization.py",
                              "pred": str(a.pred), "pred_sha256": C.sha256(a.pred), **fl}, indent=1))
    o = off[a.name]
    agree = (o["n"] == fl["n"] and abs(o["min"] - fl["min"]) < 1e-4 and abs(o["median"] - fl["median"]) < 1e-4
             and abs(o["max"] - fl["max"]) < 1e-4 and o["near_copy_lt_0.02"] == fl["near_copy_lt_0.02"])
    if not a.dry and not agree:
        C.record(a.step, "frame-level pass agrees with the official audit per clip",
                 json.dumps({"official": {k: o[k] for k in ("n", "min", "median", "max", "near_copy_lt_0.02")},
                             "frame_level": {k: fl[k] for k in ("n", "min", "median", "max", "near_copy_lt_0.02")}}), False, fj)
        return 2
    passed = "recorded"
    crit = "replay audit at 0.02 (clips with a near-copy window; frame share), positive control 641/641"
    if a.min_share_nonpad is not None:
        passed = fl["share_nonzero_frames"] >= a.min_share_nonpad
        crit = f"detector flags at least {a.min_share_nonpad:.0%} of non-padded frames (positive control on a known-replay bank)"
    num = (f"clips {o['near_copy_lt_0.02']}/{o['n']} below 0.02 (median min distance {o['median']:.4f}); frames flagged "
           f"{fl['flagged_frames']}/{fl['frames']} = {fl['share_all_frames']:.4f}, non-padded {fl['share_nonzero_frames']:.4f} "
           f"({fl['zero_frames']} zero frames); positive control {pc['near_copy_lt_0.02']}/{pc['n']}; official and frame-level agree")
    C.record(a.step, crit, num, passed, fj, extra={"official": o, "positive_control": pc})
    return 0 if passed in (True, "recorded") else 1


# ----------------------------------------------------------------------------- bootstrap
def fmt_cmp(row):
    d, i = row["point_delta"], row["bootstrap_delta"]
    return (f"dBLEU-4 {d['bleu4']:+.2f} [{i['bleu4']['ci95'][0]:+.2f}, {i['bleu4']['ci95'][1]:+.2f}], "
            f"dWER {d['corpus_wer']:+.2f} [{i['corpus_wer']['ci95'][0]:+.2f}, {i['corpus_wer']['ci95'][1]:+.2f}]")


def cmd_bootstrap(a):
    sys.path.insert(0, str(C.ROOT))
    from scripts.bootstrap_strict_cac import load_predictions, paired_bootstrap
    wd = Path(a.outdir); wd.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(Path(p).resolve().relative_to(C.ROOT))
    if C.sha256(C.FIXED_TEXT_PREDS) != C.FIXED_TEXT_PREDS_SHA:
        C.record(a.step, "fixed bank sha256 49db5307", C.sha256(C.FIXED_TEXT_PREDS)[:12], False, C.FIXED_TEXT_PREDS); return 2
    pair = f"{a.name}_vs_fixed"
    res = {}
    if not a.no_overall:
        oj = wd / f"{a.name}_vs_fixed_paired_bootstrap_test.json"
        if not a.dry:
            rc = C.run([C.PY, C.ROOT / "scripts/bootstrap_clean_route_comparisons.py", "--split", "test",
                        "--method", f"fixed={rel(C.FIXED_TEXT_PREDS)}", "--method", f"{a.name}={rel(a.text_preds)}",
                        "--pair", f"{pair}={a.name}:fixed", "--n_boot", "10000", "--seed", "30373", "--out", rel(oj)],
                       log=wd / f"bootstrap_{a.name}.log")
            if rc:
                C.record(a.step, "bootstrap runs", f"exit {rc}", False, oj); return 2
        res = C.load_json(oj)
        fx = res["methods"]["fixed"]
        if round(fx["bleu4"], 4) != C.FIXED_RECORDED["bleu4"] or round(fx["corpus_wer"], 3) != C.FIXED_RECORDED["wer"]:
            C.record(a.step, "fixed route reproduces 15.1032 / 85.582", f"{fx['bleu4']:.4f} / {fx['corpus_wer']:.3f}", False, oj)
            return 2
    # strata, fixed by the route's ledger before test
    # ids and references from the pinned files, which equal test.pt order and text on 641/641
    # (pinned_inputs_check.json: keys_equal_test_pt_order, texts_equal_test_pt_text)
    chk = C.load_json(C.RB / "pinned_inputs_check.json")
    assert chk["keys_equal_test_pt_order"] and chk["texts_equal_test_pt_text"]
    assert C.sha256(C.TEST_KEYS) == C.PINNED_SHA["test_keys.txt"] and C.sha256(C.TEST_TEXT) == C.PINNED_SHA["test_text.txt"]
    ids = C.TEST_KEYS.read_text().split()
    refs = C.TEST_TEXT.read_text(encoding="utf-8").splitlines()
    led = C.load_json(C.FIXED_LEDGER)["clips"]
    if [c["id"] for c in led] != ids:
        C.record(a.step, "ledger ids equal test.pt order", "mismatch", False, C.FIXED_LEDGER); return 2
    hyp = {"fixed": load_predictions(C.FIXED_TEXT_PREDS), a.name: load_predictions(Path(a.text_preds))}
    if len(hyp[a.name]) != len(ids) or len(hyp["fixed"]) != len(ids):
        C.record(a.step, "641/641 back-translations aligned", f"{len(hyp[a.name])}/{len(ids)}", False, a.text_preds); return 2
    # the fixed route must reproduce its recorded corpus numbers on all 641 before any comparison
    full = paired_bootstrap(hyp, refs, pairs=((pair, a.name, "fixed"),), n_boot=1, seed=30373)["methods"]["fixed"]
    if round(full["bleu4"], 4) != C.FIXED_RECORDED["bleu4"] or round(full["corpus_wer"], 3) != C.FIXED_RECORDED["wer"]:
        C.record(a.step, "fixed route reproduces 15.1032 / 85.582", f"{full['bleu4']:.4f} / {full['corpus_wer']:.3f}", False, C.FIXED_TEXT_PREDS)
        return 2
    strata = {}
    for mode in ("whole_clip_replay", "compositional_local_reuse"):
        idx = [i for i, c in enumerate(led) if c["mode"] == mode]
        sub = {k: [v[i] for i in idx] for k, v in hyp.items()}
        r = paired_bootstrap(sub, [refs[i] for i in idx], pairs=((pair, a.name, "fixed"),),
                             n_boot=(200 if a.dry else 10000), seed=30373)
        r["meta"].update({"stratum": mode, "definition": "ledger mode of clean_rerank_frame40_test_ledger.json",
                          "ledger_sha256": C.sha256(C.FIXED_LEDGER)})
        strata[mode] = r
    sj = wd / f"{a.name}_vs_fixed_strata_paired_bootstrap_test.json"
    sj.write_text(json.dumps({"inputs": {"fixed": {"path": str(C.FIXED_TEXT_PREDS), "sha256": C.FIXED_TEXT_PREDS_SHA},
                                         a.name: {"path": str(a.text_preds), "sha256": C.sha256(a.text_preds)}},
                              "strata": strata}, indent=2, sort_keys=True))
    parts = [f"fixed reproduced {full['bleu4']:.4f} / {full['corpus_wer']:.3f} (641)"]
    if res:
        fx, me = res["methods"]["fixed"], res["methods"][a.name]
        parts.append(f"all 641: {a.name} {me['bleu4']:.2f}/{me['corpus_wer']:.2f} vs fixed {fx['bleu4']:.4f}/{fx['corpus_wer']:.3f} "
                     f"(reproduced), {fmt_cmp(res['comparisons'][pair])}")
    for mode, lab in (("whole_clip_replay", "whole-replay"), ("compositional_local_reuse", "assembled")):
        r = strata[mode]
        parts.append(f"{lab} {r['meta']['n']}: {r['methods'][a.name]['bleu4']:.2f}/{r['methods'][a.name]['corpus_wer']:.2f} vs "
                     f"{r['methods']['fixed']['bleu4']:.2f}/{r['methods']['fixed']['corpus_wer']:.2f}, {fmt_cmp(r['comparisons'][pair])}")
    C.record(a.step, "paired bootstrap vs fixed 40% route, seed 30373, 10,000 resamples, overall and two strata",
             "; ".join(parts), "recorded", sj)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("ratios", cmd_ratios), ("audit", cmd_audit), ("bootstrap", cmd_bootstrap)):
        p = sub.add_parser(n); p.set_defaults(func=f)
        p.add_argument("--name", required=True); p.add_argument("--outdir", required=True); p.add_argument("--step", required=True)
        p.add_argument("--dry", action="store_true")
        if n in ("ratios", "audit"):
            p.add_argument("--pred", required=True)
        if n == "audit":
            p.add_argument("--min_share_nonpad", type=float, default=None)
        if n == "bootstrap":
            p.add_argument("--text_preds", required=True); p.add_argument("--no_overall", action="store_true")
    a = ap.parse_args()
    sys.exit(a.func(a))


if __name__ == "__main__":
    main()
