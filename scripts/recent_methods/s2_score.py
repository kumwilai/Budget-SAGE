"""Score one prediction bank with the frozen SLRTP evaluator, identity row first.

  1. split .pt sha256 must equal the pinned value;
  2. key coverage: the prediction keys must equal the ground-truth keys exactly (641/641 or 515/515,
     or the subset file given by --gt); missing items would have been written as empty outputs by
     pred_invariants.py, never dropped;
  3. identity row: main.py SPLIT.pt SPLIT.pt, difference to the recorded row must be exactly 0.0;
  4. main.py PRED GT --fps 25, then the summary line.

usage: s2_score.py --split test --pred P.pt --tag TAG --workdir DIR --step "..." [--gt GT.pt]
                   [--min_bleu4 3.0] [--max_wer 100] [--criterion "..."]
exit 0 = pass (or scored without a threshold), 1 = threshold failed, 2 = invariant failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("test", "dev"), required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--min_bleu4", type=float, default=None)
    ap.add_argument("--max_wer", type=float, default=None)
    ap.add_argument("--criterion", default=None)
    ap.add_argument("--dry", action="store_true", help="CPU test: skip the evaluator calls")
    a = ap.parse_args()

    import torch

    wd = Path(a.workdir); wd.mkdir(parents=True, exist_ok=True)
    log = wd / f"score_{a.tag}.log"
    split_pt = C.DATA / f"{a.split}.pt"
    gt_pt = Path(a.gt) if a.gt else split_pt
    split_sha = C.sha256(split_pt)
    if split_sha != C.PINNED_SHA[f"{a.split}.pt"]:
        C.record(a.step, f"{a.split}.pt sha256 pinned", split_sha[:12], False, split_pt); sys.exit(2)

    gt = torch.load(gt_pt, map_location="cpu", weights_only=True)
    pred = torch.load(a.pred, map_location="cpu", weights_only=True)
    gkeys, pkeys = list(gt.keys()), list(pred.keys())
    cover = len(set(gkeys) & set(pkeys))
    n_nan = int(sum((~torch.isfinite(torch.as_tensor(v))).sum().item() for v in pred.values()))
    shapes_ok = all(tuple(torch.as_tensor(v).shape[1:]) == (178, 3) for v in pred.values())
    del gt, pred
    inv = {"coverage": f"{cover}/{len(gkeys)}", "extra_keys": len(set(pkeys) - set(gkeys)), "nan_or_inf": n_nan,
           "shapes_178x3": shapes_ok, "pred_sha256": C.sha256(a.pred), "gt": str(gt_pt), "gt_sha256": C.sha256(gt_pt)}
    if cover != len(gkeys) or len(pkeys) != len(gkeys) or n_nan or not shapes_ok:
        C.record(a.step, "key coverage, finite, [T,178,3]", json.dumps(inv), False, a.pred); sys.exit(2)

    ident_tag = f"identity_{a.split}_before_{a.tag}"
    if not a.dry:
        rc = C.run([C.PY, C.EV / "main.py", split_pt, split_pt, C.BT_MODEL, "--tag", ident_tag, "--fps", "25"], cwd=wd, log=log)
        if rc:
            C.record(a.step, "identity row runs", f"exit {rc}", False, log); sys.exit(2)
    ij = wd / "results" / f"{ident_tag}.json"
    ir = C.load_json(ij)
    rec = C.IDENTITY[a.split]
    diff = {"bleu4": ir["bleu"]["bleu4"] - rec["bleu4"], "wer": ir["wer"] - rec["wer"], "dtw_mje": ir["dtw_mje"] - rec["dtw_mje"]}
    if any(v != 0.0 for v in diff.values()):
        C.record(a.step, f"identity row {a.split} difference exactly 0.0", json.dumps(diff), False, ij); sys.exit(2)

    if not a.dry:
        rc = C.run([C.PY, C.EV / "main.py", Path(a.pred).resolve(), gt_pt, C.BT_MODEL, "--tag", a.tag, "--fps", "25"], cwd=wd, log=log)
        if rc:
            C.record(a.step, "evaluator runs", f"exit {rc}", False, log); sys.exit(2)
    rj = wd / "results" / f"{a.tag}.json"
    r = C.load_json(rj)
    b4, wer, dtw, b1, dur = r["bleu"]["bleu4"], r["wer"], r["dtw_mje"], r["bleu"]["bleu1"], r["avg_duration"]
    passed = True
    crit = a.criterion or "scored under main.py --fps 25"
    if a.min_bleu4 is not None and b4 < a.min_bleu4:
        passed = False
    if a.max_wer is not None and wer >= a.max_wer:
        passed = False
    summary = {"identity": {"json": str(ij), "difference": diff}, "invariants": inv, "result_json": str(rj),
               "bleu1": b1, "bleu4": b4, "wer": wer, "dtw_mje": dtw, "avg_duration": dur,
               "chrf": r["chrf"], "rouge": r["rouge"], "n": len(gkeys)}
    (wd / f"score_{a.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    num = (f"BLEU-4 {b4:.3f}, WER {wer:.3f}, DTW {dtw:.5f}, BLEU-1 {b1:.2f}, duration {dur:.3f} "
           f"({cover}/{len(gkeys)} items; identity diff 0.0)")
    C.record(a.step, crit, num, passed if (a.min_bleu4 is not None or a.max_wer is not None) else "scored", rj,
             extra=summary)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
