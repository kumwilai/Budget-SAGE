"""Decision-logic helper for frame_audit_job.sh: reads a frame_replay_audit.py JSON report and prints one
line plus exits 0 (pass) or 1 (fail). Kept out of scripts/ (job-local, output artifact) and out of the bash
job's inline quoting so every check is a plain CLI call.

usage:
  audit_ctl.py positive JSON BANK             641/641 clips and >=99.9% of frames flagged
  audit_ctl.py negative JSON BANK             below 1% of frames, no clip above 10%
  audit_ctl.py range JSON BANK LO HI          share_all_frames in [LO, HI] (fixed-route prediction band)
  audit_ctl.py below JSON BANK X              share_all_frames < X (generator prediction)
  audit_ctl.py nonpad_atleast JSON BANK X     share_nonpad_frames >= X (USTC-MoE prediction)
  audit_ctl.py row JSON BANK                  human-readable summary line, always exits 0
"""
from __future__ import annotations

import json
import sys


def load(path, bank):
    return json.load(open(path))["banks"][bank]


def row_line(d):
    n_nonpad = d["n_frames"] - d["n_zero_frames"]
    return (f"clips {d['clips_flagged']}/{d['n_clips']}, frames {d['n_flagged_all']}/{d['n_frames']} = "
            f"{d['share_all_frames']:.4%}, non-padded {d['n_flagged_nonpad']}/{n_nonpad} = "
            f"{d['share_nonpad_frames']:.4%}, clip-min median {d['clip_min_median']:.4f}, "
            f"clip-min min {d['clip_min_min']:.4f}")


def main():
    cmd = sys.argv[1]
    if cmd == "positive":
        d = load(sys.argv[2], sys.argv[3])
        ok = d["clips_flagged"] == d["n_clips"] and d["share_all_frames"] >= 0.999
        print(("PASS " if ok else "FAIL ") + row_line(d))
        sys.exit(0 if ok else 1)
    if cmd == "negative":
        d = load(sys.argv[2], sys.argv[3])
        maxshare = max((c["share"] for c in d["per_clip"].values()), default=0.0)
        ok = d["share_all_frames"] < 0.01 and maxshare <= 0.10
        print(("PASS " if ok else "FAIL ") + row_line(d) + f", max_clip_share={maxshare:.4%}")
        sys.exit(0 if ok else 1)
    if cmd == "range":
        d = load(sys.argv[2], sys.argv[3])
        lo, hi = float(sys.argv[4]), float(sys.argv[5])
        ok = lo <= d["share_all_frames"] <= hi
        print(("PASS " if ok else "FAIL ") + row_line(d) + f", band=[{lo:.4%},{hi:.4%}]")
        sys.exit(0 if ok else 1)
    if cmd == "below":
        d = load(sys.argv[2], sys.argv[3])
        x = float(sys.argv[4])
        ok = d["share_all_frames"] < x
        print(("PASS " if ok else "FAIL ") + row_line(d) + f", threshold<{x:.4%}")
        sys.exit(0 if ok else 1)
    if cmd == "nonpad_atleast":
        d = load(sys.argv[2], sys.argv[3])
        x = float(sys.argv[4])
        ok = d["share_nonpad_frames"] >= x
        print(("PASS " if ok else "FAIL ") + row_line(d) + f", nonpad>={x:.4%}")
        sys.exit(0 if ok else 1)
    if cmd == "row":
        d = load(sys.argv[2], sys.argv[3])
        print(row_line(d))
        sys.exit(0)
    print(f"unknown command {cmd}", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
