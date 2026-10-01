#!/usr/bin/env python
"""Write an aggregate-only public summary of the CSL-Daily 1,077-request governance record.

The portfolio summary and the fractional source ledger of the development study
hold per-request identifiers, pose hashes and per-request source masses derived
from the licensed CSL-Daily corpus, so neither is released. This script copies
only corpus-level counts: the fixed route's whole-replay frames against its
2/5 budget, the hybrid's recorded and bridged joins, its archive and generated
mass, and a mass-conservation check recomputed over every ledger row. It
records the SHA-256 of both source files so the summary can be checked against
them on request. No identifier or per-request row is written.
"""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO = ROOT / "outputs/csl_faithful_budget_sage_20260913/dev1077_canonical_v1/portfolio"
TOL = 1e-6  # frames, the tolerance the certificate verifier uses for claimed masses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=PORTFOLIO / "summary.json")
    ap.add_argument("--ledger", type=Path, default=PORTFOLIO / "fractional_ledger.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    raw_s, raw_l = a.summary.read_bytes(), a.ledger.read_bytes()
    s, led = json.loads(raw_s), json.loads(raw_l)
    rows = list(led["rows"].values())

    budget = s["fixed_h40_budget"]
    num, den = budget["budget_fraction"]["numerator"], budget["budget_fraction"]["denominator"]
    replay, emitted = budget["replay_frames"], budget["emitted_frames"]

    frames = sum(r["frames"] for r in rows)
    arch = sum(r["archive_mass"] for r in rows)
    gen = sum(r["generated_mass"] for r in rows)
    row_err = max(abs(r["archive_mass"] + r["generated_mass"] - r["frames"]) for r in rows)
    src_err = max(abs(sum(r.get("archive_source_mass", {}).values()) - r["archive_mass"]) for r in rows)
    modes = {}
    for r in rows:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1

    summary = {
        "schema": "budget_sage.csl_dev1077_governance_summary.v1",
        "source_record_sha256": {"portfolio_summary": hashlib.sha256(raw_s).hexdigest(),
                                 "fractional_ledger": hashlib.sha256(raw_l).hexdigest()},
        "n_requests": s["n"],
        "ledger_rows": len(rows),
        "fixed_route": {
            "replay_frames": replay,
            "emitted_frames": emitted,
            "replay_frame_fraction": replay / emitted,
            "budget_fraction": f"{num}/{den}",
            "within_budget": Fraction(replay, emitted) <= Fraction(num, den),
            "whole_clip_replays": s["fixed_h40_modes"]["whole_clip_replay"],
            "compositional_local_reuse": s["fixed_h40_modes"]["compositional_local_reuse"],
            "selection_algorithm": budget["selection_algorithm"],
            "optimality_claimed": budget["optimality_claimed"],
        },
        "hybrid": {
            "total_frames": led["total_frames"],
            "recorded_joins": led["recorded_joins"],
            "bridged_joins": led["bridged_joins"],
            "row_recorded_join_cores_sum": sum(r.get("recorded_join_cores", 0) for r in rows),
            "whole_replays_unchanged": led["whole_replays_unchanged"],
            "archive_mass": led["archive_mass"],
            "generated_mass": led["generated_mass"],
            "generated_fraction": led["generated_fraction"],
            "row_modes": dict(sorted(modes.items())),
        },
        "mass_conservation": {
            "tolerance_frames": TOL,
            "row_frames_sum": frames,
            "row_archive_mass_sum": arch,
            "row_generated_mass_sum": gen,
            "total_abs_error": abs(led["archive_mass"] + led["generated_mass"] - led["total_frames"]),
            "max_row_abs_error": row_err,
            "max_row_source_sum_abs_error": src_err,
            "recorded_flag": s["fractional_mass_conserved"],
        },
        "test_inputs_opened": s["test_inputs_opened"],
        "recognizer_outputs_opened": s["recognizer_outputs_opened"],
    }
    mc = summary["mass_conservation"]
    mc["pass"] = bool(mc["recorded_flag"] and frames == led["total_frames"]
                      and mc["total_abs_error"] < TOL and row_err < TOL and src_err < TOL)
    a.out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
