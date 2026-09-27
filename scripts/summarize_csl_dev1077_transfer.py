#!/usr/bin/env python
"""Write an aggregate-only public summary of the CSL-Daily 1,077-request development study.

The full analysis record (transfer_analysis_v1.json) holds per-caption-group and
per-item material derived from the licensed CSL-Daily corpus, so it is not
released. This script copies only corpus-level aggregates: per-route and
per-head WER with its edit counts, the paired bootstrap intervals, the
preregistered gate outcome, and the claim ceiling. It records the SHA-256 of
the full record so the summary can be checked against it on request.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/csl_faithful_budget_sage_20260913/dev1077_canonical_v1/transfer_analysis_v1.json"
OUT = ROOT / "outputs/csl_faithful_budget_sage_20260913/dev1077_canonical_v1/transfer_summary_public_v1.json"
FIELDS = ("denominator", "num_ref", "num_hyp", "num_err", "num_sub", "num_del",
          "num_ins", "empty_hypotheses", "wer")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    raw = a.src.read_bytes()
    d = json.loads(raw)
    rec = {route: {head: {k: cell[k] for k in FIELDS if k in cell}
                   for head, cell in heads.items()}
           for route, heads in d["recognition"].items()}
    boot = d["bootstrap"]
    summary = {
        "schema": "budget_sage.csl_dev1077_transfer_summary.v1",
        "source_record_sha256": hashlib.sha256(raw).hexdigest(),
        "routes": {"local": "local assembly", "whole": "unrestricted retrieval",
                   "fixed_h40": "fixed 40% route", "hybrid": "Sign-JEPA hybrid",
                   "all_generated": "all-generated route"},
        "recognition": rec,
        "bootstrap": {k: boot[k] for k in ("seed", "draws", "groups", "unit",
                                          "bit_generator", "percentile_method")},
        "paired_comparisons": boot["comparisons"],
        "preregistered_gate": {k: d["strong_transfer_gate"][k]
                               for k in ("primary_comparisons", "pass", "invariants")},
        "status": d["status"],
        "claim_ceiling": d["claim_ceiling"],
        "test_inputs_opened": d["test_inputs_opened"],
    }
    a.out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
