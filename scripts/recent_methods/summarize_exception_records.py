#!/usr/bin/env python
"""Write aggregate-only public records for three per-item audits that cannot be released.

Three numbers of the manuscript and the response letter are counted from local
per-item records that hold licensed PHOENIX-2014T material or name every clip:

- E1, Table IV USTC-MoE whole replay (32 clips, 1,538 of 81,467 frames). The
  source is the per-item tag file of the length-free USTC-MoE test run, which
  holds predicted German glosses.
- E2, Table IV and Sections VI-B and VII-A, unrestricted retrieval whole replay
  (632 clips, 62,966 of 63,224 frames). The source is the route's per-request
  trace, which holds the German input sentences and clip identifiers.
- E3, the letter's Progressive Transformer timing count (630 of 641 test clips
  at half the reference length, rounded up). The source is the duration audit
  of the organizers' released bank.

This script copies only counts, denominators and shares, recomputes the
conservation checks over every row, and records the SHA-256 of each source
file and of itself, so each summary can be checked against its source on
request. No per-item row, clip identifier, German text or hypothesis string is
written. It reads JSON only and loads no model or dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
SCRIPT = "scripts/recent_methods/summarize_exception_records.py"
WORKING_COPY = "scripts/recent_baselines/summarize_exception_records.py"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def provenance() -> dict:
    return {"producing_script": SCRIPT, "working_tree_copy": WORKING_COPY,
            "producing_script_sha256": digest(Path(__file__).read_bytes())}


def e1_ustcmoe(tags_path: Path, audit_path: Path) -> tuple[dict, set]:
    raw, raw_audit = tags_path.read_bytes(), audit_path.read_bytes()
    tags = json.loads(raw)
    bank = json.loads(raw_audit)["banks"]["ustcmoe_natlen"]
    rows = list(tags.values())
    modes = {}
    for r in rows:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    total = sum(r["target_len"] for r in rows)
    whole = sum(r["archive_frames"] for r in rows if r["mode"] == "whole_clip")
    seg = sum(r["archive_frames"] for r in rows if r["mode"] == "segments")
    pad = sum(r["zero_pad_frames"] for r in rows)
    rec = {
        "schema": "budget_sage.recent_methods.ustcmoe_whole_replay_aggregate.v1",
        "supports": "Table IV, USTC-MoE row, whole-replay clips, frames and share",
        "source_records": {
            "per_item_tag_file": "USTC-MoE length-free test run, instrumentation tags (not released, "
                                 "holds predicted German glosses)",
            "frame_audit": "evidence/recent_methods/frame_audit/ustcmoe_natlen_win1_stride1_thr0.005.json "
                           "(released, gives the same frame denominator)",
        },
        "source_record_sha256": {"per_item_tag_file": digest(raw), "frame_audit": digest(raw_audit)},
        "n_requests": len(rows),
        "items_by_mode": dict(sorted(modes.items())),
        "whole_replay_clips": modes.get("whole_clip", 0),
        "whole_replay_frames": whole,
        "segment_frames": seg,
        "zero_pad_frames": pad,
        "total_frames": total,
        "whole_replay_frame_share": whole / total,
        "checks": {
            "frames_conserved": whole + seg + pad == total,
            "denominator_equals_frame_audit": total == bank["n_frames"],
            "requests_equal_frame_audit_clips": len(rows) == bank["n_clips"],
        },
        **provenance(),
    }
    return rec, set(tags)


def e2_unrestricted(trace_path: Path) -> tuple[dict, set]:
    raw = trace_path.read_bytes()
    trace = json.loads(raw)
    sources = {}
    for r in trace:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    replay = [r for r in trace if r["source"] == "retrieval"]
    total = sum(r["T"] for r in trace)
    frames = sum(r["T"] for r in replay)
    rec = {
        "schema": "budget_sage.recent_methods.unrestricted_retrieval_whole_replay_aggregate.v1",
        "supports": "Table IV unrestricted retrieval row (whole-replay clips, frames and share), and the "
                    "99.592% of Sections VI-B and VII-A",
        "source_records": {"per_request_trace": "unrestricted retrieval route, PHOENIX-2014T test, per-request "
                                                "trace (not released, holds German input text and clip "
                                                "identifiers)"},
        "source_record_sha256": {"per_request_trace": digest(raw)},
        "n_requests": len(trace),
        "requests_by_source": dict(sorted(sources.items())),
        "whole_replay_clips": len(replay),
        "whole_replay_frames": frames,
        "other_frames": total - frames,
        "total_frames": total,
        "whole_replay_frame_share": frames / total,
        "checks": {
            "requests_conserved": sum(sources.values()) == len(trace),
            "replayed_clip_emitted_whole": all(r["T"] == r["selected"]["T"] for r in replay),
        },
        **provenance(),
    }
    ids = {r["id"] for r in trace} | {str(r["selected"].get("id")) for r in replay}
    return rec, ids


def e3_pt_timing(audit_path: Path) -> dict:
    raw = audit_path.read_bytes()
    test = json.loads(raw)["results"]["test"]
    n = test["pt_clips"]
    rec = {
        "schema": "budget_sage.recent_methods.pt_bank_duration_aggregate.v1",
        "supports": "response letter, Comment 3.5: the released Progressive Transformer test bank has 630 of "
                    "641 clips at exactly half the reference length, rounded up",
        "source_records": {"duration_audit": "duration audit of the SLRTP organizers' released Progressive "
                                             "Transformer bank against the reference poses (not released, "
                                             "names local paths)"},
        "source_record_sha256": {"duration_audit": digest(raw)},
        "released_bank_sha256": test["pt_sha256"],
        "split": "test",
        "n_clips": n,
        "matched_reference_clips": test["matched_ids"],
        "exact_ceil_half_reference": test["exact_ceil_half_gt"],
        "exact_floor_half_reference": test["exact_floor_half_gt"],
        "neither": test["neither"],
        "within_one_frame_of_half_reference": test["within_one_frame_of_half_gt"],
        "exact_ceil_half_share": test["exact_ceil_half_gt"] / n,
        "checks": {
            "clips_conserved": test["exact_ceil_half_gt"] + test["exact_floor_half_gt"] + test["neither"] == n,
            "all_clips_matched": test["matched_ids"] == test["gt_clips"] == n,
        },
        **provenance(),
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rb = OUT / "recent_baselines_2026-09-28"
    ap.add_argument("--tags", type=Path, default=rb / "ustc_moe/preds/ustcmoe_mt5_natlen_test_tags.json")
    ap.add_argument("--frame_audit", type=Path,
                    default=rb / "frame_audit/ustcmoe_natlen_win1_stride1_thr0.005.json")
    ap.add_argument("--trace", type=Path, default=OUT / "revision/revision_clean_topk_test_trace.json")
    ap.add_argument("--pt_audit", type=Path, default=OUT / "baseline_protocol/pt_bank_duration_audit.json")
    ap.add_argument("--out_dir", type=Path, required=True)
    a = ap.parse_args()

    e1, ids1 = e1_ustcmoe(a.tags, a.frame_audit)
    e2, ids2 = e2_unrestricted(a.trace)
    e3 = e3_pt_timing(a.pt_audit)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rec in (("ustcmoe_whole_replay_aggregate.json", e1),
                      ("unrestricted_retrieval_whole_replay_aggregate.json", e2),
                      ("pt_bank_duration_aggregate.json", e3)):
        assert all(rec["checks"].values()), (name, rec["checks"])
        text = json.dumps(rec, indent=1, sort_keys=True) + "\n"
        leaked = [i for i in ids1 | ids2 if i and i in text]
        assert not leaked, (name, len(leaked))
        out = a.out_dir / name
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
