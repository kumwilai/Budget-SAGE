"""Audit whole-clip and local-unit reuse for Budget-SAGE PHOENIX routes.

The paper's deployment budget is explicitly a *whole-clip replay* budget.  The
symbolic fallback, however, is built from short train-pose exemplars.  This
script keeps those two provenance levels separate:

  1. route-defined whole-clip replay count;
  2. local fallback source concentration from the saved segment trace.

It emits deterministic JSON/Markdown artifacts for the TMM supplement.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "external/SLRTP-Sign-Production-Evaluation/results"
TRACE = ROOT / "outputs/sota_chase/phase40_slrtp178/trace_pgrastpp178_mt5raw_clean_native_test.json"
OUT_JSON = ROOT / "outputs/route_certificates/local_reuse_certificate.json"
OUT_MD = ROOT / "outputs/route_certificates/local_reuse_certificate.md"


ROUTES = {
    "Native fallback": {
        "kind": "fallback_only",
        "pose": RESULTS / "pgrastpp178_mt5raw_clean_native_test.pt",
    },
    "Budget40 reference": {
        "kind": "npz_in_s",
        "selector": ROOT / "outputs/learned_confidence/cac_test_in_S_b40.npz",
        "pose": RESULTS / "phase58_budget040_test.pt",
    },
    "Budget40 CAC": {
        "kind": "npz_in_s",
        "selector": ROOT / "outputs/learned_confidence/cac_alpha0.00_test_in_S.npz",
        "pose": RESULTS / "phase58_b40_CAC_a0_00_test.pt",
    },
    "BT-input-free CAC": {
        "kind": "block_ablation_in_s",
        "ablation_name": "evaluator-free CAC (drop w_i, cur_bleu4, cur_wer; 20)",
        "retrieval_pose": RESULTS / "phase57_topk20_slotonly_test.pt",
        "fallback_pose": RESULTS / "pgrastpp178_mt5raw_clean_native_test.pt",
    },
    "Pareto-knee CAC": {
        "kind": "npz_in_s",
        "selector": ROOT / "outputs/learned_confidence/cac_alpha0.25_test_in_S.npz",
        "pose": RESULTS / "phase58_b40_CAC_a0_25_test.pt",
    },
    "Replay-budgeted BT-MBR": {
        "kind": "routes_json",
        "routes": RESULTS / "bt_mbr_b40_route_selector_test_routes.json",
        "pose": RESULTS / "bt_mbr_b40_route_selector_test.pt",
    },
    "TF-IDF 1-NN": {
        "kind": "all_replay",
        "pose": RESULTS / "text1nn_tfidf_test.pt",
    },
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_pose_lengths(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"expected pose dict at {path}, got {type(data)!r}")
    return {str(k): int(v.shape[0]) for k, v in data.items()}


def segment_source(seg: dict[str, Any]) -> str:
    src = seg.get("sid", seg.get("source"))
    if src is None:
        raise KeyError(f"segment has no source id: {seg}")
    return str(src)


def segment_len(seg: dict[str, Any]) -> int:
    if "out_len" in seg:
        return int(seg["out_len"])
    return int(seg["end"]) - int(seg["start"])


def trace_stats(trace: dict[str, Any], sid: str) -> dict[str, Any]:
    rec = trace.get(sid)
    if rec is None:
        return {
            "frames": 0,
            "unique_sources": 0,
            "dominant_share": 0.0,
            "max_segment": 0,
            "source_counts": {},
            "n_segments": 0,
        }
    counts: Counter[str] = Counter()
    max_seg = 0
    for seg in rec.get("segments", []):
        n = max(0, segment_len(seg))
        counts[segment_source(seg)] += n
        max_seg = max(max_seg, n)
    frames = sum(counts.values())
    return {
        "frames": frames,
        "unique_sources": len(counts),
        "dominant_share": 0.0 if frames == 0 else max(counts.values()) / frames,
        "max_segment": max_seg,
        "source_counts": dict(counts),
        "n_segments": len(rec.get("segments", [])),
    }


def npz_replay_flags(path: Path) -> dict[str, bool]:
    z = np.load(path, allow_pickle=True)
    return {str(sid): bool(flag) for sid, flag in zip(z["sids"], z["in_S"])}


def block_ablation_replay_flags(name: str) -> dict[str, bool]:
    rows = load_json(ROOT / "outputs/learned_confidence/cac_block_ablation.json")
    for row in rows:
        if row.get("name") == name:
            test_flags = row["test_in_S"]
            fallback = load_pose_lengths(RESULTS / "pgrastpp178_mt5raw_clean_native_test.pt")
            sids = sorted(fallback)
            if len(test_flags) != len(sids):
                raise ValueError(f"{name}: {len(test_flags)} flags for {len(sids)} test ids")
            return {sid: bool(flag) for sid, flag in zip(sids, test_flags)}
    raise KeyError(name)


def mixed_pose_lengths(flags: dict[str, bool], retrieval_pose: Path, fallback_pose: Path) -> dict[str, int]:
    retrieval = load_pose_lengths(retrieval_pose)
    fallback = load_pose_lengths(fallback_pose)
    ids = sorted(fallback)
    missing = [sid for sid in ids if sid not in retrieval]
    if missing:
        raise KeyError(f"{retrieval_pose} missing {len(missing)} ids, e.g. {missing[:3]}")
    return {sid: (retrieval[sid] if flags.get(sid, False) else fallback[sid]) for sid in ids}


def same_pose(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6) -> bool:
    if tuple(a.shape) != tuple(b.shape):
        return False
    return bool(torch.max(torch.abs(a.detach().cpu() - b.detach().cpu())).item() <= atol)


def load_pose_bank_for_tag(tag: str) -> dict[str, torch.Tensor]:
    path = RESULTS / f"{tag}_test.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"expected pose dict at {path}, got {type(data)!r}")
    return data


def route_json_replay_flags(path: Path) -> dict[str, bool]:
    rows = load_json(path)
    topk = load_pose_bank_for_tag("phase57_topk20_slotonly")
    bank_cache: dict[str, dict[str, torch.Tensor]] = {}
    flags: dict[str, bool] = {}
    for r in rows:
        sid = str(r["id"])
        tag = str(r["candidate"])
        if tag == "text1nn_tfidf":
            flags[sid] = True
            continue
        if sid not in topk:
            flags[sid] = False
            continue
        if tag not in bank_cache:
            bank_cache[tag] = load_pose_bank_for_tag(tag)
        flags[sid] = same_pose(bank_cache[tag][sid], topk[sid])
    return flags


def top_share(counter: Counter[str], frac: float) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    k = max(1, int(round(frac * len(counter))))
    return sum(v for _, v in counter.most_common(k)) / total


def audit_method(name: str, spec: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    if spec["kind"] == "fallback_only":
        lengths = load_pose_lengths(spec["pose"])
        ids = sorted(lengths)
        replay = {sid: False for sid in ids}
    elif spec["kind"] == "all_replay":
        lengths = load_pose_lengths(spec["pose"])
        ids = sorted(lengths)
        replay = {sid: True for sid in ids}
    elif spec["kind"] == "npz_in_s":
        lengths = load_pose_lengths(spec["pose"])
        ids = sorted(lengths)
        replay = npz_replay_flags(spec["selector"])
    elif spec["kind"] == "block_ablation_in_s":
        replay = block_ablation_replay_flags(spec["ablation_name"])
        lengths = mixed_pose_lengths(replay, spec["retrieval_pose"], spec["fallback_pose"])
        ids = sorted(lengths)
    elif spec["kind"] == "routes_json":
        lengths = load_pose_lengths(spec["pose"])
        ids = sorted(lengths)
        replay = route_json_replay_flags(spec["routes"])
    else:
        raise ValueError(spec["kind"])

    local_frames = 0
    replay_frames = 0
    total_frames = 0
    fallback_clip_stats: list[dict[str, Any]] = []
    source_counter: Counter[str] = Counter()
    for sid in ids:
        n = int(lengths[sid])
        total_frames += n
        if replay.get(sid, False):
            replay_frames += n
            continue
        st = trace_stats(trace, sid)
        local_frames += st["frames"] or n
        fallback_clip_stats.append(st)
        source_counter.update(st["source_counts"])

    dom = [float(s["dominant_share"]) for s in fallback_clip_stats if s["frames"] > 0]
    uniq = [int(s["unique_sources"]) for s in fallback_clip_stats if s["frames"] > 0]
    maxseg = [int(s["max_segment"]) for s in fallback_clip_stats if s["frames"] > 0]
    nseg = [int(s["n_segments"]) for s in fallback_clip_stats if s["frames"] > 0]
    other_frames = max(0, total_frames - replay_frames - local_frames)
    # These PHOENIX audit rows do not have a per-frame generated-span trace.
    # We therefore keep generated mass at zero and conservatively charge every
    # residual frame to an unknown/source-derived-transform bucket rather than
    # treating it as source-free motion.
    generated_frames = 0
    unknown_or_transformed_frames = other_frames
    return {
        "method": name,
        "n_clips": len(ids),
        "whole_clip_replay_clips": int(sum(1 for sid in ids if replay.get(sid, False))),
        "fallback_clips": int(sum(1 for sid in ids if not replay.get(sid, False))),
        "total_output_frames": total_frames,
        "whole_clip_replay_frames": replay_frames,
        "local_unit_frames": local_frames,
        "generated_frames": generated_frames,
        "unknown_or_transformed_frames": unknown_or_transformed_frames,
        "whole_clip_replay_frame_fraction": replay_frames / total_frames if total_frames else 0.0,
        "local_unit_frame_fraction": local_frames / total_frames if total_frames else 0.0,
        "generated_frame_fraction": generated_frames / total_frames if total_frames else 0.0,
        "unknown_or_transformed_frame_fraction": unknown_or_transformed_frames / total_frames if total_frames else 0.0,
        "trace_covered_frame_fraction": (replay_frames + local_frames) / total_frames if total_frames else 0.0,
        "other_frame_fraction": other_frames / total_frames if total_frames else 0.0,
        "conservative_source_charged_frame_fraction": (
            replay_frames + local_frames + unknown_or_transformed_frames
        ) / total_frames if total_frames else 0.0,
        "fallback_unique_sources_mean": mean(uniq) if uniq else 0.0,
        "fallback_unique_sources_median": median(uniq) if uniq else 0.0,
        "fallback_dominant_source_share_mean": mean(dom) if dom else 0.0,
        "fallback_dominant_source_share_median": median(dom) if dom else 0.0,
        "fallback_max_segment_frames_mean": mean(maxseg) if maxseg else 0.0,
        "fallback_max_segment_frames_median": median(maxseg) if maxseg else 0.0,
        "fallback_n_segments_mean": mean(nseg) if nseg else 0.0,
        "fallback_n_segments_median": median(nseg) if nseg else 0.0,
        "unique_local_source_clips": len(source_counter),
        "local_top1_source_frame_share": top_share(source_counter, 1 / max(1, len(source_counter))),
        "local_top5pct_source_frame_share": top_share(source_counter, 0.05),
        "local_top10pct_source_frame_share": top_share(source_counter, 0.10),
    }


def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    trace = load_json(TRACE)
    rows = [audit_method(name, spec, trace) for name, spec in ROUTES.items()]
    payload = {
        "name": "Budget-SAGE local reuse certificate",
        "definition": (
            "Whole-clip replay and traced local fallback-unit reuse are "
            "reported as separate provenance levels. The frame ledger is "
            "conserved as replay + traced-local + generated + "
            "unknown/source-derived-transform output frames. For these "
            "PHOENIX rows the generated ledger coordinate is zero; every "
            "residual frame is conservatively charged to unknown/source-derived "
            "transform rather than treated as source-free. Local-unit statistics "
            "are computed from the saved "
            "PG-RAST segment trace; they are not used in the headline "
            "whole-clip replay budget."
        ),
        "trace": str(TRACE.relative_to(ROOT)),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    lines = [
        "# Budget-SAGE Local Reuse Certificate",
        "",
        payload["definition"],
        "",
        "| Method | Whole-clip replay clips | Replay frame % | Traced-local % | Generated % | Unknown/derived % | Conservative charged % | Fallback clips | Local unique src med | Dominant local src med | Top-5% local-source frame share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['whole_clip_replay_clips']}/{r['n_clips']} | "
            f"{100*r['whole_clip_replay_frame_fraction']:.2f} | "
            f"{100*r['local_unit_frame_fraction']:.2f} | "
            f"{100*r['generated_frame_fraction']:.2f} | "
            f"{100*r['unknown_or_transformed_frame_fraction']:.2f} | "
            f"{100*r['conservative_source_charged_frame_fraction']:.2f} | "
            f"{r['fallback_clips']} | {r['fallback_unique_sources_median']:.1f} | "
            f"{100*r['fallback_dominant_source_share_median']:.1f}% | "
            f"{100*r['local_top5pct_source_frame_share']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "The whole-clip budget controls route-level replay. Fallback outputs are",
            "assembled from short local exemplars; their reuse is therefore exposed",
            "through source concentration rather than counted as whole-clip replay.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
