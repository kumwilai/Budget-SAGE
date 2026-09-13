"""Assemble native SLRTP-178 poses from a source-controlled phrase lattice.

This is a stricter alternative to whole-sentence semantic retrieval:

  text/gloss plan -> train phrase lattice -> multi-source native phrase motion

The script intentionally avoids copying the semantically retrieved train clip
used to create the plan. It also caps how many target gloss tokens may be
covered by any one source clip. The goal is a falsifiable compositional row:
if it preserves the SLRTP BLEU gain, we have a broader retrieval-augmented
method; if it collapses, the current gain is whole-phrase/whole-clip memory.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/kumwilai/research/signgen-t2m")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import scripts.pg_rast_assemble as pgr


SLRTP_HAND_XY_DIMS = []
for joint in range(8, 50):
    SLRTP_HAND_XY_DIMS.extend([3 * joint, 3 * joint + 1])


def _load_slrtp_split(split: str, data_root: Path) -> dict:
    return torch.load(data_root / f"{split}.pt", map_location="cpu", weights_only=True)


def _glosses(x) -> list[str]:
    if isinstance(x, str):
        return x.split()
    return [str(t) for t in x]


def _load_plans(
    path: Path | None,
    eval_data: dict,
    *,
    allow_gt_gloss_diagnostic: bool = False,
) -> dict[str, list[str]]:
    """Load a complete query plan without silently consulting held-out glosses.

    Ground-truth glosses are useful for an explicitly labelled diagnostic, but
    they are not a valid fallback for text-input inference.  In particular, a
    partially written plan file must fail closed instead of producing a mixture
    of predicted and oracle-conditioned examples.
    """
    if path is None:
        if not allow_gt_gloss_diagnostic:
            raise ValueError(
                "--plans_json is required for inference; use "
                "--allow_gt_gloss_diagnostic only for an oracle-gloss diagnostic"
            )
        return {
            sid: _glosses(item.get("gloss", ""))
            for sid, item in eval_data.items()
        }

    data = json.loads(path.read_text())
    missing = [sid for sid in eval_data if sid not in data]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"plan file {path} is missing {len(missing)} evaluation ids "
            f"(first: {preview})"
        )
    return {sid: _glosses(data[sid]) for sid in eval_data}


def _load_exclusions(
    path: Path | None,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    if path is None:
        return {}, {}
    rows = json.load(open(path))
    if isinstance(rows, dict):
        rows = list(rows.values())
    out: dict[str, set[str]] = {}
    primary: dict[str, str] = {}
    for row in rows:
        rid = row.get("retrieved_id")
        if row.get("source") == "retrieval" and rid:
            sid = str(row["id"])
            forbidden = {str(value) for value in row.get("forbidden_source_ids", [])}
            forbidden.add(str(rid))
            out[sid] = forbidden
            primary[sid] = str(rid)
    return out, primary


def _validate_source_partition(
    pose_pool: dict,
    train_manifest_path: Path,
    eval_ids: list[str],
) -> None:
    """Require every reusable pose source to belong to the training split."""
    train_rows = json.loads(train_manifest_path.read_text())
    train_ids = {str(row["id"]) for row in train_rows}
    source_ids = {str(sid) for sid in pose_pool}
    outside = sorted(source_ids.difference(train_ids))
    overlap = sorted(source_ids.intersection(eval_ids))
    if outside:
        raise ValueError(f"pose bank contains non-training sources, first: {outside[:5]}")
    if overlap:
        raise ValueError(f"pose bank overlaps evaluation ids, first: {overlap[:5]}")


def _spans_for_clip(pred_spans_lookup, sid: str, split: str):
    if pred_spans_lookup is None:
        return None
    spans = pred_spans_lookup.get(sid)
    if spans is not None:
        return spans
    for prefix in (split, "dev", "test", "train"):
        spans = pred_spans_lookup.get(f"{prefix}/{sid}")
        if spans is not None:
            return spans
    return None


def _per_gloss_targets(spans, glosses: list[str]) -> list[int | None]:
    if spans is None:
        return [None] * len(glosses)
    pq = defaultdict(deque)
    for g, T in spans:
        pq[g].append(int(T))
    return [pq[g].popleft() if pq.get(g) else None for g in glosses]


def _build_phrase_bank(bank: dict, min_phrase: int, max_phrase: int):
    g2e = bank["gloss_to_exemplars"]
    by_sid: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for gloss, rows in g2e.items():
        for sid, s, e in rows:
            by_sid[sid].append((int(s), int(e), str(gloss)))
    for sid in list(by_sid.keys()):
        by_sid[sid].sort(key=lambda r: (r[0], r[1], r[2]))

    phrase_to_cands: dict[tuple[str, ...], list[tuple[str, int, int, int]]] = defaultdict(list)
    for sid, rows in by_sid.items():
        n = len(rows)
        for i in range(n):
            for L in range(min_phrase, max_phrase + 1):
                if i + L > n:
                    continue
                phrase = tuple(r[2] for r in rows[i:i + L])
                s = rows[i][0]
                e = rows[i + L - 1][1]
                if e <= s + 1:
                    continue
                phrase_to_cands[phrase].append((sid, s, e, e - s))

    # Stable candidate order: shorter phrases closest by duration are still
    # selected later; here we only deduplicate.
    for key in list(phrase_to_cands.keys()):
        phrase_to_cands[key] = sorted(set(phrase_to_cands[key]))
    return phrase_to_cands, by_sid


def _choose_candidate(
    candidates: list[tuple[str, int, int, int]],
    pose_pool: dict,
    target_T: int | None,
    prev_seg,
    source_counts: Counter[str],
    last_source: str | None,
    excluded_sources: set[str] | str | None,
    max_tokens_per_source: int,
    phrase_len: int,
    rng: random.Random,
    boundary_w: int,
    candidate_k: int,
    audit_counts: Counter[str],
):
    if isinstance(excluded_sources, str):
        excluded = {excluded_sources} if excluded_sources else set()
    else:
        excluded = set(excluded_sources or ())
    filt = []
    for sid, s, e, raw_len in candidates:
        if sid in excluded:
            audit_counts["excluded_source_candidate_rejections"] += 1
            continue
        if source_counts[sid] + phrase_len > max_tokens_per_source:
            audit_counts["source_limit_candidate_rejections"] += 1
            continue
        filt.append((sid, s, e, raw_len))
    if not filt:
        return None
    if target_T is not None:
        filt.sort(key=lambda r: (abs(r[3] - target_T), source_counts[r[0]], r[0], r[1], r[2]))
    else:
        filt.sort(key=lambda r: (source_counts[r[0]], r[0] == last_source, r[0], r[1], r[2]))
    pool = filt[:max(1, int(candidate_k))]

    best = None
    for sid, s, e, raw_len in pool:
        raw = pose_pool[sid][s:e].astype(np.float32)
        if raw.shape[0] < 2:
            continue
        seg = pgr._adjust_duration(raw, target_T, mode="loop", trim_mode="center")
        boundary = pgr._boundary_cost(prev_seg, seg, boundary_w=boundary_w)
        length_cost = 0.0 if target_T is None else abs(raw_len - target_T) / max(float(target_T), 1.0)
        # Source diversity is part of the method, not a post-hoc audit.
        source_cost = source_counts[sid] / max(sum(source_counts.values()), 1)
        repeat_cost = 0.5 if sid == last_source else 0.0
        score = length_cost + 0.2 * boundary + source_cost + repeat_cost
        tie = (sid, int(s), int(e))
        rec = (score, tie, sid, s, e, raw_len, seg, boundary, length_cost)
        if best is None or (rec[0], rec[1]) < (best[0], best[1]):
            best = rec
    return best


def _assemble_one(
    sid: str,
    glosses: list[str],
    phrase_to_cands: dict,
    g2e: dict,
    pose_pool: dict,
    per_targets: list[int | None],
    excluded_sources: set[str] | str | None,
    args,
    rng: random.Random,
):
    n = len(glosses)
    # Discrete source allocation rule. The minimum permits a real two-gloss
    # phrase for short plans, so this is not a literal fractional exposure cap.
    max_tokens_per_source = max(args.min_phrase, int(np.ceil(args.source_cap * max(n, 1))))
    segments = []
    trace = []
    omitted_glosses = []
    audit_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    last_source = None
    i = 0
    while i < n:
        chosen = None
        chosen_len = 1
        # Prefer longer real phrase units, but only if source-diverse.
        for L in range(min(args.max_phrase, n - i), args.min_phrase - 1, -1):
            key = tuple(glosses[i:i + L])
            cands = phrase_to_cands.get(key)
            if not cands:
                continue
            target_T = None
            if any(t is not None for t in per_targets[i:i + L]):
                target_T = int(sum(t or 0 for t in per_targets[i:i + L]))
                target_T = target_T if target_T > 1 else None
            rec = _choose_candidate(
                cands, pose_pool, target_T, segments[-1] if segments else None,
                source_counts, last_source, excluded_sources,
                max_tokens_per_source=max_tokens_per_source,
                phrase_len=L,
                rng=rng,
                boundary_w=args.boundary_w,
                candidate_k=args.candidate_k,
                audit_counts=audit_counts,
            )
            if rec is not None:
                chosen = rec
                chosen_len = L
                break

        if chosen is None:
            # Fallback to a single-gloss exemplar, still excluding the semantic
            # source where possible. Singletons are reported in the trace.
            g = glosses[i]
            cands = [(ex[0], int(ex[1]), int(ex[2]), int(ex[2] - ex[1]))
                     for ex in g2e.get(g, [])]
            target_T = per_targets[i]
            chosen = _choose_candidate(
                cands, pose_pool, target_T, segments[-1] if segments else None,
                source_counts, last_source, excluded_sources,
                max_tokens_per_source=max_tokens_per_source,
                phrase_len=1,
                rng=rng,
                boundary_w=args.boundary_w,
                candidate_k=args.candidate_k,
                audit_counts=audit_counts,
            )
            chosen_len = 1

        if chosen is None:
            omitted_glosses.append({
                "index": int(i),
                "gloss": str(glosses[i]),
                "reason": "no_allowed_training_exemplar",
            })
            i += 1
            continue
        score, _tie, src, s, e, raw_len, seg, boundary, length_cost = chosen
        segments.append(seg)
        source_counts[src] += chosen_len
        last_source = src
        trace.append({
            "i": i,
            "j": i + chosen_len,
            "glosses": glosses[i:i + chosen_len],
            "source": src,
            "start": int(s),
            "end": int(e),
            "raw_len": int(raw_len),
            "out_len": int(seg.shape[0]),
            "is_phrase": chosen_len >= args.min_phrase,
            "boundary_cost": float(boundary),
            "length_cost": float(length_cost),
            "score": float(score),
        })
        i += chosen_len

    assembly_audit = {
        "source_token_limit": int(max_tokens_per_source),
        "source_limit_formula": "max(min_phrase, ceil(source_cap * n_gloss))",
        "source_limit_candidate_rejections": int(
            audit_counts["source_limit_candidate_rejections"]
        ),
        "excluded_source_candidate_rejections": int(
            audit_counts["excluded_source_candidate_rejections"]
        ),
        "planned_gloss_tokens": int(n),
        "emitted_gloss_tokens": int(sum(row["j"] - row["i"] for row in trace)),
        "omitted_glosses": omitted_glosses,
    }
    if not segments:
        return np.zeros((4, 178 * 3), dtype=np.float32), trace, assembly_audit
    out = [segments[0]]
    for cur in segments[1:]:
        prev = out[-1]
        w = min(args.blend_w, prev.shape[0] // 2, cur.shape[0] // 2)
        if w >= 1:
            ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, w))).astype(np.float32)[:, None]
            prev[-w:] = (1.0 - ramp) * prev[-w:] + ramp * cur[:w]
        out.append(cur)
    return np.concatenate(out, axis=0).astype(np.float32), trace, assembly_audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt")
    ap.add_argument("--split", choices=["train", "dev", "test"], required=True)
    ap.add_argument("--plans_json", default="")
    ap.add_argument(
        "--allow_gt_gloss_diagnostic",
        action="store_true",
        help=("Use held-out gloss labels only for an explicitly labelled "
              "oracle-gloss diagnostic. Never enabled for text-input rows."),
    )
    ap.add_argument("--exclude_trace", default="")
    ap.add_argument(
        "--require_exclusion",
        action="store_true",
        help="Fail unless every planned evaluation item has a donor-source exclusion.",
    )
    ap.add_argument("--exclude_self", action="store_true",
                    help="Exclude the target clip id as a motion source. Used for train prompts.")
    ap.add_argument("--per_gloss_spans_json", default="")
    ap.add_argument("--tag", default="phrase_lattice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_phrase", type=int, default=2)
    ap.add_argument("--max_phrase", type=int, default=4)
    ap.add_argument("--source_cap", type=float, default=0.65)
    ap.add_argument("--candidate_k", type=int, default=32)
    ap.add_argument("--blend_w", type=int, default=4)
    ap.add_argument("--boundary_w", type=int, default=4)
    ap.add_argument("--data_root", default=(
        "external/SLRTP-Sign-Production-Evaluation/pretrained/"
        "SLRTP-Sign-Production-Evaluation-Data/data"
    ))
    ap.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    ap.add_argument("--out_dir", default="external/SLRTP-Sign-Production-Evaluation/results")
    ap.add_argument("--trace_out", default="",
                    help="Optional explicit trace path; defaults to the legacy phase43 location.")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = ap.parse_args()

    pgr.HAND_XY_DIMS_201 = SLRTP_HAND_XY_DIMS
    rng = random.Random(args.seed)
    data_root = ROOT / args.data_root
    if args.plans_json:
        # Text-input inference takes its complete ID order from the frozen plan
        # artifact.  It does not load the held-out pose/gloss ground-truth file.
        plan_path = ROOT / args.plans_json
        raw_plans = json.loads(plan_path.read_text())
        if not isinstance(raw_plans, dict) or not raw_plans:
            raise ValueError(f"invalid or empty plan file: {plan_path}")
        eval_ids = [str(sid) for sid in raw_plans]
        plans = _load_plans(plan_path, {sid: {} for sid in eval_ids})
    else:
        eval_data = _load_slrtp_split(args.split, data_root)
        eval_ids = [str(sid) for sid in eval_data]
        plans = _load_plans(
            None,
            eval_data,
            allow_gt_gloss_diagnostic=args.allow_gt_gloss_diagnostic,
        )
    empty = [sid for sid, glosses in plans.items() if not glosses]
    if empty:
        raise ValueError(f"plan file contains {len(empty)} empty gloss plans, first: {empty[:5]}")
    exclusions, primary_exclusions = _load_exclusions(
        ROOT / args.exclude_trace if args.exclude_trace else None
    )
    if args.require_exclusion:
        missing_exclusions = [sid for sid in eval_ids if not exclusions.get(sid)]
        if missing_exclusions:
            raise ValueError(
                f"missing donor-source exclusion for {len(missing_exclusions)} plans, "
                f"first: {missing_exclusions[:5]}"
            )
    if args.exclude_self:
        for sid in eval_data.keys():
            exclusions.setdefault(sid, set()).add(sid)
            primary_exclusions.setdefault(sid, sid)
    spans_lookup = None
    if args.per_gloss_spans_json:
        raw = json.load(open(ROOT / args.per_gloss_spans_json))
        spans_lookup = raw.get("spans", raw)

    print(f"Loading bank: {args.bank}")
    bank = torch.load(ROOT / args.bank, map_location="cpu", weights_only=False)
    g2e = bank["gloss_to_exemplars"]
    pose_pool = bank["exemplar_poses"]
    _validate_source_partition(pose_pool, ROOT / args.train_manifest, eval_ids)
    phrase_to_cands, _by_sid = _build_phrase_bank(bank, args.min_phrase, args.max_phrase)
    print(f"phrase types={len(phrase_to_cands)} source clips={len(pose_pool)}")

    preds = {}
    traces = {}
    for sid in eval_ids:
        gls = plans[sid]
        spans = _spans_for_clip(spans_lookup, sid, args.split)
        per_targets = _per_gloss_targets(spans, gls)
        flat, trace, assembly_audit = _assemble_one(
            sid, gls, phrase_to_cands, g2e, pose_pool,
            per_targets=per_targets,
            excluded_sources=exclusions.get(sid),
            args=args,
            rng=rng,
        )
        pose = torch.from_numpy(flat.reshape(flat.shape[0], 178, 3))
        preds[sid] = pose.half() if args.dtype == "float16" else pose
        phrase_tokens = sum(seg["j"] - seg["i"] for seg in trace if seg["is_phrase"])
        source_counts = Counter()
        for seg in trace:
            source_counts[seg["source"]] += seg["j"] - seg["i"]
        traces[sid] = {
            "id": sid,
            "excluded_source": primary_exclusions.get(sid, ""),
            "excluded_sources": sorted(exclusions.get(sid, set())),
            "glosses": gls,
            "n_gloss": len(gls),
            "phrase_token_frac": phrase_tokens / max(len(gls), 1),
            "unique_sources": len(source_counts),
            "max_source_token_frac": (max(source_counts.values()) / max(len(gls), 1)) if source_counts else 0.0,
            "source_token_counts": dict(source_counts),
            **assembly_audit,
            "segments": trace,
        }

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / f"{args.tag}_{args.split}.pt"
    torch.save(preds, out_pt)
    trace_path = (
        ROOT / args.trace_out
        if args.trace_out
        else ROOT / "outputs/sota_chase/phase43_phrase_lattice" / f"trace_{args.tag}_{args.split}.json"
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(traces, indent=2, ensure_ascii=False))

    lens = np.asarray([v.shape[0] for v in preds.values()], dtype=np.float64)
    phrase_frac = np.asarray([v["phrase_token_frac"] for v in traces.values()], dtype=np.float64)
    max_src = np.asarray([v["max_source_token_frac"] for v in traces.values()], dtype=np.float64)
    print(f"saved -> {out_pt}")
    print(f"trace -> {trace_path}")
    print(f"T mean={lens.mean():.1f} p10/50/90={np.percentile(lens, [10, 50, 90]).tolist()}")
    print(f"phrase_token_frac mean={phrase_frac.mean():.3f} p10/50/90={np.percentile(phrase_frac, [10, 50, 90]).tolist()}")
    print(f"max_source_token_frac mean={max_src.mean():.3f} p10/50/90={np.percentile(max_src, [10, 50, 90]).tolist()}")


if __name__ == "__main__":
    main()
