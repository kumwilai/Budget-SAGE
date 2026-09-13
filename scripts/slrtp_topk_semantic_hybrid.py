"""Top-k guarded semantic retrieval hybrid for native SLRTP-178 outputs.

Phase 41 used top-1 TF-IDF retrieval plus a slot guard. This script retrieves
top-k train clips and reranks safe candidates with deterministic source-text
features before falling back to clean PG-RAST++ when no candidate clears the
guard.

Selection uses only train texts/poses, eval source texts, and the clean
PG-RAST++ fallback length prior. It does not read eval references, GT poses,
GT spans, BT outputs, or MSKA outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import torch

ROOT = Path("/home/kumwilai/research/signgen-t2m")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.slrtp_semantic_clip_hybrid import slot_conflict, slots, toks  # noqa: E402
from scripts.slrtp_text_retrieval_baseline import (  # noqa: E402
    _build_features,
    _load_slrtp_split,
    _texts_and_keys,
)


def _norm_text(s: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(s)).lower()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def _f1(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    precision = inter / len(b)
    recall = inter / len(a)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def _coverage(a: set[str], b: set[str]) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _slot_score(target_text: str, cand_text: str) -> float:
    target_slots = slots(target_text)
    cand_slots = slots(cand_text)
    scores = []
    for key, target_vals in target_slots.items():
        if not target_vals:
            continue
        cand_vals = cand_slots[key]
        if cand_vals & target_vals:
            scores.append(1.0)
        elif cand_vals:
            scores.append(0.0)
        else:
            scores.append(0.25)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _len_penalty(candidate_len: int, fallback_len: int) -> float:
    if candidate_len <= 0 or fallback_len <= 0:
        return 0.0
    return abs(math.log((candidate_len + 1.0) / (fallback_len + 1.0)))


def _topk_indices(row, k: int) -> list[tuple[int, float]]:
    if row.nnz == 0:
        return []
    order = row.data.argsort()[::-1]
    out = []
    for local in order[:max(1, k)]:
        idx = int(row.indices[int(local)])
        score = float(row.data[int(local)])
        out.append((idx, score))
    return out


def _project_query_rows(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Project a frozen query trace to the only allowed evaluation inputs."""
    keys: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sid = str(row.get("id", ""))
        text = str(row.get("text", ""))
        if not sid or sid in seen:
            raise ValueError(f"invalid or duplicate query id: {sid!r}")
        if not _norm_text(text):
            raise ValueError(f"empty query text for {sid}")
        seen.add(sid)
        keys.append(sid)
        texts.append(text)
    if not keys:
        raise ValueError("query trace is empty")
    return keys, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--fallback_pt", required=True,
                    help="Clean PG-RAST++ fallback PT in native SLRTP-178 format.")
    ap.add_argument(
        "--query_trace",
        default="",
        help=("Optional JSON trace providing only evaluation id/text. When set, "
              "the held-out SLRTP pose/gloss file is not loaded."),
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_sim", type=float, default=0.0,
                    help="Fallback unless the selected candidate has at least this TF-IDF similarity.")
    ap.add_argument("--max_sim", type=float, default=1.01,
                    help="Reject candidates above this similarity; useful for exact/near-exact caps.")
    ap.add_argument("--exclude_exact", action="store_true")
    ap.add_argument("--loose_slots", action="store_true")
    ap.add_argument("--sim_w", type=float, default=1.0)
    ap.add_argument("--coverage_w", type=float, default=0.50)
    ap.add_argument("--f1_w", type=float, default=0.20)
    ap.add_argument("--slot_w", type=float, default=0.50)
    ap.add_argument("--length_w", type=float, default=0.15)
    ap.add_argument("--data_root", default=(
        "external/SLRTP-Sign-Production-Evaluation/pretrained/"
        "SLRTP-Sign-Production-Evaluation-Data/data"
    ))
    args = ap.parse_args()

    data_root = ROOT / args.data_root
    train = _load_slrtp_split("train", data_root)
    fallback = torch.load(ROOT / args.fallback_pt, map_location="cpu", weights_only=True)
    train_keys, train_texts = _texts_and_keys(train)
    if args.query_trace:
        query_rows = json.loads((ROOT / args.query_trace).read_text())
        eval_keys, eval_texts = _project_query_rows(query_rows)
    else:
        ev = _load_slrtp_split(args.split, data_root)
        eval_keys, eval_texts = _texts_and_keys(ev)
    if set(eval_keys) != set(fallback):
        missing = sorted(set(eval_keys).difference(fallback))
        extra = sorted(set(fallback).difference(eval_keys))
        raise ValueError(
            f"query/fallback ids differ: missing={missing[:5]}, extra={extra[:5]}"
        )
    overlap = sorted(set(train_keys).intersection(eval_keys))
    if overlap:
        raise ValueError(f"training/evaluation id overlap, first: {overlap[:5]}")
    x_train, x_eval = _build_features(train_texts, eval_texts)
    sims = x_eval @ x_train.T

    pred = {}
    trace = []
    counts: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()

    for i, (key, text) in enumerate(zip(eval_keys, eval_texts)):
        target_toks = toks(text)
        fallback_pose = fallback[key]
        fallback_len = int(fallback_pose.shape[0])
        candidates = []
        raw_top = _topk_indices(sims.getrow(i), args.top_k)
        target_norm = _norm_text(text)
        for rank, (j, sim) in enumerate(raw_top, start=1):
            cand_text = train_texts[j]
            cand_norm = _norm_text(cand_text)
            reasons = []
            if sim > args.max_sim:
                reasons.append("above_max_sim")
            if args.exclude_exact and cand_norm == target_norm:
                reasons.append("exact_text")
            conflict = slot_conflict(text, cand_text, strict=not args.loose_slots)
            if conflict:
                reasons.append("slot_conflict")
            if sim < args.min_sim:
                reasons.append("below_min_sim")
            if reasons:
                reject_counts.update(reasons)
                candidates.append({
                    "rank": rank,
                    "id": train_keys[j],
                    "score": sim,
                    "accepted": False,
                    "reasons": reasons,
                })
                continue

            cand_toks = toks(cand_text)
            candidate_len = int(train[train_keys[j]]["poses_3d"].shape[0])
            cov = _coverage(target_toks, cand_toks)
            f1 = _f1(target_toks, cand_toks)
            slot = _slot_score(text, cand_text)
            len_pen = _len_penalty(candidate_len, fallback_len)
            rerank = (
                args.sim_w * sim
                + args.coverage_w * cov
                + args.f1_w * f1
                + args.slot_w * slot
                - args.length_w * len_pen
            )
            candidates.append({
                "rank": rank,
                "id": train_keys[j],
                "score": sim,
                "accepted": True,
                "rerank_score": rerank,
                "coverage": cov,
                "f1": f1,
                "slot_score": slot,
                "length_penalty": len_pen,
                "T": candidate_len,
            })

        accepted = [c for c in candidates if c.get("accepted")]
        if accepted:
            best = max(
                accepted,
                key=lambda c: (
                    c["rerank_score"],
                    c["score"],
                    -c["rank"],
                    c["id"],
                ),
            )
            pred[key] = train[best["id"]]["poses_3d"].clone()
            source = "retrieval"
            counts["retrieval"] += 1
        else:
            best = None
            pred[key] = fallback_pose
            source = "pgrastpp"
            counts["pgrastpp"] += 1

        trace.append({
            "id": key,
            "text": text,
            "source": source,
            "selected": best,
            "fallback_T": fallback_len,
            "candidates": candidates[: args.top_k],
            "T": int(pred[key].shape[0]),
        })

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, out)
    trace_path = ROOT / args.trace
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))

    lens = torch.tensor([v.shape[0] for v in pred.values()], dtype=torch.float32)
    n = len(eval_keys)
    print(f"saved -> {out}")
    print(f"trace -> {trace_path}")
    print(f"counts={dict(counts)}")
    print(f"fractions={ {k: round(v / n, 4) for k, v in counts.items()} }")
    print(f"reject_counts={dict(reject_counts)}")
    print(f"T mean={lens.mean().item():.1f} p10/50/90="
          f"{[round(x, 1) for x in torch.quantile(lens, torch.tensor([0.1, 0.5, 0.9])).tolist()]}")


if __name__ == "__main__":
    main()
