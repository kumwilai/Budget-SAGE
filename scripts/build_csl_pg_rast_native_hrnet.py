"""Build CSL-Daily PG-RAST directly in MSKA's native HRNet-133 format.

The earlier CSL cascade assembled (T,179) joint-angle features and then used
a learned 179 -> HRNet lifter. That lifter has a large floor cost
(GT179-lifted dev WER ~66.7 vs real HRNet GT ~27.4). This script keeps the
same budgeted PG-RAST cascade, but retrieves/assembles train HRNet keypoints
directly and writes MSKA-ready pickles. It is a representation fix, not an
evaluator change.

Outputs:
  outputs/csl_msla_inputs/CSL-Daily.<split>_native_budget{000,040,...}
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import sys

ROOT = Path("/home/kumwilai/research/signgen-t2m")
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "data/csl-daily"
MSKA_DIR = ROOT / "external/baselines/MSKA/data/CSL-Daily"
CTC_SPANS_PATH = ROOT / "data/csl-daily/gloss_spans_ctc_train.json"
OUT_DIR = ROOT / "outputs/csl_msla_inputs"

# Phoenix-parity slot extractor + reranker. Mirrors
# scripts/slrtp_topk_semantic_hybrid.py + scripts/slrtp_semantic_clip_hybrid.py
# verbatim in CSL feature space. See scripts/csl_slot_extractor.py.
from scripts.csl_slot_extractor import (
    toks as csl_toks,
    slot_conflict as csl_slot_conflict,
    coverage as csl_coverage,
    f1_score as csl_f1,
    slot_score as csl_slot_score,
    length_penalty as csl_len_penalty,
)


def load_keypoint(v) -> np.ndarray:
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().astype(np.float32)
    return np.asarray(v, dtype=np.float32)


def linear_blend(a: np.ndarray, b: np.ndarray, w: int = 4) -> np.ndarray:
    if a.shape[0] == 0:
        return b
    if b.shape[0] == 0:
        return a
    w = min(w, a.shape[0], b.shape[0])
    if w <= 0:
        return np.concatenate([a, b], axis=0)
    fade = np.linspace(0.0, 1.0, w + 2, dtype=np.float32)[1:-1, None, None]
    cross = (1.0 - fade) * a[-w:] + fade * b[:w]
    return np.concatenate([a[:-w], cross, b[w:]], axis=0)


def char_ngrams(text: str, n_min: int = 1, n_max: int = 3) -> list[str]:
    s = "".join(text.split())
    return [s[i:i + n] for n in range(n_min, n_max + 1)
            for i in range(len(s) - n + 1)]


def build_tfidf(texts: list[str]):
    docs = [Counter(char_ngrams(t)) for t in texts]
    vocab: dict[str, int] = {}
    for d in docs:
        for tok in d:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    V, N = len(vocab), len(docs)
    df = np.zeros(V, dtype=np.float32)
    for d in docs:
        for tok in d:
            df[vocab[tok]] += 1.0
    idf = np.log((N + 1) / (df + 1)) + 1.0
    M = np.zeros((N, V), dtype=np.float32)
    for i, d in enumerate(docs):
        for tok, c in d.items():
            M[i, vocab[tok]] = c * idf[vocab[tok]]
        n = np.linalg.norm(M[i])
        if n > 0:
            M[i] /= n
    return vocab, idf, M


def tfidf_query(text: str, vocab, idf):
    d = Counter(char_ngrams(text))
    q = np.zeros(len(vocab), dtype=np.float32)
    for tok, c in d.items():
        if tok in vocab:
            q[vocab[tok]] = c * idf[vocab[tok]]
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q


def resample(arr: np.ndarray, T: int) -> np.ndarray:
    if arr.shape[0] == T:
        return arr.astype(np.float32)
    if arr.shape[0] < 2:
        return np.repeat(arr[:1], T, axis=0).astype(np.float32)
    idx = np.linspace(0, arr.shape[0] - 1, T)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, arr.shape[0] - 1)
    w = (idx - i0).astype(np.float32)[:, None, None]
    return ((1.0 - w) * arr[i0] + w * arr[i1]).astype(np.float32)


def build_bank(train_rows: list[dict], train_kp: dict, max_per_gloss: int):
    with CTC_SPANS_PATH.open() as f:
        spans = json.load(f)
    bank: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for sid, clip_spans in spans.items():
        if sid not in train_kp:
            continue
        T = int(load_keypoint(train_kp[sid]["keypoint"]).shape[0])
        for sp in clip_spans:
            s = max(0, int(sp["start"]))
            e = min(T, int(sp["end"]))
            if e - s >= 4:
                bank[sp["gloss"]].append((sid, s, e))
    rng = np.random.default_rng(0)
    for g in list(bank):
        if len(bank[g]) > max_per_gloss:
            idx = rng.choice(len(bank[g]), max_per_gloss, replace=False)
            bank[g] = [bank[g][int(i)] for i in idx]
    print(f"[native-pgrast] bank glosses={len(bank)} "
          f"mean_ex={np.mean([len(v) for v in bank.values()]):.1f}", flush=True)
    return bank


def segment_quality(train_kp: dict, sid: str, s: int, e: int,
                    median_len: float) -> float:
    kp = load_keypoint(train_kp[sid]["keypoint"])[s:e]
    if kp.shape[0] == 0:
        return -1e9
    hand_c = kp[:, 91:133, 2].mean()
    upper_c = kp[:, :13, 2].mean()
    length_penalty = abs(math.log(max(1, e - s) / max(1.0, median_len)))
    return float(0.75 * hand_c + 0.25 * upper_c - 0.10 * length_penalty)


def choose_candidate(cands: list[tuple[str, int, int]], train_kp: dict,
                     rng: random.Random, selection: str,
                     median_len: float,
                     target_signer: int | None = None,
                     signer_by_sid: dict[str, int] | None = None,
                     signer_bonus: float = 0.5) -> tuple[str, int, int]:
    if selection == "random":
        return rng.choice(cands)
    scored = []
    for sid, s, e in cands:
        if sid not in train_kp:
            continue
        length = e - s
        q = segment_quality(train_kp, sid, s, e, median_len)
        if selection == "longest":
            q = float(length)
        elif selection == "typical_len":
            q = -abs(math.log(max(1, length) / max(1.0, median_len)))
        elif selection == "highconf":
            q = q + 0.10 * abs(math.log(max(1, length) / max(1.0, median_len)))
        elif selection == "highconf_len":
            pass
        else:
            raise ValueError(f"unknown selection={selection}")
        # Same-signer preference: when target_signer is supplied and a
        # signer-by-sid lookup is provided, add a bonus iff the train
        # exemplar comes from the same signer as the test clip. MSKA-SLR
        # is signer-sensitive on CSL-Daily so this routinely shifts
        # ranking among similarly-confident candidates.
        if target_signer is not None and signer_by_sid is not None:
            if signer_by_sid.get(sid) == target_signer:
                q = q + signer_bonus
        scored.append((q, sid, s, e))
    if not scored:
        return rng.choice(cands)
    scored.sort(reverse=True)
    return scored[0][1], scored[0][2], scored[0][3]


def assemble(glosses: list[str], bank: dict[str, list[tuple[str, int, int]]],
             train_kp: dict, rng: random.Random, blend_w: int,
             selection: str,
             median_lens: dict[str, float],
             target_signer: int | None = None,
             signer_by_sid: dict[str, int] | None = None,
             signer_bonus: float = 0.5,
             hclen_mode: bool = False) -> tuple[np.ndarray | None, list[str]]:
    out = None
    uncovered: list[str] = []
    used: list[tuple[str, int]] = []   # (source train clip id, frames contributed)
    for g in glosses:
        cands = bank.get(g, [])
        if not cands:
            uncovered.append(g)
            continue
        sid, s, e = choose_candidate(
            cands, train_kp, rng, selection, median_lens.get(g, 12.0),
            target_signer=target_signer, signer_by_sid=signer_by_sid,
            signer_bonus=signer_bonus,
        )
        seg = load_keypoint(train_kp[sid]["keypoint"])[s:e]
        if seg.shape[0] < 4:
            uncovered.append(g)
            continue
        if hclen_mode:
            # native_hclen recipe: resample each gloss segment to its round-4
            # quantized per-gloss median length; plain concatenation.
            L0 = max(4, int(round(median_lens.get(g, 12.0))))
            L = max(4, int(round(L0 / 4)) * 4)
            seg = resample(seg, L)
            used.append((sid, int(L)))
            out = seg if out is None else np.concatenate([out, seg], axis=0)
        else:
            used.append((sid, int(seg.shape[0])))
            out = seg if out is None else linear_blend(out, seg, blend_w)
    return (None if out is None else out.astype(np.float32)), uncovered, used


def build_templates(train_rows: list[dict], train_kp: dict,
                    bank: dict[str, list[tuple[str, int, int]]],
                    template_T: int = 12):
    gloss_means = {}
    for g, cands in bank.items():
        pieces = []
        for sid, s, e in cands[:32]:
            pieces.append(resample(load_keypoint(train_kp[sid]["keypoint"])[s:e], template_T))
        if pieces:
            gloss_means[g] = np.mean(np.stack(pieces, axis=0), axis=0).astype(np.float32)
    sample_ids = [r["id"] for r in train_rows if r["id"] in train_kp][:128]
    global_mean = np.mean(
        np.stack([resample(load_keypoint(train_kp[sid]["keypoint"]), 32)
                  for sid in sample_ids], axis=0),
        axis=0,
    ).astype(np.float32)
    return gloss_means, global_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--budgets", default="0.0,0.4")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--blend_w", type=int, default=4)
    ap.add_argument("--max_per_gloss", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="native")
    ap.add_argument("--selection",
                    choices=["random", "highconf", "typical_len",
                             "highconf_len", "longest"],
                    default="random")
    ap.add_argument("--signer_match", action="store_true",
                    help="Add a bonus when the train exemplar's signer "
                         "matches the test clip's signer. CSL-Daily MSKA "
                         "is signer-sensitive, so this typically reduces WER.")
    ap.add_argument("--signer_bonus", type=float, default=0.5,
                    help="Bonus added to candidate score for same-signer match.")
    # Phoenix-verbatim symbolic guards + 5-feature reranker (Tier 0 only).
    # Defaults match scripts/slrtp_topk_semantic_hybrid.py and the budget
    # controller's _confidence formula in budget_semantic_retrieval_hybrid.py.
    ap.add_argument("--phoenix_rerank", action="store_true",
                    help="Use Phoenix-verbatim symbolic-guard + 5-feature "
                         "reranker on Tier 0 (slot guard, exact-match guard, "
                         "rerank_score = sim_w*sim + cov_w*cov + f1_w*f1 + "
                         "slot_w*slot - len_w*len_pen). Without this, Tier 0 "
                         "uses the legacy TF-IDF + margin selection.")
    ap.add_argument("--exclude_exact", action="store_true", default=True,
                    help="Reject train candidates whose text equals the eval "
                         "text (Phoenix guard).")
    ap.add_argument("--loose_slots", action="store_true",
                    help="Apply only HARD slot conflicts (number/date/weekday); "
                         "let SOFT conflicts (time/weather/family/body) pass. "
                         "Phoenix default is strict (this flag OFF).")
    ap.add_argument("--sim_w", type=float, default=1.0)
    ap.add_argument("--coverage_w", type=float, default=0.50)
    ap.add_argument("--f1_w", type=float, default=0.20)
    ap.add_argument("--slot_w", type=float, default=0.50)
    ap.add_argument("--length_w", type=float, default=0.15)
    ap.add_argument("--min_sim", type=float, default=0.0,
                    help="Reject candidates with TF-IDF similarity below this "
                         "threshold (Phoenix guard).")
    ap.add_argument("--max_sim", type=float, default=1.01,
                    help="Reject candidates with TF-IDF similarity above this "
                         "threshold (catches near-exact templates beyond the "
                         "exact-match guard).")
    ap.add_argument("--exclude_signers", default="",
                    help="Comma list of CSL signer ids (0-9) to remove from the "
                         "train source pool (signer-level takedown). Emitted "
                         "route provably contains 0 frames from these signers.")
    ap.add_argument("--exclude_sids_json", default="",
                    help="JSON list of train clip ids to remove from the source "
                         "pool (per-source takedown).")
    ap.add_argument("--hclen_mode", action="store_true",
                    help="native_hclen assembly: per-gloss segments resampled to "
                         "round-4 median length, plain concatenation (reproduces "
                         "the published native_hclen route byte-exactly).")
    ap.add_argument("--ledger_out", default=None,
                    help="Write a per-signer / per-source frame ledger of the "
                         "assembled (b=0) route to this JSON path.")
    ap.add_argument("--trace_out", default=None,
                    help="If set, dump per-clip 5-feature retrieval trace as "
                         "JSON to this path (mirrors Phoenix "
                         "topk_semantic_hybrid_dev_trace.json schema). Used "
                         "by the CSL learned confidence head.")
    args = ap.parse_args()

    train_rows = json.load((MANIFEST / "csl_daily_train.json").open())
    split_rows = json.load((MANIFEST / f"csl_daily_{args.split}.json").open())
    # sid -> signer_id over the full train pool (needed for the ledger and for
    # signer-level takedowns, independent of --signer_match).
    signer_of = {r["id"]: int(r["signer"]) for r in train_rows}
    # ---- Source-exclusion (takedown) filter over the train source pool ----
    banned_signers = {int(x) for x in args.exclude_signers.split(",") if x.strip() != ""}
    banned_sids = set(json.load(open(args.exclude_sids_json))) if args.exclude_sids_json else set()
    if banned_signers or banned_sids:
        n0 = len(train_rows)
        train_rows = [r for r in train_rows
                      if int(r["signer"]) not in banned_signers
                      and r["id"] not in banned_sids]
        kept = {r["id"] for r in train_rows}
        print(f"[native-pgrast] takedown: signers={sorted(banned_signers)} "
              f"sids={len(banned_sids)} -> train pool {n0}->{len(train_rows)}",
              flush=True)
    else:
        kept = None
    # Build signer lookup (sid -> signer_id) from train manifest
    signer_by_sid = {r["id"]: int(r["signer"]) for r in train_rows} if args.signer_match else None
    if args.signer_match:
        from collections import Counter as _C
        sc = _C(signer_by_sid.values())
        print(f"[native-pgrast] signer-match ON  bonus={args.signer_bonus}  "
              f"train signers={dict(sc)}", flush=True)
    with (MSKA_DIR / "CSL-Daily.train").open("rb") as f:
        train_kp = pickle.load(f)
    if kept is not None:
        train_kp = {sid: v for sid, v in train_kp.items() if sid in kept}
    split_gt_path = MSKA_DIR / f"CSL-Daily.{args.split}.bak_gt_hrnet"
    if not split_gt_path.exists():
        split_gt_path = MSKA_DIR / f"CSL-Daily.{args.split}"
    with split_gt_path.open("rb") as f:
        split_gt = pickle.load(f)
    train_texts = [r["text"] for r in train_rows]
    vocab, idf, M = build_tfidf(train_texts)
    bank = build_bank(train_rows, train_kp, args.max_per_gloss)
    median_lens = {
        g: float(np.median([e - s for _, s, e in cands]))
        for g, cands in bank.items()
    }
    gloss_means, global_mean = build_templates(train_rows, train_kp, bank)
    rng = random.Random(args.seed)

    # Precompute train-clip frame counts once — avoids materializing each
    # candidate's full keypoint array inside the per-query rerank loop.
    train_T = {}
    for sid, rec in train_kp.items():
        kp = rec.get("keypoint") if isinstance(rec, dict) else rec
        if torch.is_tensor(kp):
            train_T[sid] = int(kp.shape[0])
        elif kp is not None:
            try:
                train_T[sid] = int(np.asarray(kp).shape[0])
            except Exception:
                train_T[sid] = 0

    tier0 = []
    # Counters for transparency in --phoenix_rerank mode (mirrors Phoenix
    # reject_counts logging). Plain dict to avoid an extra import.
    reject_counts = {"above_max_sim": 0, "exact_text": 0,
                      "slot_conflict": 0, "below_min_sim": 0}
    n_fallback_to_pgrast = 0
    trace_rows: list[dict] = []   # populated only if --trace_out is set

    for r in split_rows:
        text = r["text"]
        q = tfidf_query(text, vocab, idf)
        sims = M @ q
        order = np.argsort(-sims)[: args.top_k]

        if not args.phoenix_rerank:
            # Legacy path — TF-IDF score + margin (existing behavior).
            chosen = int(order[0])
            for j in order:
                jj = int(j)
                if train_texts[jj] != text:
                    chosen = jj
                    break
            margin = (float(sims[chosen] - sims[int(order[1])])
                      if len(order) > 1 else float(sims[chosen]))
            src_id = train_rows[chosen]["id"]
            pose = (load_keypoint(train_kp[src_id]["keypoint"])
                    if src_id in train_kp else global_mean)
            tier0.append({
                "id": r["id"],
                "src_id": src_id,
                "ret_pose": pose,
                "conf": float(sims[chosen]) + 0.10 * margin,
                "glosses": r["gloss"].split(),
            })
            continue

        # Phoenix-verbatim: per-candidate guard, then rerank.
        # PG-RAST fallback length predicted from per-gloss median spans —
        # avoids ordering dependency on pgrast_pose (which is built later).
        glosses_r = r["gloss"].split()
        fallback_len = max(8, int(round(sum(
            median_lens.get(g, 12.0) for g in glosses_r
        ))))

        accepted = []
        for rank, j in enumerate(order, start=1):
            jj = int(j)
            sim = float(sims[jj])
            cand_text = train_texts[jj]
            reasons = []
            if sim > args.max_sim:
                reasons.append("above_max_sim")
            if args.exclude_exact and cand_text == text:
                reasons.append("exact_text")
            if csl_slot_conflict(text, cand_text,
                                  strict=not args.loose_slots):
                reasons.append("slot_conflict")
            if sim < args.min_sim:
                reasons.append("below_min_sim")
            if reasons:
                for rr in reasons:
                    reject_counts[rr] = reject_counts.get(rr, 0) + 1
                continue
            cand_id = train_rows[jj]["id"]
            cand_T = train_T.get(cand_id, fallback_len)
            cov = csl_coverage(text, cand_text)
            f1v = csl_f1(text, cand_text)
            slv = csl_slot_score(text, cand_text)
            lpn = csl_len_penalty(fallback_len, cand_T)
            rerank = (
                args.sim_w * sim
                + args.coverage_w * cov
                + args.f1_w * f1v
                + args.slot_w * slv
                - args.length_w * lpn
            )
            accepted.append({
                "rank": rank, "j": jj, "id": cand_id,
                "score": sim, "rerank_score": rerank,
                "coverage": cov, "f1": f1v, "slot_score": slv,
                "length_penalty": lpn, "T": cand_T,
            })

        if accepted:
            best = max(accepted,
                       key=lambda c: (c["rerank_score"], c["score"],
                                       -c["rank"], c["id"]))
            src_id = best["id"]
            pose = (load_keypoint(train_kp[src_id]["keypoint"])
                    if src_id in train_kp else global_mean)
            conf = best["rerank_score"]
        else:
            # Fall through to PG-RAST symbolic fallback (assemble step below
            # always runs; budget controller will route this clip there).
            n_fallback_to_pgrast += 1
            src_id = None
            pose = global_mean   # placeholder; budget controller will skip
            conf = float("-inf")
        if args.trace_out:
            trace_rows.append({
                "id": r["id"],
                "text": text,
                "source": "csl_phoenix_rerank",
                "selected": (dict(best, accepted=True) if accepted else {}),
                "fallback_T": int(fallback_len),
                "candidates": [dict(c, accepted=True) for c in accepted],
                "n_rejected": int(sum(reject_counts.values())),
            })
        tier0.append({
            "id": r["id"],
            "src_id": src_id,
            "ret_pose": pose,
            "conf": float(conf),
            "glosses": r["gloss"].split(),
        })

    if args.phoenix_rerank:
        print(f"[native-pgrast] Phoenix-rerank ON  rejects={reject_counts}  "
              f"fallback_to_pgrast={n_fallback_to_pgrast}", flush=True)

    pgrast_pose = {}
    pgrast_used: dict[str, list[tuple[str, int]]] = {}
    full = partial = 0
    for r in split_rows:
        target_signer = int(r["signer"]) if args.signer_match and "signer" in r else None
        pose, uncov, used = assemble(r["gloss"].split(), bank, train_kp, rng,
                                     args.blend_w, args.selection, median_lens,
                                     target_signer=target_signer,
                                     signer_by_sid=signer_by_sid,
                                     signer_bonus=args.signer_bonus,
                                     hclen_mode=args.hclen_mode)
        if pose is None:
            continue
        pgrast_pose[r["id"]] = pose
        pgrast_used[r["id"]] = used
        full += int(not uncov)
        partial += int(bool(uncov))
    print(f"[native-pgrast] fallback full={full} partial={partial}", flush=True)

    budgets = [float(x) for x in args.budgets.split(",")]
    tier0_sorted = sorted(tier0, key=lambda x: -x["conf"])
    N = len(tier0_sorted)
    summary = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for b in budgets:
        m = int(round(b * N))
        allowed = {tier0_sorted[i]["id"] for i in range(m)}
        tc = Counter()
        out = {}
        for rec in tier0:
            sid = rec["id"]
            sample = {
                "name": split_gt[sid]["name"],
                "gloss": split_gt[sid]["gloss"],
                "text": split_gt[sid].get("text", ""),
            }
            if sid in allowed:
                kp = rec["ret_pose"]
                tc["T0"] += 1
            elif sid in pgrast_pose:
                kp = pgrast_pose[sid]
                tc["T1"] += 1
            else:
                pieces = [gloss_means[g] for g in rec["glosses"] if g in gloss_means]
                if pieces:
                    kp = np.concatenate(pieces, axis=0).astype(np.float32)
                    tc["T2"] += 1
                else:
                    kp = global_mean
                    tc["T4"] += 1
            sample["keypoint"] = torch.from_numpy(kp.astype(np.float32))
            sample["num_frames"] = int(kp.shape[0])
            out[sid] = sample
        tag = f"{args.prefix}_budget{int(round(b * 100)):03d}"
        outp = OUT_DIR / f"CSL-Daily.{args.split}_{tag}"
        with outp.open("wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        summary[tag] = dict(tc)
        print(f"[native-pgrast] {args.split} {tag}: {dict(tc)} -> {outp}", flush=True)
    (ROOT / f"outputs/csl_native_pgrast_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    if args.trace_out and trace_rows:
        trace_path = Path(args.trace_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace_rows, indent=2, ensure_ascii=False)
        )
        print(f"[native-pgrast] wrote 5-feature trace ({len(trace_rows)} rows) "
              f"-> {trace_path}", flush=True)

    if args.ledger_out:
        # Per-signer / per-source frame ledger of the deployed b=0 assembled
        # route (every clip is T1 PG-RAST assembly when the replay budget is 0).
        signer_frames: dict[int, int] = defaultdict(int)
        source_frames: dict[str, int] = defaultdict(int)
        total = 0
        for sid, used in pgrast_used.items():
            for src, fr in used:
                total += fr
                source_frames[src] += fr
                signer_frames[signer_of.get(src, -1)] += fr
        per_signer_pct = {str(k): 100.0 * v / max(1, total)
                          for k, v in sorted(signer_frames.items())}
        top_sources = sorted(source_frames.items(), key=lambda kv: -kv[1])[:20]
        max_src_pct = 100.0 * (top_sources[0][1] / max(1, total)) if top_sources else 0.0
        ledger = {
            "split": args.split,
            "route": "b=0 assembled (T1 PG-RAST)",
            "excluded_signers": sorted(banned_signers),
            "excluded_sids": len(banned_sids),
            "total_traced_local_frames": int(total),
            "n_source_clips": len(source_frames),
            "per_signer_frame_pct": per_signer_pct,
            "residual_banned_signer_pct": sum(
                per_signer_pct.get(str(s), 0.0) for s in banned_signers),
            "max_single_source_pct": max_src_pct,
            "top20_source_clips": [{"id": s, "frames": int(f)} for s, f in top_sources],
        }
        Path(args.ledger_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.ledger_out).write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
        print(f"[native-pgrast] ledger: {len(source_frames)} sources, "
              f"max-source {max_src_pct:.3f}%, residual banned-signer "
              f"{ledger['residual_banned_signer_pct']:.3f}% -> {args.ledger_out}",
              flush=True)


if __name__ == "__main__":
    main()
