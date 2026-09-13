"""Item 3: Learned confidence head replacing the hand-tuned linear weights.

Hand-tuned formula (scripts/budget_semantic_retrieval_hybrid.py:_confidence):
  conf = rerank_score
       + 0.10*coverage + 0.05*slot_score + 0.03*f1
       - 0.05*length_penalty + 0.01*score

This script replaces those fixed weights with a head trained on a per-clip
target: the chrF gain of using the retrieval branch over the symbolic
fallback branch on the same dev clip. We use sentence-chrF (deterministic,
no BT decode here — we use existing per-clip BT text for both branches),
not corpus BLEU, because chrF is additive and matches Theorem 4's gain
form exactly.

Pipeline (no BT decoding required — both branches' per-clip texts are on
disk):
  1. Read 5-feature retrieval trace (505 clips with selected candidates).
  2. Read per-clip retrieval text (semantic_hybrid_clean_dev_text_preds.pt)
     and per-clip fallback text (pgrastpp178_mt5raw_clean_native_dev_text_preds.pt).
  3. Read GT references (the dev.pt's text field).
  4. Compute per-clip chrF for both branches and the gain delta.
  5. Train a linear head + a small MLP head leave-one-out on the 505 clips.
  6. Rank clips by predicted gain and compare the implied operating curve
     vs the hand-tuned ranking.

Output: outputs/learned_confidence/dev_loo.json with the learned weights,
the LOO predictions, and a ranking-quality summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


FEATURES = ["rerank_score", "coverage", "slot_score", "f1",
            "length_penalty", "score"]
HAND_WEIGHTS = np.array([1.0, 0.10, 0.05, 0.03, -0.05, 0.01], dtype=np.float64)


# ---------------------------- chrF ----------------------------

def _char_ngrams(s: str, n: int) -> list[str]:
    """Character n-grams over the input string, ignoring whitespace runs."""
    s = " ".join(s.split())
    return [s[i:i + n] for i in range(0, len(s) - n + 1)]


def sentence_chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Deterministic sentence-level chrF (chrF++ style, no smoothing)."""
    if not hyp.strip() or not ref.strip():
        return 0.0
    precs, recs = [], []
    for n in range(1, max_n + 1):
        h = _char_ngrams(hyp, n)
        r = _char_ngrams(ref, n)
        if not h or not r:
            continue
        h_count: dict[str, int] = {}
        for x in h:
            h_count[x] = h_count.get(x, 0) + 1
        r_count: dict[str, int] = {}
        for x in r:
            r_count[x] = r_count.get(x, 0) + 1
        match = sum(min(h_count[x], r_count.get(x, 0)) for x in h_count)
        precs.append(match / max(1, len(h)))
        recs.append(match / max(1, len(r)))
    if not precs:
        return 0.0
    p = float(np.mean(precs)); r = float(np.mean(recs))
    if p + r == 0:
        return 0.0
    f = (1 + beta * beta) * p * r / (beta * beta * p + r)
    return float(f)


# --------------------------- training ---------------------------

def linear_loo(X: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """Closed-form ridge regression with leave-one-out via Sherman-Morrison.
    X: [N, D], y: [N]. Returns yhat_loo [N]."""
    N, D = X.shape
    # Add intercept
    X1 = np.concatenate([X, np.ones((N, 1), dtype=X.dtype)], axis=1)
    A = X1.T @ X1 + ridge * np.eye(D + 1)
    b = X1.T @ y
    A_inv = np.linalg.inv(A)
    beta = A_inv @ b
    yhat_full = X1 @ beta
    # Hat matrix diagonal
    H_diag = np.einsum("nd,de,ne->n", X1, A_inv, X1)
    # LOO via standard ridge formula
    H_diag = np.clip(H_diag, 0.0, 1.0 - 1e-9)
    yhat_loo = (yhat_full - H_diag * y) / (1.0 - H_diag)
    return yhat_loo, beta


def operating_curve_from_ranks(rank: np.ndarray, deltas: np.ndarray,
                                budgets: list[float]) -> dict:
    """Given a ranking of clips by predicted score (descending), compute the
    cumulative chrF gain achieved at each budget level (top-bN clips)."""
    order = np.argsort(-rank, kind="stable")          # descending
    sorted_deltas = deltas[order]
    N = len(deltas)
    out = {}
    for b in budgets:
        m = int(round(b * N))
        m = max(0, min(N, m))
        out[f"b{int(b*100):03d}"] = {
            "n": m,
            "cum_gain": float(sorted_deltas[:m].sum()) if m > 0 else 0.0,
            "mean_gain": float(sorted_deltas[:m].mean()) if m > 0 else 0.0,
        }
    return out


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=(
        "external/SLRTP-Sign-Production-Evaluation/results/"
        "topk_semantic_hybrid_dev_trace.json"))
    ap.add_argument("--retrieval_text",
                    default=("external/SLRTP-Sign-Production-Evaluation/results/"
                             "semantic_hybrid_clean_dev_text_preds.pt"))
    ap.add_argument("--fallback_text",
                    default=("external/SLRTP-Sign-Production-Evaluation/results/"
                             "pgrastpp178_mt5raw_clean_native_dev_text_preds.pt"))
    ap.add_argument("--gt_pt",
                    default=("external/SLRTP-Sign-Production-Evaluation/pretrained/"
                             "SLRTP-Sign-Production-Evaluation-Data/data/dev.pt"))
    ap.add_argument("--out", default="outputs/learned_confidence/dev_loo.json")
    ap.add_argument("--ridge", type=float, default=1e-2)
    args = ap.parse_args()

    print(f"[item3] loading trace from {args.trace}")
    rows = json.loads(Path(args.trace).read_text())
    # Order by id; this is the dev sample order
    print(f"[item3] {len(rows)} trace rows")

    # Load retrieval and fallback per-clip texts (lists of strings, ordered by dev sample order)
    retr_text = torch.load(args.retrieval_text, map_location="cpu", weights_only=False)
    fall_text = torch.load(args.fallback_text, map_location="cpu", weights_only=False)
    assert isinstance(retr_text, list) and isinstance(fall_text, list)
    print(f"[item3] retrieval text: {len(retr_text)}; fallback text: {len(fall_text)}")

    # Load GT references in the same dev order. dev.pt is a dict of
    # sid -> {name, text, gloss, poses_3d, speaker}; the iteration order
    # of this dict IS the dev sample order the BT pipeline uses.
    gt = torch.load(args.gt_pt, map_location="cpu", weights_only=False)
    ids = list(gt.keys())
    gt_text = [gt[sid]["text"] for sid in ids]
    print(f"[item3] gt: {len(ids)} ids / {len(gt_text)} ref texts")

    # Build id -> index for the dev sample order
    id_to_idx = {sid: i for i, sid in enumerate(ids)}

    # For each trace row with a selected retrieval candidate, get its 5 features
    # AND compute the per-clip retrieval/fallback chrF + delta.
    feats: list[list[float]] = []
    deltas: list[float] = []
    rids: list[str] = []
    skipped = {"no_selected": 0, "no_eval_id": 0, "no_index": 0}
    for row in rows:
        sid = row.get("id")
        sel = row.get("selected") or {}
        if not sel:
            skipped["no_selected"] += 1
            continue
        if sid not in id_to_idx:
            skipped["no_eval_id"] += 1
            continue
        idx = id_to_idx[sid]
        if idx >= len(retr_text) or idx >= len(fall_text):
            skipped["no_index"] += 1
            continue
        x = [
            float(sel.get("rerank_score", 0.0)),
            float(sel.get("coverage", 0.0)),
            float(sel.get("slot_score", 0.0)),
            float(sel.get("f1", 0.0)),
            float(sel.get("length_penalty", 0.0)),
            float(sel.get("score", 0.0)),
        ]
        rt = retr_text[idx]
        ft = fall_text[idx]
        ref = gt_text[idx]
        chrf_r = sentence_chrf(rt, ref)
        chrf_f = sentence_chrf(ft, ref)
        feats.append(x)
        deltas.append(chrf_r - chrf_f)
        rids.append(sid)

    X = np.array(feats, dtype=np.float64)
    y = np.array(deltas, dtype=np.float64)
    print(f"[item3] N={len(X)} clips with features; skipped={skipped}")
    print(f"[item3] chrF gain stats: mean={y.mean():.4f} std={y.std():.4f} "
          f"p10/50/90={np.percentile(y, [10,50,90]).tolist()}  "
          f"frac_positive={(y>0).mean():.3f}")

    # Hand-tuned ranking
    hand_score = X @ HAND_WEIGHTS
    # Standardize features for the learned head (improves conditioning)
    X_mean = X.mean(0); X_std = X.std(0) + 1e-9
    Xs = (X - X_mean) / X_std

    yhat_loo, beta_full = linear_loo(Xs, y, ridge=args.ridge)

    # Operating curve at named budgets
    budgets = [0.10, 0.20, 0.40, 0.60, 0.80]
    hand_curve = operating_curve_from_ranks(hand_score, y, budgets)
    learned_curve = operating_curve_from_ranks(yhat_loo, y, budgets)
    oracle_curve = operating_curve_from_ranks(y, y, budgets)        # upper bound

    # Spearman-style rank correlation
    def _ranks(z): return np.argsort(np.argsort(z))
    rho_hand = float(np.corrcoef(_ranks(hand_score), _ranks(y))[0, 1])
    rho_learned = float(np.corrcoef(_ranks(yhat_loo), _ranks(y))[0, 1])

    # Learned weights in raw-feature space (un-standardize)
    learned_w_std = beta_full[:-1]
    learned_w_raw = learned_w_std / X_std
    learned_b = float(beta_full[-1] - float(np.sum(learned_w_std * X_mean / X_std)))

    out = {
        "n_clips": int(len(X)),
        "skipped": skipped,
        "delta_stats": {
            "mean": float(y.mean()), "std": float(y.std()),
            "p10": float(np.percentile(y, 10)),
            "p50": float(np.percentile(y, 50)),
            "p90": float(np.percentile(y, 90)),
            "frac_positive": float((y > 0).mean()),
        },
        "hand_weights":     {f: float(w) for f, w in zip(FEATURES, HAND_WEIGHTS)},
        "learned_weights":  {f: float(w) for f, w in zip(FEATURES, learned_w_raw)},
        "learned_intercept": learned_b,
        "rank_correlation_with_oracle_delta": {
            "hand": rho_hand,
            "learned_loo": rho_learned,
        },
        "operating_curve_chrf_gain": {
            "hand":         hand_curve,
            "learned_loo":  learned_curve,
            "oracle":       oracle_curve,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[item3] saved {args.out}")

    # Pretty print
    print()
    print("=== Hand-tuned vs learned-LOO ranking quality ===")
    print(f"  Spearman rho with oracle delta:")
    print(f"    hand-tuned   : {rho_hand:+.4f}")
    print(f"    learned LOO  : {rho_learned:+.4f}")
    print()
    print("=== Cumulative chrF gain at named budgets (higher = better ranking) ===")
    print(f"  {'budget':>6s}  {'hand':>10s}  {'learned LOO':>12s}  {'oracle':>10s}")
    for b in budgets:
        k = f"b{int(b*100):03d}"
        print(f"  {b:>6.2f}  "
              f"{hand_curve[k]['cum_gain']:>10.4f}  "
              f"{learned_curve[k]['cum_gain']:>12.4f}  "
              f"{oracle_curve[k]['cum_gain']:>10.4f}")
    print()
    print("=== Learned weights (raw-feature scale) ===")
    for f, w in zip(FEATURES, learned_w_raw):
        print(f"  {f:>15s}: hand={dict(zip(FEATURES, HAND_WEIGHTS))[f]:+.4f}   learned={w:+.4f}")
    print(f"  intercept      : learned={learned_b:+.4f}")


if __name__ == "__main__":
    main()
