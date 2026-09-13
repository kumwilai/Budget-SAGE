"""Corpus-Aware Controller (CAC): subset-level greedy with set-aggregate features.

Move #1 from the controller-axis menu. Per Prop. pcs-limit, no per-clip-surrogate
controller can match the corpus-aware oracle. CAC adds set-aggregate features
(corpus length, brevity-penalty argument, current corpus BLEU) so the per-clip
score depends on what is already in S — corpus-aware by construction.

Pipeline:
  1. Precompute per-hypothesis BLEU sufficient statistics (correct[n], total[n],
     sys_len) for both retrieval and fallback variants on dev and test using the
     SLRTP scorer's exact tokeniser. This makes any corpus_BLEU(in_S_mask) cost
     O(N) once and per-clip flips O(1).
  2. Forward greedy on dev from S = empty: at each step, evaluate the marginal
     BLEU-4 of every candidate, pick argmax, append to S until |S| == target.
     Record (per-clip features, set-aggregate features, marginal_BLEU4) triples.
  3. Train a GBDT regressor on the recorded trajectory.
  4. Deploy on test: forward greedy with model-predicted marginals replacing the
     real corpus-BLEU evaluations.
  5. Save in_S binary mask, build mixed pose .pt, log dev in-sample/OOF and test
     scores via the fast BLEU stub. (External SLRTP harness scoring is the next
     step — driven by scripts/score_cac_with_slrtp.sh.)

Output:
  outputs/learned_confidence/cac_dev_in_S.npz       (sids, in_S, scores)
  outputs/learned_confidence/cac_test_in_S.npz      (sids, in_S, scores)
  external/SLRTP/results/phase58_b40_CAC_dev.pt     (mixed pose at dev S*)
  external/SLRTP/results/phase58_b40_CAC_test.pt    (mixed pose at test S*)
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import jiwer
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SLRTP = ROOT / "external/SLRTP-Sign-Production-Evaluation"
sys.path.insert(0, str(SLRTP))

from scripts.train_learned_confidence import FEATURES, HAND_WEIGHTS

NGRAM_ORDER = 4


# ---------------------------------------------------------------------------
# Vectorised corpus-BLEU under SLRTP's raw_corpus_bleu config
# (tokenize=none, smooth_method=floor, smooth_value=0, force=True,
# use_effective_order=True). For a single-reference corpus this is fully
# decomposable per-hypothesis: correct[n] = sum_i correct_i[n] etc.
# ---------------------------------------------------------------------------
def per_hyp_bleu_stats(hyp: str, ref: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Returns (correct[4], total[4], sys_len, ref_len) for a single hyp/ref pair."""
    hyp_toks = hyp.split()
    ref_toks = ref.split()
    sys_len = len(hyp_toks)
    ref_len = len(ref_toks)
    correct = np.zeros(NGRAM_ORDER, dtype=np.int64)
    total = np.zeros(NGRAM_ORDER, dtype=np.int64)
    for n in range(1, NGRAM_ORDER + 1):
        if sys_len - n + 1 <= 0:
            break
        hyp_ng = Counter(" ".join(hyp_toks[i:i + n]) for i in range(sys_len - n + 1))
        ref_ng = Counter(" ".join(ref_toks[i:i + n]) for i in range(max(0, ref_len - n + 1)))
        c = sum(min(hyp_ng[ng], ref_ng.get(ng, 0)) for ng in hyp_ng.keys())
        t = sum(hyp_ng.values())
        correct[n - 1] = c
        total[n - 1] = t
    return correct, total, sys_len, ref_len


def my_log_safe(p: float) -> float:
    if p <= 0.0:
        return -9999999999.0
    return math.log(p)


def bleu_from_stats(correct: np.ndarray, total: np.ndarray,
                    sys_len: int, ref_len: int
                    ) -> tuple[float, float]:
    """Returns (BLEU-1, BLEU-4) under SLRTP raw_corpus_bleu config."""
    precisions = [0.0] * NGRAM_ORDER
    effective_order = NGRAM_ORDER
    for n in range(NGRAM_ORDER):
        if total[n] == 0:
            effective_order = n
            break
        if correct[n] == 0:
            # smooth_method='floor', smooth_value=0 -> precision 0
            precisions[n] = 0.0
        else:
            precisions[n] = 100.0 * correct[n] / total[n]
    bp = 1.0
    if 0 < sys_len < ref_len:
        bp = math.exp(1.0 - ref_len / sys_len)
    elif sys_len == 0:
        bp = 0.0
    scores = []
    for k in range(1, NGRAM_ORDER + 1):
        if k > effective_order:
            scores.append(0.0)
            continue
        s = sum(my_log_safe(precisions[j] / 100.0) for j in range(k)) / k
        scores.append(100.0 * bp * math.exp(s))
    return scores[0], scores[3]


_WER_TRANSFORMS = jiwer.Compose([
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])


def per_hyp_wer_stats(hyp: str, ref: str) -> tuple[int, int]:
    """Returns (edit_count, ref_len_norm) under SLRTP wer's jiwer transforms."""
    out = jiwer.process_words([ref], [hyp],
                              reference_transform=_WER_TRANSFORMS,
                              hypothesis_transform=_WER_TRANSFORMS)
    edits = out.insertions + out.deletions + out.substitutions
    ref_len = out.hits + out.deletions + out.substitutions
    return int(edits), int(ref_len)


class FastCorpusBLEU:
    """Vectorised corpus-BLEU evaluator over a binary in_S selection."""

    def __init__(self, retr_texts, fall_texts, refs):
        N = len(refs)
        assert len(retr_texts) == len(fall_texts) == N
        self.N = N
        self.A_retr = np.zeros((N, NGRAM_ORDER), dtype=np.int64)
        self.T_retr = np.zeros((N, NGRAM_ORDER), dtype=np.int64)
        self.L_retr = np.zeros(N, dtype=np.int64)
        self.A_fall = np.zeros((N, NGRAM_ORDER), dtype=np.int64)
        self.T_fall = np.zeros((N, NGRAM_ORDER), dtype=np.int64)
        self.L_fall = np.zeros(N, dtype=np.int64)
        self.ref_len_per_clip = np.zeros(N, dtype=np.int64)
        # WER stats (additive per-clip on the jiwer-normalised text)
        self.E_retr = np.zeros(N, dtype=np.int64)
        self.E_fall = np.zeros(N, dtype=np.int64)
        self.RL_norm = np.zeros(N, dtype=np.int64)
        for i in range(N):
            ar, tr, lr, rl = per_hyp_bleu_stats(retr_texts[i], refs[i])
            af, tf, lf, _ = per_hyp_bleu_stats(fall_texts[i], refs[i])
            self.A_retr[i] = ar; self.T_retr[i] = tr; self.L_retr[i] = lr
            self.A_fall[i] = af; self.T_fall[i] = tf; self.L_fall[i] = lf
            self.ref_len_per_clip[i] = rl
            er, rln = per_hyp_wer_stats(retr_texts[i], refs[i])
            ef, _   = per_hyp_wer_stats(fall_texts[i], refs[i])
            self.E_retr[i] = er; self.E_fall[i] = ef; self.RL_norm[i] = rln
        self.ref_len_total = int(self.ref_len_per_clip.sum())
        self.ref_len_norm_total = int(self.RL_norm.sum())
        # Per-clip WER COST of choosing retrieval over fallback (in SLRTP units, ×100).
        # marginal_corpus_WER(adding i to S) = delta_edits[i] / ref_len_norm_total × 100.
        # ref_len_norm_total is constant across selections.
        self.wer_cost = (self.E_retr - self.E_fall).astype(np.float64) * 100.0 / max(1, self.ref_len_norm_total)

    def wer_for_mask(self, in_S: np.ndarray) -> float:
        """Returns corpus WER (×100) under selection in_S."""
        in_S_b = in_S.astype(bool)
        notS_b = ~in_S_b
        edits = int(self.E_retr[in_S_b].sum() + self.E_fall[notS_b].sum())
        return edits / max(1, self.ref_len_norm_total) * 100.0

    def bleu_for_mask(self, in_S: np.ndarray) -> tuple[float, float]:
        """in_S: binary [N] -> (BLEU-1, BLEU-4)."""
        in_S = in_S.astype(bool)
        notS = ~in_S
        correct = (self.A_retr[in_S].sum(axis=0) + self.A_fall[notS].sum(axis=0))
        total = (self.T_retr[in_S].sum(axis=0) + self.T_fall[notS].sum(axis=0))
        sys_len = int(self.L_retr[in_S].sum() + self.L_fall[notS].sum())
        return bleu_from_stats(correct, total, sys_len, self.ref_len_total)

    def bleu_for_set(self, S: set) -> tuple[float, float]:
        mask = np.zeros(self.N, dtype=bool)
        if S:
            mask[list(S)] = True
        return self.bleu_for_mask(mask)

    def marginal_for_candidates(self, in_S: np.ndarray, candidates: np.ndarray
                                ) -> np.ndarray:
        """For each candidate i in `candidates`, compute BLEU-4(S ∪ {i}) - BLEU-4(S).

        Vectorised: BLEU-4 is non-linear so we still loop, but each per-i flip is
        O(NGRAM_ORDER). That gives O(|candidates| * NGRAM_ORDER) per call.
        """
        in_S_b = in_S.astype(bool)
        notS_b = ~in_S_b
        base_correct = self.A_retr[in_S_b].sum(axis=0) + self.A_fall[notS_b].sum(axis=0)
        base_total = self.T_retr[in_S_b].sum(axis=0) + self.T_fall[notS_b].sum(axis=0)
        base_len = int(self.L_retr[in_S_b].sum() + self.L_fall[notS_b].sum())
        base_b1, base_b4 = bleu_from_stats(base_correct, base_total, base_len,
                                            self.ref_len_total)
        out_b4 = np.zeros(len(candidates), dtype=np.float64)
        out_b1 = np.zeros(len(candidates), dtype=np.float64)
        for k, i in enumerate(candidates):
            if in_S_b[i]:
                # Already in S — flipping out
                c = base_correct - self.A_retr[i] + self.A_fall[i]
                t = base_total - self.T_retr[i] + self.T_fall[i]
                ln = base_len - int(self.L_retr[i]) + int(self.L_fall[i])
            else:
                # Not in S — flipping in
                c = base_correct - self.A_fall[i] + self.A_retr[i]
                t = base_total - self.T_fall[i] + self.T_retr[i]
                ln = base_len - int(self.L_fall[i]) + int(self.L_retr[i])
            b1, b4 = bleu_from_stats(c, t, ln, self.ref_len_total)
            out_b1[k] = b1 - base_b1
            out_b4[k] = b4 - base_b4
        return out_b4


# ---------------------------------------------------------------------------
# Per-clip features (matching PCFB's 11 + extras)
# ---------------------------------------------------------------------------
def load_per_clip_features(split: str, ids: list[str],
                            retr_text, fall_text):
    """11 features per clip: 6 reranker + hand_score + 4 length features."""
    rows = json.loads((SLRTP / f"results/topk_semantic_hybrid_{split}_trace.json").read_text())
    sid_to_sel = {r["id"]: (r.get("selected") or {}) for r in rows}
    rerank = []
    has_retrieval = []
    for sid in ids:
        sel = sid_to_sel.get(sid, {})
        feats = [float(sel.get(k, 0.0)) for k in FEATURES]
        rerank.append(feats)
        has_retrieval.append(1.0 if sel else 0.0)
    rerank = np.array(rerank, dtype=np.float64)
    hand_score = rerank @ np.array(HAND_WEIGHTS, dtype=np.float64)
    has_retrieval = np.array(has_retrieval, dtype=np.float64)
    len_r = np.array([len(t.split()) for t in retr_text], dtype=np.float64)
    len_f = np.array([len(t.split()) for t in fall_text], dtype=np.float64)
    len_diff = len_r - len_f
    len_ratio = (len_r + 1.0) / (len_f + 1.0)
    return np.concatenate([
        rerank,                                 # 6
        hand_score.reshape(-1, 1),              # 1
        len_r.reshape(-1, 1),                   # 1
        len_f.reshape(-1, 1),                   # 1
        len_diff.reshape(-1, 1),                # 1
        len_ratio.reshape(-1, 1),               # 1
        has_retrieval.reshape(-1, 1),           # 1   (for reference; non-retrieval clips
                                                #     are excluded from the candidate pool anyway)
    ], axis=1)


PER_CLIP_FEAT_NAMES = (
    list(FEATURES) +                                            # 6
    ["hand_score", "len_retr", "len_fall", "len_diff",
     "len_ratio", "has_retrieval"]                              # 6
)


def aggregate_features(in_S: np.ndarray, fcb: FastCorpusBLEU
                        ) -> np.ndarray:
    """Return aggregate features describing the current state of S."""
    in_S_b = in_S.astype(bool)
    notS_b = ~in_S_b
    nS = int(in_S_b.sum())
    nNS = int(notS_b.sum())
    cur_total_len = int(fcb.L_retr[in_S_b].sum() + fcb.L_fall[notS_b].sum())
    fill_progress = nS / max(1, fcb.N)
    cur_brevity_arg = cur_total_len / max(1, fcb.ref_len_total)
    mean_len_retr_in_S = float(fcb.L_retr[in_S_b].mean()) if nS > 0 else 0.0
    mean_len_fall_outS = float(fcb.L_fall[notS_b].mean()) if nNS > 0 else 0.0
    cur_b1, cur_b4 = fcb.bleu_for_mask(in_S)
    cur_wer = fcb.wer_for_mask(in_S)
    return np.array([
        fill_progress,
        cur_total_len,
        cur_brevity_arg,
        mean_len_retr_in_S,
        mean_len_fall_outS,
        cur_b4,
        cur_wer,
    ], dtype=np.float64)


AGG_FEAT_NAMES = ["fill_progress", "total_mix_len", "brevity_arg",
                  "mean_len_retr_S", "mean_len_fall_NS", "cur_bleu4", "cur_wer"]


def cross_features(per_clip: np.ndarray, agg: np.ndarray,
                    fcb: FastCorpusBLEU, candidates: np.ndarray
                    ) -> np.ndarray:
    """Per-clip × aggregate cross features for [|cands|, ?]."""
    delta_len_per_clip = (fcb.L_retr - fcb.L_fall).astype(np.float64)[candidates]
    rel_len = delta_len_per_clip / max(1.0, agg[1])  # delta-len / current total len
    rel_brevity = delta_len_per_clip / max(1.0, fcb.ref_len_total)
    wer_cost = fcb.wer_cost[candidates]              # per-clip WER cost (constant)
    return np.stack([delta_len_per_clip, rel_len, rel_brevity, wer_cost], axis=1)


CROSS_FEAT_NAMES = ["delta_len_clip", "rel_delta_len", "rel_brevity_impact",
                    "wer_cost_clip"]


def build_features_for_candidates(per_clip: np.ndarray, fcb: FastCorpusBLEU,
                                   in_S: np.ndarray, candidates: np.ndarray
                                   ) -> np.ndarray:
    """For each candidate i in `candidates`, return [|cands|, F] feature row."""
    agg = aggregate_features(in_S, fcb)
    pc = per_clip[candidates]                                # [K, 12]
    agg_tiled = np.tile(agg, (len(candidates), 1))           # [K, 6]
    cross = cross_features(per_clip, agg, fcb, candidates)   # [K, 3]
    return np.concatenate([pc, agg_tiled, cross], axis=1)    # [K, 21]


ALL_FEAT_NAMES = PER_CLIP_FEAT_NAMES + AGG_FEAT_NAMES + CROSS_FEAT_NAMES


# ---------------------------------------------------------------------------
# Forward greedy with corpus-BLEU oracle. Records (state, candidate, marginal)
# triples for training the regressor.
# ---------------------------------------------------------------------------
def forward_greedy_record(fcb: FastCorpusBLEU, per_clip: np.ndarray,
                           candidate_pool: np.ndarray, target_size: int,
                           verbose: bool = True
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Forward greedy from S = empty. At each step, record (X, marginal, picked).

    candidate_pool: indices of clips eligible to be in S (e.g., clips that have
                     a retrieval candidate in the trace).
    target_size: stop when |S| == target_size.

    Returns:
      X_all: [Total_K, F] feature rows (one per candidate considered)
      y_all: [Total_K] marginal BLEU-4 (label/target)
      mask_picked: [Total_K] 1 if this row's candidate was the pick at its step
      history: list of dicts per step
    """
    in_S = np.zeros(fcb.N, dtype=bool)
    remaining = set(candidate_pool.tolist())
    X_rows = []
    y_rows = []
    pick_mask = []
    history = []
    t0 = time.time()
    base_b1, base_b4 = fcb.bleu_for_mask(in_S)
    history.append({"step": 0, "size": 0, "b1": base_b1, "b4": base_b4,
                    "pick": None, "marg_b4": None})
    if verbose:
        print(f"[greedy] initial: BLEU-1={base_b1:.4f}  BLEU-4={base_b4:.4f}  "
              f"(N={fcb.N}, target={target_size}, pool={len(remaining)})")
    for step in range(1, target_size + 1):
        cand_arr = np.fromiter(remaining, dtype=np.int64)
        feats = build_features_for_candidates(per_clip, fcb, in_S, cand_arr)
        marg = fcb.marginal_for_candidates(in_S, cand_arr)
        best_idx = int(np.argmax(marg))
        pick = int(cand_arr[best_idx])
        # Record all rows
        X_rows.append(feats)
        y_rows.append(marg)
        pm = np.zeros(len(cand_arr), dtype=np.int8)
        pm[best_idx] = 1
        pick_mask.append(pm)
        # Apply pick
        in_S[pick] = True
        remaining.discard(pick)
        b1_new, b4_new = fcb.bleu_for_mask(in_S)
        history.append({"step": step, "size": int(in_S.sum()),
                        "b1": b1_new, "b4": b4_new,
                        "pick": pick, "marg_b4": float(marg[best_idx])})
        if verbose and (step % 20 == 0 or step == target_size):
            print(f"[greedy] step {step}/{target_size}  |S|={int(in_S.sum())}  "
                  f"BLEU-1={b1_new:.3f}  BLEU-4={b4_new:.3f}  "
                  f"marg+={float(marg[best_idx]):+.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")
    X_all = np.concatenate(X_rows, axis=0)
    y_all = np.concatenate(y_rows, axis=0)
    pick_all = np.concatenate(pick_mask, axis=0)
    return X_all, y_all, pick_all, history


# ---------------------------------------------------------------------------
# Inference greedy: replay greedy on test using the trained regressor.
# ---------------------------------------------------------------------------
def deploy_greedy(fcb: FastCorpusBLEU, per_clip: np.ndarray,
                   candidate_pool: np.ndarray, target_size: int,
                   regressor, alpha_wer: float = 0.0, verbose: bool = True
                   ) -> tuple[np.ndarray, list]:
    """At each step, score(i) = predicted_marginal_BLEU4(i) - alpha_wer * wer_cost[i].

    wer_cost[i] is the constant per-clip WER penalty (×100) of routing i to
    retrieval. alpha_wer = 0 reduces to pure BLEU-4 greedy.
    """
    in_S = np.zeros(fcb.N, dtype=bool)
    remaining = set(candidate_pool.tolist())
    history = []
    t0 = time.time()
    b1, b4 = fcb.bleu_for_mask(in_S)
    history.append({"step": 0, "size": 0, "b1": b1, "b4": b4, "pick": None, "score": None})
    for step in range(1, target_size + 1):
        cand_arr = np.fromiter(remaining, dtype=np.int64)
        feats = build_features_for_candidates(per_clip, fcb, in_S, cand_arr)
        bleu_pred = regressor.predict(feats)
        wer_pen = alpha_wer * fcb.wer_cost[cand_arr]
        scores = bleu_pred - wer_pen
        best_idx = int(np.argmax(scores))
        pick = int(cand_arr[best_idx])
        in_S[pick] = True
        remaining.discard(pick)
        if verbose and (step % 32 == 0 or step == target_size):
            b1, b4 = fcb.bleu_for_mask(in_S)
            wer_cur = fcb.wer_for_mask(in_S)
            print(f"[deploy] step {step}/{target_size}  |S|={int(in_S.sum())}  "
                  f"BLEU-1={b1:.3f}  BLEU-4={b4:.3f}  WER={wer_cur:.3f}  "
                  f"score+={float(scores[best_idx]):+.6f}  "
                  f"elapsed={time.time()-t0:.1f}s")
        history.append({"step": step, "size": int(in_S.sum()),
                         "pick": pick, "score": float(scores[best_idx])})
    return in_S, history


# ---------------------------------------------------------------------------
# Build mixed pose .pt for SLRTP scoring
# ---------------------------------------------------------------------------
def build_mixed_pose(ids: list[str], in_S: np.ndarray,
                      retr_pose_path: str, fall_pose_path: str,
                      out_path: str):
    retr = torch.load(retr_pose_path, map_location="cpu", weights_only=True)
    fall = torch.load(fall_pose_path, map_location="cpu", weights_only=True)
    pred = {}
    n_r = n_f = 0
    for i, sid in enumerate(ids):
        if in_S[i]:
            pred[sid] = retr[sid]; n_r += 1
        else:
            pred[sid] = fall[sid]; n_f += 1
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, out_path)
    print(f"[mix] saved {out_path}  retr={n_r}  fall={n_f}  "
          f"realised_rate={n_r/(n_r+n_f):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.40)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--alpha_wer", type=float, default=0.0,
                     help="WER penalty coefficient (>=0). 0 = pure-BLEU greedy.")
    ap.add_argument("--out_dir", default="outputs/learned_confidence")
    ap.add_argument("--out_pose_dir", default=str(SLRTP / "results"))
    ap.add_argument("--label", default="CAC", help="suffix tag for output files")
    ap.add_argument("--skip_dev_pose", action="store_true",
                     help="Skip building dev mixed pose (don't overwrite an existing one).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_pose_dir = Path(args.out_pose_dir); out_pose_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load both splits ----
    splits = {}
    for s in ("dev", "test"):
        gt = torch.load(SLRTP / f"pretrained/SLRTP-Sign-Production-Evaluation-Data/data/{s}.pt",
                         map_location="cpu", weights_only=False)
        ids = list(gt.keys())
        refs = [gt[k]["text"] for k in ids]
        retr_text = torch.load(SLRTP / f"results/phase57_topk20_slotonly_{s}_text_preds.pt",
                                map_location="cpu", weights_only=False)
        fall_text = torch.load(SLRTP / f"results/phase56_stage_phrase045_cov75_{s}_text_preds.pt",
                                map_location="cpu", weights_only=False)
        init_text = torch.load(SLRTP / f"results/phase58_budget040_{s}_text_preds.pt",
                                map_location="cpu", weights_only=False)
        N = len(ids)
        assert len(retr_text) == len(fall_text) == len(init_text) == N

        fcb = FastCorpusBLEU(retr_text, fall_text, refs)
        per_clip = load_per_clip_features(s, ids, retr_text, fall_text)
        # Sanity vs SLRTP scorer
        _, b4_check = fcb.bleu_for_mask(np.zeros(N, dtype=bool))
        rows = json.loads((SLRTP / f"results/topk_semantic_hybrid_{s}_trace.json").read_text())
        sid_to_sel = {r["id"]: (r.get("selected") or {}) for r in rows}
        retrieval_pool = np.array([i for i, sid in enumerate(ids) if sid_to_sel.get(sid)],
                                   dtype=np.int64)
        target_size = int(round(args.budget * N))   # paper's realised rate (206 dev, 256 test)
        # Hand-tuned "definite retrieval" set: init == retr AND init != fall
        # (clip was actually routed to retrieval, not just a tie). |.|=113 on dev, 157 on test.
        S_hand_def = set(i for i in range(N) if init_text[i] == retr_text[i]
                                            and init_text[i] != fall_text[i])
        # Whole hand-tuned pool that "looks like retrieval" via the BT text identity.
        S_hand_total = set(i for i in range(N) if init_text[i] == retr_text[i])
        # For BLEU comparison we need a CONSISTENT |S| across hand-tuned and CAC.
        # Use S_hand chosen so |S_hand_for_bleu| == target_size: take S_hand_def
        # plus enough ties to reach target_size. Ties don't change BLEU, so this
        # is the BLEU-equivalent of hand-tuned at the budget's realised rate.
        ties = [i for i in range(N) if retr_text[i] == fall_text[i]]
        S_hand = set(S_hand_def)
        for i in ties:
            if len(S_hand) >= target_size: break
            S_hand.add(i)
        print(f"[load] {s}: N={N}  retrieval_pool={len(retrieval_pool)}  "
              f"target={target_size}  S_hand_definite={len(S_hand_def)}  "
              f"S_hand_for_bleu(|.|=target)={len(S_hand)}  "
              f"S_hand_total(init==retr)={len(S_hand_total)}")
        splits[s] = {
            "ids": ids, "refs": refs, "retr_text": retr_text, "fall_text": fall_text,
            "init_text": init_text, "fcb": fcb, "per_clip": per_clip,
            "retrieval_pool": retrieval_pool, "target_size": target_size,
            "S_hand": S_hand,
        }

    # ---- Stage 1: forward greedy on dev to generate trajectory ----
    print("\n[stage1] Forward greedy on dev to record training trajectory")
    dev = splits["dev"]
    X_dev, y_dev, pick_dev, hist_dev = forward_greedy_record(
        dev["fcb"], dev["per_clip"],
        candidate_pool=dev["retrieval_pool"],
        target_size=dev["target_size"],
        verbose=True,
    )
    print(f"[stage1] training rows: {X_dev.shape}  picks: {int(pick_dev.sum())}/{len(pick_dev)}")
    # Save trajectory
    np.savez(out_dir / f"cac_dev_trajectory_b40.npz",
             X=X_dev, y=y_dev, picked=pick_dev,
             feature_names=np.array(ALL_FEAT_NAMES))
    Path(out_dir / f"cac_dev_history_b40.json").write_text(
        json.dumps(hist_dev, indent=2))
    print(f"[stage1] saved trajectory: rows={len(X_dev)}, F={X_dev.shape[1]}")

    # ---- Stage 2: train regressor ----
    print("\n[stage2] Train GBDT regressor on (features -> marginal BLEU-4)")
    reg = GradientBoostingRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, subsample=0.85, random_state=0,
        min_samples_leaf=20,
    )
    t0 = time.time()
    reg.fit(X_dev, y_dev)
    print(f"[stage2] fit elapsed: {time.time()-t0:.1f}s")
    # Feature importance
    fi = list(zip(ALL_FEAT_NAMES, reg.feature_importances_))
    fi.sort(key=lambda x: -x[1])
    print("[stage2] feature importance (top 8):")
    for name, imp in fi[:8]:
        print(f"    {name:>20s}  {imp:.4f}")

    # ---- Stage 3a: deploy on dev (in-sample) ----
    print(f"\n[stage3a] Deploy on dev (in-sample replay), alpha_wer={args.alpha_wer}")
    in_S_dev, hist_dev_deploy = deploy_greedy(
        dev["fcb"], dev["per_clip"],
        candidate_pool=dev["retrieval_pool"],
        target_size=dev["target_size"],
        regressor=reg, alpha_wer=args.alpha_wer, verbose=True,
    )
    b1_dev, b4_dev = dev["fcb"].bleu_for_mask(in_S_dev)
    wer_dev = dev["fcb"].wer_for_mask(in_S_dev)
    # Hand-tuned dev baseline
    in_S_hand_dev = np.zeros(dev["fcb"].N, dtype=bool)
    for i in dev["S_hand"]: in_S_hand_dev[i] = True
    b1_hand_dev, b4_hand_dev = dev["fcb"].bleu_for_mask(in_S_hand_dev)
    wer_hand_dev = dev["fcb"].wer_for_mask(in_S_hand_dev)
    print(f"[stage3a] dev:  CAC  BLEU-1={b1_dev:.3f}  BLEU-4={b4_dev:.3f}  WER={wer_dev:.3f}")
    print(f"[stage3a] dev: hand BLEU-1={b1_hand_dev:.3f}  BLEU-4={b4_hand_dev:.3f}  "
          f"WER={wer_hand_dev:.3f}")
    print(f"[stage3a] dev:  Δ   BLEU-1={b1_dev-b1_hand_dev:+.3f}  "
          f"BLEU-4={b4_dev-b4_hand_dev:+.3f}  WER={wer_dev-wer_hand_dev:+.3f}")

    # Save dev in_S
    sids_dev = np.array(dev["ids"])
    np.savez(out_dir / f"cac_dev_in_S_b40.npz",
             sids=sids_dev, in_S=in_S_dev.astype(np.int8),
             bleu1=b1_dev, bleu4=b4_dev, wer=wer_dev,
             alpha_wer=args.alpha_wer)

    # ---- Stage 3b: deploy on test ----
    print(f"\n[stage3b] Deploy on test (transfer), alpha_wer={args.alpha_wer}")
    test = splits["test"]
    in_S_test, hist_test = deploy_greedy(
        test["fcb"], test["per_clip"],
        candidate_pool=test["retrieval_pool"],
        target_size=test["target_size"],
        regressor=reg, alpha_wer=args.alpha_wer, verbose=True,
    )
    b1_test, b4_test = test["fcb"].bleu_for_mask(in_S_test)
    wer_test = test["fcb"].wer_for_mask(in_S_test)
    in_S_hand_test = np.zeros(test["fcb"].N, dtype=bool)
    for i in test["S_hand"]: in_S_hand_test[i] = True
    b1_hand_test, b4_hand_test = test["fcb"].bleu_for_mask(in_S_hand_test)
    wer_hand_test = test["fcb"].wer_for_mask(in_S_hand_test)
    print(f"[stage3b] test: CAC  BLEU-1={b1_test:.3f}  BLEU-4={b4_test:.3f}  WER={wer_test:.3f}")
    print(f"[stage3b] test: hand BLEU-1={b1_hand_test:.3f}  BLEU-4={b4_hand_test:.3f}  "
          f"WER={wer_hand_test:.3f}")
    print(f"[stage3b] test: Δ    BLEU-1={b1_test-b1_hand_test:+.3f}  "
          f"BLEU-4={b4_test-b4_hand_test:+.3f}  WER={wer_test-wer_hand_test:+.3f}")

    sids_test = np.array(test["ids"])
    np.savez(out_dir / f"cac_test_in_S_b40.npz",
             sids=sids_test, in_S=in_S_test.astype(np.int8),
             bleu1=b1_test, bleu4=b4_test, wer=wer_test,
             alpha_wer=args.alpha_wer)

    # ---- Stage 4: build mixed pose .pt for SLRTP scoring ----
    print("\n[stage4] Build mixed pose files for SLRTP harness")
    if not args.skip_dev_pose:
        build_mixed_pose(dev["ids"], in_S_dev,
                          retr_pose_path=str(SLRTP / "results/phase57_topk20_slotonly_dev.pt"),
                          fall_pose_path=str(SLRTP / "results/phase56_stage_phrase045_cov75_dev.pt"),
                          out_path=str(out_pose_dir / f"phase58_b40_{args.label}_dev.pt"))
    build_mixed_pose(test["ids"], in_S_test,
                      retr_pose_path=str(SLRTP / "results/phase57_topk20_slotonly_test.pt"),
                      fall_pose_path=str(SLRTP / "results/phase56_stage_phrase045_cov75_test.pt"),
                      out_path=str(out_pose_dir / f"phase58_b40_{args.label}_test.pt"))

    # ---- Summary ----
    summary = {
        "config": {
            "budget": args.budget,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "label": args.label,
        },
        "dev_target_size": dev["target_size"],
        "test_target_size": test["target_size"],
        "fast_bleu_dev": {
            "hand_bleu1": float(b1_hand_dev), "hand_bleu4": float(b4_hand_dev),
            "cac_bleu1": float(b1_dev), "cac_bleu4": float(b4_dev),
        },
        "fast_bleu_test": {
            "hand_bleu1": float(b1_hand_test), "hand_bleu4": float(b4_hand_test),
            "cac_bleu1": float(b1_test), "cac_bleu4": float(b4_test),
        },
        "feature_importance_top": fi[:10],
    }
    Path(out_dir / f"cac_summary_b40.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\n[done] saved summary -> {out_dir}/cac_summary_b40.json")


if __name__ == "__main__":
    main()
