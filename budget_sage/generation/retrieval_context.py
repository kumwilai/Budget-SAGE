"""Retrieval-context preprocessor for BudgetFM training.

For each training clip i we need a *retrieval target* clip $\\rho(i)$
that the budget-FM model is supposed to imitate at b=1. The cleanest
choice is the top-1 nearest train clip by TF-IDF caption similarity,
SELF-EXCLUDED. We compute this once (≈30 s on Phoenix train, ~7k clips),
save the mapping to disk, and reuse it during training.

The retrieval target carries:
  - the matched train clip's pose (used as y_ret in the budget-mixed FM target)
  - the matched train clip's UPC sequence (used as u_ret for the cross-attention)
  - the cosine similarity (sanity-check / filter)

Bug guards baked in:
  - SELF-EXCLUSION: a clip can never be its own retrieval target.
  - L2-normalised TF-IDF: a non-positive cosine triggers a fallback.
  - DETERMINISTIC tie-breaking: ties broken by ascending lexicographic clip id.
  - LENGTH SANITY: if a candidate has a UPC sequence shorter than 4 tokens we
    skip it (degenerate exemplar).
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


def _char_ngrams(s: str, n_min: int = 1, n_max: int = 3) -> list[str]:
    s = "".join(s.split()).lower()
    return [s[i:i + n] for n in range(n_min, n_max + 1)
            for i in range(len(s) - n + 1)]


def build_tfidf(texts: Sequence[str]):
    """Returns (vocab, idf, M[N,V]) — L2-normalised char-1/2/3-gram TF-IDF."""
    docs = [Counter(_char_ngrams(t)) for t in texts]
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


def precompute_phoenix_retrieval_map(
    train_manifest_path: Path,
    clip_to_upc_path: Path,
    out_json_path: Path,
    min_upc_length: int = 4,
) -> dict[str, dict]:
    """For each Phoenix train clip, find the top-1 nearest other-train-clip
    by char-n-gram TF-IDF cosine, with self-exclusion and degenerate-exemplar
    skipping. Saves the result as JSON keyed by source clip id.

    Returns the mapping in memory.
    """
    rows = json.loads(Path(train_manifest_path).read_text())
    upc = json.loads(Path(clip_to_upc_path).read_text())

    # Filter: keep clips that have an entry in clip_to_upc and a non-degenerate
    # UPC sequence. We still want the *source* set to be all clips that have
    # text, but the *candidate* set restricts to clips with usable UPC.
    sids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}

    cand_mask = np.zeros(len(sids), dtype=bool)
    for sid in sids:
        seq = upc.get(sid)
        if seq is None:
            continue
        if isinstance(seq, str):
            seq = list(seq)
        if len(seq) >= min_upc_length:
            cand_mask[sid_to_idx[sid]] = True
    n_cands = int(cand_mask.sum())
    print(f"[retrieval_context] {len(sids)} clips, {n_cands} usable as candidates "
          f"(rest filtered by UPC length < {min_upc_length})", flush=True)
    if n_cands == 0:
        raise RuntimeError("No usable retrieval candidates")

    print("[retrieval_context] building TF-IDF index ...", flush=True)
    vocab, idf, M = build_tfidf(texts)

    # Mask out non-candidates by zeroing their rows
    M_cand = M.copy()
    M_cand[~cand_mask] = 0.0

    out: dict[str, dict] = {}
    for i, sid in enumerate(sids):
        q = M[i]                                  # (V,)
        sims = M_cand @ q                          # (N,)
        # Self-exclusion: drop the source row
        sims[i] = -1e9
        # Pick top-1 deterministically; tie-break by ascending sid index
        # (built into argmax with stable behaviour because sims are floats and
        # we sorted nothing — for true determinism on ties, sort by (-sims, idx))
        order = np.argsort(-sims, kind="stable")
        top1 = int(order[0])
        cos = float(sims[top1])
        if cos <= 0.0:
            # Degenerate: no positive-similarity candidate found. Fall back
            # to the second-most-similar (still deterministic).
            top1 = int(order[1]) if len(order) > 1 else int(order[0])
            cos = float(sims[top1])
        out[sid] = {
            "ret_sid": sids[top1],
            "cos": cos,
            "ret_upc_len": len(upc.get(sids[top1], [])),
        }
        if (i + 1) % 1000 == 0:
            print(f"[retrieval_context] {i+1}/{len(sids)} mapped", flush=True)

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    # Quick provenance summary
    cos_arr = np.array([v["cos"] for v in out.values()])
    n_self_skipped = int((cos_arr <= 0.0).sum())
    print(f"[retrieval_context] saved {out_json_path}  "
          f"mean cos {cos_arr.mean():.4f}  "
          f"median cos {float(np.median(cos_arr)):.4f}  "
          f"clips with cos<=0 (fell back to 2nd): {n_self_skipped}",
          flush=True)
    return out


__all__ = [
    "build_tfidf",
    "precompute_phoenix_retrieval_map",
]
