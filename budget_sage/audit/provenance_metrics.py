"""Copy-Leakage / Provenance audit metrics for Budget-SAGE.

Quantifies how much each generated output resembles training data, beyond
the nominal retrieval-rate disclosure. Five per-sample signals are combined
into a single Copy-Leakage Index (CLI) in [0, 1]:

    CLI = 0.30 * pose_nn_similarity
        + 0.30 * upc_ngram_overlap
        + 0.25 * frame_source_concentration
        + 0.15 * text_nn_similarity

The module is CPU-only and stream-friendly: every function operates on a
single (caption_i, output_pose_i, optional UPC sequence, optional source
trace) tuple and returns numpy floats. The driver `score_pickle_against_train`
walks an on-disk MSKA-format pickle (the format produced by
`build_csl_pg_rast_native_hrnet.py` and `build_csl_pg_rast_with_tier2.py`)
and writes per-sample + corpus-level CSV/Markdown reports.

No new training is required; the metrics consume pose tensors that already
exist on disk plus the train pose pool used by the same pipeline.
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 1. Pose similarity (DTW-MJE on a small, fast 2D-MJE proxy)
# ---------------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _flatten_pose(p: np.ndarray) -> np.ndarray:
    """(T, J, C) -> (T, J*C). xy only if C==3 (drop confidence)."""
    p = _to_np(p)
    if p.ndim == 3:
        if p.shape[-1] == 3:
            p = p[..., :2]
        T, J, C = p.shape
        p = p.reshape(T, J * C)
    return p


def _resample_T(p: np.ndarray, T_target: int) -> np.ndarray:
    if p.shape[0] == T_target:
        return p
    src_T = p.shape[0]
    if src_T < 2:
        return np.repeat(p[:1], T_target, axis=0)
    idx = np.linspace(0, src_T - 1, T_target)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, src_T - 1)
    w = (idx - i0).astype(np.float32)[:, None]
    return ((1.0 - w) * p[i0] + w * p[i1]).astype(np.float32)


def fast_pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Resampled mean joint error in pixel units. Approximates DTW-MJE for
    fast nearest-neighbor scans; good enough for ranking the top-1 source."""
    af = _flatten_pose(a)
    bf = _flatten_pose(b)
    T = max(af.shape[0], bf.shape[0])
    af = _resample_T(af, T)
    bf = _resample_T(bf, T)
    diff = af - bf
    return float(np.sqrt((diff * diff).mean()))


def pose_nn_similarity(out_pose: np.ndarray,
                        train_pool: dict[str, np.ndarray],
                        scale: float = 80.0) -> tuple[Optional[str], float]:
    """Find nearest train clip by fast pose distance. Returns (sid, sim) where
    sim = exp(-d/scale) in [0, 1]. `scale` is set so a typical-MJE pose
    (~80 px) gives sim ~0.37; identical clips give sim ~1.0."""
    best_sid, best_d = None, float("inf")
    for sid, p in train_pool.items():
        d = fast_pose_distance(out_pose, p)
        if d < best_d:
            best_d = d; best_sid = sid
    if best_sid is None:
        return None, 0.0
    sim = float(math.exp(-best_d / scale))
    return best_sid, sim


# ---------------------------------------------------------------------------
# 2. UPC sequence overlap (Jaccard set + n-gram)
# ---------------------------------------------------------------------------

def _upc_set(seq: Sequence[int]) -> set[int]:
    return set(int(c) for c in seq)


def _upc_ngrams(seq: Sequence[int], n: int) -> set[tuple]:
    return set(tuple(int(c) for c in seq[i:i + n])
               for i in range(len(seq) - n + 1))


def upc_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    A, B = _upc_set(a), _upc_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def upc_ngram_overlap(a: Sequence[int], b: Sequence[int],
                       n: int = 3) -> float:
    A, B = _upc_ngrams(a, n), _upc_ngrams(b, n)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, min(len(A), len(B)))


def upc_max_ngram_overlap(out_seq: Sequence[int],
                           train_seqs: dict[str, Sequence[int]],
                           ns: Sequence[int] = (2, 3, 4)) -> dict[str, float]:
    """Max overlap over the train pool for each n. Returns per-n max plus
    a combined "upc_ngram_overlap" = mean over the requested n's."""
    out: dict[str, float] = {}
    for n in ns:
        best = 0.0
        for tseq in train_seqs.values():
            if len(tseq) < n:
                continue
            v = upc_ngram_overlap(out_seq, tseq, n)
            if v > best:
                best = v
        out[f"upc_ngram_max_n{n}"] = float(best)
    out["upc_ngram_overlap"] = float(np.mean([out[f"upc_ngram_max_n{n}"]
                                               for n in ns]))
    # Token-set Jaccard over the entire train pool (cheap).
    out["upc_jaccard_max"] = max((upc_jaccard(out_seq, t)
                                   for t in train_seqs.values()),
                                  default=0.0)
    return out


# ---------------------------------------------------------------------------
# 3. Frame-source concentration
# ---------------------------------------------------------------------------

def frame_source_concentration(source_trace: Optional[Sequence[Optional[str]]]
                                ) -> float:
    """Given a per-frame source-clip-id trace (None for generated frames),
    return the dominant source's fractional share. Whole-clip retrieval
    yields 1.0; pure generation yields 0.0; PG-RAST assembly yields
    something in between depending on per-source share."""
    if not source_trace:
        return 0.0
    counts: dict[str, int] = {}
    n_known = 0
    for s in source_trace:
        if s is None:
            continue
        counts[s] = counts.get(s, 0) + 1
        n_known += 1
    if n_known == 0:
        return 0.0
    return max(counts.values()) / len(source_trace)


# ---------------------------------------------------------------------------
# 4. Caption / gloss text NN similarity (TF-IDF char-bigram)
# ---------------------------------------------------------------------------

def _char_ngrams(s: str, n_min: int = 1, n_max: int = 3) -> list[str]:
    s = "".join(s.split())
    return [s[i:i + n] for n in range(n_min, n_max + 1)
            for i in range(len(s) - n + 1)]


def _build_tfidf_index(texts: Sequence[str]):
    from collections import Counter
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


def _tfidf_query(text: str, vocab, idf):
    from collections import Counter
    d = Counter(_char_ngrams(text))
    q = np.zeros(len(vocab), dtype=np.float32)
    for tok, c in d.items():
        if tok in vocab:
            q[vocab[tok]] = c * idf[vocab[tok]]
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q


def text_nn_similarity(query: str, vocab, idf, M: np.ndarray) -> float:
    """Max cosine similarity over the train pool, in [0, 1]."""
    q = _tfidf_query(query, vocab, idf)
    sims = M @ q
    return float(sims.max()) if sims.size > 0 else 0.0


def gloss_ngram_overlap(out_glosses: Sequence[str],
                         train_glosses: Iterable[Sequence[str]],
                         n: int = 3) -> float:
    if len(out_glosses) < n:
        return 0.0
    out_ngrams = set(tuple(out_glosses[i:i + n])
                      for i in range(len(out_glosses) - n + 1))
    if not out_ngrams:
        return 0.0
    best = 0
    for tg in train_glosses:
        if len(tg) < n:
            continue
        tg_ngrams = set(tuple(tg[i:i + n])
                         for i in range(len(tg) - n + 1))
        ov = len(out_ngrams & tg_ngrams)
        if ov > best:
            best = ov
    return best / max(1, len(out_ngrams))


# ---------------------------------------------------------------------------
# 5. Composite Copy-Leakage Index
# ---------------------------------------------------------------------------

@dataclass
class CLISignals:
    pose_nn_similarity: float = 0.0
    pose_nn_source_id: Optional[str] = None
    upc_ngram_overlap: float = 0.0
    upc_jaccard_max: float = 0.0
    upc_ngram_max_n2: float = 0.0
    upc_ngram_max_n3: float = 0.0
    upc_ngram_max_n4: float = 0.0
    frame_source_concentration: float = 0.0
    text_nn_similarity: float = 0.0
    text_nn_source_id: Optional[str] = None
    gloss_ngram_overlap: float = 0.0


@dataclass
class CLIWeights:
    pose: float = 0.30
    upc: float = 0.30
    frame_source: float = 0.25
    text: float = 0.15

    def normalize(self) -> "CLIWeights":
        s = self.pose + self.upc + self.frame_source + self.text
        return CLIWeights(self.pose / s, self.upc / s,
                           self.frame_source / s, self.text / s)


def cli_score(s: CLISignals, w: CLIWeights = CLIWeights()) -> float:
    w = w.normalize()
    return float(w.pose * s.pose_nn_similarity
                 + w.upc * s.upc_ngram_overlap
                 + w.frame_source * s.frame_source_concentration
                 + w.text * s.text_nn_similarity)


# ---------------------------------------------------------------------------
# 6. Driver: run the audit on an on-disk MSKA-format pickle
# ---------------------------------------------------------------------------

@dataclass
class CorpusSummary:
    n_clips: int
    mean_cli: float
    median_cli: float
    p90_cli: float
    pct_high_cli: float          # % with cli > 0.80
    pct_high_frame_source: float # % with frame_source > 0.50
    realized_retrieval_rate: float = 0.0
    requested_budget: float = 0.0
    extras: dict = field(default_factory=dict)


def score_pickle_against_train(
    out_pickle: Path,
    train_pose_pool: dict[str, np.ndarray],
    train_texts: dict[str, str],
    train_upc: Optional[dict[str, Sequence[int]]] = None,
    train_glosses: Optional[dict[str, Sequence[str]]] = None,
    out_upc: Optional[dict[str, Sequence[int]]] = None,
    source_trace: Optional[dict[str, Sequence[Optional[str]]]] = None,
    weights: CLIWeights = CLIWeights(),
    pose_subsample: int = 256,
) -> tuple[list[dict], CorpusSummary]:
    """Walk an MSKA pickle (sid -> {keypoint, text, gloss, ...}) and compute
    per-sample provenance signals + a corpus-level summary.

    `pose_subsample` caps the train pose pool used for nearest-neighbor pose
    similarity (the full pool can be ~18k clips on CSL; 256 is enough to
    rank the top match within an order-of-magnitude scale)."""
    with out_pickle.open("rb") as f:
        out_data = pickle.load(f)

    rng = np.random.RandomState(0)
    train_ids_pose = list(train_pose_pool.keys())
    if pose_subsample and len(train_ids_pose) > pose_subsample:
        sub_ids = list(rng.choice(train_ids_pose, size=pose_subsample,
                                    replace=False))
        train_pool_sub = {sid: train_pose_pool[sid] for sid in sub_ids}
    else:
        train_pool_sub = dict(train_pose_pool)

    train_id_list = list(train_texts.keys())
    train_text_list = [train_texts[s] for s in train_id_list]
    vocab, idf, M = _build_tfidf_index(train_text_list)

    rows: list[dict] = []
    cli_values: list[float] = []
    frame_conc_values: list[float] = []

    for sid, rec in out_data.items():
        kp = rec.get("keypoint") if isinstance(rec, dict) else rec
        text = rec.get("text", "") if isinstance(rec, dict) else ""
        glosses = rec.get("gloss", "").split() if isinstance(rec, dict) else []

        sig = CLISignals()
        # Pose NN
        nn_sid, nn_sim = pose_nn_similarity(kp, train_pool_sub)
        sig.pose_nn_similarity = nn_sim
        sig.pose_nn_source_id = nn_sid
        # UPC overlap
        if out_upc is not None and train_upc is not None and sid in out_upc:
            d = upc_max_ngram_overlap(out_upc[sid], train_upc)
            sig.upc_ngram_overlap = d["upc_ngram_overlap"]
            sig.upc_jaccard_max = d["upc_jaccard_max"]
            sig.upc_ngram_max_n2 = d["upc_ngram_max_n2"]
            sig.upc_ngram_max_n3 = d["upc_ngram_max_n3"]
            sig.upc_ngram_max_n4 = d["upc_ngram_max_n4"]
        # Frame-source concentration
        if source_trace is not None and sid in source_trace:
            sig.frame_source_concentration = frame_source_concentration(
                source_trace[sid])
        # Text NN
        if text:
            sig.text_nn_similarity = text_nn_similarity(text, vocab, idf, M)
            # Trace the matched source id by recomputing the argmax (cheap).
            q = _tfidf_query(text, vocab, idf)
            sims = M @ q
            sig.text_nn_source_id = train_id_list[int(np.argmax(sims))] \
                if sims.size > 0 else None
        # Gloss n-gram overlap
        if glosses and train_glosses is not None:
            sig.gloss_ngram_overlap = gloss_ngram_overlap(
                glosses, train_glosses.values())
        cli = cli_score(sig, weights)
        cli_values.append(cli)
        frame_conc_values.append(sig.frame_source_concentration)
        rows.append({
            "sid": sid,
            **asdict(sig),
            "cli": cli,
        })

    summary = CorpusSummary(
        n_clips=len(rows),
        mean_cli=float(np.mean(cli_values)),
        median_cli=float(np.median(cli_values)),
        p90_cli=float(np.percentile(cli_values, 90)),
        pct_high_cli=float(np.mean([v > 0.80 for v in cli_values])),
        pct_high_frame_source=float(np.mean([v > 0.50
                                              for v in frame_conc_values])),
    )
    return rows, summary


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_md(summary: CorpusSummary, path: Path,
                      header: str = "Provenance audit") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {header}",
        "",
        f"- Clips scored: **{summary.n_clips}**",
        f"- Mean CLI: **{summary.mean_cli:.4f}**",
        f"- Median CLI: **{summary.median_cli:.4f}**",
        f"- 90th-percentile CLI: **{summary.p90_cli:.4f}**",
        f"- Fraction CLI > 0.80: **{summary.pct_high_cli*100:.2f}%**",
        f"- Fraction frame-source-concentration > 0.50: "
        f"**{summary.pct_high_frame_source*100:.2f}%**",
        f"- Requested budget b: {summary.requested_budget:.2f}",
        f"- Realized retrieval rate: {summary.realized_retrieval_rate:.4f}",
        "",
        "CLI = 0.30 pose-NN + 0.30 UPC-ngram + 0.25 frame-source + 0.15 text-NN.",
    ]
    if summary.extras:
        lines += ["", "Extras:"]
        for k, v in summary.extras.items():
            lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n")


__all__ = [
    "fast_pose_distance",
    "pose_nn_similarity",
    "upc_jaccard",
    "upc_ngram_overlap",
    "upc_max_ngram_overlap",
    "frame_source_concentration",
    "text_nn_similarity",
    "gloss_ngram_overlap",
    "CLISignals", "CLIWeights", "cli_score",
    "score_pickle_against_train",
    "CorpusSummary",
    "write_csv", "write_summary_md",
]
