"""Build a dev-only CAC supervision trajectory for the clean source substrate.

The target is the exact change in development corpus BLEU-4 when a clean
whole-clip retrieval replaces the clean compositional fallback.  The saved
cache retains only the nine route-safe inputs plus the development WER-cost
target needed by the optional Pareto head.  No test artifact is accepted by
this command-line interface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.corpus_aware_controller import (
    ALL_FEAT_NAMES,
    FastCorpusBLEU,
    build_features_for_candidates,
)
from scripts.strict_route_only_cac import FEATURE_NAMES
from scripts.train_learned_confidence import FEATURES, HAND_WEIGHTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_text_predictions(path: Path) -> list[str]:
    values = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(values, (list, tuple)) or not all(isinstance(x, str) for x in values):
        raise TypeError(f"expected a sequence of text predictions: {path}")
    return list(values)


def per_clip_features(
    trace_path: Path,
    ids: list[str],
    retrieval_text: list[str],
    fallback_text: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(trace_path.read_text())
    if not isinstance(rows, list):
        raise TypeError("retrieval trace must be a list")
    by_id = {str(row.get("id", "")): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError("retrieval trace ids do not match development ids")
    rerank: list[list[float]] = []
    eligible: list[bool] = []
    for sid in ids:
        selected = by_id[sid].get("selected") or {}
        values = [float(selected.get(name, 0.0)) for name in FEATURES]
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite trace feature for {sid}")
        rerank.append(values)
        eligible.append(bool(selected))
    rerank_array = np.asarray(rerank, dtype=np.float64)
    hand_score = rerank_array @ np.asarray(HAND_WEIGHTS, dtype=np.float64)
    len_retrieval = np.asarray([len(text.split()) for text in retrieval_text], dtype=np.float64)
    len_fallback = np.asarray([len(text.split()) for text in fallback_text], dtype=np.float64)
    matrix = np.concatenate(
        [
            rerank_array,
            hand_score[:, None],
            len_retrieval[:, None],
            len_fallback[:, None],
            (len_retrieval - len_fallback)[:, None],
            ((len_retrieval + 1.0) / (len_fallback + 1.0))[:, None],
            np.asarray(eligible, dtype=np.float64)[:, None],
        ],
        axis=1,
    )
    return matrix, np.flatnonzero(eligible)


def stable_forward_greedy(
    fcb: FastCorpusBLEU,
    per_clip: np.ndarray,
    candidate_pool: np.ndarray,
    target_size: int,
    ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if target_size < 0 or target_size > len(candidate_pool):
        raise ValueError("target size exceeds the eligible development pool")
    selected = np.zeros(fcb.N, dtype=bool)
    remaining = {int(index) for index in candidate_pool}
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    history: list[dict] = []
    start = time.time()
    for step in range(1, target_size + 1):
        candidates = np.asarray(sorted(remaining, key=lambda index: ids[index]), dtype=np.int64)
        rows = build_features_for_candidates(per_clip, fcb, selected, candidates)
        marginal = fcb.marginal_for_candidates(selected, candidates)
        best_value = float(marginal.max())
        tied = candidates[np.isclose(marginal, best_value, rtol=0.0, atol=1e-14)]
        pick = min((int(index) for index in tied), key=lambda index: ids[index])
        features.append(rows)
        labels.append(marginal)
        selected[pick] = True
        remaining.remove(pick)
        history.append({
            "step": step,
            "pick_index": pick,
            "pick_id": ids[pick],
            "marginal_bleu4": best_value,
        })
        if step % 32 == 0 or step == target_size:
            _, bleu4 = fcb.bleu_for_mask(selected)
            print(
                f"step={step}/{target_size} bleu4={bleu4:.4f} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    return np.concatenate(features), np.concatenate(labels), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_gt", required=True)
    parser.add_argument("--retrieval_text", required=True)
    parser.add_argument("--fallback_text", required=True)
    parser.add_argument("--retrieval_trace", required=True)
    parser.add_argument("--budget", type=float, default=0.40)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history_out", required=True)
    parser.add_argument("--provenance_out", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.budget <= 1.0:
        raise ValueError("budget must lie in [0, 1]")

    gt_path = ROOT / args.dev_gt
    retrieval_text_path = ROOT / args.retrieval_text
    fallback_text_path = ROOT / args.fallback_text
    trace_path = ROOT / args.retrieval_trace
    gt = torch.load(gt_path, map_location="cpu", weights_only=False)
    if not isinstance(gt, dict):
        raise TypeError("development ground truth must be keyed by clip id")
    ids = [str(sid) for sid in gt]
    references = [str(gt[sid]["text"]) for sid in ids]
    retrieval_text = load_text_predictions(retrieval_text_path)
    fallback_text = load_text_predictions(fallback_text_path)
    if len(retrieval_text) != len(ids) or len(fallback_text) != len(ids):
        raise ValueError("development predictions do not align with ground truth")

    fcb = FastCorpusBLEU(retrieval_text, fallback_text, references)
    per_clip, candidates = per_clip_features(
        trace_path, ids, retrieval_text, fallback_text
    )
    target_size = int(round(args.budget * len(ids)))
    full_x, y_bleu, history = stable_forward_greedy(
        fcb, per_clip, candidates, target_size, ids
    )
    retained_names = list(FEATURE_NAMES) + ["wer_cost_clip"]
    retained_indices = [ALL_FEAT_NAMES.index(name) for name in retained_names]
    retained_x = full_x[:, retained_indices]
    if not np.isfinite(retained_x).all() or not np.isfinite(y_bleu).all():
        raise ValueError("non-finite trajectory values")

    out_path = ROOT / args.out
    history_path = ROOT / args.history_out
    provenance_path = ROOT / args.provenance_out
    for path in (out_path, history_path, provenance_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=retained_x,
        y=y_bleu,
        feature_names=np.asarray(retained_names),
    )
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    provenance = {
        "training_split": "development only",
        "test_artifacts_loaded": [],
        "n_clips": len(ids),
        "n_eligible": int(len(candidates)),
        "budget": args.budget,
        "target_size": target_size,
        "trajectory_rows": int(len(retained_x)),
        "retained_feature_names": retained_names,
        "bleu_target": "exact marginal development corpus BLEU-4",
        "wer_target": "development corpus WER edit-cost change",
        "inputs": {
            "dev_gt": {"path": str(gt_path), "sha256": sha256(gt_path)},
            "retrieval_text": {
                "path": str(retrieval_text_path), "sha256": sha256(retrieval_text_path)
            },
            "fallback_text": {
                "path": str(fallback_text_path), "sha256": sha256(fallback_text_path)
            },
            "retrieval_trace": {"path": str(trace_path), "sha256": sha256(trace_path)},
        },
        "trajectory": {"path": str(out_path), "sha256": sha256(out_path)},
        "history": {"path": str(history_path), "sha256": sha256(history_path)},
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "n_clips": len(ids), "n_eligible": len(candidates),
        "target_size": target_size, "trajectory_rows": len(retained_x),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
