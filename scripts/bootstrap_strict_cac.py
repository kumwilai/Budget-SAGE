"""Paired bootstrap uncertainty for the strict CAC revision routes.

The script evaluates the exact corpus BLEU and corpus WER functionals used by
the retained SLRTP evaluator.  It resamples aligned clips once per replicate,
so every method comparison is paired and uses the same bootstrap draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SLRTP = ROOT / "external/SLRTP-Sign-Production-Evaluation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SLRTP))

from external_metrics.sacrebleu import raw_corpus_bleu  # noqa: E402
from scripts.corpus_aware_controller import (  # noqa: E402
    bleu_from_stats,
    per_hyp_bleu_stats,
    per_hyp_wer_stats,
)


DEFAULT_METHODS = {
    "reference": SLRTP / "results/phase58_budget040_test_text_preds.pt",
    "strict_bleu": SLRTP / "results/revision_strict_route_only_cac_test_text_preds.pt",
    "strict_pareto": SLRTP / "results/revision_strict_route_only_cac_pareto_test_text_preds.pt",
}

DEFAULT_PAIRS = (
    ("strict_bleu_vs_reference", "strict_bleu", "reference"),
    ("strict_pareto_vs_reference", "strict_pareto", "reference"),
    ("strict_pareto_vs_strict_bleu", "strict_pareto", "strict_bleu"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(path: Path) -> list[str]:
    values = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"predictions must be a list or tuple: {path}")
    if not all(isinstance(value, str) for value in values):
        raise TypeError(f"predictions contain a non-string value: {path}")
    return list(values)


def sufficient_stats(hypotheses: list[str], references: list[str]) -> dict[str, np.ndarray]:
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references are not aligned")
    bleu_rows = [per_hyp_bleu_stats(hyp, ref) for hyp, ref in zip(hypotheses, references)]
    wer_rows = [per_hyp_wer_stats(hyp, ref) for hyp, ref in zip(hypotheses, references)]
    return {
        "correct": np.stack([row[0] for row in bleu_rows]),
        "total": np.stack([row[1] for row in bleu_rows]),
        "sys_len": np.asarray([row[2] for row in bleu_rows], dtype=np.int64),
        "ref_len_bleu": np.asarray([row[3] for row in bleu_rows], dtype=np.int64),
        "edits": np.asarray([row[0] for row in wer_rows], dtype=np.int64),
        "ref_len_wer": np.asarray([row[1] for row in wer_rows], dtype=np.int64),
    }


def metrics_from_stats(stats: dict[str, np.ndarray], indices: np.ndarray | None = None) -> np.ndarray:
    """Return rows ``[BLEU-1, BLEU-4, corpus WER]``.

    With no indices, one row describes the original corpus.  With an integer
    matrix of shape ``[B, N]``, one row is returned for each paired bootstrap
    replicate.
    """
    if indices is None:
        indices = np.arange(len(stats["sys_len"]), dtype=np.int64)[None, :]
    if indices.ndim != 2:
        raise ValueError("indices must have shape [replicates, clips]")
    correct = stats["correct"][indices].sum(axis=1)
    total = stats["total"][indices].sum(axis=1)
    sys_len = stats["sys_len"][indices].sum(axis=1)
    ref_len_bleu = stats["ref_len_bleu"][indices].sum(axis=1)
    edits = stats["edits"][indices].sum(axis=1)
    ref_len_wer = stats["ref_len_wer"][indices].sum(axis=1)
    out = np.empty((len(indices), 3), dtype=np.float64)
    for row in range(len(indices)):
        bleu1, bleu4 = bleu_from_stats(
            correct[row], total[row], int(sys_len[row]), int(ref_len_bleu[row])
        )
        out[row, 0] = bleu1
        out[row, 1] = bleu4
    out[:, 2] = 100.0 * edits / np.maximum(1, ref_len_wer)
    return out


def interval(draws: np.ndarray) -> dict[str, object]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    n = len(draws)
    p_le_zero = (int(np.count_nonzero(draws <= 0.0)) + 1) / (n + 1)
    p_ge_zero = (int(np.count_nonzero(draws >= 0.0)) + 1) / (n + 1)
    return {
        "mean": float(draws.mean()),
        "ci95": [float(lo), float(hi)],
        "fraction_positive": float(np.mean(draws > 0.0)),
        "fraction_negative": float(np.mean(draws < 0.0)),
        "two_sided_p": float(min(1.0, 2.0 * min(p_le_zero, p_ge_zero))),
    }


def paired_bootstrap(
    hypotheses: dict[str, list[str]],
    references: list[str],
    pairs: tuple[tuple[str, str, str], ...] = DEFAULT_PAIRS,
    n_boot: int = 2000,
    seed: int = 30373,
) -> dict[str, object]:
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    n = len(references)
    if n < 1 or any(len(values) != n for values in hypotheses.values()):
        raise ValueError("all methods must align one-to-one with nonempty references")
    for _, comparison, baseline in pairs:
        if comparison not in hypotheses or baseline not in hypotheses:
            raise KeyError(f"unknown comparison pair: {comparison}, {baseline}")

    stats = {name: sufficient_stats(values, references) for name, values in hypotheses.items()}
    point = {name: metrics_from_stats(value)[0] for name, value in stats.items()}

    # This assertion guards against a drift between our additive sufficient
    # statistics and the exact raw SacreBLEU implementation used by SLRTP.
    for name, values in hypotheses.items():
        official = raw_corpus_bleu(sys_stream=values, ref_streams=[references]).scores
        if not np.allclose(point[name][:2], [official[0], official[3]], rtol=0.0, atol=1e-10):
            raise AssertionError(f"BLEU sufficient statistics disagree for {name}")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)
    draws = {name: metrics_from_stats(value, indices) for name, value in stats.items()}
    metric_names = ("bleu1", "bleu4", "corpus_wer")
    methods = {
        name: {metric: float(point[name][j]) for j, metric in enumerate(metric_names)}
        for name in hypotheses
    }
    comparisons: dict[str, object] = {}
    for pair_name, comparison, baseline in pairs:
        delta = draws[comparison] - draws[baseline]
        comparisons[pair_name] = {
            "comparison": comparison,
            "baseline": baseline,
            "orientation": "comparison minus baseline; lower corpus WER is better",
            "point_delta": {
                metric: float(point[comparison][j] - point[baseline][j])
                for j, metric in enumerate(metric_names)
            },
            "bootstrap_delta": {
                metric: interval(delta[:, j]) for j, metric in enumerate(metric_names)
            },
        }
    return {
        "meta": {"n": n, "n_boot": n_boot, "seed": seed, "paired": True},
        "methods": methods,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=30373)
    parser.add_argument("--out", default="outputs/revision/strict_cac_paired_bootstrap.json")
    args = parser.parse_args()

    gt_path = SLRTP / "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/test.pt"
    gt = torch.load(gt_path, map_location="cpu", weights_only=False)
    if not isinstance(gt, dict):
        raise TypeError(f"expected keyed SLRTP ground truth: {gt_path}")
    ids = list(gt)
    references = [gt[sid]["text"] for sid in ids]
    hypotheses = {name: load_predictions(path) for name, path in DEFAULT_METHODS.items()}
    result = paired_bootstrap(
        hypotheses, references, n_boot=args.n_boot, seed=args.seed
    )
    result["meta"].update({
        "split": "test",
        "ids_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "ground_truth": str(gt_path),
        "ground_truth_sha256": sha256(gt_path),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in DEFAULT_METHODS.items()
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    })
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, row in result["comparisons"].items():
        d = row["point_delta"]
        b = row["bootstrap_delta"]
        print(
            f"{name}: delta BLEU-4={d['bleu4']:+.3f} "
            f"CI={b['bleu4']['ci95']}; delta WER={d['corpus_wer']:+.3f} "
            f"CI={b['corpus_wer']['ci95']}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
