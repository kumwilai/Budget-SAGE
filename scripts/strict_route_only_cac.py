"""Strict evaluator-output-independent CAC routing.

The route is deliberately isolated from torch, references, decoded text, and
FastCorpusBLEU. A GBDT is fit to the retained dev trajectory, but deployment
is static: scores are computed once from trace metadata and eligible IDs are
ranked with an explicit deterministic ID tie-break. Admission then obeys either
a clip-count budget or an exact replay-frame-mass constraint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
FEATURE_NAMES = (
    "rerank_score", "coverage", "slot_score", "f1", "length_penalty", "score",
    "hand_score", "has_retrieval", "fill_progress",
)
TRACE_FEATURES = FEATURE_NAMES[:6]
FORBIDDEN_MARKERS = (
    "len_retr", "len_fall", "len_diff", "len_ratio", "total_mix_len",
    "brevity_arg", "mean_len_retr_S", "mean_len_fall_NS", "cur_bleu4",
    "cur_wer", "delta_len_clip", "rel_delta_len", "rel_brevity_impact",
    "wer_cost_clip",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    z = np.load(path, allow_pickle=False)
    required = {"X", "y", "feature_names"}
    missing = required.difference(z.files)
    if missing:
        raise ValueError(f"trajectory missing {sorted(missing)}: {path}")
    names = [str(x) for x in z["feature_names"].tolist()]
    if len(names) != z["X"].shape[1]:
        raise ValueError("trajectory feature_names do not match X columns")
    if len(set(names)) != len(names):
        raise ValueError("trajectory feature names are not unique")
    if any(n not in names for n in FEATURE_NAMES):
        raise ValueError("retained trajectory cannot support exact nine-feature mapping")
    if any(n in names for n in FORBIDDEN_MARKERS):
        # The retained cache is allowed to contain forbidden columns; only the
        # explicit allow-list may reach the estimator.
        pass
    X = np.asarray(z["X"], dtype=np.float64)
    y = np.asarray(z["y"], dtype=np.float64).reshape(-1)
    if X.shape[0] != y.shape[0] or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("invalid retained trajectory values")
    return X, y, names


def _allow_columns(X: np.ndarray, names: list[str]) -> np.ndarray:
    idx = [names.index(n) for n in FEATURE_NAMES]
    return X[:, idx]


def load_trace_features(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read only IDs and selected retrieval-trace fields; never text fields."""
    rows = json.loads(path.read_text())
    ids: list[str] = []
    rows_x: list[list[float]] = []
    eligible: list[bool] = []
    seen: set[str] = set()
    hand = np.array([1.0, 0.10, 0.05, 0.03, -0.05, 0.01], dtype=np.float64)
    for row in rows:
        sid = row.get("id")
        if not isinstance(sid, str) or sid in seen:
            raise ValueError(f"invalid or duplicate trace id in {path}: {sid!r}")
        seen.add(sid)
        sel = row.get("selected")
        ok = isinstance(sel, dict) and bool(sel)
        vals = [float(sel.get(k, 0.0)) if ok else 0.0 for k in TRACE_FEATURES]
        if not np.isfinite(vals).all():
            raise ValueError(f"non-finite retrieval trace feature for {sid}")
        ids.append(sid)
        rows_x.append(vals + [float(np.dot(vals, hand)), float(ok), 0.0])
        eligible.append(ok)
    return ids, np.asarray(rows_x, dtype=np.float64), np.asarray(eligible, dtype=bool)


def fit_route_model(trajectory: Path, **kwargs) -> GradientBoostingRegressor:
    X, y, names = _trajectory(trajectory)
    Xa = _allow_columns(X, names)
    return GradientBoostingRegressor(
        n_estimators=int(kwargs.get("n_estimators", 400)),
        max_depth=int(kwargs.get("max_depth", 4)),
        learning_rate=float(kwargs.get("learning_rate", 0.05)),
        subsample=0.85, random_state=0, min_samples_leaf=20,
    ).fit(Xa, y)


def fit_pareto_model(trajectory: Path, alpha_wer: float, **kwargs):
    """Fit dev-supervised BLEU and WER-cost heads on the same safe inputs.

    ``wer_cost_clip`` is used only as a development-set training target.  It
    never enters the feature matrix and is never needed when the frozen model
    ranks an evaluation split.
    """
    X, y_bleu, names = _trajectory(trajectory)
    Xa = _allow_columns(X, names)
    if "wer_cost_clip" not in names:
        raise ValueError("trajectory lacks the declared dev WER-cost target")
    y_wer = X[:, names.index("wer_cost_clip")]
    common = dict(
        n_estimators=int(kwargs.get("n_estimators", 400)),
        max_depth=int(kwargs.get("max_depth", 4)),
        learning_rate=float(kwargs.get("learning_rate", 0.05)),
        subsample=0.85,
        random_state=0,
        min_samples_leaf=20,
    )
    bleu_model = GradientBoostingRegressor(**common).fit(Xa, y_bleu)
    wer_model = GradientBoostingRegressor(**common).fit(Xa, y_wer)

    class _ParetoModel:
        def predict(self, features: np.ndarray) -> np.ndarray:
            return bleu_model.predict(features) - float(alpha_wer) * wer_model.predict(features)

    return _ParetoModel()


def select_static_topk(model, ids: list[str], features: np.ndarray,
                       eligible: np.ndarray, k: int) -> np.ndarray:
    """Return a full-length bool mask; ties resolve by lexicographically low ID."""
    if features.shape != (len(ids), len(FEATURE_NAMES)):
        raise ValueError("feature matrix is not exactly the nine allowed columns")
    if len(eligible) != len(ids) or k < 0:
        raise ValueError("invalid candidate selection inputs")
    cand = np.flatnonzero(eligible)
    k = min(int(k), len(cand))
    scores = np.asarray(model.predict(features), dtype=np.float64)
    order = sorted(cand.tolist(), key=lambda i: (-float(scores[i]), ids[i]))
    out = np.zeros(len(ids), dtype=np.int8)
    out[order[:k]] = 1
    return out


def load_length_metadata(path: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(path.read_text())
    meta_ids = meta.get("ids")
    if (
        not isinstance(meta_ids, list)
        or not all(isinstance(sid, str) for sid in meta_ids)
        or len(meta_ids) != len(set(meta_ids))
        or meta_ids != sorted(meta_ids)
        or set(meta_ids) != set(ids)
        or set(meta.get("r", {})) != set(meta_ids)
        or set(meta.get("f", {})) != set(meta_ids)
    ):
        raise ValueError("length metadata IDs do not match trace IDs")
    try:
        r = np.asarray([int(meta["r"][sid]) for sid in ids], dtype=np.int64)
        f = np.asarray([int(meta["f"][sid]) for sid in ids], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("invalid length metadata") from e
    if np.any(r <= 0) or np.any(f <= 0):
        raise ValueError("length metadata must be positive")
    return r, f


def select_replay_frames(
    model,
    ids: list[str],
    features: np.ndarray,
    eligible: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    beta: str | float,
) -> tuple[np.ndarray, dict]:
    """Rank once, admitting candidates satisfying exact Fraction capacity."""
    if features.shape != (len(ids), len(FEATURE_NAMES)):
        raise ValueError("feature matrix is not exactly the nine allowed columns")
    if len(eligible) != len(ids) or r.shape != (len(ids),) or f.shape != (len(ids),):
        raise ValueError("replay-frame inputs do not align")
    if np.any(r <= 0) or np.any(f <= 0):
        raise ValueError("candidate lengths must be positive")
    frac = Fraction(Decimal(str(beta)))
    p, q = frac.numerator, frac.denominator
    if not (0 <= p <= q):
        raise ValueError("replay-frame budget must lie in [0,1]")
    scores = np.asarray(model.predict(features), dtype=np.float64)
    if scores.shape != (len(ids),) or not np.isfinite(scores).all():
        raise ValueError("model scores must be finite and aligned")
    order = sorted(np.flatnonzero(eligible).tolist(), key=lambda i: (-float(scores[i]), ids[i]))
    costs = (q - p) * r + p * f
    capacity = p * int(f.sum())
    used = 0
    mask = np.zeros(len(ids), dtype=np.int8)
    for i in order:
        if used + int(costs[i]) <= capacity:
            mask[i] = 1
            used += int(costs[i])
    selected = mask.astype(bool)
    replay_frames = int(r[selected].sum())
    emitted_frames = int(np.where(selected, r, f).sum())
    return mask, {
        "budget_fraction": {"numerator": p, "denominator": q},
        "scaled_capacity": int(capacity),
        "scaled_used_cost": int(used),
        "scaled_unused_capacity": int(capacity - used),
        "selected_clips": int(mask.sum()),
        "replay_frames": replay_frames,
        "emitted_frames": emitted_frames,
        "realized_replay_frame_fraction": replay_frames / max(1, emitted_frames),
        "selection_algorithm": "one-pass score-ranked feasible admission",
        "optimality_claimed": False,
    }


def run(args: argparse.Namespace) -> dict:
    if not 0.0 <= float(args.budget) <= 1.0:
        raise ValueError(f"budget must lie in [0, 1], got {args.budget}")
    traj = Path(args.trajectory)
    if float(args.alpha_wer) == 0.0:
        model = fit_route_model(traj, n_estimators=args.n_estimators,
                                max_depth=args.max_depth, learning_rate=args.learning_rate)
    else:
        model = fit_pareto_model(
            traj,
            alpha_wer=float(args.alpha_wer),
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
    if getattr(args, "dev_only", False) and getattr(args, "test_only", False):
        raise ValueError("dev_only and test_only are mutually exclusive")
    split_out = {}
    if getattr(args, "test_only", False):
        split_inputs = [("test", args.test_trace)]
    else:
        split_inputs = [("dev", args.dev_trace)]
        if not getattr(args, "dev_only", False):
            split_inputs.append(("test", args.test_trace))
    for split, trace_arg in split_inputs:
        trace = Path(trace_arg)
        ids, feats, eligible = load_trace_features(trace)
        unit = getattr(args, "budget_unit", "clips")
        if unit == "replay_frames":
            metadata_arg = getattr(args, f"{split}_length_metadata", None)
            if not metadata_arg:
                raise ValueError(f"missing {split} length metadata for replay_frames")
            r, f = load_length_metadata(Path(metadata_arg), ids)
            mask, budget_info = select_replay_frames(model, ids, feats, eligible, r, f, str(args.budget))
            length_metadata = {
                "path": str(metadata_arg),
                "sha256": _sha256(Path(metadata_arg)),
            }
            k = int(mask.sum())
        elif unit == "clips":
            k = int(round(float(args.budget) * len(ids)))
            mask = select_static_topk(model, ids, feats, eligible, k)
            budget_info = {"capacity": k, "used": int(mask.sum()), "unused_capacity": k - int(mask.sum()),
                           "selected_clips": int(mask.sum()),
                           "realized_replay_frame_fraction": None}
            length_metadata = None
        else:
            raise ValueError("budget_unit must be clips or replay_frames")
        npz_path = Path(args.out_dir) / f"{args.route_name}_{split}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, ids=np.asarray(ids), sids=np.asarray(ids), in_S=mask,
                            selected_feature_names=np.asarray(FEATURE_NAMES),
                            scores=model.predict(feats), eligible=eligible)
        selected_ids = sorted(ids[i] for i in np.flatnonzero(mask))
        selected_ids_sha256 = hashlib.sha256(
            ("\n".join(selected_ids) + "\n").encode("utf-8")
        ).hexdigest()
        split_out[split] = {"trace": str(trace), "trace_sha256": _sha256(trace),
                            "npz": str(npz_path), "npz_sha256": _sha256(npz_path),
                            "selected_ids_sha256": selected_ids_sha256,
                            "n_ids": len(ids), "n_eligible": int(eligible.sum()),
                            "k": k, "selected": int(mask.sum()), "budget": budget_info,
                            "length_metadata": length_metadata}
    out = {"route": str(args.route_name), "budget": str(args.budget),
           "budget_unit": getattr(args, "budget_unit", "clips"),
           "alpha_wer": float(args.alpha_wer),
           "trajectory": str(traj), "trajectory_sha256": _sha256(traj),
           "implementation": str(Path(__file__).resolve()),
           "implementation_sha256": _sha256(Path(__file__).resolve()),
           "estimator": {
               "class": "sklearn.ensemble.GradientBoostingRegressor",
               "n_estimators": int(args.n_estimators),
               "max_depth": int(args.max_depth),
               "learning_rate": float(args.learning_rate),
               "subsample": 0.85,
               "random_state": 0,
               "min_samples_leaf": 20,
           },
           "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
           "selected_feature_names": list(FEATURE_NAMES),
           "development_supervision": {
               "bleu_target": "marginal corpus BLEU-4 in the retained dev trajectory",
               "wer_target": ("per-candidate corpus WER edit cost in the retained dev trajectory"
                              if float(args.alpha_wer) != 0.0 else "not used"),
               "test_evaluator_input": "none",
           },
           "test_evaluator_outputs_loaded_at_route_time": [],
           "route_time_inputs": (
               "candidate trace fields and candidate-length metadata only; "
               "development targets are confined to model fitting"
           ),
           "splits": split_out}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="outputs/learned_confidence/cac_dev_trajectory_b40.npz")
    ap.add_argument("--dev_trace", default="external/SLRTP-Sign-Production-Evaluation/results/topk_semantic_hybrid_dev_trace.json")
    ap.add_argument("--test_trace", default="external/SLRTP-Sign-Production-Evaluation/results/topk_semantic_hybrid_test_trace.json")
    ap.add_argument(
        "--dev_only",
        action="store_true",
        help="Fit and select development routes without reading or writing a test artifact.",
    )
    ap.add_argument(
        "--test_only",
        action="store_true",
        help="Fit from the frozen trajectory and select only the test route.",
    )
    ap.add_argument("--budget", default="0.40")
    ap.add_argument("--budget_unit", choices=("clips", "replay_frames"), default="clips")
    ap.add_argument("--dev_length_metadata", default=None)
    ap.add_argument("--test_length_metadata", default=None)
    ap.add_argument("--alpha_wer", type=float, default=0.0)
    ap.add_argument("--route_name", default="strict_route_only_cac")
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--out", default="outputs/revision/strict_route_only_cac.json")
    ap.add_argument("--out_dir", default="outputs/revision")
    args = ap.parse_args()
    out = run(args)
    print(json.dumps({s: v["selected"] for s, v in out["splits"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
