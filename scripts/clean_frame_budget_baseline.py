"""Build a non-learned matched-frame-budget route.

The baseline ranks eligible whole-clip candidates by one declared safe trace
feature and applies the same exact replay-frame constraint as strict CAC. It
does not load a learned trajectory, decoded text, references, or evaluator
outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.strict_route_only_cac import (
    FEATURE_NAMES,
    _sha256,
    load_length_metadata,
    load_trace_features,
    select_replay_frames,
)


class _ColumnScore:
    def __init__(self, column: int):
        self.column = int(column)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features[:, self.column], dtype=np.float64)


def build_route(
    trace_path: Path,
    length_path: Path,
    budget: str,
    score_feature: str,
    route_name: str,
    mask_path: Path,
    record_path: Path,
) -> dict:
    if score_feature not in FEATURE_NAMES:
        raise ValueError(f"score feature must be one of {FEATURE_NAMES}")
    ids, features, eligible = load_trace_features(trace_path)
    replay_lengths, fallback_lengths = load_length_metadata(length_path, ids)
    model = _ColumnScore(FEATURE_NAMES.index(score_feature))
    mask, budget_info = select_replay_frames(
        model,
        ids,
        features,
        eligible,
        replay_lengths,
        fallback_lengths,
        budget,
    )
    scores = model.predict(features)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mask_path,
        ids=np.asarray(ids),
        sids=np.asarray(ids),
        in_S=mask,
        selected_feature_names=np.asarray([score_feature]),
        scores=scores,
        eligible=eligible,
    )
    selected_ids = sorted(ids[index] for index in np.flatnonzero(mask))
    record = {
        "route": route_name,
        "learned": False,
        "score_feature": score_feature,
        "budget": str(budget),
        "budget_unit": "replay_frames",
        "selection": budget_info,
        "n_ids": len(ids),
        "n_eligible": int(eligible.sum()),
        "selected_ids_sha256": hashlib.sha256(
            ("\n".join(selected_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "inputs": {
            "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
            "length_metadata": {
                "path": str(length_path),
                "sha256": _sha256(length_path),
            },
        },
        "mask": {"path": str(mask_path), "sha256": _sha256(mask_path)},
        "test_evaluator_outputs_loaded_at_route_time": [],
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "numpy_version": np.__version__,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--length_metadata", required=True)
    parser.add_argument("--budget", default="0.40")
    parser.add_argument("--score_feature", default="rerank_score")
    parser.add_argument("--route_name", default="clean_rerank_frame40")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--record", required=True)
    args = parser.parse_args()
    record = build_route(
        Path(args.trace),
        Path(args.length_metadata),
        args.budget,
        args.score_feature,
        args.route_name,
        Path(args.mask),
        Path(args.record),
    )
    print(json.dumps({
        "selected": record["selection"]["selected_clips"],
        "replay_frame_fraction": record["selection"]["realized_replay_frame_fraction"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
