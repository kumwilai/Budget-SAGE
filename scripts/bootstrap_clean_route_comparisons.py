"""Paired corpus-metric bootstrap for declared clean-route comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bootstrap_strict_cac import (  # noqa: E402
    load_predictions,
    paired_bootstrap,
    sha256,
)

SLRTP = ROOT / "external/SLRTP-Sign-Production-Evaluation"


def parse_assignment(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, path


def parse_pair(value: str) -> tuple[str, str, str]:
    label, separator, methods = value.partition("=")
    comparison, colon, baseline = methods.partition(":")
    if not separator or not colon or not label or not comparison or not baseline:
        raise argparse.ArgumentTypeError("expected LABEL=COMPARISON:BASELINE")
    return label, comparison, baseline


def run(
    split: str,
    method_specs: list[tuple[str, str]],
    pairs: list[tuple[str, str, str]],
    n_boot: int,
    seed: int,
    out_path: Path,
) -> dict:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if len({name for name, _ in method_specs}) != len(method_specs):
        raise ValueError("method names must be unique")
    method_paths = {name: ROOT / path for name, path in method_specs}
    gt_path = (
        SLRTP
        / "pretrained/SLRTP-Sign-Production-Evaluation-Data/data"
        / f"{split}.pt"
    )
    gt = torch.load(gt_path, map_location="cpu", weights_only=False)
    if not isinstance(gt, dict):
        raise TypeError(f"expected keyed ground truth: {gt_path}")
    ids = [str(sid) for sid in gt]
    references = [str(gt[sid]["text"]) for sid in ids]
    hypotheses = {
        name: load_predictions(path) for name, path in method_paths.items()
    }
    result = paired_bootstrap(
        hypotheses,
        references,
        pairs=tuple(pairs),
        n_boot=n_boot,
        seed=seed,
    )
    result["meta"].update({
        "split": split,
        "ids_sha256": hashlib.sha256(
            ("\n".join(ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "ground_truth": str(gt_path),
        "ground_truth_sha256": sha256(gt_path),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in method_paths.items()
        },
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "paired_bootstrap_implementation": str(
            ROOT / "scripts/bootstrap_strict_cac.py"
        ),
        "paired_bootstrap_implementation_sha256": sha256(
            ROOT / "scripts/bootstrap_strict_cac.py"
        ),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--method", action="append", type=parse_assignment, required=True)
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=30373)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(
        args.split,
        args.method,
        args.pair,
        args.n_boot,
        args.seed,
        ROOT / args.out,
    )
    for name, row in result["comparisons"].items():
        delta = row["point_delta"]
        interval = row["bootstrap_delta"]
        print(
            f"{name}: delta BLEU-4={delta['bleu4']:+.3f} "
            f"CI={interval['bleu4']['ci95']}; "
            f"delta WER={delta['corpus_wer']:+.3f} "
            f"CI={interval['corpus_wer']['ci95']}"
        )


if __name__ == "__main__":
    main()
