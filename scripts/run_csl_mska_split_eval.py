"""Supported test-only entry point for the sealed CSL MSKA evaluator.

The evaluated implementation is retained byte-for-byte in
``run_csl_mska_split_eval_57b74fa1.py``.  This wrapper rejects development
runs before any output is created and delegates test runs only after checking
that immutable implementation's complete SHA-256 digest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).with_name("run_csl_mska_split_eval_57b74fa1.py")
SNAPSHOT_SHA256 = "57b74fa11242cc3b11fa2c190eeceb9da7a3e91db1b386305c67e0aa862bf7b9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_snapshot():
    if not SNAPSHOT.is_file() or sha256(SNAPSHOT) != SNAPSHOT_SHA256:
        raise RuntimeError("sealed CSL evaluator snapshot is missing or changed")
    spec = importlib.util.spec_from_file_location(
        "sealed_csl_mska_eval_57b74fa1", SNAPSHOT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed CSL evaluator snapshot")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FROZEN = _load_snapshot()
DEFAULT_CONFIG = _FROZEN.DEFAULT_CONFIG
DEFAULT_CHECKPOINT = _FROZEN.DEFAULT_CHECKPOINT
MSKA_DATA = _FROZEN.MSKA_DATA
cpu_cuda_compat = _FROZEN.cpu_cuda_compat
evaluate_ids = _FROZEN.evaluate_ids


def run_one(
    input_pickle: str | Path,
    split: str,
    out_dir: str | Path,
    config: str | Path = DEFAULT_CONFIG,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    input_sha256: str | None = None,
    config_sha256: str | None = None,
    checkpoint_sha256: str | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    beam_size: int = 5,
    evaluator: Callable[..., dict] = evaluate_ids,
) -> dict[str, Any]:
    if split != "test":
        raise ValueError("supported CSL evaluator wrapper is test-only")
    # These assignments support confined synthetic tests; production values
    # remain the repository root and the frozen MSKA data directory.
    _FROZEN.ROOT = ROOT
    _FROZEN.MSKA_DATA = MSKA_DATA
    return _FROZEN.run_one(
        input_pickle,
        "test",
        out_dir,
        config,
        checkpoint,
        input_sha256,
        config_sha256,
        checkpoint_sha256,
        batch_size,
        num_workers,
        beam_size,
        evaluator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the hash-pinned CSL MSKA evaluator on test inputs only."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", required=True, choices=("test",))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args()
    result = run_one(
        args.input,
        args.split,
        args.out_dir,
        args.config,
        args.checkpoint,
        args.input_sha256,
        args.config_sha256,
        args.checkpoint_sha256,
        args.batch_size,
        args.num_workers,
        args.beam_size,
    )
    print(json.dumps({key: value for key, value in result.items()
                      if key != "per_id"}, indent=2))


if __name__ == "__main__":
    main()
