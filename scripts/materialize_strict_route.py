"""Materialize a strict route mask into an evaluator-ready pose dictionary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(mask_path: Path, retrieval_path: Path, fallback_path: Path) -> tuple[dict, dict]:
    mask_data = np.load(mask_path, allow_pickle=False)
    required = {"ids", "in_S"}
    if not required.issubset(mask_data.files):
        raise ValueError(f"route mask lacks {sorted(required.difference(mask_data.files))}")
    ids = [str(value) for value in mask_data["ids"].tolist()]
    mask = np.asarray(mask_data["in_S"], dtype=bool)
    if len(ids) != len(mask) or len(ids) != len(set(ids)):
        raise ValueError("route ids and mask are invalid")
    retrieval = torch.load(retrieval_path, map_location="cpu", weights_only=True)
    fallback = torch.load(fallback_path, map_location="cpu", weights_only=True)
    expected = set(ids)
    if set(retrieval) != expected or set(fallback) != expected:
        raise ValueError("route branches do not have exactly the masked ids")

    output = {}
    replay_frames = 0
    local_frames = 0
    for index, sid in enumerate(ids):
        value = retrieval[sid] if mask[index] else fallback[sid]
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[1:] != (178, 3):
            raise ValueError(f"invalid pose tensor for {sid}: {getattr(value, 'shape', None)}")
        if value.shape[0] < 1 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"empty or non-finite pose tensor for {sid}")
        output[sid] = value
        if mask[index]:
            replay_frames += int(value.shape[0])
        else:
            local_frames += int(value.shape[0])
    total_frames = replay_frames + local_frames
    summary = {
        "n_clips": len(ids),
        "whole_clip_replay_clips": int(mask.sum()),
        "compositional_fallback_clips": int((~mask).sum()),
        "whole_clip_replay_clip_fraction": float(mask.mean()),
        "whole_clip_replay_frames": replay_frames,
        "compositional_fallback_frames": local_frames,
        "total_frames": total_frames,
        "whole_clip_replay_frame_fraction": replay_frames / max(1, total_frames),
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provenance_out", required=True)
    args = parser.parse_args()
    mask_path = ROOT / args.mask
    retrieval_path = ROOT / args.retrieval
    fallback_path = ROOT / args.fallback
    out_path = ROOT / args.out
    provenance_path = ROOT / args.provenance_out
    output, summary = materialize(mask_path, retrieval_path, fallback_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    provenance = {
        **summary,
        "inputs": {
            "mask": {"path": str(mask_path), "sha256": sha256(mask_path)},
            "retrieval": {"path": str(retrieval_path), "sha256": sha256(retrieval_path)},
            "fallback": {"path": str(fallback_path), "sha256": sha256(fallback_path)},
        },
        "output": {"path": str(out_path), "sha256": sha256(out_path)},
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
