"""Build a conserved source ledger for a clean strict route.

Whole-clip route frames are charged to the selected training donor.  Frames in
the compositional branch are charged to their traced training segments.  A
cross-faded boundary carries fractional mass from both adjacent sources; the
weights for each emitted frame sum to one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def normalize_caption(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).lower()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compositional_ledger(
    segments: list[dict],
    output_length: int,
    blend_width: int,
) -> tuple[dict[str, float], list[dict]]:
    if not segments:
        return {"unknown_or_derived": float(output_length)}, []
    lengths = [int(segment["out_len"]) for segment in segments]
    if any(length < 1 for length in lengths) or sum(lengths) != output_length:
        raise ValueError("segment lengths do not conserve emitted frame count")
    mass: defaultdict[str, float] = defaultdict(float)
    for segment, length in zip(segments, lengths):
        mass[str(segment["source"])] += float(length)

    blends: list[dict] = []
    offset = 0
    for index in range(len(segments) - 1):
        previous = segments[index]
        current = segments[index + 1]
        previous_length = lengths[index]
        current_length = lengths[index + 1]
        width = min(int(blend_width), previous_length // 2, current_length // 2)
        if width > 0:
            alpha = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, width)))
            previous_source = str(previous["source"])
            current_source = str(current["source"])
            transferred = float(alpha.sum())
            mass[previous_source] -= transferred
            mass[current_source] += transferred
            for local, weight in enumerate(alpha.tolist()):
                frame_sources: defaultdict[str, float] = defaultdict(float)
                frame_sources[previous_source] += float(1.0 - weight)
                frame_sources[current_source] += float(weight)
                blends.append({
                    "frame": offset + previous_length - width + local,
                    "sources": dict(frame_sources),
                })
        offset += previous_length
    cleaned = {source: value for source, value in mass.items() if abs(value) > 1e-12}
    if any(value < -1e-9 for value in cleaned.values()):
        raise ValueError("negative source mass after boundary attribution")
    if not math.isclose(sum(cleaned.values()), output_length, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("compositional source mass is not conserved")
    for blend in blends:
        if not math.isclose(sum(blend["sources"].values()), 1.0, abs_tol=1e-12):
            raise ValueError("blend-frame source weights are not conserved")
    return cleaned, blends


def build_ledger(
    mask_path: Path,
    route_pose_path: Path,
    retrieval_trace_path: Path,
    fallback_trace_path: Path,
    train_manifest_path: Path,
    blend_width: int,
    require_no_exact_caption_sources: bool = False,
) -> dict:
    mask_data = np.load(mask_path, allow_pickle=False)
    ids = [str(value) for value in mask_data["ids"].tolist()]
    mask = np.asarray(mask_data["in_S"], dtype=bool)
    poses = torch.load(route_pose_path, map_location="cpu", weights_only=True)
    retrieval_rows = json.loads(retrieval_trace_path.read_text())
    fallback_rows = json.loads(fallback_trace_path.read_text())
    train_rows = json.loads(train_manifest_path.read_text())
    retrieval = {str(row["id"]): row for row in retrieval_rows}
    if isinstance(fallback_rows, list):
        fallback = {str(row["id"]): row for row in fallback_rows}
    else:
        fallback = {str(sid): row for sid, row in fallback_rows.items()}
    train = {str(row["id"]): row for row in train_rows}
    train_caption = {
        source: normalize_caption(row.get("text", ""))
        for source, row in train.items()
    }
    expected = set(ids)
    if len(ids) != len(mask) or set(poses) != expected:
        raise ValueError("route mask and pose ids do not align")
    if set(retrieval) != expected or set(fallback) != expected:
        raise ValueError("route traces do not cover exactly the emitted ids")

    clips: list[dict] = []
    source_total: defaultdict[str, float] = defaultdict(float)
    signer_total: defaultdict[str, float] = defaultdict(float)
    mode_frames: Counter[str] = Counter()
    for index, sid in enumerate(ids):
        length = int(poses[sid].shape[0])
        query_caption = normalize_caption(retrieval[sid].get("text", ""))
        if require_no_exact_caption_sources and not query_caption:
            raise ValueError(f"missing query text for exact-caption audit: {sid}")
        exact_sources = {
            source for source, caption in train_caption.items()
            if query_caption and caption == query_caption
        }
        if mask[index]:
            selected = retrieval[sid].get("selected") or {}
            donor = str(selected.get("id", ""))
            if donor not in train:
                raise ValueError(f"invalid whole-clip donor for {sid}: {donor}")
            if require_no_exact_caption_sources and donor in exact_sources:
                raise ValueError(f"exact-caption whole-clip donor for {sid}: {donor}")
            source_mass = {donor: float(length)}
            blends: list[dict] = []
            mode = "whole_clip_replay"
        else:
            row = fallback[sid]
            segments = list(row.get("segments", []))
            excluded = {str(value) for value in row.get("excluded_sources", [])}
            legacy_excluded = str(row.get("excluded_source", ""))
            if legacy_excluded:
                excluded.add(legacy_excluded)
            used_sources = {str(segment["source"]) for segment in segments}
            reused = sorted(excluded.intersection(used_sources))
            if reused:
                raise ValueError(f"excluded donor reused by fallback for {sid}: {reused[:5]}")
            exact_reused = sorted(exact_sources.intersection(used_sources))
            if require_no_exact_caption_sources and exact_reused:
                raise ValueError(
                    f"exact-caption fallback source for {sid}: {exact_reused[:5]}"
                )
            outside = sorted(used_sources.difference(train))
            if outside:
                raise ValueError(f"non-training fallback source for {sid}: {outside[:5]}")
            source_mass, blends = compositional_ledger(segments, length, blend_width)
            mode = "compositional_local_reuse"
        for source, value in source_mass.items():
            source_total[source] += value
            signer = str(train.get(source, {}).get("signer", "unknown"))
            signer_total[signer] += value
        mode_frames[mode] += length
        clips.append({
            "id": sid,
            "mode": mode,
            "frames": length,
            "source_mass": source_mass,
            "source_mass_sum": float(sum(source_mass.values())),
            "blend_frames": blends,
        })

    total_frames = int(sum(int(row["frames"]) for row in clips))
    if not math.isclose(sum(source_total.values()), total_frames, abs_tol=1e-6):
        raise ValueError("release-level source ledger is not conserved")
    top_sources = sorted(source_total.items(), key=lambda item: (-item[1], item[0]))
    top_signers = sorted(signer_total.items(), key=lambda item: (-item[1], item[0]))
    return {
        "definition": (
            "Each emitted frame has unit provenance mass. Whole-clip frames charge one "
            "training donor. Local boundary blends split unit mass between adjacent "
            "training sources according to the deterministic cosine blend."
        ),
        "n_clips": len(ids),
        "total_frames": total_frames,
        "whole_clip_replay_clips": int(mask.sum()),
        "whole_clip_replay_clip_fraction": float(mask.mean()),
        "whole_clip_replay_frames": int(mode_frames["whole_clip_replay"]),
        "whole_clip_replay_frame_fraction": (
            mode_frames["whole_clip_replay"] / max(1, total_frames)
        ),
        "compositional_local_frames": int(mode_frames["compositional_local_reuse"]),
        "generated_frames": 0,
        "unknown_or_untraced_frames": float(source_total.get("unknown_or_derived", 0.0)),
        "exact_caption_exclusion_enforced": bool(require_no_exact_caption_sources),
        "normalized_exact_caption_source_links": 0 if require_no_exact_caption_sources else None,
        "source_mass_sum": float(sum(source_total.values())),
        "unique_training_sources": len([key for key in source_total if key in train]),
        "top_sources": [
            {"source": source, "frames": value, "fraction": value / max(1, total_frames)}
            for source, value in top_sources[:20]
        ],
        "signers": [
            {"signer": signer, "frames": value, "fraction": value / max(1, total_frames)}
            for signer, value in top_signers
        ],
        "clips": clips,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--route_pose", required=True)
    parser.add_argument("--retrieval_trace", required=True)
    parser.add_argument("--fallback_trace", required=True)
    parser.add_argument("--train_manifest", default="data/phoenix/phoenix_train.json")
    parser.add_argument("--blend_width", type=int, default=4)
    parser.add_argument("--require_no_exact_caption_sources", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = {
        "mask": ROOT / args.mask,
        "route_pose": ROOT / args.route_pose,
        "retrieval_trace": ROOT / args.retrieval_trace,
        "fallback_trace": ROOT / args.fallback_trace,
        "train_manifest": ROOT / args.train_manifest,
    }
    ledger = build_ledger(
        paths["mask"], paths["route_pose"], paths["retrieval_trace"],
        paths["fallback_trace"], paths["train_manifest"], args.blend_width,
        args.require_no_exact_caption_sources,
    )
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger["provenance"] = {
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "blend_width": args.blend_width,
        "require_no_exact_caption_sources": args.require_no_exact_caption_sources,
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    out_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: ledger[key] for key in (
            "n_clips", "total_frames", "whole_clip_replay_clips",
            "whole_clip_replay_frame_fraction", "unique_training_sources",
            "unknown_or_untraced_frames",
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
