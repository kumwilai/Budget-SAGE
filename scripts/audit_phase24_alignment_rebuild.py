"""Audit a rebuilt Phase-24 interval bank against the preserved bank.

The byte representation produced by ``torch.save`` is not a stable semantic
contract.  This audit therefore hashes the ordered interval tuples and every
stored pose tensor in a canonical, length-delimited representation.  A PASS
means that the rebuilt bank has the same source clips, arrays, intervals, and
reported counts as the preserved pre-UPC bank.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _put_bytes(digest: "hashlib._Hash", value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _put_text(digest: "hashlib._Hash", value: Any) -> None:
    _put_bytes(digest, str(value).encode("utf-8"))


def canonical_bank_summary(path: Path) -> dict[str, Any]:
    bank = torch.load(path, map_location="cpu", weights_only=False)
    required = {"gloss_to_exemplars", "exemplar_poses", "n_glosses", "n_exemplars"}
    missing = sorted(required.difference(bank))
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")

    digest = hashlib.sha256()
    digest.update(b"phase24_interval_bank_semantics_v1\0")
    glosses = bank["gloss_to_exemplars"]
    poses = bank["exemplar_poses"]

    interval_count = 0
    for gloss in sorted(glosses, key=str):
        _put_text(digest, gloss)
        rows = list(glosses[gloss])
        digest.update(len(rows).to_bytes(8, "big"))
        interval_count += len(rows)
        # Preserve list order because candidate ordering affects assembly ties.
        for row in rows:
            if len(row) != 3:
                raise ValueError(f"{path}: malformed interval {row!r}")
            source_id, start, end = row
            _put_text(digest, source_id)
            digest.update(int(start).to_bytes(8, "big", signed=True))
            digest.update(int(end).to_bytes(8, "big", signed=True))

    pose_frames = 0
    pose_bytes = 0
    pose_dtypes: set[str] = set()
    for source_id in sorted(poses, key=str):
        _put_text(digest, source_id)
        array = poses[source_id]
        if torch.is_tensor(array):
            array = array.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(array))
        _put_text(digest, array.dtype.str)
        digest.update(len(array.shape).to_bytes(4, "big"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=False))
        raw = memoryview(array).cast("B")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        pose_frames += int(array.shape[0]) if array.ndim else 0
        pose_bytes += int(array.nbytes)
        pose_dtypes.add(str(array.dtype))

    summary = {
        "semantic_sha256": digest.hexdigest(),
        "top_level_keys": sorted(str(key) for key in bank),
        "declared_n_glosses": int(bank["n_glosses"]),
        "observed_n_glosses": len(glosses),
        "declared_n_exemplars": int(bank["n_exemplars"]),
        "observed_n_exemplars": interval_count,
        "n_source_clips": len(poses),
        "pose_frames": pose_frames,
        "pose_bytes": pose_bytes,
        "pose_dtypes": sorted(pose_dtypes),
    }
    del bank, glosses, poses
    gc.collect()
    return summary


def canonical_interval_summary(
    path: Path,
    allowed_sources: set[str] | None = None,
    include_source_ids: bool = False,
) -> dict[str, Any]:
    """Summarize interval identity, optionally after a source-ID projection."""
    bank = torch.load(path, map_location="cpu", weights_only=False)
    glosses = bank.get("gloss_to_exemplars")
    poses = bank.get("exemplar_poses")
    if not isinstance(glosses, dict) or not isinstance(poses, dict):
        raise ValueError(f"{path}: interval bank fields are missing")
    source_ids = {str(source_id) for source_id in poses}

    ordered = hashlib.sha256()
    multiset = hashlib.sha256()
    ordered.update(b"phase24_ordered_intervals_v1\0")
    multiset.update(b"phase24_interval_multiset_v1\0")
    interval_count = 0
    nonempty_glosses = 0
    referenced_sources: set[str] = set()
    for gloss in sorted(glosses, key=str):
        rows = [
            (str(source_id), int(start), int(end))
            for source_id, start, end in glosses[gloss]
            if allowed_sources is None or str(source_id) in allowed_sources
        ]
        if not rows:
            continue
        nonempty_glosses += 1
        interval_count += len(rows)
        referenced_sources.update(source_id for source_id, _, _ in rows)
        for digest, selected_rows in ((ordered, rows), (multiset, sorted(rows))):
            _put_text(digest, gloss)
            digest.update(len(selected_rows).to_bytes(8, "big"))
            for source_id, start, end in selected_rows:
                _put_text(digest, source_id)
                digest.update(start.to_bytes(8, "big", signed=True))
                digest.update(end.to_bytes(8, "big", signed=True))
    out = {
        "ordered_interval_sha256": ordered.hexdigest(),
        "interval_multiset_sha256": multiset.hexdigest(),
        "n_glosses": nonempty_glosses,
        "n_intervals": interval_count,
        "n_pose_source_ids": len(source_ids),
        "n_referenced_source_ids": len(referenced_sources),
        "all_referenced_sources_have_poses": referenced_sources.issubset(source_ids),
    }
    if include_source_ids:
        out["_source_ids"] = sorted(source_ids)
    del bank, glosses, poses
    gc.collect()
    return out


def input_commitment(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not path.is_dir():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()
    digest.update(b"directory_tree_sha256_v1\0")
    count = 0
    total_bytes = 0
    for child in sorted((p for p in path.rglob("*") if p.is_file()),
                        key=lambda p: p.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix()
        child_hash = sha256_file(child)
        _put_text(digest, relative)
        _put_text(digest, child_hash)
        size = child.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        count += 1
        total_bytes += size
    return {
        "path": str(path),
        "kind": "directory",
        "files": count,
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--consumed", type=Path)
    parser.add_argument("--lineage-builder", type=Path)
    parser.add_argument("--lineage-input", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    consumed_lineage = None
    lineage_mismatches: dict[str, Any] = {}
    if args.consumed:
        consumed_lineage = canonical_interval_summary(
            args.consumed, include_source_ids=True
        )
        consumed_source_ids = set(consumed_lineage.pop("_source_ids"))
        rebuilt_projection = canonical_interval_summary(
            args.rebuilt, allowed_sources=consumed_source_ids
        )
        lineage_fields = [
            "ordered_interval_sha256",
            "interval_multiset_sha256",
            "n_glosses",
            "n_intervals",
            "n_referenced_source_ids",
        ]
        lineage_mismatches = {
            key: {
                "consumed": consumed_lineage[key],
                "rebuilt_projection": rebuilt_projection[key],
            }
            for key in lineage_fields
            if consumed_lineage[key] != rebuilt_projection[key]
        }
        consumed_lineage = {
            "consumed_bank": consumed_lineage,
            "rebuilt_projected_to_consumed_sources": rebuilt_projection,
            "source_projection_count": len(consumed_source_ids),
            "mismatches": lineage_mismatches,
        }

    reference = canonical_bank_summary(args.reference)
    rebuilt = canonical_bank_summary(args.rebuilt)
    invariant_fields = [
        "semantic_sha256",
        "top_level_keys",
        "declared_n_glosses",
        "observed_n_glosses",
        "declared_n_exemplars",
        "observed_n_exemplars",
        "n_source_clips",
        "pose_frames",
        "pose_bytes",
        "pose_dtypes",
    ]
    mismatches = {
        key: {"reference": reference[key], "rebuilt": rebuilt[key]}
        for key in invariant_fields
        if reference[key] != rebuilt[key]
    }
    internal_count_checks = {
        "reference": (
            reference["declared_n_glosses"] == reference["observed_n_glosses"]
            and reference["declared_n_exemplars"] == reference["observed_n_exemplars"]
        ),
        "rebuilt": (
            rebuilt["declared_n_glosses"] == rebuilt["observed_n_glosses"]
            and rebuilt["declared_n_exemplars"] == rebuilt["observed_n_exemplars"]
        ),
    }
    passed = (
        not mismatches
        and not lineage_mismatches
        and all(internal_count_checks.values())
    )
    report = {
        "schema": "phase24_alignment_rebuild_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "claim_supported": (
            "The preserved Phase-24 interval bank is semantically reproducible "
            "from the committed builder and inputs."
            if passed else
            "The rebuild does not reproduce the preserved Phase-24 interval bank."
        ),
        "claim_not_supported": (
            "This newly captured run does not prove the content of an absent "
            "historical execution log or restore later frozen script bytes."
        ),
        "reference": {
            "path": str(args.reference),
            "bytes": args.reference.stat().st_size,
            "file_sha256": sha256_file(args.reference),
            **reference,
        },
        "rebuilt": {
            "path": str(args.rebuilt),
            "bytes": args.rebuilt.stat().st_size,
            "file_sha256": sha256_file(args.rebuilt),
            **rebuilt,
        },
        "builder": input_commitment(args.builder),
        "execution_log": input_commitment(args.log),
        "inputs": [input_commitment(path) for path in args.input],
        "consumed_bank_lineage": consumed_lineage,
        "lineage_builder": (
            input_commitment(args.lineage_builder) if args.lineage_builder else None
        ),
        "lineage_inputs": [input_commitment(path) for path in args.lineage_input],
        "internal_count_checks": internal_count_checks,
        "mismatches": mismatches,
        "environment": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "semantic_sha256": rebuilt["semantic_sha256"],
        "mismatches": sorted(mismatches),
        "report": str(args.report),
    }, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
