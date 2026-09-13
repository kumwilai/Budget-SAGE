from pathlib import Path

import numpy as np
import torch

from scripts.audit_phase24_alignment_rebuild import (
    canonical_bank_summary,
    canonical_interval_summary,
)


def _bank(offset: float = 0.0):
    poses = {
        "b": (np.arange(12, dtype=np.float16).reshape(2, 6) + offset),
        "a": np.ones((3, 6), dtype=np.float16),
    }
    return {
        "gloss_to_exemplars": {
            "B": [("b", 0, 2)],
            "A": [("a", 0, 2), ("a", 1, 3)],
        },
        "exemplar_poses": poses,
        "n_glosses": 2,
        "n_exemplars": 3,
    }


def test_canonical_summary_ignores_mapping_insertion_order(tmp_path: Path):
    first = _bank()
    second = {
        **first,
        "gloss_to_exemplars": dict(reversed(list(first["gloss_to_exemplars"].items()))),
        "exemplar_poses": dict(reversed(list(first["exemplar_poses"].items()))),
    }
    p1, p2 = tmp_path / "one.pt", tmp_path / "two.pt"
    torch.save(first, p1)
    torch.save(second, p2)
    assert canonical_bank_summary(p1) == canonical_bank_summary(p2)


def test_canonical_summary_detects_pose_change(tmp_path: Path):
    p1, p2 = tmp_path / "one.pt", tmp_path / "two.pt"
    torch.save(_bank(), p1)
    torch.save(_bank(offset=1.0), p2)
    assert (
        canonical_bank_summary(p1)["semantic_sha256"]
        != canonical_bank_summary(p2)["semantic_sha256"]
    )


def test_canonical_summary_preserves_interval_order(tmp_path: Path):
    first = _bank()
    second = _bank()
    second["gloss_to_exemplars"]["A"] = list(
        reversed(second["gloss_to_exemplars"]["A"])
    )
    p1, p2 = tmp_path / "one.pt", tmp_path / "two.pt"
    torch.save(first, p1)
    torch.save(second, p2)
    assert (
        canonical_bank_summary(p1)["semantic_sha256"]
        != canonical_bank_summary(p2)["semantic_sha256"]
    )


def test_interval_projection_matches_consumed_source_subset(tmp_path: Path):
    source = _bank()
    consumed = {
        "gloss_to_exemplars": {"A": [("a", 0, 2), ("a", 1, 3)]},
        "exemplar_poses": {"a": source["exemplar_poses"]["a"]},
        "n_glosses": 1,
        "n_exemplars": 2,
    }
    p1, p2 = tmp_path / "source.pt", tmp_path / "consumed.pt"
    torch.save(source, p1)
    torch.save(consumed, p2)
    actual = canonical_interval_summary(p2, include_source_ids=True)
    allowed = set(actual.pop("_source_ids"))
    projected = canonical_interval_summary(p1, allowed_sources=allowed)
    for key in (
        "ordered_interval_sha256",
        "interval_multiset_sha256",
        "n_glosses",
        "n_intervals",
        "n_referenced_source_ids",
    ):
        assert actual[key] == projected[key]
