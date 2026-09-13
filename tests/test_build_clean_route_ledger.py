import json

import numpy as np
import pytest
import torch

from scripts.build_clean_route_ledger import build_ledger, compositional_ledger


def test_compositional_blend_conserves_fractional_source_mass():
    segments = [
        {"source": "a", "out_len": 4},
        {"source": "b", "out_len": 4},
    ]
    mass, blends = compositional_ledger(segments, output_length=8, blend_width=2)
    assert mass == {"a": 3.0, "b": 5.0}
    assert [row["frame"] for row in blends] == [2, 3]
    assert all(abs(sum(row["sources"].values()) - 1.0) < 1e-12 for row in blends)


def test_adjacent_segments_from_same_source_keep_unit_frame_mass():
    segments = [
        {"source": "train-a", "out_len": 4},
        {"source": "train-a", "out_len": 4},
    ]
    mass, blends = compositional_ledger(segments, output_length=8, blend_width=2)
    assert mass == {"train-a": 8.0}
    assert [row["sources"] for row in blends] == [
        {"train-a": 1.0},
        {"train-a": 1.0},
    ]


def test_route_ledger_rejects_excluded_source_reuse(tmp_path):
    mask_path = tmp_path / "mask.npz"
    np.savez(mask_path, ids=np.asarray(["q"]), in_S=np.asarray([0]))
    pose_path = tmp_path / "pose.pt"
    torch.save({"q": torch.zeros(3, 178, 3)}, pose_path)
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(json.dumps([{"id": "q", "selected": {"id": "train-a"}}]))
    fallback_path = tmp_path / "fallback.json"
    fallback_path.write_text(json.dumps({
        "q": {
            "excluded_source": "train-a",
            "segments": [{"source": "train-a", "out_len": 3}],
        }
    }))
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([{"id": "train-a", "signer": "s"}]))
    with pytest.raises(ValueError, match="excluded donor reused"):
        build_ledger(mask_path, pose_path, retrieval_path, fallback_path, train_path, 2)


def test_route_ledger_rejects_any_normalized_exact_caption_source(tmp_path):
    mask_path = tmp_path / "mask.npz"
    np.savez(mask_path, ids=np.asarray(["q"]), in_S=np.asarray([0]))
    pose_path = tmp_path / "pose.pt"
    torch.save({"q": torch.zeros(3, 178, 3)}, pose_path)
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(json.dumps([{"id": "q", "text": "Guten Abend!", "selected": {"id": "safe"}}]))
    fallback_path = tmp_path / "fallback.json"
    fallback_path.write_text(json.dumps({
        "q": {
            "excluded_source": "safe",
            "excluded_sources": ["safe"],
            "segments": [{"source": "exact", "out_len": 3}],
        }
    }))
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps([
        {"id": "safe", "text": "Guten Morgen", "signer": "s"},
        {"id": "exact", "text": "guten abend", "signer": "s"},
    ]))
    with pytest.raises(ValueError, match="exact-caption fallback source"):
        build_ledger(
            mask_path, pose_path, retrieval_path, fallback_path, train_path, 2,
            require_no_exact_caption_sources=True,
        )
