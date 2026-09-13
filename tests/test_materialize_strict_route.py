import numpy as np
import pytest
import torch

from scripts.materialize_strict_route import materialize


def test_materialize_follows_mask_and_counts_frames(tmp_path):
    mask = tmp_path / "mask.npz"
    np.savez(mask, ids=np.asarray(["a", "b"]), in_S=np.asarray([1, 0]))
    retrieval = tmp_path / "retrieval.pt"
    fallback = tmp_path / "fallback.pt"
    torch.save({"a": torch.ones(3, 178, 3), "b": torch.ones(4, 178, 3)}, retrieval)
    torch.save({"a": torch.zeros(5, 178, 3), "b": torch.zeros(6, 178, 3)}, fallback)
    output, summary = materialize(mask, retrieval, fallback)
    assert torch.equal(output["a"], torch.ones(3, 178, 3))
    assert torch.equal(output["b"], torch.zeros(6, 178, 3))
    assert summary["whole_clip_replay_frames"] == 3
    assert summary["compositional_fallback_frames"] == 6


def test_materialize_rejects_branch_id_mismatch(tmp_path):
    mask = tmp_path / "mask.npz"
    np.savez(mask, ids=np.asarray(["a"]), in_S=np.asarray([1]))
    retrieval = tmp_path / "retrieval.pt"
    fallback = tmp_path / "fallback.pt"
    torch.save({"wrong": torch.ones(3, 178, 3)}, retrieval)
    torch.save({"a": torch.zeros(3, 178, 3)}, fallback)
    with pytest.raises(ValueError, match="exactly the masked ids"):
        materialize(mask, retrieval, fallback)
