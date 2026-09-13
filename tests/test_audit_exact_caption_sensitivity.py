import pytest

from scripts.audit_exact_caption_sensitivity import assert_frame_budget


def test_exact_integer_frame_budget_accepts_boundary():
    row = {
        "replay_frames": 40,
        "emitted_frames": 100,
        "realized_replay_frame_fraction": 0.4,
    }
    assert_frame_budget(row)


def test_exact_integer_frame_budget_rejects_violation():
    row = {
        "replay_frames": 41,
        "emitted_frames": 100,
        "realized_replay_frame_fraction": 0.41,
    }
    with pytest.raises(AssertionError, match="budget failed"):
        assert_frame_budget(row)
