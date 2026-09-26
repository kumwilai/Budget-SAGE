"""Hand-computed check of the motion-ratio kinematic primitives.

Pure CPU, no file loaded. The synthetic pose puts every hand joint (indices
8:50) at the same x-only trajectory x(t) = t**3 for t = 0, 1, 2, 3, 4, with
y = z = 0, so each function reduces to a scalar arithmetic series that can be
checked by hand:

  positions:      0, 1, 8, 27, 64
  1st differences (speed terms):  1, 7, 19, 37          -> mean = 16.0
  2nd differences:                6, 12, 18
  3rd differences (jerk terms):   6, 6                  -> mean |.| = 6.0
  sample std (ddof=1) of the 5 positions                -> sqrt(722.5)

hand_posestd averages the per-joint std over all three xyz channels (not just
the x channel that carries the signal here), and y = z = 0 for every joint and
frame contribute a std of 0. So the expected mean is sqrt(722.5) / 3, one third
of the x-only figure above.
"""
import math

import torch

from scripts.exp_revision_motion_ratios import hand_jerk, hand_posestd, hand_speed


def _synthetic_pose() -> torch.Tensor:
    t = torch.arange(5, dtype=torch.float32)
    x = t**3  # 0, 1, 8, 27, 64
    pose = torch.zeros(5, 178, 3, dtype=torch.float32)
    pose[:, 8:50, 0] = x[:, None]
    return pose


def test_hand_speed_matches_hand_computation():
    pose = _synthetic_pose()
    assert math.isclose(hand_speed(pose), 16.0, rel_tol=1e-6)


def test_hand_jerk_matches_hand_computation():
    pose = _synthetic_pose()
    assert math.isclose(hand_jerk(pose), 6.0, rel_tol=1e-6)


def test_hand_posestd_matches_hand_computation():
    pose = _synthetic_pose()
    expected = math.sqrt(722.5) / 3.0
    assert math.isclose(hand_posestd(pose), expected, rel_tol=1e-5)
