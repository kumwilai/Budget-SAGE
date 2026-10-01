"""SLRTP-178 <-> Progressive-Transformer-50 joint mapping and the target-independent fills.

SLRTP-178 layout (section9_answers.md item 1, scripts/slrtp178_to_phoenix201.py lines 44-54,
edge list external/SLRTP-Sign-Production-Evaluation/skeleton_def.py RELEASE_EDGES):
  0 RShoulder, 1 RElbow, 2 RWrist, 3 LShoulder, 4 LElbow, 5 LWrist, 6 RHip, 7 LHip,
  8..28 right hand (8 = its wrist root, identical to body 2 in the release),
  29..49 left hand (29 = its wrist root, identical to body 5), 50..177 face (128 points).

Progressive-Transformer 50 layout, as walked by getSkeletalModelStructure() in
external/baselines/Sign-IDD/helpers.py (OpenPose body order):
  0 nose, 1 neck, 2 RShoulder, 3 RElbow, 4 RWrist, 5 LShoulder, 6 LElbow, 7 LWrist,
  8..28 the hand attached at joint 7 (bone (7, 8)) = left hand,
  29..49 the hand attached at joint 4 (bone (4, 29)) = right hand.
  Within a hand both layouts use the same 21-point order (0 wrist, 1-4 thumb, 5-8 index,
  9-12 middle, 13-16 ring, 17-20 pinky); only the bone topology differs.

Mapping (section9_answers.md item 1):
  PT neck = midpoint of SLRTP shoulders 0 and 3.
  PT nose = one fixed SLRTP face landmark. The 128-point face contains no nose landmark
  (make_128_face_from_478.py: MediaPipe 1, 4, 5, 6, 168, 195, 197 all absent), so the
  nearest fixed midline landmark is used: MediaPipe 0 (upper-lip centre) = SLRTP 82.

Fills at export (DESIGN.md Section 4, extended by section9_answers.md item 1):
  face: the train-split mean face (128 points). The release stores the face per frame
        centred at the origin (centroid exactly 0 in every dev frame, checked 2026-09-28),
        so the per-frame rigid translation that keeps it attached to its anchor is zero:
        the fill is the constant train-mean face in the release's own face frame.
  hips: neck(t) + train-mean (hip - neck) offset, neck = the PT neck joint of the pose.
"""
from __future__ import annotations

import numpy as np

N_SLRTP = 178
N_PT = 50
FACE = slice(50, 178)
HIPS = (6, 7)
PT_NOSE_FROM_SLRTP_FACE = 82  # MediaPipe landmark 0, upper-lip centre (no nose tip in the 128)

# PT index -> SLRTP index for the joints that are copied one-to-one
PT_FROM_SLRTP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
for _j in range(21):
    PT_FROM_SLRTP[8 + _j] = 29 + _j   # PT left hand  <- SLRTP left hand
    PT_FROM_SLRTP[29 + _j] = 8 + _j   # PT right hand <- SLRTP right hand
# SLRTP manual index -> PT index (inverse of the above; SLRTP 0-5 and 8-49)
SLRTP_FROM_PT = {s: p for p, s in PT_FROM_SLRTP.items()}


def slrtp_neck(pose178: np.ndarray) -> np.ndarray:
    return 0.5 * (pose178[..., 0, :] + pose178[..., 3, :])


def slrtp178_to_pt50(pose178: np.ndarray) -> np.ndarray:
    """[T, 178, 3] -> [T, 50, 3] in native SLRTP coordinates."""
    pose178 = np.asarray(pose178)
    assert pose178.shape[-2:] == (N_SLRTP, 3), pose178.shape
    out = np.zeros(pose178.shape[:-2] + (N_PT, 3), dtype=pose178.dtype)
    out[..., 0, :] = pose178[..., PT_NOSE_FROM_SLRTP_FACE, :]
    out[..., 1, :] = slrtp_neck(pose178)
    for p, s in PT_FROM_SLRTP.items():
        out[..., p, :] = pose178[..., s, :]
    return out


def pt50_to_slrtp178(pose50: np.ndarray, stats: dict) -> np.ndarray:
    """[T, 50, 3] -> [T, 178, 3]: manual joints copied, hips and face filled.

    The PT nose (joint 0) is not part of the SLRTP body block and is discarded.
    """
    pose50 = np.asarray(pose50)
    assert pose50.shape[-2:] == (N_PT, 3), pose50.shape
    out = np.zeros(pose50.shape[:-2] + (N_SLRTP, 3), dtype=np.float32)
    for p, s in PT_FROM_SLRTP.items():
        out[..., s, :] = pose50[..., p, :]
    neck = pose50[..., 1, :]
    out[..., HIPS[0], :] = neck + stats["hip_offset"][0]
    out[..., HIPS[1], :] = neck + stats["hip_offset"][1]
    out[..., FACE, :] = stats["face_mean"]
    return out


def apply_fills(pose178: np.ndarray, stats: dict, face: bool = True, hips: bool = True) -> np.ndarray:
    """Gate 0: replace the face and/or hips of a (ground-truth) SLRTP pose with the fills."""
    out = np.array(pose178, dtype=np.float32, copy=True)
    if face:
        out[..., FACE, :] = stats["face_mean"]
    if hips:
        neck = slrtp_neck(out)
        out[..., HIPS[0], :] = neck + stats["hip_offset"][0]
        out[..., HIPS[1], :] = neck + stats["hip_offset"][1]
    return out


def compute_fill_stats(poses: list) -> dict:
    """Train-split statistics for the fills. `poses` is a list of [T, 178, 3] arrays."""
    n = 0
    face_sum = np.zeros((128, 3), dtype=np.float64)
    hip_sum = np.zeros((2, 3), dtype=np.float64)
    for p in poses:
        p = np.asarray(p, dtype=np.float64)
        neck = slrtp_neck(p)
        face_sum += p[:, FACE, :].sum(0)
        hip_sum += (p[:, list(HIPS), :] - neck[:, None, :]).sum(0)
        n += p.shape[0]
    return {"face_mean": (face_sum / n).astype(np.float32),
            "hip_offset": (hip_sum / n).astype(np.float32),
            "n_frames": np.int64(n)}


def load_stats(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}
