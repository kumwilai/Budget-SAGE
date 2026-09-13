"""Exhaustive finite-domain checks for the deterministic certificate claims."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from budget_sage.audit.certificate import (
    CertificateError,
    canonical_json,
    pose_sha256,
)
from scripts.clean_route_certificate import (
    REPLAY_BUDGET_UNIT,
    reconstruct_from_ledger,
    source_mass_from_segments,
    validate_replay_budget,
)


def _policy(numerator: int, denominator: int) -> dict:
    return {
        "route": "clean_rerank_frame40",
        "selection": "frozen before evaluation",
        "whole_clip_replay_budget": {
            "numerator": numerator,
            "denominator": denominator,
            "unit": REPLAY_BUDGET_UNIT,
        },
    }


def _local_ledger(length_a: int, length_b: int, same_source: bool, width: int):
    total = length_a + length_b
    source_b = "a" if same_source else "b"
    segments = [
        {
            "route": "local_unit",
            "start": 0,
            "end": length_a,
            "source_id": "a",
            "source_start": 0,
            "source_end": length_a,
        },
        {
            "route": "local_unit",
            "start": length_a,
            "end": total,
            "source_id": source_b,
            "source_start": 0,
            "source_end": length_b,
        },
    ]
    mass = source_mass_from_segments(segments, width)
    ledger = {
        "route": "fallback",
        "T": total,
        "frame_counts": {
            "whole_clip": 0,
            "local_unit": total,
            "generated": 0,
            "unknown_or_derived": 0,
        },
        "claimed_source_ids": sorted(set(("a", source_b))),
        "segments": segments,
        "fractional_source_mass": mass,
    }
    return ledger, mass


def test_exact_integer_budget_exhaustive_small_domain():
    """The implementation agrees with qR <= pT for every small valid case."""
    checked = 0
    for denominator in range(1, 10):
        for numerator in range(denominator + 1):
            policy = _policy(numerator, denominator)
            for total in range(21):
                for replay in range(total + 1):
                    feasible = denominator * replay <= numerator * total
                    if feasible:
                        result = validate_replay_budget(policy, replay, total)
                        assert result["verified"] is True
                        assert result["integer_check"] == (
                            f"{denominator * replay} <= {numerator * total}"
                        )
                    else:
                        with pytest.raises(CertificateError, match="exceeds"):
                            validate_replay_budget(policy, replay, total)
                    checked += 1
    assert checked == 12474


@pytest.mark.parametrize("same_source", [False, True])
def test_cosine_reconstruction_and_mass_exhaustive_short_intervals(same_source):
    """Check every length 1..6 and configured width 0..6, including repeats."""
    for length_a in range(1, 7):
        for length_b in range(1, 7):
            for configured_width in range(7):
                source_len = max(length_a, length_b) if same_source else length_a
                archive = {
                    "a": torch.arange(
                        source_len * 178 * 3, dtype=torch.float32
                    ).reshape(source_len, 178, 3)
                }
                if not same_source:
                    archive["b"] = (
                        torch.arange(length_b * 178 * 3, dtype=torch.float32)
                        .reshape(length_b, 178, 3)
                        .add_(100000.0)
                    )
                ledger, mass = _local_ledger(
                    length_a, length_b, same_source, configured_width
                )
                actual = reconstruct_from_ledger(
                    ledger, archive, blend_width=configured_width
                )

                first = archive["a"][:length_a].numpy().reshape(length_a, -1).copy()
                second_key = "a" if same_source else "b"
                second = archive[second_key][:length_b].numpy().reshape(length_b, -1).copy()
                width = min(configured_width, length_a // 2, length_b // 2)
                if width:
                    ramp = 0.5 * (
                        1.0 - np.cos(np.linspace(0.0, np.pi, width))
                    ).astype(np.float32)[:, None]
                    first[-width:] = (
                        (1.0 - ramp) * first[-width:] + ramp * second[:width]
                    )
                expected = torch.from_numpy(
                    np.concatenate((first, second), axis=0).reshape(-1, 178, 3)
                )
                assert torch.equal(actual, expected)
                assert actual.dtype == torch.float32
                assert actual.shape == (length_a + length_b, 178, 3)
                assert sum(mass.values()) == pytest.approx(length_a + length_b)
                assert all(value >= 0.0 for value in mass.values())
                if same_source:
                    assert mass == {"a": pytest.approx(length_a + length_b)}


def test_canonical_serialization_and_pose_dtype_binding():
    left = {"z": [3, {"beta": "β"}], "a": {"x": 1, "y": 2.0}}
    right = {"a": {"y": 2.0, "x": 1}, "z": [3, {"beta": "β"}]}
    assert canonical_json(left) == canonical_json(right)
    assert json.loads(canonical_json(left)) == left

    pose32 = np.arange(24, dtype=np.float32).reshape(3, 8)
    pose64 = pose32.astype(np.float64)
    assert pose_sha256(pose32) != pose_sha256(pose64)
    assert pose_sha256(np.asfortranarray(pose32)) == pose_sha256(pose32)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_serialization_rejects_nonfinite_nested_values(value):
    with pytest.raises(CertificateError, match="non-finite"):
        canonical_json({"outer": [{"mass": value}]})
