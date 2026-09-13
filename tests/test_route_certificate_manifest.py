import copy
import tempfile

import numpy as np

from budget_sage.audit.certificate import (
    CertificateError,
    canonical_json,
    issue_certificate,
    sha256_file,
    validate_ledger,
    verify_certificate,
)


def _make():
    pose = np.arange(24, dtype=np.float32).reshape(3, 8)
    kwargs = dict(
        certificate_id="cert-1", clip_id="clip-1", pose=pose,
        ledger={"route": "generated", "T": 3,
                "nested": {"source": "s7", "mass": 0.25},
                "frame_counts": {"whole_clip": 0, "local_unit": 0, "generated": 3,
                                  "unknown_or_derived": 0}, "claimed_source_ids": []},
        archive_commitment={"algorithm": "sha256", "digest": "a" * 64},
        archive_id="archive-7",
        policy={"max_reuse": 0.2, "allow": ["generated"]},
        detector_config={"window": 16, "threshold": 0.05},
        watermark_key_id="wm-kid-1", manifest_key_id="manifest-kid-1",
        manifest_key="manifest-secret",
    )
    return pose, kwargs, issue_certificate(**kwargs)


def test_certificate_success_and_required_bindings():
    pose, kw, cert = _make()
    assert verify_certificate(cert, pose=pose, manifest_key=kw["manifest_key"])
    for field in ("archive_commitment", "archive_id", "policy", "detector_config", "watermark"):
        changed = copy.deepcopy(cert)
        if isinstance(changed[field], dict):
            changed[field][next(iter(changed[field]))] = "tampered"
        else:
            changed[field] = "tampered"
        assert not verify_certificate(changed, pose=pose, manifest_key=kw["manifest_key"])


def test_pose_and_every_nested_ledger_field_tamper():
    pose, kw, cert = _make()
    bad_pose = pose.copy(); bad_pose[0, 0] += 1
    assert not verify_certificate(cert, pose=bad_pose, manifest_key=kw["manifest_key"])
    other_shape = pose.reshape(2, 12)
    assert not verify_certificate(cert, pose=other_shape, manifest_key=kw["manifest_key"])
    assert not verify_certificate(cert, pose=pose.astype(np.float64),
                                  manifest_key=kw["manifest_key"])
    for path in (("route",), ("nested", "source"), ("nested", "mass")):
        bad = copy.deepcopy(cert)
        target = bad["ledger"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = "tampered"
        assert not verify_certificate(bad, pose=pose, manifest_key=kw["manifest_key"])


def test_wrong_keys_missing_manifest_and_nonfinite_input():
    pose, kw, cert = _make()
    assert not verify_certificate(cert, pose=pose, manifest_key="wrong")
    assert not verify_certificate({}, pose=pose, manifest_key=kw["manifest_key"])
    with np.testing.assert_raises(CertificateError):
        canonical_json({"x": float("nan")})
    with np.testing.assert_raises(CertificateError):
        issue_certificate(**{**kw, "policy": {"bad": float("inf")}})


def test_expected_claims_are_checked_after_authentication():
    pose, kw, cert = _make()
    expected = {
        "ledger": kw["ledger"],
        "archive_commitment": kw["archive_commitment"],
        "detector_config": kw["detector_config"],
        "watermark": {"detector": "scripts.pose_watermark.detect",
                      "key_id": kw["watermark_key_id"]},
    }
    assert verify_certificate(cert, pose=pose, manifest_key=kw["manifest_key"],
                              expected_claims=expected)
    wrong = copy.deepcopy(expected)
    wrong["ledger"]["route"] = "fallback"
    assert not verify_certificate(cert, pose=pose, manifest_key=kw["manifest_key"],
                                  expected_claims=wrong)


def test_ledger_T_route_source_ids_and_key_separation():
    _, kw, _ = _make()
    for mutation in ("T", "route", "duplicate_source"):
        bad = copy.deepcopy(kw)
        if mutation == "T":
            bad["ledger"]["T"] = 2
        elif mutation == "route":
            bad["ledger"]["route"] = "made-up"
        else:
            bad["ledger"]["claimed_source_ids"] = ["x", "x"]
        with np.testing.assert_raises(CertificateError):
            issue_certificate(**bad)
    with np.testing.assert_raises(CertificateError):
        issue_certificate(**{**kw, "watermark_key_id": kw["manifest_key_id"]})


def test_conservation_and_archive_commitment_helper():
    _, kw, _ = _make()
    validate_ledger(kw["ledger"], 3)
    bad = copy.deepcopy(kw["ledger"])
    bad["frame_counts"]["generated"] = 2
    with np.testing.assert_raises(CertificateError):
        validate_ledger(bad, 3)
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"archive")
        f.flush()
        import hashlib
        assert sha256_file(f.name) == hashlib.sha256(b"archive").hexdigest()


def test_optional_segments_must_partition_pose_and_reproduce_counts():
    _, kw, _ = _make()
    segmented = copy.deepcopy(kw["ledger"])
    segmented["route"] = "mixed"
    segmented["frame_counts"] = {
        "whole_clip": 1, "local_unit": 1, "generated": 0,
        "unknown_or_derived": 1,
    }
    segmented["claimed_source_ids"] = ["source-a"]
    segmented["segments"] = [
        {"start": 0, "end": 1, "route": "whole_clip",
         "source_id": "source-a", "source_start": 4},
        {"start": 1, "end": 2, "route": "local_unit",
         "source_id": "source-a", "source_start": 9},
        {"start": 2, "end": 3, "route": "unknown_or_derived"},
    ]
    validate_ledger(segmented, 3)
    for mutation in ("gap", "wrong_counts", "wrong_sources"):
        bad = copy.deepcopy(segmented)
        if mutation == "gap":
            bad["segments"][1]["start"] = 2
        elif mutation == "wrong_counts":
            bad["frame_counts"]["local_unit"] = 0
            bad["frame_counts"]["unknown_or_derived"] = 2
        else:
            bad["claimed_source_ids"] = ["source-b"]
        with np.testing.assert_raises(CertificateError):
            validate_ledger(bad, 3)
