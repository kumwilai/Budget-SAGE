import copy
import hashlib

import numpy as np
import pytest
import torch

from budget_sage.audit.certificate import (
    CertificateError,
    issue_certificate,
    verify_certificate,
)
from scripts.clean_route_certificate import (
    MANIFEST_KEY_ID,
    SCHEMA_VERSION,
    WATERMARK_KEY_ID,
    WITHDRAWAL_SCHEMA_VERSION,
    _close_mass,
    _release_mac,
    canonical_withdrawn_sources,
    reconstruct_from_ledger,
    source_mass_from_segments,
    tamper_drills,
    validate_replay_budget,
    validate_withdrawal_commitment,
    verify_release,
    withdrawal_commitment,
)


def _fallback_ledger():
    return {
        "route": "fallback",
        "T": 8,
        "frame_counts": {
            "whole_clip": 0,
            "local_unit": 8,
            "generated": 0,
            "unknown_or_derived": 0,
        },
        "claimed_source_ids": ["a", "b"],
        "segments": [
            {"route": "local_unit", "start": 0, "end": 4,
             "source_id": "a", "source_start": 1, "source_end": 5},
            {"route": "local_unit", "start": 4, "end": 8,
             "source_id": "b", "source_start": 0, "source_end": 4},
        ],
        "fractional_source_mass": {"a": 3.0, "b": 5.0},
    }


def test_fallback_reconstruction_and_mass_include_blend():
    archive = {
        "a": torch.arange(5 * 178 * 3, dtype=torch.float32).reshape(5, 178, 3),
        "b": torch.arange(4 * 178 * 3, dtype=torch.float32).reshape(4, 178, 3) + 100,
    }
    ledger = _fallback_ledger()
    pose = reconstruct_from_ledger(ledger, archive, blend_width=2)
    assert pose.shape == (8, 178, 3)
    mass = source_mass_from_segments(ledger["segments"], blend_width=2)
    assert mass == {"a": pytest.approx(3.0), "b": pytest.approx(5.0)}
    assert sum(mass.values()) == pytest.approx(8.0)


def test_reconstruction_rejects_missing_or_out_of_bounds_source():
    archive = {
        "a": torch.zeros(5, 178, 3),
        "b": torch.zeros(4, 178, 3),
    }
    ledger = _fallback_ledger()
    with pytest.raises(CertificateError, match="out of bounds"):
        reconstruct_from_ledger(ledger, {"a": archive["a"], "b": archive["b"][:2]}, 2)
    changed = copy.deepcopy(ledger)
    changed["segments"][0]["source_id"] = "missing"
    changed["claimed_source_ids"] = ["missing", "b"]
    with pytest.raises(CertificateError, match="absent"):
        reconstruct_from_ledger(changed, archive, 2)
    changed = copy.deepcopy(ledger)
    changed["segments"][0]["source_end"] = 4
    with pytest.raises(CertificateError, match="source_end is inconsistent"):
        reconstruct_from_ledger(changed, archive, 2)


def test_hmac_binds_blend_and_fractional_source_claims():
    pose = torch.zeros(8, 178, 3)
    ledger = _fallback_ledger()
    cert = issue_certificate(
        certificate_id="q",
        clip_id="q",
        pose=pose,
        ledger=ledger,
        archive_commitment={"algorithm": "sha256", "digest": "1" * 64},
        archive_id="archive",
        policy={"budget": "2/5"},
        detector_config={"blend": "cosine", "blend_width": 2},
        watermark_key_id=WATERMARK_KEY_ID,
        manifest_key_id=MANIFEST_KEY_ID,
        manifest_key=b"test-key",
    )
    assert cert["ledger"]["fractional_source_mass"] == {"a": 3.0, "b": 5.0}
    changed = copy.deepcopy(cert)
    changed["detector_config"]["blend_width"] = 3
    assert not verify_certificate(changed, pose=pose, manifest_key=b"test-key")


def test_source_mass_same_source_boundary_is_conserved():
    segments = [
        {"start": 0, "end": 4, "source_id": "a", "source_start": 0, "source_end": 4},
        {"start": 4, "end": 8, "source_id": "a", "source_start": 0, "source_end": 4},
    ]
    assert source_mass_from_segments(segments, 2) == {"a": pytest.approx(8.0)}


def test_authenticated_replay_budget_is_enforced_with_integer_arithmetic():
    policy = _policy((2, 5))
    result = validate_replay_budget(policy, replay_frames=40, total_frames=100)
    assert result["verified"] is True
    assert result["integer_check"] == "200 <= 200"
    with pytest.raises(CertificateError, match="exceeds"):
        validate_replay_budget(policy, replay_frames=41, total_frames=100)


def _policy(budget=(1, 1), route="clean_strict_cac_frame40"):
    return {
        "route": route,
        "selection": "frozen before evaluation",
        "whole_clip_replay_budget": {
            "numerator": budget[0], "denominator": budget[1],
            "unit": "whole_clip_replay_frames_per_emitted_frame",
        },
    }


def _detector():
    return {
        "whole_clip": "byte_exact_reconstruction",
        "local_unit": "byte_exact_archive_reconstruction",
        "blend": "cosine", "blend_width": 2,
        "generated": "inactive_for_this_release",
    }


def _signed_single_interval_release(tmp_path, *, source_start, source_end,
                                    route, budget=(1, 1), frame_route=None):
    """Sign even dishonest labels, keeping all generic HMAC/schema checks valid."""
    key = b"test-key"
    archive_path = tmp_path / "archive.pt"
    pose_path = tmp_path / "poses.pt"
    source = torch.arange(8 * 178 * 3, dtype=torch.float32).reshape(8, 178, 3)
    pose = source[source_start:source_end].clone()
    torch.save({"a": source}, archive_path)
    torch.save({"q": pose}, pose_path)
    total = len(pose)
    frame_route = frame_route or ("whole_clip" if route == "replay" else "local_unit")
    ledger = {
        "route": route, "T": total,
        "frame_counts": {name: total if name == frame_route else 0 for name in
                         ("whole_clip", "local_unit", "generated", "unknown_or_derived")},
        "claimed_source_ids": ["a"],
        "segments": [{"route": frame_route, "start": 0, "end": total,
                      "source_id": "a", "source_start": source_start,
                      "source_end": source_end}],
        "fractional_source_mass": {"a": float(total)},
    }
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    commitment = {"algorithm": "sha256", "digest": digest}
    archive_id = f"archive.pt:{digest[:16]}"
    policy = _policy(budget)
    detector = _detector()
    cert = issue_certificate(
        certificate_id="q", clip_id="q", pose=pose, ledger=ledger,
        archive_commitment=commitment, archive_id=archive_id,
        policy=policy, detector_config=detector,
        watermark_key_id=WATERMARK_KEY_ID,
        manifest_key_id=MANIFEST_KEY_ID, manifest_key=key,
    )
    assert verify_certificate(cert, pose=pose, manifest_key=key)
    body = {
        "schema_version": SCHEMA_VERSION, "archive_commitment": commitment,
        "archive_id": archive_id, "policy": policy, "detector_config": detector,
        "manifest_key_id": MANIFEST_KEY_ID,
        "pose_bank_sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
        "certificate_ids": ["q"], "certificates": {"q": cert},
        "issuance_inputs": {},
    }
    return {"pose_bank_path": pose_path, "archive_path": archive_path,
            "manifest_key": key, "manifest": {**body, "release_mac": _release_mac(key, body)}}


def _resign_release(release):
    """Use the authorized key to authenticate the changed payload verbatim."""
    manifest = release["manifest"]
    for cert in manifest["certificates"].values():
        body = {k: v for k, v in cert.items() if k != "manifest_mac"}
        cert["manifest_mac"] = _release_mac(release["manifest_key"], body)
    body = {k: v for k, v in manifest.items() if k != "release_mac"}
    manifest["release_mac"] = _release_mac(release["manifest_key"], body)


@pytest.mark.parametrize("component,field,value", [
    ("detector_config", "blend", "linear"),
    ("detector_config", "whole_clip", "approximate"),
    ("detector_config", "local_unit", "untraced"),
    ("detector_config", "generated", "active"),
    ("detector_config", "blend_width", True),
    ("detector_config", "blend_width", 2.0),
    ("detector_config", "blend_width", "2"),
    ("detector_config", "blend_width", -1),
    ("policy", "route", "arbitrary"),
    ("policy", "selection", "selected after evaluation"),
    ("archive_commitment", "algorithm", "sha512"),
])
def test_release_rejects_resigned_unsupported_schema_declarations(tmp_path, component,
                                                                field, value):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=4, route="fallback",
    )
    release["manifest"][component][field] = value
    release["manifest"]["certificates"]["q"][component] = copy.deepcopy(
        release["manifest"][component]
    )
    _resign_release(release)
    with pytest.raises(CertificateError):
        verify_release(**release)


@pytest.mark.parametrize("field,value", [
    ("numerator", True), ("numerator", "1"), ("numerator", 1.9),
    ("denominator", True), ("denominator", "1"), ("denominator", 1.9),
    ("unit", "whole_clips_per_clip"),
])
def test_release_rejects_resigned_budget_coercion_and_wrong_unit(tmp_path, field, value):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=4, route="fallback",
    )
    release["manifest"]["policy"]["whole_clip_replay_budget"][field] = value
    release["manifest"]["certificates"]["q"]["policy"] = copy.deepcopy(
        release["manifest"]["policy"]
    )
    _resign_release(release)
    with pytest.raises(CertificateError):
        verify_release(**release)


@pytest.mark.parametrize("component,path", [
    ("detector_config", ("blend_width",)),
    ("policy", ("whole_clip_replay_budget", "numerator")),
])
def test_release_rejects_certificate_only_numeric_type_alias(tmp_path, component, path):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=4, route="fallback",
    )
    cert = release["manifest"]["certificates"]["q"]
    cert[component] = copy.deepcopy(release["manifest"][component])
    target = cert[component]
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = float(target[path[-1]])
    assert cert[component] == release["manifest"][component]
    _resign_release(release)
    with pytest.raises(CertificateError, match="exact integer"):
        verify_release(**release)


@pytest.mark.parametrize("component", ["detector_config", "policy", "budget", "archive_commitment"])
@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_release_rejects_resigned_missing_or_extra_configuration_fields(tmp_path,
                                                                      component, operation):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=4, route="fallback",
    )
    manifest = release["manifest"]
    target = (manifest["policy"]["whole_clip_replay_budget"]
              if component == "budget" else manifest[component])
    if operation == "missing":
        target.pop(next(iter(target)))
    else:
        target["unimplemented_option"] = "new_rule"
    for field in ("detector_config", "policy", "archive_commitment"):
        manifest["certificates"]["q"][field] = copy.deepcopy(manifest[field])
    _resign_release(release)
    with pytest.raises(CertificateError):
        verify_release(**release)


@pytest.mark.parametrize("path,value", [
    (("T",), True),
    (("frame_counts", "local_unit"), True),
    (("frame_counts", "whole_clip"), False),
    (("segments", 0, "start"), False),
    (("segments", 0, "end"), True),
    (("segments", 0, "source_start"), False),
    (("segments", 0, "source_end"), True),
])
def test_release_rejects_resigned_boolean_ledger_integers(tmp_path, path, value):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=1, route="fallback",
    )
    cert = release["manifest"]["certificates"]["q"]
    target = cert["ledger"]
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    _resign_release(release)
    pose = torch.load(release["pose_bank_path"], weights_only=True)["q"]
    assert not verify_certificate(cert, pose=pose, manifest_key=release["manifest_key"])
    with pytest.raises(CertificateError, match="certificate authentication failed"):
        verify_release(**release)


@pytest.mark.parametrize("value", [True, "1.0"])
def test_release_rejects_resigned_nonnumeric_source_mass(tmp_path, value):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=1, route="fallback",
    )
    release["manifest"]["certificates"]["q"]["ledger"]["fractional_source_mass"]["a"] = value
    _resign_release(release)
    with pytest.raises(CertificateError, match="fractional source ledger failed"):
        verify_release(**release)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 10 ** 1000])
def test_source_mass_comparison_rejects_nonfinite_or_unrepresentable_numbers(value):
    assert not _close_mass({"a": 1.0}, {"a": value})


@pytest.mark.parametrize("scope", ["release", "certificate", "both"])
def test_release_rejects_resigned_wrong_manifest_key_version(tmp_path, scope):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=4, route="fallback",
    )
    if scope in ("release", "both"):
        release["manifest"]["manifest_key_id"] = "other-key-version"
    if scope in ("certificate", "both"):
        release["manifest"]["certificates"]["q"]["manifest_key_id"] = "other-key-version"
    _resign_release(release)
    with pytest.raises(CertificateError, match="key identifier"):
        verify_release(**release)


@pytest.mark.parametrize("replay_frames,total_frames", [(True, 100), (40, 100.0), ("40", 100)])
def test_replay_budget_rejects_noninteger_measured_totals(replay_frames, total_frames):
    with pytest.raises(CertificateError, match="exact integers"):
        validate_replay_budget(_policy((2, 5)), replay_frames, total_frames)


@pytest.mark.parametrize("source_start,source_end,route", [
    (0, 8, "replay"), (0, 4, "fallback"), (2, 6, "fallback"),
])
def test_release_accepts_structurally_correct_route_class(tmp_path, source_start,
                                                         source_end, route):
    result = verify_release(**_signed_single_interval_release(
        tmp_path, source_start=source_start, source_end=source_end, route=route,
    ))
    total = source_end - source_start
    assert result["replay_frames"] == (total if route == "replay" else 0)
    assert result["fallback_frames"] == (total if route == "fallback" else 0)
    assert result["total_frames"] == total


def test_release_accepts_supported_nonlearned_policy_route(tmp_path):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=4, route="fallback",
    )
    release["manifest"]["policy"] = _policy(
        (1, 1), route="clean_rerank_frame40"
    )
    release["manifest"]["certificates"]["q"]["policy"] = copy.deepcopy(
        release["manifest"]["policy"]
    )
    _resign_release(release)
    result = verify_release(**release)
    assert result["verdict"] == "PASS"


@pytest.mark.parametrize("source_start,source_end,route", [
    (0, 8, "fallback"), (0, 4, "replay"), (2, 6, "replay"),
])
def test_release_rejects_resigned_route_class_mismatch(tmp_path, source_start,
                                                       source_end, route):
    release = _signed_single_interval_release(
        tmp_path, source_start=source_start, source_end=source_end, route=route,
        budget=(0, 1) if route == "fallback" else (1, 1),
    )
    with pytest.raises(CertificateError, match="archive-derived recipe class"):
        verify_release(**release)


def test_release_rejects_resigned_frame_labels_disagreeing_with_derived_replay(tmp_path):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=8, route="replay", frame_route="local_unit",
    )
    with pytest.raises(CertificateError, match="route frame labels differ"):
        verify_release(**release)


def test_release_charges_full_archive_interval_to_replay_budget(tmp_path):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=8, route="replay", budget=(2, 5),
    )
    with pytest.raises(CertificateError, match="exceeds.*replay-frame budget"):
        verify_release(**release)


def test_release_rejects_resigned_certificate_with_different_archive_commitment(tmp_path):
    key = b"test-key"
    archive_path = tmp_path / "archive.pt"
    pose_path = tmp_path / "poses.pt"
    archive = {
        "a": torch.arange(5 * 178 * 3, dtype=torch.float32).reshape(5, 178, 3),
        "b": torch.arange(4 * 178 * 3, dtype=torch.float32).reshape(4, 178, 3) + 100,
    }
    torch.save(archive, archive_path)
    ledger = _fallback_ledger()
    pose = reconstruct_from_ledger(ledger, archive, 2)
    torch.save({"q": pose}, pose_path)
    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_commitment = {"algorithm": "sha256", "digest": archive_digest}
    archive_id = f"archive.pt:{archive_digest[:16]}"
    policy = _policy((2, 5))
    detector = _detector()
    wrong_commitment = {"algorithm": "sha256", "digest": "0" * 64}
    cert = issue_certificate(
        certificate_id="q", clip_id="q", pose=pose, ledger=ledger,
        archive_commitment=wrong_commitment, archive_id="wrong",
        policy=policy, detector_config=detector,
        watermark_key_id=WATERMARK_KEY_ID,
        manifest_key_id=MANIFEST_KEY_ID, manifest_key=key,
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "archive_commitment": archive_commitment,
        "archive_id": archive_id,
        "policy": policy,
        "detector_config": detector,
        "manifest_key_id": MANIFEST_KEY_ID,
        "pose_bank_sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
        "certificate_ids": ["q"],
        "certificates": {"q": cert},
        "issuance_inputs": {},
    }
    manifest = {**body, "release_mac": _release_mac(key, body)}
    with pytest.raises(CertificateError, match="certificate archive commitment differs"):
        verify_release(
            pose_bank_path=pose_path,
            manifest=manifest,
            archive_path=archive_path,
            manifest_key=key,
        )


def _signed_withdrawal_release(tmp_path, withdrawn):
    """Sign a two-source local-assembly release that declares a withdrawn set."""
    key = b"test-key"
    archive_path = tmp_path / "archive.pt"
    pose_path = tmp_path / "poses.pt"
    archive = {
        "a": torch.arange(5 * 178 * 3, dtype=torch.float32).reshape(5, 178, 3),
        "b": torch.arange(4 * 178 * 3, dtype=torch.float32).reshape(4, 178, 3) + 100,
    }
    torch.save(archive, archive_path)
    ledger = _fallback_ledger()
    pose = reconstruct_from_ledger(ledger, archive, 2)
    torch.save({"q": pose}, pose_path)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    commitment = {"algorithm": "sha256", "digest": digest}
    archive_id = f"archive.pt:{digest[:16]}"
    policy = {**_policy(), "withdrawal": withdrawal_commitment(withdrawn)}
    detector = _detector()
    cert = issue_certificate(
        certificate_id="q", clip_id="q", pose=pose, ledger=ledger,
        archive_commitment=commitment, archive_id=archive_id,
        policy=policy, detector_config=detector,
        watermark_key_id=WATERMARK_KEY_ID,
        manifest_key_id=MANIFEST_KEY_ID, manifest_key=key,
    )
    body = {
        "schema_version": WITHDRAWAL_SCHEMA_VERSION,
        "archive_commitment": commitment, "archive_id": archive_id,
        "policy": policy, "detector_config": detector,
        "manifest_key_id": MANIFEST_KEY_ID,
        "pose_bank_sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
        "certificate_ids": ["q"], "certificates": {"q": cert},
        "issuance_inputs": {},
        "withdrawn_sources": canonical_withdrawn_sources(withdrawn),
    }
    return {"pose_bank_path": pose_path, "archive_path": archive_path,
            "manifest_key": key,
            "manifest": {**body, "release_mac": _release_mac(key, body)}}


@pytest.mark.parametrize("withdrawn", [[], ["z"], ["y", "z"]])
def test_release_verifies_a_declared_withdrawal_with_zero_exposure(tmp_path, withdrawn):
    release = _signed_withdrawal_release(tmp_path, withdrawn)
    report = verify_release(**release)
    block = report["withdrawn_source_set"]
    assert report["verdict"] == "PASS"
    assert block["declared"] is True and block["verified"] is True
    assert block["withdrawn_source_count"] == len(withdrawn)
    assert block["emitted_positions_checked"] == 8
    assert block["emitted_positions_drawing_on_withdrawn_source"] == 0
    assert block["withdrawn_source_mass_in_release"] == 0.0
    assert block["withdrawn_sources_present_in_committed_archive"] == 0


@pytest.mark.parametrize("withdrawn,position", [(["a"], 0), (["b"], 2)])
def test_release_rejects_a_release_drawing_on_a_declared_withdrawn_source(
        tmp_path, withdrawn, position):
    """A correctly signed release that still uses a withdrawn source fails.

    Source ``b`` starts at emitted position 4 but the cosine blend makes
    positions 2 and 3 draw on it as well, so the first exposed position is 2.
    """
    release = _signed_withdrawal_release(tmp_path, withdrawn)
    with pytest.raises(CertificateError,
                       match=f"emitted position {position} of q draws on a withdrawn source"):
        verify_release(**release)


def test_release_rejects_withdrawn_set_swapped_after_signing(tmp_path):
    release = _signed_withdrawal_release(tmp_path, ["z"])
    manifest = release["manifest"]
    manifest["withdrawn_sources"] = ["y"]
    body = {k: v for k, v in manifest.items() if k != "release_mac"}
    manifest["release_mac"] = _release_mac(release["manifest_key"], body)
    with pytest.raises(CertificateError, match="does not match the authenticated commitment"):
        verify_release(**release)


def test_release_rejects_withdrawn_set_reordered_after_signing(tmp_path):
    release = _signed_withdrawal_release(tmp_path, ["y", "z"])
    release["manifest"]["withdrawn_sources"] = ["z", "y"]
    _resign_release(release)
    with pytest.raises(CertificateError, match="not in canonical order"):
        verify_release(**release)


def test_release_rejects_dropped_withdrawal_declaration(tmp_path):
    release = _signed_withdrawal_release(tmp_path, ["z"])
    release["manifest"].pop("withdrawn_sources")
    _resign_release(release)
    with pytest.raises(CertificateError, match="requires a withdrawn-source declaration"):
        verify_release(**release)


def test_membership_tamper_drill_reaches_entry_guard_after_resigning(tmp_path):
    release = _signed_withdrawal_release(tmp_path, ["z"])
    drills = tamper_drills(
        pose_bank_path=release["pose_bank_path"],
        manifest=release["manifest"],
        archive_path=release["archive_path"],
        manifest_key=release["manifest_key"],
    )
    assert drills["release_membership_edit_caught"] is True


def test_release_rejects_withdrawal_declaration_under_the_v1_schema(tmp_path):
    release = _signed_withdrawal_release(tmp_path, ["z"])
    release["manifest"]["schema_version"] = SCHEMA_VERSION
    _resign_release(release)
    with pytest.raises(CertificateError, match="release policy fields differ"):
        verify_release(**release)


def test_release_rejects_unpaired_withdrawn_list_under_the_v1_schema(tmp_path):
    release = _signed_single_interval_release(
        tmp_path, source_start=0, source_end=4, route="fallback",
    )
    release["manifest"]["withdrawn_sources"] = ["z"]
    _resign_release(release)
    with pytest.raises(CertificateError, match="unsupported by this release schema"):
        verify_release(**release)


@pytest.mark.parametrize("value", [
    [True], [1], [1.0], [""], ["a", "a"], {"a": 1}, "a", None,
])
def test_withdrawn_source_set_rejects_coerced_or_duplicate_ids(value):
    with pytest.raises(CertificateError):
        canonical_withdrawn_sources(value)


@pytest.mark.parametrize("field,value", [
    ("count", True), ("count", 1.0), ("count", "1"), ("count", -1),
    ("digest", "0" * 63), ("digest", "z" * 64), ("digest", 0),
    ("algorithm", "sha512"), ("schema", "withdrawn-source-set-v0"),
])
def test_withdrawal_commitment_rejects_coerced_fields(field, value):
    commitment = withdrawal_commitment(["z"])
    commitment[field] = value
    with pytest.raises(CertificateError):
        validate_withdrawal_commitment(commitment)


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("count"),
    lambda c: c.__setitem__("extra", 1),
])
def test_withdrawal_commitment_rejects_missing_or_extra_fields(mutate):
    commitment = withdrawal_commitment(["z"])
    mutate(commitment)
    with pytest.raises(CertificateError, match="fields differ from the clean schema"):
        validate_withdrawal_commitment(commitment)
