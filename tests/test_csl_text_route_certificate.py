import copy

import numpy as np
import pytest

from budget_sage.audit.certificate import CertificateError
from scripts.csl_text_route_certificate import (
    issue as _issue, verify as _verify, mac, converted_ledger, reconstruct, path_inside,
    POLICY, CONFIG,
)
from scripts.build_csl_pg_rast_native_hrnet import resample

KEY = b"public-test-key"
DIGESTS = ("1"*64, "2"*64, "3"*64)
COMMITMENTS = {"protocol_sha256": "4"*64, "archive_admission_sha256": "5"*64,
               "prescore_sha256": "6"*64, "allocation_protocol_sha256": "7"*64}


def issue(*args, **kwargs):
    return _issue(*args, construction_commitments=COMMITMENTS, **kwargs)


def verify(*args, **kwargs):
    return _verify(*args, construction_commitments=COMMITMENTS, **kwargs)


def release(local_length=6):
    archive = {"a": np.arange(4*133*3, dtype=np.float32).reshape(4, 133, 3),
               "b": np.arange(8*133*3, dtype=np.float32).reshape(8, 133, 3)}
    rows = {
        "q_local": {"whole_replay": False, "recipe": [{"source_id": "b", "start": 1, "end": 5, "out_len": local_length}],
                    "source_mass": {"b": local_length}},
        "q_replay": {"whole_replay": True, "recipe": [{"source_id": "a", "start": 0, "end": 4, "out_len": 4}],
                     "source_mass": {"a": 4}},
    }
    poses = {"q_local": resample(archive["b"][1:5], local_length), "q_replay": archive["a"].copy()}
    return poses, rows, archive


def issue_fixture():
    poses, rows, archive = release()
    manifest = issue(poses, rows, archive, set(archive), *DIGESTS, KEY)
    return poses, manifest, archive


def resign(manifest):
    for cert in manifest["certificates"].values():
        cert["manifest_mac"] = mac(KEY, {k: v for k, v in cert.items() if k != "manifest_mac"})
    manifest["release_mac"] = mac(KEY, {k: v for k, v in manifest.items() if k != "release_mac"})


def test_independent_native_resampling_and_exact_budget_roundtrip():
    poses, manifest, archive = issue_fixture()
    result = verify(poses, manifest, archive, set(archive), *DIGESTS, KEY)
    assert result["entries"] == 2 and result["exact_reconstructions"] == 2
    assert result["integer_budget"] == "20 <= 20"
    assert result["replay_frames"] == 4 and result["fallback_frames"] == 6
    assert result["within_source_interpolated_positions"] == 4


def test_valid_but_over_budget_issuance_rejected():
    poses, rows, archive = release(local_length=5)
    with pytest.raises(CertificateError, match="exceeds exact replay"):
        issue(poses, rows, archive, set(archive), *DIGESTS, KEY)


@pytest.mark.parametrize("component,field,value", [
    ("policy", "route", "unimplemented"),
    ("policy", "selection", "claiming fresh test"),
    ("policy", "budget", {"numerator": 2., "denominator": 5, "unit": "whole_clip_replay_frames_per_emitted_frame"}),
    ("policy", "budget", {"numerator": True, "denominator": 5, "unit": "whole_clip_replay_frames_per_emitted_frame"}),
    ("detector_config", "interpolation", "nearest"),
    ("detector_config", "join", "cosine"),
    ("detector_config", "representation", "float16"),
    ("archive_commitment", "algorithm", "sha512"),
])
def test_resigned_unsupported_schema_rejected(component, field, value):
    poses, original, archive = issue_fixture()
    manifest = copy.deepcopy(original)
    manifest[component][field] = value
    for cert in manifest["certificates"].values():
        cert[component][field] = value
    resign(manifest)
    with pytest.raises(CertificateError):
        verify(poses, manifest, archive, set(archive), *DIGESTS, KEY)


@pytest.mark.parametrize("field,value", [("source_start", True), ("source_start", 1.),
                                         ("source_end", 999), ("end", 6.), ("start", False)])
def test_resigned_invalid_intervals_rejected(field, value):
    poses, manifest, archive = issue_fixture()
    manifest["certificates"]["q_local"]["ledger"]["segments"][0][field] = value
    resign(manifest)
    with pytest.raises(CertificateError):
        verify(poses, manifest, archive, set(archive), *DIGESTS, KEY)


def test_resigned_whole_clip_relabel_is_rejected():
    poses, manifest, archive = issue_fixture()
    ledger = manifest["certificates"]["q_replay"]["ledger"]
    ledger["route"] = "fallback"
    ledger["segments"][0]["route"] = "local_unit"
    ledger["frame_counts"].update(whole_clip=0, local_unit=4)
    resign(manifest)
    with pytest.raises(CertificateError, match="derived archive extents"):
        verify(poses, manifest, archive, set(archive), *DIGESTS, KEY)


@pytest.mark.parametrize("mass", [6., True, "6", float("nan")])
def test_noninteger_source_masses_rejected(mass):
    _, rows, archive = release()
    ledger = converted_ledger(rows["q_local"])
    ledger["source_mass"]["b"] = mass
    with pytest.raises(CertificateError):
        reconstruct(ledger, archive, set(archive))


def test_pose_membership_key_and_source_admission_tampering_rejected():
    poses, manifest, archive = issue_fixture()
    changed = copy.deepcopy(poses)
    changed["q_local"][0, 0, 0] += 1
    for p, allowed, key in ((changed, set(archive), KEY), (poses, {"a"}, KEY),
                            (poses, set(archive), b"wrong"), ({"q_replay": poses["q_replay"]}, set(archive), KEY)):
        with pytest.raises(CertificateError):
            verify(p, manifest, archive, allowed, *DIGESTS, key)


def test_verifier_source_array_tamper_fails_reconstruction_even_with_same_hash_argument():
    poses, manifest, archive = issue_fixture()
    changed = copy.deepcopy(archive)
    changed["b"][2, 0, 0] += 1
    with pytest.raises(CertificateError, match="reconstruction"):
        verify(poses, manifest, changed, set(archive), *DIGESTS, KEY)


@pytest.mark.parametrize("path", ["/tmp/csl-certificate-escape", "../outside-csl"])
def test_cli_paths_cannot_escape_project(path):
    with pytest.raises(CertificateError, match="inside the project"):
        path_inside(path)


@pytest.mark.parametrize("field", sorted(COMMITMENTS))
def test_resigned_construction_commitment_tamper_rejected(field):
    poses, manifest, archive = issue_fixture()
    manifest["construction_commitments"][field] = "9"*64
    resign(manifest)
    with pytest.raises(CertificateError, match="construction commitments"):
        verify(poses, manifest, archive, set(archive), *DIGESTS, KEY)


def test_native_schema_has_no_watermark_claim():
    poses, manifest, archive = issue_fixture()
    assert all("watermark" not in cert for cert in manifest["certificates"].values())
    manifest["certificates"]["q_local"]["watermark"] = {"detector": "invented"}
    resign(manifest)
    with pytest.raises(CertificateError, match="no watermark"):
        verify(poses, manifest, archive, set(archive), *DIGESTS, KEY)


def test_issued_payload_has_no_alias_to_closed_constants_or_issuer_inputs():
    poses, rows, archive = release()
    expected_policy, expected_config = copy.deepcopy(POLICY), copy.deepcopy(CONFIG)
    expected_rows, expected_commitments = copy.deepcopy(rows), copy.deepcopy(COMMITMENTS)
    manifest = issue(poses, rows, archive, set(archive), *DIGESTS, KEY)
    manifest["policy"]["budget"]["numerator"] = 99
    manifest["detector_config"]["join"] = "invented"
    manifest["construction_commitments"]["protocol_sha256"] = "9"*64
    manifest["certificates"]["q_local"]["ledger"]["source_mass"]["b"] = 99
    assert POLICY == expected_policy and CONFIG == expected_config
    assert rows == expected_rows and COMMITMENTS == expected_commitments
    resign(manifest)
    with pytest.raises(CertificateError):
        verify(poses, manifest, archive, set(archive), *DIGESTS, KEY)


def test_explicit_reduced_policy_supports_same_reconstruction_contract():
    poses, rows, archive = release()
    manifest = issue(poses, rows, archive, set(archive), *DIGESTS, KEY,
                     policy_route="csl_text_native_reduced_gbdt40_v1")
    assert verify(poses, manifest, archive, set(archive), *DIGESTS, KEY)["verdict"] == "PASS"
