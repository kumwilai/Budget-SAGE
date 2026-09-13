"""Closed cooperative certificate for native CSL interval-resampled pose releases.

This schema is separate from the PHOENIX cosine-join certificate. The auditor
reads committed archive/manifest/poses and pre-issued certificates, not route logs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import pickle
import resource
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from budget_sage.audit.certificate import (
    CertificateError, canonical_json, pose_descriptor, sha256_file, validate_ledger,
)

SCHEMA = "csl-native-resampled-source-release-v1"
KEY_ID = "authorized-csl-auditor-hmac-v1"
ITEM_SCHEMA = "csl-native-source-item-v1"
CONFIG = {"representation": "float32_hrnet133_xyconfidence",
          "interpolation": "numpy_endpoint_linear_float32_v1",
          "join": "concatenation", "route_class": "archive_extent_strict_subinterval_v1"}
POLICY = {"route": "csl_text_native_fixed40_v4",
          "selection": "post-test feasibility; committed v4 fixed ranking",
          "budget": {"numerator": 2, "denominator": 5,
                     "unit": "whole_clip_replay_frames_per_emitted_frame"}}
POLICIES = {POLICY["route"]: POLICY,
            "csl_text_native_reduced_gbdt40_v1": {
                "route": "csl_text_native_reduced_gbdt40_v1",
                "selection": "post-test reduced dev-supervised similarity; see committed allocation protocol",
                "budget": POLICY["budget"]}}
COMMITMENT_FIELDS = {"protocol_sha256", "archive_admission_sha256", "prescore_sha256", "allocation_protocol_sha256"}


def strict_equal(actual, expected, label):
    if canonical_json(actual) != canonical_json(expected):
        raise CertificateError(f"unsupported or inconsistent {label}")


def mac(key, value):
    if not isinstance(key, bytes) or not key:
        raise CertificateError("nonempty HMAC key required")
    return hmac.new(key, canonical_json(value), hashlib.sha256).hexdigest()


def validate_commitments(value):
    if not isinstance(value, dict) or set(value) != COMMITMENT_FIELDS:
        raise CertificateError("construction commitment fields differ")
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in value.values()):
        raise CertificateError("construction commitments require SHA-256 hexadecimal digests")


def path_inside(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT/path).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise CertificateError("path must be inside the project root")
    return path


def pose_array(value, *, archive=False):
    if isinstance(value, dict):
        value = value["keypoint"]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if not archive and arr.dtype != np.dtype("float32"):
        raise CertificateError("released pose dtype must be float32")
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 3 or arr.shape[1:] != (133, 3) or len(arr) < 1 or not np.isfinite(arr).all():
        raise CertificateError("pose must be finite nonempty [T,133,3]")
    return arr


def reconstruct(ledger, archive, admitted_ids):
    """Independently implement the declared endpoint-linear source recipe."""
    if not isinstance(ledger, dict) or set(ledger) != {
        "route", "T", "frame_counts", "claimed_source_ids", "segments", "source_mass"
    }:
        raise CertificateError("unsupported CSL ledger fields")
    validate_ledger(ledger, ledger["T"])
    segments = ledger["segments"]
    if not segments:
        raise CertificateError("empty recipe")
    pieces, masses, full_flags = [], Counter(), []
    interpolated_positions = 0
    for segment in segments:
        if set(segment) != {"route", "start", "end", "source_id", "source_start", "source_end"}:
            raise CertificateError("unsupported CSL segment fields")
        sid = segment["source_id"]
        if sid not in admitted_ids or sid not in archive:
            raise CertificateError("source absent from committed training admission")
        source = pose_array(archive[sid], archive=True)
        start, end = segment["source_start"], segment["source_end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source):
            raise CertificateError("invalid native source interval")
        length = segment["end"] - segment["start"]
        raw = source[start:end]
        full_flags.append(start == 0 and end == len(source))
        if len(raw) == length:
            value = raw.copy()
        elif len(raw) == 1:
            value = np.repeat(raw, length, axis=0)
        else:
            coordinate = np.linspace(0, len(raw)-1, length)
            lower = np.floor(coordinate).astype(np.int64)
            upper = np.minimum(lower+1, len(raw)-1)
            weight = (coordinate-lower).astype(np.float32)[:, None, None]
            value = ((1.-weight)*raw[lower] + weight*raw[upper]).astype(np.float32)
            interpolated_positions += int(((weight > 0) & (weight < 1)).sum())
        pieces.append(value)
        masses[sid] += length
    derived = "replay" if len(segments) == 1 and full_flags[0] and len(pieces[0]) == len(pose_array(archive[segments[0]["source_id"]], archive=True)) else "fallback"
    if derived == "fallback" and any(full_flags):
        raise CertificateError("local recipe must use strict source subintervals")
    if ledger["route"] != derived:
        raise CertificateError("route label differs from derived archive extents")
    segment_route = "whole_clip" if derived == "replay" else "local_unit"
    expected_counts = {name: ledger["T"] if name == segment_route else 0
                       for name in ("whole_clip", "local_unit", "generated", "unknown_or_derived")}
    strict_equal(ledger["frame_counts"], expected_counts, "derived route counts")
    if any(segment["route"] != segment_route for segment in segments):
        raise CertificateError("segment class differs from derived route")
    strict_equal(ledger["source_mass"], dict(masses), "integer source mass")
    result = np.concatenate(pieces).astype(np.float32)
    if len(result) != ledger["T"] or sum(masses.values()) != len(result):
        raise CertificateError("source mass or tensor length mismatch")
    return result, derived, masses, interpolated_positions


def converted_ledger(row):
    if type(row.get("whole_replay")) is not bool:
        raise CertificateError("issuer route declaration must be Boolean")
    replay, cursor = row["whole_replay"], 0
    segments = []
    for unit in row["recipe"]:
        length = unit["out_len"]
        if type(length) is not int or length < 1:
            raise CertificateError("issuer output length must be positive exact integer")
        segments.append({"route": "whole_clip" if replay else "local_unit",
                         "start": cursor, "end": cursor+length,
                         "source_id": unit["source_id"],
                         "source_start": unit["start"], "source_end": unit["end"]})
        cursor += length
    kind = "whole_clip" if replay else "local_unit"
    return {"route": "replay" if replay else "fallback", "T": cursor,
            "frame_counts": {k: cursor if k == kind else 0 for k in
                             ("whole_clip", "local_unit", "generated", "unknown_or_derived")},
            "claimed_source_ids": sorted({s["source_id"] for s in segments}),
            "segments": segments, "source_mass": row["source_mass"]}


def issue(poses, rows, archive, admitted_ids, archive_digest, train_digest, bank_digest, key,
          *, construction_commitments, policy_route=POLICY["route"]):
    validate_commitments(construction_commitments)
    if policy_route not in POLICIES:
        raise CertificateError("unsupported allocation policy")
    policy = POLICIES[policy_route]
    if set(poses) != set(rows) or set(poses) & admitted_ids:
        raise CertificateError("release membership differs or overlaps training")
    commitment = {"algorithm": "sha256", "digest": archive_digest}
    archive_id = f"csl-native:{archive_digest[:16]}"
    certs = {}
    for sid, value in poses.items():
        pose = pose_array(value)
        ledger = converted_ledger(rows[sid])
        rebuilt, _, _, _ = reconstruct(ledger, archive, admitted_ids)
        if not np.array_equal(pose, rebuilt):
            raise CertificateError(f"issuer reconstruction mismatch: {sid}")
        item = {"schema_version": ITEM_SCHEMA, "certificate_id": sid, "clip_id": sid,
                "pose": pose_descriptor(pose), "ledger": ledger,
                "archive_commitment": commitment, "archive_id": archive_id,
                "policy": policy, "detector_config": CONFIG, "manifest_key_id": KEY_ID}
        certs[sid] = {**item, "manifest_mac": mac(key, item)}
    body = {"schema_version": SCHEMA, "policy": policy, "detector_config": CONFIG,
            "archive_commitment": commitment, "archive_id": archive_id,
            "training_manifest_sha256": train_digest, "manifest_key_id": KEY_ID,
            "pose_bank_sha256": bank_digest, "certificate_ids": list(poses), "certificates": certs,
            "construction_commitments": construction_commitments}
    manifest = {**body, "release_mac": mac(key, body)}
    # Issuance also rejects an over-budget release, without altering recipes.
    verify(poses, manifest, archive, admitted_ids, archive_digest, train_digest, bank_digest, key,
           construction_commitments=construction_commitments)
    # A returned certificate must not share mutable claims with the caller's
    # expected commitments, ledger rows, or the closed schema constants.
    return copy.deepcopy(manifest)


def verify(poses, manifest, archive, admitted_ids, archive_digest, train_digest, bank_digest, key,
           *, construction_commitments):
    fields = {"schema_version", "policy", "detector_config", "archive_commitment", "archive_id",
              "training_manifest_sha256", "manifest_key_id", "pose_bank_sha256", "certificate_ids", "certificates", "release_mac", "construction_commitments"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise CertificateError("unsupported release schema fields")
    body = {k: v for k, v in manifest.items() if k != "release_mac"}
    if not hmac.compare_digest(str(manifest["release_mac"]), mac(key, body)):
        raise CertificateError("release HMAC failed")
    strict_equal(manifest["schema_version"], SCHEMA, "schema")
    policy = manifest["policy"]
    if not isinstance(policy, dict) or policy.get("route") not in POLICIES:
        raise CertificateError("unsupported allocation policy")
    strict_equal(policy, POLICIES[policy["route"]], "policy")
    validate_commitments(construction_commitments)
    validate_commitments(manifest["construction_commitments"])
    strict_equal(manifest["construction_commitments"], construction_commitments, "construction commitments")
    strict_equal(manifest["detector_config"], CONFIG, "reconstruction configuration")
    strict_equal(manifest["archive_commitment"], {"algorithm": "sha256", "digest": archive_digest}, "archive commitment")
    strict_equal(manifest["archive_id"], f"csl-native:{archive_digest[:16]}", "archive identifier")
    strict_equal(manifest["training_manifest_sha256"], train_digest, "training admission hash")
    strict_equal(manifest["manifest_key_id"], KEY_ID, "key identifier")
    strict_equal(manifest["pose_bank_sha256"], bank_digest, "pose-bank commitment")
    if any(not isinstance(sid, str) or not sid for sid in poses) or not isinstance(manifest["certificates"], dict) or manifest["certificate_ids"] != list(poses) or set(manifest["certificates"]) != set(poses) or set(poses) & admitted_ids or not poses:
        raise CertificateError("release membership/order differs or overlaps training")
    counts, sources, fractional = Counter(), set(), 0
    for sid, value in poses.items():
        pose = pose_array(value)
        cert = manifest["certificates"][sid]
        item_fields = {"schema_version", "certificate_id", "clip_id", "pose", "ledger", "archive_commitment", "archive_id", "policy", "detector_config", "manifest_key_id", "manifest_mac"}
        if not isinstance(cert, dict) or set(cert) != item_fields:
            raise CertificateError("unsupported CSL item fields; no watermark metadata is supported")
        item = {k: v for k, v in cert.items() if k != "manifest_mac"}
        if cert["schema_version"] != ITEM_SCHEMA or not hmac.compare_digest(str(cert["manifest_mac"]), mac(key, item)):
            raise CertificateError(f"item authentication failed: {sid}")
        for field in ("policy", "detector_config", "archive_commitment", "archive_id", "manifest_key_id"):
            strict_equal(cert[field], manifest[field], f"item {field}")
        strict_equal(cert["pose"], pose_descriptor(pose), "pose descriptor")
        if cert["certificate_id"] != sid or cert["clip_id"] != sid:
            raise CertificateError("item identifier differs")
        rebuilt, route, mass, nfrac = reconstruct(cert["ledger"], archive, admitted_ids)
        if not np.array_equal(pose, rebuilt):
            raise CertificateError(f"pose reconstruction failed: {sid}")
        counts[f"{route}_clips"] += 1
        counts[f"{route}_frames"] += len(pose)
        sources.update(mass)
        fractional += nfrac
    replay, total = counts["replay_frames"], counts["replay_frames"]+counts["fallback_frames"]
    if 5*replay > 2*total:
        raise CertificateError("release exceeds exact replay-frame budget")
    return {"verdict": "PASS", "entries": len(poses), "exact_reconstructions": len(poses),
            **counts, "total_frames": total, "unique_training_sources": len(sources),
            "within_source_interpolated_positions": fractional, "untraced_frames": 0,
            "integer_budget": f"{5*replay} <= {2*total}", "issuer_route_records_read": False,
            "archive_sha256": archive_digest, "train_manifest_sha256": train_digest,
            "pose_bank_sha256": bank_digest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("issue", "verify"))
    parser.add_argument("--poses", required=True)
    parser.add_argument("--ledger")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report")
    parser.add_argument("--archive", default="external/baselines/MSKA/data/CSL-Daily/CSL-Daily.train")
    parser.add_argument("--train_manifest", default="data/csl-daily/csl_daily_train.json")
    parser.add_argument("--key_env", default="BUDGET_SAGE_MANIFEST_KEY")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--prescore", required=True)
    parser.add_argument("--allocation_protocol")
    parser.add_argument("--policy_route", choices=sorted(POLICIES), default=POLICY["route"])
    args = parser.parse_args()
    paths = {name: path_inside(getattr(args, name)) for name in
             ("poses", "manifest", "archive", "train_manifest", "protocol", "admission", "prescore")}
    allocation_path = path_inside(args.allocation_protocol) if args.allocation_protocol else paths["protocol"]
    report_path = path_inside(args.report) if args.report else None
    key = os.environ.get(args.key_env, "").encode()
    mac(key, {})
    poses = torch.load(paths["poses"], map_location="cpu", weights_only=True)
    with paths["archive"].open("rb") as handle:
        archive = pickle.load(handle)
    admitted = {r["id"] for r in json.loads(paths["train_manifest"].read_text())}
    digests = [sha256_file(str(paths[k])) for k in ("archive", "train_manifest", "poses")]
    protocol = json.loads(paths["protocol"].read_text())
    admission = json.loads(paths["admission"].read_text())
    prescore = json.loads(paths["prescore"].read_text())
    if protocol.get("version") != "CSL-text-native-fixed40-v4":
        raise CertificateError("expected v4 source-construction protocol")
    strict_equal(protocol.get("budget"), [2, 5], "v4 exact budget")
    if protocol["inputs"]["pose_archive"]["sha256"] != digests[0] or protocol["inputs"]["train_manifest"]["sha256"] != digests[1]:
        raise CertificateError("protocol input commitments differ from supplied archive/admission")
    if prescore.get(paths["poses"].name) != digests[2]:
        raise CertificateError("pose bank absent from committed prescore hashes")
    strict_equal(admission.get("excluded_pose_sources"), sorted(set(archive)-admitted), "archive admission exclusions")
    strict_equal(admission.get("admitted_pose_sources"), len(set(archive)&admitted), "archive admission count")
    commitments = {"protocol_sha256": sha256_file(str(paths["protocol"])),
                   "archive_admission_sha256": sha256_file(str(paths["admission"])),
                   "prescore_sha256": sha256_file(str(paths["prescore"])),
                   "allocation_protocol_sha256": sha256_file(str(allocation_path))}
    if args.mode == "issue":
        if not args.ledger or paths["manifest"].exists():
            raise CertificateError("issuance requires a ledger and a new manifest path")
        rows = json.loads(path_inside(args.ledger).read_text())
        manifest = issue(poses, rows, archive, admitted, *digests, key,
                         construction_commitments=commitments, policy_route=args.policy_route)
        paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
        paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
        print(json.dumps({"issued": len(poses), "manifest_sha256": sha256_file(str(paths["manifest"]))}))
    else:
        if report_path is None:
            raise CertificateError("verification report output required")
        report = verify(poses, json.loads(paths["manifest"].read_text()), archive, admitted, *digests, key,
                        construction_commitments=commitments)
        report.update(implementation_sha256=sha256_file(__file__), helper_sha256=sha256_file(str(ROOT/"budget_sage/audit/certificate.py")),
                      manifest_sha256=sha256_file(str(paths["manifest"])), construction_commitments=commitments,
                      peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
