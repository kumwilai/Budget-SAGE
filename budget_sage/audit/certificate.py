"""Authenticated route certificates.

This module deliberately implements cooperative, shared-secret authentication
(HMAC), not a public signature scheme.  A certificate is issued before an
audit and the verifier only checks that pre-issued object; it never mints a
replacement certificate while checking a route.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "route-certificate-v4"
LEDGER_ROUTES = ("whole_clip", "local_unit", "generated", "unknown_or_derived")


class CertificateError(ValueError):
    """Raised when a certificate or one of its bound inputs is invalid."""


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CertificateError("canonical JSON rejects non-finite floats")
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str):
                raise CertificateError("canonical JSON requires string object keys")
            _finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _finite(v)


def canonical_json(value: Any) -> bytes:
    """Return deterministic JSON bytes, rejecting NaN and +/-Infinity."""
    _finite(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CertificateError(f"value is not canonical JSON: {exc}") from exc


def pose_sha256(pose: Any) -> str:
    """Hash the exact contiguous bytes of a numpy-like or torch pose tensor."""
    if hasattr(pose, "detach"):
        pose = pose.detach().cpu().numpy()
    try:
        raw = pose if isinstance(pose, (bytes, bytearray)) else pose.tobytes(order="C")
    except AttributeError as exc:
        raise CertificateError("pose must provide tobytes()") from exc
    return hashlib.sha256(raw).hexdigest()


def pose_descriptor(pose: Any) -> dict[str, Any]:
    if hasattr(pose, "detach"):
        pose = pose.detach().cpu().numpy()
    return {"dtype": str(pose.dtype), "shape": list(pose.shape),
            "sha256": pose_sha256(pose)}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ledger(ledger: Mapping[str, Any], frame_count: int) -> None:
    if not isinstance(ledger, Mapping):
        raise CertificateError("ledger must be a mapping")
    if type(frame_count) is not int or frame_count < 0:
        raise CertificateError("pose frame count must be a non-negative exact integer")
    counts = ledger.get("frame_counts")
    if not isinstance(counts, Mapping) or set(counts) != set(LEDGER_ROUTES):
        raise CertificateError("ledger must contain exactly four route frame counts")
    if any(type(counts[k]) is not int or counts[k] < 0 for k in LEDGER_ROUTES):
        raise CertificateError("ledger frame counts must be non-negative integers")
    if sum(counts[k] for k in LEDGER_ROUTES) != frame_count:
        raise CertificateError("ledger frame counts must sum exactly to pose length")
    if type(ledger.get("T")) is not int or ledger["T"] != frame_count:
        raise CertificateError("ledger T must equal pose length")
    if ledger.get("route") not in {"replay", "fallback", "generated", "mixed"}:
        raise CertificateError("ledger route is invalid")
    source_ids = ledger.get("claimed_source_ids")
    if (not isinstance(source_ids, list)
            or any(not isinstance(s, str) or not s for s in source_ids)
            or len(source_ids) != len(set(source_ids))):
        raise CertificateError("claimed_source_ids must be unique non-empty strings")
    segments = ledger.get("segments")
    if segments is None:
        return
    if not isinstance(segments, list) or not segments:
        raise CertificateError("ledger segments must be a non-empty list when supplied")
    cursor = 0
    segment_counts = {route: 0 for route in LEDGER_ROUTES}
    segment_sources: list[str] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise CertificateError("each ledger segment must be a mapping")
        route = segment.get("route")
        start, end = segment.get("start"), segment.get("end")
        if route not in LEDGER_ROUTES:
            raise CertificateError("ledger segment route is invalid")
        if type(start) is not int or type(end) is not int:
            raise CertificateError("ledger segment bounds must be integers")
        if start != cursor or end <= start or end > frame_count:
            raise CertificateError("ledger segments must form an ordered exact partition")
        segment_counts[route] += end - start
        cursor = end
        if route in {"whole_clip", "local_unit"}:
            source_id = segment.get("source_id")
            source_start = segment.get("source_start")
            if not isinstance(source_id, str) or not source_id:
                raise CertificateError("source-backed segments require source_id")
            if type(source_start) is not int or source_start < 0:
                raise CertificateError("source-backed segments require non-negative source_start")
            if "source_end" in segment and type(segment["source_end"]) is not int:
                raise CertificateError("source-backed segment endpoints must be exact integers")
            segment_sources.append(source_id)
        elif "source_id" in segment or "source_start" in segment:
            raise CertificateError("generated or unknown segments cannot assert a source")
    if cursor != frame_count or segment_counts != dict(counts):
        raise CertificateError("ledger segments must reproduce the four frame counts")
    if sorted(set(segment_sources)) != sorted(source_ids):
        raise CertificateError("claimed_source_ids must equal segment source ids")


def _key(key: str | bytes, name: str) -> bytes:
    if isinstance(key, str):
        key = key.encode("utf-8")
    if not isinstance(key, bytes) or not key:
        raise CertificateError(f"{name} must be supplied and non-empty")
    return key


def _mac(key: str | bytes, payload: Mapping[str, Any]) -> str:
    return hmac.new(_key(key, "manifest_key"), canonical_json(payload), hashlib.sha256).hexdigest()


def _payload(*, certificate_id: str, clip_id: str, pose: Any,
             ledger: Mapping[str, Any], archive_commitment: Any,
             archive_id: str, policy: Mapping[str, Any],
             detector_config: Mapping[str, Any], manifest_key_id: str,
             watermark_metadata: Mapping[str, Any]) -> dict[str, Any]:
    validate_ledger(ledger, len(pose))
    if not certificate_id or not clip_id or not archive_id:
        raise CertificateError("certificate_id, clip_id, and archive_id are required")
    if not manifest_key_id:
        raise CertificateError("manifest_key_id is required")
    if (not isinstance(archive_commitment, Mapping)
            or archive_commitment.get("algorithm") != "sha256"
            or not isinstance(archive_commitment.get("digest"), str)
            or len(archive_commitment["digest"]) != 64):
        raise CertificateError("archive_commitment must contain a SHA-256 digest")
    # Do not normalize, filter, or rebuild ledger: every supplied nested field
    # is authenticated as supplied.
    return {
        "schema_version": SCHEMA_VERSION,
        "certificate_id": certificate_id,
        "clip_id": clip_id,
        "pose": pose_descriptor(pose),
        "ledger": ledger,
        "archive_commitment": archive_commitment,
        "archive_id": archive_id,
        "policy": policy,
        "detector_config": detector_config,
        "manifest_key_id": manifest_key_id,
        "watermark": watermark_metadata,
    }


def issue_certificate(*, certificate_id: str, clip_id: str, pose: Any,
                      ledger: Mapping[str, Any], archive_commitment: Any,
                      archive_id: str, policy: Mapping[str, Any],
                      detector_config: Mapping[str, Any],
                      watermark_key_id: str, manifest_key_id: str,
                      manifest_key: str | bytes) -> dict[str, Any]:
    """Issue one cooperative HMAC-authenticated certificate.

    The watermark secret is intentionally absent.  The actual pose watermark
    is measured separately by ``pose_watermark.detect``; this object binds
    only the detector metadata and the identifier of that distinct key.
    """
    if not watermark_key_id or watermark_key_id == manifest_key_id:
        raise CertificateError("watermark and manifest key identifiers must be distinct")
    body = _payload(certificate_id=certificate_id, clip_id=clip_id,
                    pose=pose, ledger=ledger,
                    archive_commitment=archive_commitment, archive_id=archive_id,
                    policy=policy, detector_config=detector_config,
                    manifest_key_id=manifest_key_id,
                    watermark_metadata={"detector": "scripts.pose_watermark.detect",
                                        "key_id": watermark_key_id})
    return {**body, "manifest_mac": _mac(manifest_key, body)}


def verify_certificate(certificate: Mapping[str, Any], *, pose: Any,
                       manifest_key: str | bytes,
                       expected_claims: Mapping[str, Any] | None = None) -> bool:
    """Authenticate a pre-issued certificate and optional expected claims.

    HMAC authentication alone establishes integrity relative to a cooperating
    issuer.  ``expected_claims`` lets an auditor additionally require exact
    agreement with claims reconstructed from the pose/archive audit.
    """
    try:
        if not isinstance(certificate, Mapping) or "manifest_mac" not in certificate:
            return False
        body = {k: certificate[k] for k in certificate if k != "manifest_mac"}
        if body.get("schema_version") != SCHEMA_VERSION:
            return False
        desc = pose_descriptor(pose)
        if body.get("pose") != desc:
            return False
        validate_ledger(body["ledger"], len(pose))
        if expected_claims is not None:
            if not isinstance(expected_claims, Mapping):
                return False
            for field, expected in expected_claims.items():
                if body.get(field) != expected:
                    return False
        expected_mac = _mac(manifest_key, body)
        return hmac.compare_digest(str(certificate["manifest_mac"]), expected_mac)
    except (CertificateError, KeyError, TypeError, ValueError, AttributeError):
        return False


def verify_manifest(manifest: Mapping[str, Any], *, poses: Mapping[str, Any],
                    manifest_key: str | bytes) -> bool:
    """Verify all entries in an already-issued manifest; missing entries fail."""
    try:
        entries = manifest["certificates"]
        if not isinstance(entries, Mapping) or set(entries) != set(poses):
            return False
        return all(verify_certificate(entries[cid], pose=poses[cid],
                                      manifest_key=manifest_key)
                   for cid in entries)
    except (KeyError, TypeError):
        return False


# Explicit aliases make the issuance/verification boundary discoverable to
# callers without changing the compact public API.
issue_route_certificate = issue_certificate
verify_route_certificate = verify_certificate
