"""Issue and verify cooperative certificates for a clean source route.

Issuance reads the frozen route records and converts them into authenticated
per-clip source instructions. Verification is deliberately separate: it reads
only the emitted pose bank, the pre-issued manifest, the declared source
archive snapshot, and an HMAC key. It reconstructs every replay and locally
assembled output from that archive, including deterministic boundary blends.

HMAC provides integrity for an authorized cooperative auditor. It is not a
public signature because any holder of the shared key can issue another tag.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from budget_sage.audit.certificate import (  # noqa: E402
    CertificateError,
    canonical_json,
    issue_certificate,
    pose_sha256,
    sha256_file,
    verify_certificate,
    validate_ledger,
)

SCHEMA_VERSION = "clean-source-release-v1"
WITHDRAWAL_SCHEMA_VERSION = "clean-source-release-v2"
# A release either makes no withdrawal claim at all (v1) or carries a complete
# authenticated withdrawn-source declaration (v2).  Pairing the requirement
# with the schema version keeps "no claim was made" and "the claim was
# honoured" from collapsing into the same verdict.
SUPPORTED_SCHEMA_VERSIONS = {
    SCHEMA_VERSION: False,
    WITHDRAWAL_SCHEMA_VERSION: True,
}
WITHDRAWAL_SET_SCHEMA = "withdrawn-source-set-v1"
WITHDRAWAL_COMMITMENT_FIELDS = {"schema", "algorithm", "digest", "count"}
POLICY_FIELDS = {"route", "whole_clip_replay_budget", "selection"}
DEFAULT_ARCHIVE = (
    "outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt"
)
DEFAULT_KEY_ENV = "BUDGET_SAGE_MANIFEST_KEY"
WATERMARK_KEY_ID = "not-used-clean-source-route-v1"
MANIFEST_KEY_ID = "authorized-auditor-hmac-v1"
POLICY_ROUTE = "clean_strict_cac_frame40"
POLICY_SELECTION = "frozen before evaluation"
# These are the only route declarations for which this certificate schema has
# an audited, pre-evaluation selection record.  Keep the selection paired with
# each route so adding a route cannot silently inherit an untruthful label.
SUPPORTED_POLICY_DECLARATIONS = {
    "clean_strict_cac_frame40": "frozen before evaluation",
    "clean_rerank_frame40": "frozen before evaluation",
}
REPLAY_BUDGET_UNIT = "whole_clip_replay_frames_per_emitted_frame"
RECONSTRUCTION_IDENTIFIERS = {
    "whole_clip": "byte_exact_reconstruction",
    "local_unit": "byte_exact_archive_reconstruction",
    "blend": "cosine",
    "generated": "inactive_for_this_release",
}


def _path(value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise CertificateError(f"path is outside the project root: {resolved}")
    return resolved


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _pose(value: Any) -> torch.Tensor:
    pose = torch.as_tensor(value).detach().cpu()
    if pose.ndim == 2:
        pose = pose.reshape(pose.shape[0], 178, 3)
    if pose.ndim != 3 or tuple(pose.shape[1:]) != (178, 3):
        raise CertificateError(f"expected [T,178,3] pose, got {tuple(pose.shape)}")
    if pose.shape[0] < 1 or not bool(torch.isfinite(pose).all()):
        raise CertificateError("pose must be non-empty and finite")
    return pose.contiguous()


def load_archive(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, Mapping) and "exemplar_poses" in raw:
        raw = raw["exemplar_poses"]
    if not isinstance(raw, Mapping) or not raw:
        raise CertificateError("source archive must contain a non-empty pose mapping")
    return {str(sid): _pose(value) for sid, value in raw.items()}


def _release_mac(key: bytes, body: Mapping[str, Any]) -> str:
    return hmac.new(key, canonical_json(body), hashlib.sha256).hexdigest()


def _key_from_env(name: str) -> bytes:
    value = os.environ.get(name, "")
    if not value:
        raise CertificateError(f"environment variable {name} is required")
    return value.encode("utf-8")


def source_mass_from_segments(segments: list[dict], blend_width: int) -> dict[str, float]:
    mass: defaultdict[str, float] = defaultdict(float)
    for segment in segments:
        length = int(segment["end"]) - int(segment["start"])
        mass[str(segment["source_id"])] += float(length)
    for previous, current in zip(segments, segments[1:]):
        previous_length = int(previous["end"]) - int(previous["start"])
        current_length = int(current["end"]) - int(current["start"])
        width = min(int(blend_width), previous_length // 2, current_length // 2)
        if width:
            alpha = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, width)))
            transferred = float(alpha.sum())
            mass[str(previous["source_id"])] -= transferred
            mass[str(current["source_id"])] += transferred
    return {source: value for source, value in mass.items() if abs(value) > 1e-12}


def derive_route_from_ledger(
    ledger: Mapping[str, Any],
    archive: Mapping[str, torch.Tensor],
) -> str:
    """Classify a valid source recipe without trusting its route labels.

    A single interval spanning one entire archive clip is whole replay. All
    other valid interval recipes are local assembly. This classifies the
    authenticated recipe, not every alternative decomposition of its bytes.
    """
    total = ledger.get("T")
    if type(total) is not int or total < 1:
        raise CertificateError("clean source recipe requires a positive integer T")
    validate_ledger(ledger, total)
    segments = ledger.get("segments")
    if not segments:
        raise CertificateError("clean source certificate requires source segments")
    for segment in segments:
        source_id = segment.get("source_id")
        if source_id not in archive:
            raise CertificateError(f"source is absent from archive: {source_id}")
        source_start = segment.get("source_start")
        source_end = segment.get("source_end")
        length = segment["end"] - segment["start"]
        if (type(source_start) is not int or type(source_end) is not int
                or source_end != source_start + length):
            raise CertificateError(f"source_end is inconsistent: {source_id}")
        if source_start < 0 or source_end > len(archive[source_id]):
            raise CertificateError(f"source interval is out of bounds: {source_id}")
    first = segments[0]
    replay = (
        len(segments) == 1
        and first["source_start"] == 0
        and first["source_end"] == len(archive[first["source_id"]])
    )
    derived_route = "replay" if replay else "fallback"
    segment_route = "whole_clip" if replay else "local_unit"
    if ledger["route"] != derived_route:
        raise CertificateError("route label differs from the archive-derived recipe class")
    expected_counts = {
        "whole_clip": total if replay else 0,
        "local_unit": 0 if replay else total,
        "generated": 0,
        "unknown_or_derived": 0,
    }
    if (any(segment["route"] != segment_route for segment in segments)
            or ledger["frame_counts"] != expected_counts):
        raise CertificateError("route frame labels differ from the archive-derived recipe class")
    return derived_route


def reconstruct_from_ledger(
    ledger: Mapping[str, Any],
    archive: Mapping[str, torch.Tensor],
    blend_width: int,
) -> torch.Tensor:
    route = derive_route_from_ledger(ledger, archive)
    if type(blend_width) is not int or blend_width < 0:
        raise CertificateError("blend width must be a non-negative integer")
    segments = ledger["segments"]
    parts: list[np.ndarray] = []
    for segment in segments:
        source_id = str(segment["source_id"])
        if source_id not in archive:
            raise CertificateError(f"source is absent from archive: {source_id}")
        source_start = int(segment["source_start"])
        length = int(segment["end"]) - int(segment["start"])
        source_end = source_start + length
        if "source_end" not in segment or int(segment["source_end"]) != source_end:
            raise CertificateError(f"source_end is inconsistent: {source_id}")
        source = archive[source_id]
        if source_start < 0 or source_end > len(source):
            raise CertificateError(f"source interval is out of bounds: {source_id}")
        part = source[source_start:source_end].numpy().astype(np.float32, copy=True)
        parts.append(part.reshape(length, -1))
    if route == "fallback":
        output = [parts[0]]
        for current in parts[1:]:
            previous = output[-1]
            width = min(int(blend_width), len(previous) // 2, len(current) // 2)
            if width:
                ramp = 0.5 * (
                    1.0 - np.cos(np.linspace(0.0, np.pi, width))
                ).astype(np.float32)[:, None]
                previous[-width:] = (
                    (1.0 - ramp) * previous[-width:] + ramp * current[:width]
                )
            output.append(current)
        array = np.concatenate(output, axis=0)
    else:
        array = parts[0]
    return torch.from_numpy(array.reshape(-1, 178, 3)).contiguous()


def _close_mass(actual: Mapping[str, float], expected: Any) -> bool:
    if not isinstance(expected, Mapping) or set(actual) != set(expected):
        return False
    try:
        return all(
            type(expected[key]) in (int, float)
            and math.isfinite(expected[key])
            and math.isclose(actual[key], expected[key], rel_tol=0.0, abs_tol=1e-6)
            for key in actual
        )
    except OverflowError:
        return False


def _replay_budget_ratio(policy: Mapping[str, Any]) -> tuple[int, int]:
    budget = policy.get("whole_clip_replay_budget")
    if not isinstance(budget, Mapping):
        raise CertificateError("structured replay-frame budget is missing")
    if set(budget) != {"numerator", "denominator", "unit"}:
        raise CertificateError("replay-frame budget fields differ from the clean schema")
    if budget["unit"] != REPLAY_BUDGET_UNIT:
        raise CertificateError("unsupported replay-frame budget unit")
    numerator, denominator = budget["numerator"], budget["denominator"]
    if type(numerator) is not int or type(denominator) is not int:
        raise CertificateError("replay-frame budget numerator and denominator must be exact integers")
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise CertificateError("invalid replay-frame budget")
    return numerator, denominator


def canonical_withdrawn_sources(values: Any) -> list[str]:
    """Return the canonical withdrawn-source list: sorted, unique, strings.

    Coerced types are rejected exactly as the ledger schema rejects them, so
    a boolean or a number cannot pose as a source identifier, and a duplicate
    cannot change the committed count without changing the set.
    """
    if not isinstance(values, list):
        raise CertificateError("withdrawn source set must be a list")
    for value in values:
        if type(value) is not str or not value:
            raise CertificateError("withdrawn source ids must be non-empty strings")
    ordered = sorted(set(values))
    if len(ordered) != len(values):
        raise CertificateError("withdrawn source set contains duplicate ids")
    return ordered


def withdrawal_commitment(sources: Any) -> dict[str, Any]:
    """Commit to a withdrawn-source set as one authenticated digest."""
    ordered = canonical_withdrawn_sources(sources)
    body = {"schema": WITHDRAWAL_SET_SCHEMA, "sources": ordered}
    return {
        "schema": WITHDRAWAL_SET_SCHEMA,
        "algorithm": "sha256",
        "digest": hashlib.sha256(canonical_json(body)).hexdigest(),
        "count": int(len(ordered)),
    }


def validate_withdrawal_commitment(commitment: Any) -> None:
    if (not isinstance(commitment, Mapping)
            or set(commitment) != WITHDRAWAL_COMMITMENT_FIELDS):
        raise CertificateError("withdrawal commitment fields differ from the clean schema")
    if commitment["schema"] != WITHDRAWAL_SET_SCHEMA:
        raise CertificateError("unsupported withdrawn-source set schema")
    if commitment["algorithm"] != "sha256":
        raise CertificateError("unsupported withdrawal commitment algorithm")
    digest = commitment["digest"]
    if (type(digest) is not str or len(digest) != 64
            or set(digest) - set("0123456789abcdef")):
        raise CertificateError("withdrawal commitment must carry a SHA-256 digest")
    if type(commitment["count"]) is not int or commitment["count"] < 0:
        raise CertificateError("withdrawn source count must be a non-negative exact integer")


def bind_withdrawn_sources(policy: Mapping[str, Any], declared: Any) -> frozenset[str]:
    """Recover the withdrawn set that the authenticated policy commits to.

    The list itself travels once in the release body; its digest travels
    inside every certificate.  Recomputing the digest here is what stops the
    list from being swapped for another one after the certificates were
    signed, and what stops a re-signed envelope from re-binding it alone.
    """
    commitment = policy.get("withdrawal")
    validate_withdrawal_commitment(commitment)
    ordered = canonical_withdrawn_sources(declared)
    if list(declared) != ordered:
        raise CertificateError("declared withdrawn sources are not in canonical order")
    if withdrawal_commitment(ordered) != dict(commitment):
        raise CertificateError(
            "withdrawn-source list does not match the authenticated commitment")
    return frozenset(ordered)


def withdrawn_source_exposure(
    ledger: Mapping[str, Any],
    blend_width: int,
    withdrawn: frozenset[str],
) -> dict[str, int]:
    """Count emitted positions that draw on a withdrawn source.

    This reads the authenticated recipe only.  Each segment owns its own
    interval, and a cosine blend of width w additionally makes the last w
    positions of a segment draw on the next segment's source.  Contact is
    charged even where the blend weight is zero, and a position with no
    source at all is treated as a hole in the recipe and fails closed.
    """
    total = ledger.get("T")
    if type(total) is not int or total < 1:
        raise CertificateError("withdrawal check requires a positive integer T")
    if type(blend_width) is not int or blend_width < 0:
        raise CertificateError("blend width must be a non-negative integer")
    segments = ledger.get("segments")
    if not segments:
        raise CertificateError("withdrawal check requires source segments")
    drawn: list[set[str]] = [set() for _ in range(total)]
    for segment in segments:
        source_id = str(segment["source_id"])
        for position in range(int(segment["start"]), int(segment["end"])):
            drawn[position].add(source_id)
    for previous, current in zip(segments, segments[1:]):
        previous_length = int(previous["end"]) - int(previous["start"])
        current_length = int(current["end"]) - int(current["start"])
        width = min(blend_width, previous_length // 2, current_length // 2)
        for position in range(int(previous["end"]) - width, int(previous["end"])):
            drawn[position].add(str(current["source_id"]))
    if any(not sources for sources in drawn):
        raise CertificateError("recipe leaves an emitted position without a source")
    exposed = [index for index, sources in enumerate(drawn) if sources & withdrawn]
    return {
        "positions": total,
        "exposed_positions": len(exposed),
        "first_exposed_position": exposed[0] if exposed else -1,
    }


def validate_release_configuration(
    policy: Mapping[str, Any], detector: Mapping[str, Any],
    *, expect_withdrawal: bool = False,
) -> None:
    """Accept only the implemented clean schema, not merely matching labels.

    The selection value is a required authenticated declaration, not evidence
    of when selection actually happened. Other reconstruction rules need an
    explicitly implemented and versioned schema.
    """
    expected_policy_fields = (
        POLICY_FIELDS | {"withdrawal"} if expect_withdrawal else POLICY_FIELDS
    )
    if set(policy) != expected_policy_fields:
        raise CertificateError("release policy fields differ from the clean schema")
    route = policy["route"]
    if (type(route) is not str or route not in SUPPORTED_POLICY_DECLARATIONS
            or policy["selection"] != SUPPORTED_POLICY_DECLARATIONS[route]):
        raise CertificateError("unsupported clean release policy declaration")
    _replay_budget_ratio(policy)
    if set(detector) != set(RECONSTRUCTION_IDENTIFIERS) | {"blend_width"}:
        raise CertificateError("reconstruction fields differ from the clean schema")
    if any(detector[field] != value for field, value in RECONSTRUCTION_IDENTIFIERS.items()):
        raise CertificateError("unsupported clean reconstruction identifier")
    if type(detector["blend_width"]) is not int or detector["blend_width"] < 0:
        raise CertificateError("blend width must be a non-negative exact integer")
    if expect_withdrawal:
        validate_withdrawal_commitment(policy["withdrawal"])


def validate_replay_budget(
    policy: Mapping[str, Any], replay_frames: int, total_frames: int
) -> dict[str, Any]:
    numerator, denominator = _replay_budget_ratio(policy)
    if (type(replay_frames) is not int or type(total_frames) is not int
            or not 0 <= replay_frames <= total_frames):
        raise CertificateError("release frame totals must be non-negative exact integers")
    left = denominator * replay_frames
    right = numerator * total_frames
    if left > right:
        raise CertificateError("release exceeds the authenticated replay-frame budget")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "integer_check": f"{left} <= {right}",
        "verified": True,
    }


def _certificate_ledger(
    sid: str,
    emitted: torch.Tensor,
    replay: bool,
    retrieval: Mapping[str, dict],
    fallback: Mapping[str, dict],
    frozen_ledger: Mapping[str, dict],
) -> dict:
    total = int(emitted.shape[0])
    frozen = frozen_ledger[sid]
    if int(frozen["frames"]) != total:
        raise CertificateError(f"frozen ledger length mismatch for {sid}")
    if replay:
        selected = retrieval[sid].get("selected") or {}
        source_id = str(selected.get("id", ""))
        segments = [{
            "route": "whole_clip",
            "start": 0,
            "end": total,
            "source_id": source_id,
            "source_start": 0,
            "source_end": total,
        }]
        route = "replay"
        counts = {
            "whole_clip": total,
            "local_unit": 0,
            "generated": 0,
            "unknown_or_derived": 0,
        }
    else:
        cursor = 0
        segments = []
        for row in fallback[sid].get("segments", []):
            length = int(row["out_len"])
            segments.append({
                "route": "local_unit",
                "start": cursor,
                "end": cursor + length,
                "source_id": str(row["source"]),
                "source_start": int(row["start"]),
                "source_end": int(row["end"]),
                "gloss_count": int(row["j"]) - int(row["i"]),
            })
            cursor += length
        if cursor != total or not segments:
            raise CertificateError(f"fallback segments do not cover {sid}")
        route = "fallback"
        counts = {
            "whole_clip": 0,
            "local_unit": total,
            "generated": 0,
            "unknown_or_derived": 0,
        }
    claimed = sorted({str(row["source_id"]) for row in segments})
    expected_mode = "whole_clip_replay" if replay else "compositional_local_reuse"
    if frozen["mode"] != expected_mode:
        raise CertificateError(f"frozen ledger mode mismatch for {sid}")
    return {
        "route": route,
        "T": total,
        "frame_counts": counts,
        "claimed_source_ids": claimed,
        "segments": segments,
        "fractional_source_mass": frozen["source_mass"],
    }


def issue_release(
    *,
    pose_bank_path: Path,
    mask_path: Path,
    retrieval_trace_path: Path,
    fallback_trace_path: Path,
    ledger_path: Path,
    archive_path: Path,
    manifest_key: bytes,
    blend_width: int,
    withdrawn_sources_path: Path | None = None,
    policy_route: str = POLICY_ROUTE,
) -> dict:
    poses = torch.load(pose_bank_path, map_location="cpu", weights_only=True)
    poses = {str(sid): _pose(value) for sid, value in poses.items()}
    mask_data = np.load(mask_path, allow_pickle=False)
    ids = [str(value) for value in mask_data["ids"].tolist()]
    flags = np.asarray(mask_data["in_S"], dtype=bool)
    if len(ids) != len(flags) or set(ids) != set(poses):
        raise CertificateError("mask and emitted pose IDs do not align")
    retrieval_rows = _read_json(retrieval_trace_path)
    fallback_rows = _read_json(fallback_trace_path)
    frozen_raw = _read_json(ledger_path)
    retrieval = {str(row["id"]): row for row in retrieval_rows}
    fallback = (
        {str(row["id"]): row for row in fallback_rows}
        if isinstance(fallback_rows, list)
        else {str(sid): row for sid, row in fallback_rows.items()}
    )
    frozen = {str(row["id"]): row for row in frozen_raw["clips"]}
    if any(set(rows) != set(ids) for rows in (retrieval, fallback, frozen)):
        raise CertificateError("issuer records do not cover the emitted IDs exactly")
    archive = load_archive(archive_path)
    archive_commitment = {"algorithm": "sha256", "digest": sha256_file(str(archive_path))}
    archive_id = f"{archive_path.name}:{archive_commitment['digest'][:16]}"
    if (type(policy_route) is not str
            or policy_route not in SUPPORTED_POLICY_DECLARATIONS):
        raise CertificateError("unsupported clean release policy route")
    declared_withdrawn: list[str] | None = None
    withdrawn: frozenset[str] | None = None
    if withdrawn_sources_path is not None:
        declared_withdrawn = canonical_withdrawn_sources(
            _read_json(withdrawn_sources_path))
        withdrawn = frozenset(declared_withdrawn)
    policy = {
        "route": policy_route,
        "whole_clip_replay_budget": {
            "numerator": 2,
            "denominator": 5,
            "unit": REPLAY_BUDGET_UNIT,
        },
        "selection": SUPPORTED_POLICY_DECLARATIONS[policy_route],
    }
    if declared_withdrawn is not None:
        policy["withdrawal"] = withdrawal_commitment(declared_withdrawn)
    detector = {
        **RECONSTRUCTION_IDENTIFIERS,
        "blend_width": blend_width,
    }
    validate_release_configuration(
        policy, detector, expect_withdrawal=declared_withdrawn is not None
    )
    certificates: dict[str, dict] = {}
    flag_by_id = {sid: bool(flag) for sid, flag in zip(ids, flags)}
    for sid in ids:
        pose = poses[sid]
        ledger = _certificate_ledger(
            sid, pose, flag_by_id[sid], retrieval, fallback, frozen
        )
        reconstructed = reconstruct_from_ledger(ledger, archive, blend_width)
        if not torch.equal(pose, reconstructed.to(dtype=pose.dtype)):
            raise CertificateError(f"issuer reconstruction failed for {sid}")
        measured_mass = (
            {ledger["segments"][0]["source_id"]: float(len(pose))}
            if ledger["route"] == "replay"
            else source_mass_from_segments(ledger["segments"], blend_width)
        )
        if not _close_mass(measured_mass, ledger["fractional_source_mass"]):
            raise CertificateError(f"source mass mismatch for {sid}")
        if withdrawn is not None:
            exposure = withdrawn_source_exposure(ledger, blend_width, withdrawn)
            if exposure["exposed_positions"]:
                raise CertificateError(
                    f"issuer recipe draws on a withdrawn source at emitted position "
                    f"{exposure['first_exposed_position']} of {sid}")
        certificates[sid] = issue_certificate(
            certificate_id=sid,
            clip_id=sid,
            pose=pose,
            ledger=ledger,
            archive_commitment=archive_commitment,
            archive_id=archive_id,
            policy=policy,
            detector_config=detector,
            watermark_key_id=WATERMARK_KEY_ID,
            manifest_key_id=MANIFEST_KEY_ID,
            manifest_key=manifest_key,
        )
    body = {
        "schema_version": (
            SCHEMA_VERSION if declared_withdrawn is None
            else WITHDRAWAL_SCHEMA_VERSION
        ),
        "archive_commitment": archive_commitment,
        "archive_id": archive_id,
        "policy": policy,
        "detector_config": detector,
        "manifest_key_id": MANIFEST_KEY_ID,
        "pose_bank_sha256": sha256_file(str(pose_bank_path)),
        "certificate_ids": ids,
        "certificates": certificates,
        "issuance_inputs": {
            "pose_bank": sha256_file(str(pose_bank_path)),
            "mask": sha256_file(str(mask_path)),
            "retrieval_trace": sha256_file(str(retrieval_trace_path)),
            "fallback_trace": sha256_file(str(fallback_trace_path)),
            "ledger": sha256_file(str(ledger_path)),
            "archive": sha256_file(str(archive_path)),
        },
    }
    if declared_withdrawn is not None:
        body["withdrawn_sources"] = declared_withdrawn
        body["issuance_inputs"]["withdrawn_sources"] = sha256_file(
            str(withdrawn_sources_path))
    return {**body, "release_mac": _release_mac(manifest_key, body)}


def verify_release(
    *,
    pose_bank_path: Path,
    manifest: Mapping[str, Any],
    archive_path: Path,
    manifest_key: bytes,
) -> dict:
    release_schema = manifest.get("schema_version")
    if (type(release_schema) is not str
            or release_schema not in SUPPORTED_SCHEMA_VERSIONS):
        raise CertificateError("release schema mismatch")
    expects_withdrawal = SUPPORTED_SCHEMA_VERSIONS[release_schema]
    body = {key: value for key, value in manifest.items() if key != "release_mac"}
    expected_mac = _release_mac(manifest_key, body)
    if not hmac.compare_digest(str(manifest.get("release_mac", "")), expected_mac):
        raise CertificateError("release HMAC failed")
    commitment = manifest.get("archive_commitment")
    if (not isinstance(commitment, Mapping)
            or set(commitment) != {"algorithm", "digest"}
            or commitment["algorithm"] != "sha256"
            or not isinstance(commitment["digest"], str)):
        raise CertificateError("unsupported archive commitment schema or algorithm")
    if manifest.get("manifest_key_id") != MANIFEST_KEY_ID:
        raise CertificateError("unsupported release manifest key identifier")
    if sha256_file(str(archive_path)) != commitment["digest"]:
        raise CertificateError("archive snapshot commitment failed")
    expected_archive_id = (
        f"{archive_path.name}:{manifest['archive_commitment']['digest'][:16]}"
    )
    if manifest.get("archive_id") != expected_archive_id:
        raise CertificateError("archive identifier does not match the committed snapshot")
    poses_raw = torch.load(pose_bank_path, map_location="cpu", weights_only=True)
    poses = {str(sid): _pose(value) for sid, value in poses_raw.items()}
    certs = manifest.get("certificates")
    ids = manifest.get("certificate_ids")
    if not isinstance(certs, Mapping) or not isinstance(ids, list) or not ids:
        raise CertificateError("release entries are malformed")
    if ids != list(poses) or set(certs) != set(poses):
        raise CertificateError("manifest membership or pose order differs")
    if manifest.get("pose_bank_sha256") != sha256_file(str(pose_bank_path)):
        raise CertificateError("pose-bank commitment failed")
    release_policy = manifest.get("policy")
    release_detector = manifest.get("detector_config")
    if not isinstance(release_policy, Mapping) or not isinstance(release_detector, Mapping):
        raise CertificateError("release policy or detector configuration is missing")
    validate_release_configuration(
        release_policy, release_detector, expect_withdrawal=expects_withdrawal
    )
    # The withdrawn set is authenticated twice: the list is covered by the
    # release MAC and its digest is covered by every certificate MAC.  A
    # release that makes no declaration asserts no withdrawal property.
    declared_withdrawn = manifest.get("withdrawn_sources")
    withdrawn: frozenset[str] | None = None
    if expects_withdrawal:
        if declared_withdrawn is None:
            raise CertificateError(
                "this release schema requires a withdrawn-source declaration")
        withdrawn = bind_withdrawn_sources(release_policy, declared_withdrawn)
    elif declared_withdrawn is not None or "withdrawal" in release_policy:
        raise CertificateError(
            "withdrawn-source declaration is unsupported by this release schema")
    archive = load_archive(archive_path)
    replay_clips = replay_frames = fallback_clips = fallback_frames = 0
    withdrawn_positions_checked = 0
    withdrawn_mass = 0.0
    all_sources: set[str] = set()
    blend_boundary_positions = 0
    fractional_blend_positions = 0
    for sid in ids:
        pose = poses[sid]
        cert = certs[sid]
        if not verify_certificate(cert, pose=pose, manifest_key=manifest_key):
            raise CertificateError(f"certificate authentication failed for {sid}")
        if cert.get("certificate_id") != sid or cert.get("clip_id") != sid:
            raise CertificateError(f"certificate identity differs from the release entry: {sid}")
        if cert.get("manifest_key_id") != manifest["manifest_key_id"]:
            raise CertificateError(f"certificate manifest key identifier differs: {sid}")
        if cert.get("archive_commitment") != manifest["archive_commitment"]:
            raise CertificateError(f"certificate archive commitment differs: {sid}")
        if cert.get("archive_id") != manifest["archive_id"]:
            raise CertificateError(f"certificate archive identifier differs: {sid}")
        if cert.get("policy") != release_policy:
            raise CertificateError(f"certificate policy differs: {sid}")
        if cert.get("detector_config") != release_detector:
            raise CertificateError(f"certificate detector configuration differs: {sid}")
        # Equality alone treats e.g. 1, 1.0, and True as interchangeable.
        # Validate each authenticated mapping as well as the release envelope.
        validate_release_configuration(
            cert["policy"], cert["detector_config"],
            expect_withdrawal=expects_withdrawal,
        )
        ledger = cert["ledger"]
        derived_route = derive_route_from_ledger(ledger, archive)
        blend_width = cert["detector_config"]["blend_width"]
        reconstructed = reconstruct_from_ledger(ledger, archive, blend_width)
        if not torch.equal(pose, reconstructed.to(dtype=pose.dtype)):
            raise CertificateError(f"pose-side source reconstruction failed for {sid}")
        measured_mass = (
            {ledger["segments"][0]["source_id"]: float(len(pose))}
            if derived_route == "replay"
            else source_mass_from_segments(ledger["segments"], blend_width)
        )
        if not _close_mass(measured_mass, ledger.get("fractional_source_mass")):
            raise CertificateError(f"fractional source ledger failed for {sid}")
        if not math.isclose(sum(measured_mass.values()), len(pose), rel_tol=0.0, abs_tol=1e-6):
            raise CertificateError(f"source mass is not conserved for {sid}")
        if withdrawn is not None:
            exposure = withdrawn_source_exposure(ledger, blend_width, withdrawn)
            if exposure["exposed_positions"]:
                raise CertificateError(
                    f"emitted position {exposure['first_exposed_position']} of {sid} "
                    f"draws on a withdrawn source")
            withdrawn_positions_checked += exposure["positions"]
            withdrawn_mass += sum(
                value for source, value in measured_mass.items() if source in withdrawn
            )
        all_sources.update(measured_mass)
        if derived_route == "replay":
            replay_clips += 1
            replay_frames += len(pose)
        else:
            fallback_clips += 1
            fallback_frames += len(pose)
            segments = list(ledger["segments"])
            for previous, current in zip(segments, segments[1:]):
                previous_length = int(previous["end"]) - int(previous["start"])
                current_length = int(current["end"]) - int(current["start"])
                width = min(blend_width, previous_length // 2, current_length // 2)
                blend_boundary_positions += width
                if width:
                    alpha = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, width)))
                    fractional_blend_positions += int(((alpha > 0.0) & (alpha < 1.0)).sum())
    total_frames = replay_frames + fallback_frames
    verified_budget = validate_replay_budget(
        release_policy, replay_frames, total_frames
    )
    withdrawal_report: dict[str, Any] = {
        "declared": withdrawn is not None,
        "release_schema_version": release_schema,
    }
    if withdrawn is None:
        withdrawal_report.update({
            "verified": False,
            "reason": "release makes no authenticated withdrawn-source declaration",
        })
    else:
        withdrawal_report.update({
            "set_schema": WITHDRAWAL_SET_SCHEMA,
            "commitment": dict(release_policy["withdrawal"]),
            "withdrawn_source_count": len(withdrawn),
            "withdrawn_sources_present_in_committed_archive":
                len(withdrawn & set(archive)),
            "clips_checked": len(ids),
            "emitted_positions_checked": withdrawn_positions_checked,
            "emitted_positions_drawing_on_withdrawn_source": 0,
            "withdrawn_source_mass_in_release": withdrawn_mass,
            "verified": True,
        })
    return {
        "verdict": "PASS",
        "withdrawn_source_set": withdrawal_report,
        "manifest_entries": len(ids),
        "replay_clips": replay_clips,
        "fallback_clips": fallback_clips,
        "replay_frames": replay_frames,
        "fallback_frames": fallback_frames,
        "total_frames": total_frames,
        "replay_frame_fraction": replay_frames / max(1, total_frames),
        "unique_training_sources": len(all_sources),
        "unknown_or_untraced_frames": 0,
        "blend_boundary_positions": blend_boundary_positions,
        "fractional_blend_positions": fractional_blend_positions,
        "replay_frame_budget": verified_budget,
        "router_records_read_by_verifier": False,
        "archive_sha256": sha256_file(str(archive_path)),
        "pose_bank_sha256": sha256_file(str(pose_bank_path)),
    }


def _resign_release(manifest: dict, manifest_key: bytes) -> dict:
    """Re-authenticate an edited release with the authorized key.

    Used only by the drills, to model a dishonest issuer rather than an
    outsider: every MAC is valid afterwards, so only the pose-side,
    archive-side and schema checks can still reject the release.
    """
    for cert in manifest["certificates"].values():
        cert.pop("manifest_mac", None)
        cert["manifest_mac"] = _release_mac(manifest_key, cert)
    body = {key: value for key, value in manifest.items() if key != "release_mac"}
    manifest["release_mac"] = _release_mac(manifest_key, body)
    return manifest


def tamper_drills(
    *,
    pose_bank_path: Path,
    manifest: Mapping[str, Any],
    archive_path: Path,
    manifest_key: bytes,
) -> dict[str, bool]:
    poses = torch.load(pose_bank_path, map_location="cpu", weights_only=True)
    sid = str(manifest["certificate_ids"][0])
    cert = manifest["certificates"][sid]
    pose = _pose(poses[sid])
    drills: dict[str, bool] = {}
    drills["wrong_key_caught"] = not verify_certificate(
        cert, pose=pose, manifest_key=manifest_key + b"-wrong"
    )
    changed_pose = pose.clone()
    changed_pose.reshape(-1)[0] += 1e-4
    drills["pose_edit_caught"] = not verify_certificate(
        cert, pose=changed_pose, manifest_key=manifest_key
    )
    for name, mutate in (
        ("ledger_edit_caught", lambda value: value["ledger"]["frame_counts"].__setitem__(
            "unknown_or_derived", 1
        )),
        ("route_relabel_caught", lambda value: value["ledger"].__setitem__(
            "route", "generated"
        )),
        ("blend_config_edit_caught", lambda value: value["detector_config"].__setitem__(
            "blend_width", int(value["detector_config"]["blend_width"]) + 1
        )),
        ("archive_commitment_edit_caught", lambda value: value["archive_commitment"].__setitem__(
            "digest", "0" * 64
        )),
    ):
        changed = copy.deepcopy(cert)
        mutate(changed)
        drills[name] = not verify_certificate(
            changed, pose=pose, manifest_key=manifest_key
        )
    removed = copy.deepcopy(manifest)
    removed_id = removed["certificate_ids"].pop()
    removed["certificates"].pop(removed_id)
    # Re-authenticate the edited envelope so this drill reaches the membership
    # guard instead of passing incidentally at the earlier release-HMAC check.
    removed_body = {key: value for key, value in removed.items() if key != "release_mac"}
    removed["release_mac"] = _release_mac(manifest_key, removed_body)
    try:
        verify_release(
            pose_bank_path=pose_bank_path,
            manifest=removed,
            archive_path=archive_path,
            manifest_key=manifest_key,
        )
    except CertificateError as exc:
        drills["release_membership_edit_caught"] = str(exc) in {
            "release entries are malformed",
            "manifest membership or pose order differs",
        }
    else:
        drills["release_membership_edit_caught"] = False

    if not SUPPORTED_SCHEMA_VERSIONS.get(str(manifest.get("schema_version")), False):
        return drills
    declared = manifest["withdrawn_sources"]

    def rejected(changed: Mapping[str, Any]) -> bool:
        try:
            verify_release(
                pose_bank_path=pose_bank_path,
                manifest=changed,
                archive_path=archive_path,
                manifest_key=manifest_key,
            )
        except CertificateError:
            return True
        return False

    # The committed list is swapped after the certificates were signed and the
    # envelope is re-authenticated; the per-certificate digest still binds the
    # original set, so the swap has to be visible.
    swapped = copy.deepcopy(manifest)
    swapped["withdrawn_sources"] = canonical_withdrawn_sources(
        list(declared) + ["withdrawn-set-tamper-probe"]
    )
    swapped_body = {k: v for k, v in swapped.items() if k != "release_mac"}
    swapped["release_mac"] = _release_mac(manifest_key, swapped_body)
    drills["withdrawn_set_edit_caught"] = rejected(swapped)

    # The per-certificate withdrawal commitment is edited after signing.
    commitment_cert = copy.deepcopy(cert)
    commitment_cert["policy"]["withdrawal"]["digest"] = "0" * 64
    drills["withdrawn_commitment_edit_caught"] = not verify_certificate(
        commitment_cert, pose=pose, manifest_key=manifest_key
    )

    # The declaration is dropped entirely and the release is re-authenticated
    # with the authorized key.
    dropped = copy.deepcopy(manifest)
    dropped.pop("withdrawn_sources")
    drills["withdrawal_declaration_dropped_caught"] = rejected(
        _resign_release(dropped, manifest_key)
    )

    # A dishonest issuer declares a source withdrawn that the release still
    # draws on, and signs every certificate correctly.  Authentication has to
    # succeed and the release still has to be rejected.
    dishonest = copy.deepcopy(manifest)
    used_source = str(cert["ledger"]["segments"][0]["source_id"])
    dishonest["withdrawn_sources"] = canonical_withdrawn_sources(
        sorted(set(declared) | {used_source})
    )
    commitment = withdrawal_commitment(dishonest["withdrawn_sources"])
    dishonest["policy"]["withdrawal"] = commitment
    for entry in dishonest["certificates"].values():
        entry["policy"]["withdrawal"] = copy.deepcopy(commitment)
    _resign_release(dishonest, manifest_key)
    authenticated = verify_certificate(
        dishonest["certificates"][sid], pose=pose, manifest_key=manifest_key
    )
    drills["withdrawn_source_still_used_caught"] = bool(
        authenticated and rejected(dishonest)
    )
    return drills


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    issue = subparsers.add_parser("issue")
    issue.add_argument("--pose_bank", required=True)
    issue.add_argument("--mask", required=True)
    issue.add_argument("--retrieval_trace", required=True)
    issue.add_argument("--fallback_trace", required=True)
    issue.add_argument("--ledger", required=True)
    issue.add_argument("--archive", default=DEFAULT_ARCHIVE)
    issue.add_argument("--blend_width", type=int, default=4)
    issue.add_argument(
        "--withdrawn_sources", default=None,
        help="JSON list of withdrawn source ids; omit to make no withdrawal claim",
    )
    issue.add_argument("--manifest_out", required=True)
    issue.add_argument("--key_env", default=DEFAULT_KEY_ENV)
    issue.add_argument(
        "--policy_route", choices=tuple(SUPPORTED_POLICY_DECLARATIONS),
        default=POLICY_ROUTE,
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--pose_bank", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--archive", default=DEFAULT_ARCHIVE)
    verify.add_argument("--report_out", required=True)
    verify.add_argument("--key_env", default=DEFAULT_KEY_ENV)
    verify.add_argument("--tamper_drills", action="store_true")
    args = parser.parse_args()
    key = _key_from_env(args.key_env)
    pose_bank_path = _path(args.pose_bank)
    archive_path = _path(args.archive)
    if args.mode == "issue":
        manifest = issue_release(
            pose_bank_path=pose_bank_path,
            mask_path=_path(args.mask),
            retrieval_trace_path=_path(args.retrieval_trace),
            fallback_trace_path=_path(args.fallback_trace),
            ledger_path=_path(args.ledger),
            archive_path=archive_path,
            manifest_key=key,
            blend_width=args.blend_width,
            withdrawn_sources_path=(
                _path(args.withdrawn_sources) if args.withdrawn_sources else None
            ),
            policy_route=args.policy_route,
        )
        output = _path(args.manifest_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"issued": len(manifest["certificate_ids"]), "out": str(output)}))
    else:
        manifest_path = _path(args.manifest)
        manifest = _read_json(manifest_path)
        report = verify_release(
            pose_bank_path=pose_bank_path,
            manifest=manifest,
            archive_path=archive_path,
            manifest_key=key,
        )
        if args.tamper_drills:
            drills = tamper_drills(
                pose_bank_path=pose_bank_path,
                manifest=manifest,
                archive_path=archive_path,
                manifest_key=key,
            )
            if not all(drills.values()):
                raise CertificateError("one or more tamper drills failed")
            report["tamper_drills"] = drills
        report["manifest_sha256"] = sha256_file(str(manifest_path))
        report["implementation"] = str(Path(__file__).resolve())
        report["implementation_sha256"] = sha256_file(str(Path(__file__).resolve()))
        report["certificate_helper_sha256"] = sha256_file(
            str(ROOT / "budget_sage/audit/certificate.py")
        )
        output = _path(args.report_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
