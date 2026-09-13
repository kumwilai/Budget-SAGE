#!/usr/bin/env python
"""Executable boundary examples for cooperative HMAC verification.

The final scenario intentionally passes: possession of the shared key permits
coherent re-issuance and is therefore outside the stated threat model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from budget_sage.audit.certificate import (
    issue_certificate,
    sha256_file,
    verify_certificate,
)

DEFAULT_OUTPUT = (
    ROOT
    / "outputs/revision/nonhuman_closure_20260907"
    / "certificate_threat_boundary.json"
)


def _ledger(source_id: str, length: int) -> dict:
    return {
        "route": "fallback",
        "T": length,
        "frame_counts": {
            "whole_clip": 0,
            "local_unit": length,
            "generated": 0,
            "unknown_or_derived": 0,
        },
        "claimed_source_ids": [source_id],
        "segments": [
            {
                "route": "local_unit",
                "start": 0,
                "end": length,
                "source_id": source_id,
                "source_start": 1,
                "source_end": length + 1,
            }
        ],
        "fractional_source_mass": {source_id: float(length)},
    }


def _issue(pose, ledger, key: bytes, archive_digest: str):
    return issue_certificate(
        certificate_id="synthetic-q",
        clip_id="synthetic-q",
        pose=pose,
        ledger=ledger,
        archive_commitment={"algorithm": "sha256", "digest": archive_digest},
        archive_id=f"synthetic:{archive_digest[:16]}",
        policy={
            "route": "clean_rerank_frame40",
            "selection": "frozen before evaluation",
            "whole_clip_replay_budget": {
                "numerator": 2,
                "denominator": 5,
                "unit": "whole_clip_replay_frames_per_emitted_frame",
            },
        },
        detector_config={
            "whole_clip": "byte_exact_reconstruction",
            "local_unit": "byte_exact_archive_reconstruction",
            "blend": "cosine",
            "blend_width": 2,
            "generated": "inactive_for_this_release",
        },
        watermark_key_id="not-used-clean-source-route-v1",
        manifest_key_id="authorized-auditor-hmac-v1",
        manifest_key=key,
    )


def run() -> dict:
    started = time.time_ns()
    key = b"synthetic-authorized-auditor-key"
    wrong_key = b"synthetic-unauthorized-key"
    pose = np.arange(4 * 178 * 3, dtype=np.float32).reshape(4, 178, 3)
    ledger = _ledger("archive-source-a", len(pose))
    archive_digest = hashlib.sha256(b"committed-synthetic-archive-a").hexdigest()
    certificate = _issue(pose, ledger, key, archive_digest)

    tampered_pose = pose.copy()
    tampered_pose[0, 0, 0] += np.float32(1.0)
    tampered_ledger = json.loads(json.dumps(certificate))
    tampered_ledger["ledger"]["segments"][0]["source_id"] = "other-source"

    hostile_pose = np.full_like(pose, 7.0)
    hostile_ledger = _ledger("issuer-controlled-source", len(hostile_pose))
    hostile_digest = hashlib.sha256(b"issuer-controlled-archive").hexdigest()
    hostile_certificate = _issue(hostile_pose, hostile_ledger, key, hostile_digest)

    scenarios = [
        {
            "name": "honest_preissued_exact_pose",
            "inside_threat_model": True,
            "expected_accept": True,
            "observed_accept": verify_certificate(
                certificate, pose=pose, manifest_key=key
            ),
        },
        {
            "name": "postissuance_pose_change",
            "inside_threat_model": True,
            "expected_accept": False,
            "observed_accept": verify_certificate(
                certificate, pose=tampered_pose, manifest_key=key
            ),
        },
        {
            "name": "postissuance_ledger_change_without_reissue",
            "inside_threat_model": True,
            "expected_accept": False,
            "observed_accept": verify_certificate(
                tampered_ledger, pose=pose, manifest_key=key
            ),
        },
        {
            "name": "unauthorized_wrong_key",
            "inside_threat_model": True,
            "expected_accept": False,
            "observed_accept": verify_certificate(
                certificate, pose=pose, manifest_key=wrong_key
            ),
        },
        {
            "name": "coherent_malicious_keyholder_reissue",
            "inside_threat_model": False,
            "expected_accept": True,
            "observed_accept": verify_certificate(
                hostile_certificate, pose=hostile_pose, manifest_key=key
            ),
            "interpretation": (
                "Expected pass demonstrates the excluded malicious-issuer/key-holder "
                "capability; HMAC is cooperative integrity, not public nonrepudiation."
            ),
        },
    ]
    passed = all(row["expected_accept"] == row["observed_accept"] for row in scenarios)
    return {
        "schema": "certificate-threat-boundary-audit-v1",
        "status": "PASS" if passed else "FAIL",
        "claim": (
            "The cooperative verifier detects post-issuance alteration and wrong-key "
            "use; it does not constrain a coherent issuer who holds the shared key."
        ),
        "not_claimed": [
            "public nonrepudiation",
            "protection from malicious issuance",
            "semantic consent",
            "unique source decomposition",
            "rendered-video provenance",
        ],
        "scenarios": scenarios,
        "implementation": {
            "certificate_module": "budget_sage/audit/certificate.py",
            "certificate_module_sha256": sha256_file(
                str(ROOT / "budget_sage/audit/certificate.py")
            ),
            "audit_script_sha256": sha256_file(__file__),
            "python": platform.python_version(),
        },
        "started_unix_ns": started,
        "completed_unix_ns": time.time_ns(),
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "scenarios": len(report["scenarios"]),
        "peak_rss_kb": report["peak_rss_kb"],
    }, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
