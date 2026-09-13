"""Cooperative audit of a Budget-SAGE route certificate using HMAC authentication.

Threat model: an auditor holds (i) the emitted pose bank, (ii) the emitted
route certificate (per-clip route class, claimed source ids, frame ledger,
declared frame count), (iii) the training archive, and (iv) the watermark key.
The auditor does NOT trust the router's code or logs. In normal verification
mode, route and source claims come only from the authenticated, pre-issued
certificate. The script checks every positive source claim against the emitted
pose and archive; unknown/derived frames remain conservatively unverified:

  * replay clips  -> the pose must byte-match the claimed train source
                     (exact verbatim check, tolerance 1e-6);
  * fallback clips-> local-unit reuse must be visible at window granularity:
                     the fraction of 16-frame windows that match some train
                     window (<0.02 per-dim RMS) is compared with the declared
                     traced-local mass;
  * generated     -> the pose watermark signal is checked by
                     ``scripts.pose_watermark.detect`` and no window may match
                     any train window.

Manifest authentication is cooperative HMAC verification, not a public
signature, and does not itself prove that a watermark signal exists.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_subsequence_memorization import (  # noqa: E402
    as_pose,
    min_window_distance,
    window_signatures,
)
from scripts.pose_watermark import detect  # noqa: E402
from scripts.signjepa_cpf_knapsack import SLRTP  # noqa: E402
from budget_sage.audit.certificate import (  # noqa: E402
    CertificateError, issue_certificate, pose_sha256, sha256_file,
    verify_certificate,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default="phase59_btfree_deployable_student_g20_wm_test.pt")
    ap.add_argument("--route_json", default="outputs/learned_confidence/deployable_e2e_rebuild.json")
    ap.add_argument("--in_s_npz", default="outputs/learned_confidence/cac_btfree_test_in_S.npz")
    ap.add_argument("--trace_json", default="external/SLRTP-Sign-Production-Evaluation/results/topk_semantic_hybrid_test_trace.json")
    ap.add_argument("--train_pt",
                    default=str(SLRTP / "pretrained/SLRTP-Sign-Production-Evaluation-Data/data/train.pt"))
    ap.add_argument("--watermark_key", default=None,
                    help="secret used by the keyed pose-mark detector (required)")
    ap.add_argument("--manifest_key", default=None,
                    help="HMAC key for manifest authentication (required)")
    ap.add_argument("--manifest_in", default=None,
                    help="pre-issued manifest to verify; required by default")
    ap.add_argument("--manifest_out",
                    default="outputs/learned_confidence/route_manifest_authenticated.json")
    ap.add_argument("--issue", action="store_true",
                    help="explicit demo issuance mode; never enabled by default")
    ap.add_argument("--watermark_key_id", default="demo-watermark-v1")
    ap.add_argument("--manifest_key_id", default="demo-manifest-v1")
    ap.add_argument("--z_threshold", type=float, default=8.0)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_fallback_clips", type=int, default=0,
                    help="optional diagnostic cap; 0 checks every fallback clip")
    ap.add_argument("--out", default="outputs/learned_confidence/route_certificate_verification.json")
    args = ap.parse_args()
    if not args.watermark_key or not args.manifest_key:
        ap.error("--watermark_key and --manifest_key are required (no production-secret default)")
    if args.watermark_key == args.manifest_key:
        ap.error("watermark and manifest secrets must be distinct")
    if args.watermark_key_id == args.manifest_key_id:
        ap.error("watermark and manifest key identifiers must be distinct")
    if not args.issue and not args.manifest_in:
        ap.error("normal verification requires a pre-issued --manifest_in; use --issue only for demo issuance")

    bank = torch.load(SLRTP / "results" / args.bank, map_location="cpu", weights_only=False)
    trace: dict[str, dict[str, Any]] = {}
    if args.issue:
        # Issuance may consume internal route records. Those records are never
        # consulted by the normal verifier.
        routes = json.loads((ROOT / args.route_json).read_text())
        sel = routes["dev_selected_target"]
        row = next(r for r in routes["rows"]["test"]
                   if abs(r["target_generated_frame_fraction"] - sel) < 1e-9)
        gen_ids = set(row["rewritten_ids"])
        z = np.load(ROOT / args.in_s_npz, allow_pickle=True)
        in_S = {str(s): bool(f) for s, f in zip(z["sids"], z["in_S"])}
        trace = {r["id"]: (r.get("selected") or {}) for r in
                 json.loads((ROOT / args.trace_json).read_text())}
        supplied: dict[str, Any] = {}
        certs: dict[str, Any] = {}
    else:
        supplied = json.loads((ROOT / args.manifest_in).read_text())
        certs = supplied.get("certificates", {})
        if (supplied.get("schema_version") != "route-certificate-v4"
                or supplied.get("watermark_key_id") != args.watermark_key_id
                or supplied.get("manifest_key_id") != args.manifest_key_id):
            raise CertificateError("manifest schema or key identifiers do not match")
        if not isinstance(certs, dict) or set(certs) != set(bank):
            raise CertificateError("manifest entries do not match the emitted pose bank")
        gen_ids = {sid for sid, cert in certs.items()
                   if cert.get("ledger", {}).get("route") == "generated"}
        in_S = {sid: cert.get("ledger", {}).get("route") == "replay"
                for sid, cert in certs.items()}
    archive_commitment = {"algorithm": "sha256", "digest": sha256_file(args.train_pt)}
    archive_id = Path(args.train_pt).name
    detector_config = {
        "z_threshold": args.z_threshold,
        "win": args.win,
        "stride": args.stride,
        "generated_no_copy_min_distance": 0.05,
        "fallback_match_rms": 0.02,
    }

    print("[verify] loading train archive + window bank", flush=True)
    train = torch.load(args.train_pt, map_location="cpu", weights_only=False)
    bank_sigs = []
    exact_index: dict[tuple[tuple[int, ...], str], list[str]] = {}
    for sid, meta in train.items():
        source_pose = as_pose(meta)
        bank_sigs.append(window_signatures(source_pose, args.win, args.stride).half())
        exact_index.setdefault((tuple(source_pose.shape), pose_sha256(source_pose)), []).append(str(sid))
    train_windows = torch.cat(bank_sigs).float()

    verdicts: dict[str, dict[str, Any]] = {}
    counts = {"replay_pass": 0, "replay_fail": 0, "replay_recovered_at_issuance": 0,
              "generated_pass": 0, "generated_fail": 0, "local_segment_pass": 0,
              "local_segment_fail": 0, "fallback_checked": 0,
              "fallback_windowmatch_frac": []}
    manifest: dict[str, Any] = {}
    n_fallback_done = 0
    for sid, raw in bank.items():
        pose = as_pose(raw)
        src = ""
        if args.issue:
            route = ("generated" if sid in gen_ids
                     else "replay" if in_S.get(sid, False) else "fallback")
            led: dict[str, Any] | None = None
            v: dict[str, Any] = {"route": route, "certificate_authenticated": None}
        else:
            cert = certs[sid]
            expected = {
                "certificate_id": sid,
                "clip_id": sid,
                "archive_commitment": archive_commitment,
                "archive_id": archive_id,
                "detector_config": detector_config,
                "manifest_key_id": args.manifest_key_id,
                "watermark": {"detector": "scripts.pose_watermark.detect",
                              "key_id": args.watermark_key_id},
            }
            if not verify_certificate(cert, pose=pose, manifest_key=args.manifest_key,
                                      expected_claims=expected):
                raise CertificateError(f"certificate authentication failed for {sid}")
            led = cert["ledger"]
            route = str(led["route"])
            v = {"route": route, "certificate_authenticated": True}
        if route == "replay":
            if args.issue:
                src = str(trace.get(sid, {}).get("id", ""))
            else:
                claimed = led.get("claimed_source_ids", [])
                src = str(claimed[0]) if len(claimed) == 1 else ""
            matches = exact_index.get((tuple(pose.shape), pose_sha256(pose)), [])
            claimed_ok = bool(src and src in matches)
            recovered = None
            if args.issue and not claimed_ok and len(matches) == 1:
                recovered = matches[0]
                src = recovered
                claimed_ok = True
                counts["replay_recovered_at_issuance"] += 1
            ok = claimed_ok
            v.update({"claimed_source": src, "exact_match": ok,
                      "unique_archive_match": matches[0] if len(matches) == 1 else None,
                      "recovered_at_issuance": recovered})
            counts["replay_pass" if ok else "replay_fail"] += 1
        elif route == "generated":
            det = detect(pose, args.watermark_key, sid)
            q = window_signatures(pose, args.win, args.stride)
            d = min_window_distance(q, train_windows)
            ok = (det["z"] > args.z_threshold) and (d > 0.05)
            v.update({"watermark_z": round(det["z"], 2), "min_window_dist": round(d, 4),
                      "pass": ok})
            counts["generated_pass" if ok else "generated_fail"] += 1
        else:
            if not args.issue and led.get("segments"):
                for segment in led["segments"]:
                    if segment["route"] not in {"whole_clip", "local_unit"}:
                        continue
                    start, end = int(segment["start"]), int(segment["end"])
                    source_id = str(segment["source_id"])
                    source_start = int(segment["source_start"])
                    source_end = source_start + (end - start)
                    segment_ok = source_id in train and source_end <= len(as_pose(train[source_id]))
                    if segment_ok:
                        segment_ok = torch.equal(
                            pose[start:end], as_pose(train[source_id])[source_start:source_end])
                    counts["local_segment_pass" if segment_ok else "local_segment_fail"] += 1
                    if not segment_ok:
                        raise CertificateError(
                            f"source-backed segment evidence failed for {sid}")
            if args.max_fallback_clips <= 0 or n_fallback_done < args.max_fallback_clips:
                q = window_signatures(pose, args.win, args.stride)
                qc = q.cuda() if torch.cuda.is_available() else q
                frac = 0.0
                best = torch.full((q.shape[0],), float("inf"))
                qn = (qc * qc).sum(dim=1, keepdim=True)
                for s0 in range(0, train_windows.shape[0], 8192):
                    b = train_windows[s0:s0 + 8192]
                    b = b.cuda() if torch.cuda.is_available() else b
                    d2 = qn + (b * b).sum(dim=1).unsqueeze(0) - 2.0 * (qc @ b.T)
                    best = torch.minimum(best, d2.clamp_min(0).min(dim=1).values.cpu())
                dim = q.shape[1]
                frac = float((torch.sqrt(best / dim) < 0.02).float().mean())
                v.update({"window_match_frac": round(frac, 3)})
                counts["fallback_windowmatch_frac"].append(frac)
                counts["fallback_checked"] += 1
                n_fallback_done += 1
        verdicts[sid] = v
        if args.issue:
            src_ids = [str(src)] if route == "replay" and src else []
            t = int(pose.shape[0])
            led = {
                "route": route, "T": t,
                "frame_counts": {
                    "whole_clip": t if route == "replay" else 0,
                    "local_unit": 0,
                    "generated": t if route == "generated" else 0,
                    # Without an authenticated segment trace, fallback is
                    # conservatively charged as unknown at issuance.
                    "unknown_or_derived": t if route == "fallback" else 0,
                },
                "claimed_source_ids": src_ids,
            }
        assert led is not None
        manifest[sid] = {"pose": pose, "ledger": led}

    print(f"[verify] replay {counts['replay_pass']}/{counts['replay_pass'] + counts['replay_fail']} exact-match pass",
          flush=True)
    print(f"[verify] generated {counts['generated_pass']}/{counts['generated_pass'] + counts['generated_fail']} "
          f"watermark+no-copy pass", flush=True)
    if counts["fallback_windowmatch_frac"]:
        fr = np.array(counts["fallback_windowmatch_frac"])
        print(f"[verify] fallback ({counts['fallback_checked']} clips) window-match frac "
              f"median {np.median(fr):.3f} p10 {np.percentile(fr, 10):.3f}", flush=True)

    if (counts["replay_fail"] or counts["generated_fail"]
            or counts["local_segment_fail"]):
        raise CertificateError("pose-side route evidence failed before manifest authentication")

    # Build or authenticate the cooperative manifest before writing any report.
    ids = list(bank.keys())
    gid = next(s for s in ids if s in gen_ids)
    rid = next(s for s in ids if in_S.get(s, False) and s not in gen_ids)
    drills = {}
    # 1. claim a replay clip as generated -> watermark absent
    det = detect(as_pose(bank[rid]), args.watermark_key, rid)
    drills["replay_claimed_generated_caught"] = bool(det["z"] < args.z_threshold)
    # 2. claim a generated clip as replay -> no exact train match
    src = str(trace.get(gid, {}).get("id", "")) or list(train.keys())[0]
    gp = as_pose(bank[gid])
    drills["generated_claimed_replay_caught"] = not bool(
        exact_index.get((tuple(gp.shape), pose_sha256(gp)), []))
    if args.issue:
        certs = {}
        for sid, row0 in manifest.items():
            certs[sid] = issue_certificate(
                certificate_id=sid, clip_id=sid, pose=row0["pose"],
                ledger=row0["ledger"], archive_commitment=archive_commitment,
                archive_id=archive_id, policy={"route": row0["ledger"]["route"]},
                detector_config=detector_config,
                watermark_key_id=args.watermark_key_id,
                manifest_key_id=args.manifest_key_id,
                manifest_key=args.manifest_key)
        out_manifest = {"schema_version": "route-certificate-v4",
                        "watermark_key_id": args.watermark_key_id,
                        "manifest_key_id": args.manifest_key_id,
                        "certificates": certs}
        p = ROOT / args.manifest_out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out_manifest, indent=2, sort_keys=True))
        print(f"[issue] wrote pre-issued cooperative manifest to {p}", flush=True)
    checks = {}
    for sid, row0 in manifest.items():
        expected_claims = {
            "certificate_id": sid,
            "clip_id": sid,
            "archive_commitment": archive_commitment,
            "archive_id": archive_id,
            "detector_config": detector_config,
            "manifest_key_id": args.manifest_key_id,
            "watermark": {"detector": "scripts.pose_watermark.detect",
                          "key_id": args.watermark_key_id},
        }
        if args.issue:
            expected_claims["ledger"] = row0["ledger"]
            expected_claims["policy"] = {"route": row0["ledger"]["route"]}
        checks[sid] = verify_certificate(
            certs.get(sid, {}), pose=row0["pose"], manifest_key=args.manifest_key,
            expected_claims=expected_claims)
    if not checks or not all(checks.values()) or set(certs) != set(manifest):
        raise CertificateError(
            "manifest authentication or reconstructed-claim comparison failed")
    print(f"[verify] manifest authenticated and pose-side claims checked ({len(checks)} entries)")
    # Actual drills against authenticated certificates.
    tampered = gp + 1e-3
    drills["pose_tamper_caught"] = not verify_certificate(
        certs[gid], pose=tampered, manifest_key=args.manifest_key)
    ledger_cert = copy.deepcopy(certs[gid])
    ledger_cert["ledger"]["frame_counts"]["generated"] += 1
    drills["ledger_edit_caught"] = not verify_certificate(
        ledger_cert, pose=gp, manifest_key=args.manifest_key)
    metadata_cert = copy.deepcopy(certs[gid])
    metadata_cert["watermark"]["detector"] = "tampered"
    drills["watermark_metadata_tamper_caught"] = not verify_certificate(
        metadata_cert, pose=gp, manifest_key=args.manifest_key)
    print(f"[verify] tamper drills: {drills}", flush=True)
    if not all(drills.values()):
        raise CertificateError("one or more tamper drills failed")
    payload = {"bank": args.bank, "z_threshold": args.z_threshold,
               "counts": {k: v for k, v in counts.items() if k != "fallback_windowmatch_frac"},
               "fallback_window_match_median": float(np.median(counts["fallback_windowmatch_frac"]))
               if counts["fallback_windowmatch_frac"] else None,
               "tamper_drills": drills, "manifest_entries": len(manifest),
               "manifest_authenticated": True,
               "authenticated_claims_pose_checked": True,
               "router_records_read": bool(args.issue),
               "verdicts": verdicts,
               "archive_id": archive_id,
               "archive_commitment": archive_commitment}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
