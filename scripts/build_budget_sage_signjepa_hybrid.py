"""Build the frozen within-sequence Budget-SAGE/Sign-JEPA hybrid.

This materializer deliberately has no evaluator or ground-truth inputs.  It
replaces every recorded local-assembly join by a temporally aligned Sign-JEPA
bridge and uses an eight-frame quintic cross-fade on each side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/revision/clean_rerank_frame40_test.pt"
GENERATED = ROOT / "outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt"
LEDGER = ROOT / "outputs/revision/clean_rerank_frame40_test_ledger.json"
CERT = ROOT / "outputs/revision/astra_open_closure_20260906/nonlearned_certificate/nonlearned_manifest.json"
OUT = ROOT / "outputs/reviewer_closure_takeover_20260912/hybrid_signjepa"
FLANK = 8
BODY = slice(0, 8)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def smoothstep5(x: torch.Tensor) -> torch.Tensor:
    """Quintic 6x^5-15x^4+10x^3, with zero slope/curvature at 0 and 1."""
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def resize_pose(pose: torch.Tensor, length: int) -> torch.Tensor:
    if int(pose.shape[0]) == int(length):
        return pose.float().clone()
    flat = pose.float().reshape(pose.shape[0], -1).T.unsqueeze(0)
    out = F.interpolate(flat, size=int(length), mode="linear", align_corners=True)
    return out.squeeze(0).T.reshape(int(length), 178, 3).contiguous()


def join_groups(blend_frames: list[dict[str, Any]], flank: int = FLANK) -> list[tuple[int, int]]:
    frames = sorted({int(row["frame"]) for row in blend_frames})
    groups: list[list[int]] = []
    for frame in frames:
        if not groups or frame > groups[-1][1] + 1:
            groups.append([frame, frame])
        else:
            groups[-1][1] = frame
    merged: list[list[int]] = []
    for start, end in groups:
        if merged and start - flank <= merged[-1][1] + flank:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def bridge_weight(length: int, groups: list[tuple[int, int]], flank: int = FLANK) -> torch.Tensor:
    weight = torch.zeros(int(length), dtype=torch.float32)
    for start, end in groups:
        left, right = max(0, start - flank), min(length - 1, end + flank)
        if start > left:
            u = torch.linspace(0.0, 1.0, start - left + 1)
            weight[left:start + 1] = smoothstep5(u)
        else:
            weight[start] = 1.0
        weight[start:end + 1] = 1.0
        if right > end:
            u = torch.linspace(1.0, 0.0, right - end + 1)
            weight[end:right + 1] = smoothstep5(u)
    return weight


def aligned_generated(base: torch.Tensor, generated: torch.Tensor,
                      groups: list[tuple[int, int]], flank: int = FLANK) -> torch.Tensor:
    """Apply a geometry-preserving, minimum-jerk body-anchor translation."""
    out = resize_pose(generated, int(base.shape[0]))
    for start, end in groups:
        left, right = max(0, start - flank), min(base.shape[0] - 1, end + flank)
        dl = (base[left, BODY] - out[left, BODY]).mean(dim=0)
        dr = (base[right, BODY] - out[right, BODY]).mean(dim=0)
        u = smoothstep5(torch.linspace(0.0, 1.0, right - left + 1)).view(-1, 1)
        shift = (1.0 - u) * dl.view(1, 3) + u * dr.view(1, 3)
        out[left:right + 1] += shift[:, None, :]
    return out


def source_weights(length: int, cert: dict[str, Any],
                   blend_frames: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = [{} for _ in range(length)]
    for seg in cert["ledger"]["segments"]:
        source = str(seg["source_id"])
        for frame in range(max(0, int(seg["start"])), min(length, int(seg["end"]))):
            rows[frame] = {source: 1.0}
    for rec in blend_frames:
        frame = int(rec["frame"])
        if 0 <= frame < length:
            rows[frame] = {str(k): float(v) for k, v in rec["sources"].items() if float(v) > 0}
    return rows


def seam_diagnostics(base: torch.Tensor, aligned: torch.Tensor, hybrid: torch.Tensor,
                     groups: list[tuple[int, int]]) -> list[dict[str, float]]:
    rows = []
    for start, end in groups:
        hard = []
        smooth = []
        if start > 0:
            hard.append(float(torch.linalg.vector_norm(aligned[start] - base[start - 1], dim=-1).mean()))
            left = max(1, start - FLANK + 1)
            smooth.append(float(torch.linalg.vector_norm(hybrid[left] - hybrid[left - 1], dim=-1).mean()))
        if end + 1 < base.shape[0]:
            hard.append(float(torch.linalg.vector_norm(base[end + 1] - aligned[end], dim=-1).mean()))
            right = min(base.shape[0] - 1, end + FLANK)
            smooth.append(float(torch.linalg.vector_norm(hybrid[right] - hybrid[right - 1], dim=-1).mean()))
        rows.append({
            "start": start, "end": end,
            "hard_boundary_jump_mean": sum(hard) / max(1, len(hard)),
            "smooth_outer_jump_mean": sum(smooth) / max(1, len(smooth)),
        })
    return rows


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    x = sorted(values)
    n = len(x)
    return x[n // 2] if n % 2 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    base = torch.load(BASE, map_location="cpu", weights_only=True)
    generated = torch.load(GENERATED, map_location="cpu", weights_only=True)
    ledger = json.loads(LEDGER.read_text())
    cert_manifest = json.loads(CERT.read_text())
    by_id = {str(row["id"]): row for row in ledger["clips"]}
    if set(base) != set(generated) or set(base) != set(by_id):
        raise RuntimeError("base/generated/ledger ID sets differ")
    if set(base) != set(cert_manifest["certificates"]):
        raise RuntimeError("certificate ID set differs")

    hybrid: dict[str, torch.Tensor] = {}
    clip_rows = []
    source_totals: Counter[str] = Counter()
    generated_mass = archive_mass = 0.0
    original_joins = merged_bridges = 0
    hard_jumps: list[float] = []
    smooth_jumps: list[float] = []
    unchanged_whole = 0

    for sid, raw_base in base.items():
        a = raw_base.float().contiguous()
        g = generated[sid].float().contiguous()
        if a.ndim != 3 or tuple(a.shape[1:]) != (178, 3) or g.ndim != 3 or tuple(g.shape[1:]) != (178, 3):
            raise RuntimeError(f"invalid pose layout for {sid}: {tuple(a.shape)}, {tuple(g.shape)}")
        if not torch.isfinite(a).all() or not torch.isfinite(g).all():
            raise RuntimeError(f"nonfinite input for {sid}")
        rec = by_id[sid]
        raw_groups = join_groups(rec.get("blend_frames", []), flank=0)
        groups = join_groups(rec.get("blend_frames", []), flank=FLANK)
        original_joins += len(raw_groups)
        merged_bridges += len(groups)
        weight = bridge_weight(int(a.shape[0]), groups)
        if groups:
            aligned = aligned_generated(a, g, groups)
            h = ((1.0 - weight[:, None, None]) * a + weight[:, None, None] * aligned).contiguous()
            diag = seam_diagnostics(a, aligned, h, groups)
            hard_jumps.extend(row["hard_boundary_jump_mean"] for row in diag)
            smooth_jumps.extend(row["smooth_outer_jump_mean"] for row in diag)
        else:
            h = a.clone()
            diag = []
        if not torch.isfinite(h).all():
            raise RuntimeError(f"nonfinite hybrid output for {sid}")
        if rec["mode"] == "whole_clip_replay":
            if not torch.equal(h, a):
                raise RuntimeError(f"whole replay changed: {sid}")
            unchanged_whole += 1
        hybrid[sid] = h

        frame_sources = source_weights(int(a.shape[0]), cert_manifest["certificates"][sid], rec.get("blend_frames", []))
        clip_sources: Counter[str] = Counter()
        for frame, sw in enumerate(frame_sources):
            keep = 1.0 - float(weight[frame])
            for source, mass in sw.items():
                clip_sources[source] += keep * mass
        source_totals.update(clip_sources)
        gm = float(weight.sum())
        am = float(a.shape[0]) - gm
        generated_mass += gm
        archive_mass += am
        clip_rows.append({
            "id": sid, "frames": int(a.shape[0]), "original_mode": rec["mode"],
            "recorded_joins": len(raw_groups), "merged_bridges": len(groups),
            "generated_mass": gm, "archive_mass": am,
            "generated_fraction": gm / max(1, int(a.shape[0])),
            "source_mass": dict(sorted(clip_sources.items())),
            "bridges": [{"start": x, "end": y} for x, y in groups],
            "seam_diagnostics": diag,
        })

    total_frames = sum(int(x.shape[0]) for x in hybrid.values())
    if abs((archive_mass + generated_mass) - total_frames) > 1e-4:
        raise RuntimeError("archive/generated mass is not conserved")
    if original_joins != 948:
        raise RuntimeError(f"expected 948 recorded joins, got {original_joins}")
    if unchanged_whole != int(ledger["whole_clip_replay_clips"]):
        raise RuntimeError("whole replay identity count differs")

    pose_path = OUT / "budget_sage_signjepa_hybrid_test.pt"
    ledger_path = OUT / "budget_sage_signjepa_hybrid_test_ledger.json"
    torch.save(hybrid, pose_path)
    hybrid_ledger = {
        "schema": "budget-sage-signjepa-hybrid-ledger-v1",
        "n_clips": len(hybrid), "total_frames": total_frames,
        "recorded_joins": original_joins, "merged_bridges": merged_bridges,
        "flank_frames": FLANK, "generated_mass": generated_mass,
        "archive_mass": archive_mass,
        "generated_frame_fraction": generated_mass / total_frames,
        "archive_frame_fraction": archive_mass / total_frames,
        "whole_replay_clips_unchanged": unchanged_whole,
        "unique_archive_sources": len(source_totals),
        "source_mass": dict(sorted(source_totals.items())),
        "seam": {
            "n": len(hard_jumps),
            "hard_boundary_jump_median": median(hard_jumps),
            "smooth_outer_jump_median": median(smooth_jumps),
            "median_reduction_fraction": 1.0 - median(smooth_jumps) / max(median(hard_jumps), 1e-12),
        },
        "clips": clip_rows,
    }
    ledger_path.write_text(json.dumps(hybrid_ledger, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "budget-sage-signjepa-hybrid-manifest-v1",
        "status": "authenticated provenance record; not an archive reconstruction certificate",
        "policy": "bridge every recorded local-assembly join; no evaluator-based selection",
        "flank_frames": FLANK, "fps": 25,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in (BASE, GENERATED, LEDGER, CERT)},
        "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in (pose_path, ledger_path)},
        "implementation": str(Path(__file__).relative_to(ROOT)),
        "implementation_sha256": sha256(Path(__file__)),
        "forbidden_route_inputs_loaded": [],
        "summary": {k: hybrid_ledger[k] for k in (
            "n_clips", "total_frames", "recorded_joins", "merged_bridges",
            "generated_mass", "archive_mass", "generated_frame_fraction",
            "whole_replay_clips_unchanged", "unique_archive_sources", "seam")},
    }
    manifest_path = OUT / "materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    args = parser.parse_args()
    if args.command == "build":
        build()


if __name__ == "__main__":
    main()
