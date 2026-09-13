"""Audit full transition-window dynamics against an unsmoothed hard splice."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_budget_sage_signjepa_hybrid import FLANK, aligned_generated, join_groups


def window_stats(x: torch.Tensor, left: int, right: int) -> dict[str, float]:
    segment = x[left:right + 1].float()
    values = {}
    for name, order in (("velocity", 1), ("acceleration", 2), ("jerk", 3)):
        delta = torch.diff(segment, n=order, dim=0)
        per_frame = torch.linalg.vector_norm(delta, dim=-1).mean(dim=-1)
        values[f"peak_{name}"] = float(per_frame.max()) if per_frame.numel() else 0.0
    velocity = torch.linalg.vector_norm(torch.diff(segment, dim=0), dim=-1).mean(dim=-1)
    values["velocity_energy"] = float((velocity * velocity).sum())
    return values


def main() -> None:
    base = torch.load(ROOT / "outputs/revision/clean_rerank_frame40_test.pt", weights_only=True, map_location="cpu")
    gen = torch.load(ROOT / "outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt", weights_only=True, map_location="cpu")
    hybrid = torch.load(ROOT / "outputs/reviewer_closure_takeover_20260912/hybrid_signjepa/budget_sage_signjepa_hybrid_test.pt", weights_only=True, map_location="cpu")
    ledger = json.loads((ROOT / "outputs/revision/clean_rerank_frame40_test_ledger.json").read_text())
    records = {str(row["id"]): row for row in ledger["clips"]}
    rows = []
    for sid, base_pose in base.items():
        groups = join_groups(records[sid].get("blend_frames", []), FLANK)
        if not groups:
            continue
        aligned = aligned_generated(base_pose, gen[sid], groups, FLANK)
        for start, end in groups:
            left, right = max(0, start - FLANK), min(base_pose.shape[0] - 1, end + FLANK)
            hard = base_pose.float().clone()
            hard[start:end + 1] = aligned[start:end + 1]
            rows.append({
                "id": sid, "start": start, "end": end,
                "hard": window_stats(hard, left, right),
                "smooth": window_stats(hybrid[sid], left, right),
            })
    summary = {"schema": "hybrid-seam-window-audit-v1", "n_merged_bridges": len(rows), "metrics": {}}
    for metric in ("peak_velocity", "peak_acceleration", "peak_jerk", "velocity_energy"):
        hard = np.asarray([row["hard"][metric] for row in rows])
        smooth = np.asarray([row["smooth"][metric] for row in rows])
        hm, sm = float(np.median(hard)), float(np.median(smooth))
        summary["metrics"][metric] = {
            "hard_median": hm, "smooth_median": sm,
            "median_reduction_fraction": 1.0 - sm / max(hm, 1e-12),
            "fraction_bridges_improved": float(np.mean(smooth < hard)),
        }
    summary["scope"] = (
        "Discrete trajectory diagnostics over every complete transition window; "
        "not visual quality, naturalness, intelligibility, or a human result."
    )
    out = ROOT / "outputs/reviewer_closure_takeover_20260912/hybrid_signjepa/seam_window_audit.json"
    out.write_text(json.dumps({"summary": summary, "bridges": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
