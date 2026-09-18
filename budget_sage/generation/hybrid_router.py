"""Executable archive/Sign-JEPA routing hierarchy.

The evaluated hybrid has an admissible archive base and uses ``join_hybrid``.
``full_generation`` is the deterministic escape when no archive plan is
admissible.  It is implemented here and tested separately; the 641-item hybrid
release did not need to activate that missing-plan escape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from scripts.build_budget_sage_signjepa_hybrid import (
    FLANK,
    aligned_generated,
    bridge_weight,
    join_groups,
)


@dataclass(frozen=True)
class HybridDecision:
    pose: torch.Tensor
    generated_weight: torch.Tensor
    mode: str
    bridges: tuple[tuple[int, int], ...]


def route_motion(
    generated: torch.Tensor,
    base: torch.Tensor | None,
    blend_frames: list[dict[str, Any]] | None = None,
    *,
    archive_admissible: bool = True,
    flank: int = FLANK,
    spatial_dims: int | None = None,
) -> HybridDecision:
    """Route one request without references, evaluator output, or target pose."""
    if base is None or not archive_admissible:
        pose = generated.float().contiguous().clone()
        return HybridDecision(
            pose=pose,
            generated_weight=torch.ones(pose.shape[0], dtype=torch.float32),
            mode="full_generation",
            bridges=(),
        )
    base = base.float().contiguous()
    groups = join_groups(blend_frames or [], flank=flank)
    if not groups:
        return HybridDecision(
            pose=base.clone(),
            generated_weight=torch.zeros(base.shape[0], dtype=torch.float32),
            mode="archive",
            bridges=(),
        )
    aligned = aligned_generated(
        base, generated.float(), groups, flank=flank, spatial_dims=spatial_dims
    )
    weight = bridge_weight(base.shape[0], groups, flank=flank)
    pose = ((1.0 - weight[:, None, None]) * base + weight[:, None, None] * aligned).contiguous()
    return HybridDecision(pose=pose, generated_weight=weight,
                          mode="join_hybrid", bridges=tuple(groups))
