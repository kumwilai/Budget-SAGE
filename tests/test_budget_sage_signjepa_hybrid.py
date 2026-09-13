import torch

from scripts.build_budget_sage_signjepa_hybrid import (
    aligned_generated,
    bridge_weight,
    join_groups,
    smoothstep5,
)


def test_quintic_end_conditions():
    x = torch.tensor([0.0, 1.0])
    y = smoothstep5(x)
    assert torch.equal(y, x)


def test_join_merge_and_weight_are_deterministic():
    rows = [{"frame": x} for x in (10, 11, 12, 13, 26, 27, 28, 29)]
    assert join_groups(rows, flank=8) == [(10, 29)]
    w = bridge_weight(40, [(10, 29)], flank=8)
    assert w[2] == 0 and w[10] == 1 and w[29] == 1 and w[37] == 0
    assert torch.all((w >= 0) & (w <= 1))


def test_body_anchor_alignment_preserves_generated_geometry():
    base = torch.zeros(30, 178, 3)
    gen = torch.randn(20, 178, 3)
    out = aligned_generated(base, gen, [(10, 13)], flank=8)
    # A shared translation cannot alter within-frame joint differences.
    resized_delta = out[10, 20] - out[10, 21]
    from scripts.build_budget_sage_signjepa_hybrid import resize_pose
    reference = resize_pose(gen, 30)
    assert torch.allclose(resized_delta, reference[10, 20] - reference[10, 21])


def test_router_uses_full_generation_when_archive_is_unavailable():
    from budget_sage.generation.hybrid_router import route_motion
    gen = torch.randn(17, 178, 3)
    decision = route_motion(gen, None)
    assert decision.mode == "full_generation"
    assert torch.equal(decision.pose, gen)
    assert torch.equal(decision.generated_weight, torch.ones(17))


def test_router_blends_when_archive_has_a_join():
    from budget_sage.generation.hybrid_router import route_motion
    base = torch.zeros(30, 178, 3)
    gen = torch.ones(20, 178, 3)
    rows = [{"frame": frame} for frame in range(12, 16)]
    decision = route_motion(gen, base, rows)
    assert decision.mode == "join_hybrid"
    assert decision.generated_weight[12] == 1
    assert decision.generated_weight[4] == 0
