# Independent review: Budget-SAGE / Sign-JEPA hybrid

Verdict: **PASS at the bounded claim ceiling below.**

- `route_motion()` has explicit evaluator/ground-truth-free all-generated
  semantics: `base is None` or `archive_admissible=False` returns the unchanged
  generated tensor, unit generated weights, and `mode="full_generation"`.
  The evaluated 641-item release did not exercise this branch; it is a tested
  fallback, not a measured fallback-rate result.
- The evaluated hybrid is join-level. Archive routes without joins remain
  unchanged; joined local routes blend. “Within-sequence hybrid” is supported
  for sequences with recorded local joins, not as an every-frame/all-clip
  property.
- The corrected seam audit covers all 695 merged transition windows. It
  compares complete-window peak velocity, acceleration, jerk, and velocity
  energy with the defined hard splice. Median reductions are 61.9--72.6%, and
  98.6--99.6% of bridges improve. These are discrete trajectory diagnostics,
  not visual quality, naturalness, or intelligibility.
- Recomputed invariants: 641/641 identical IDs; finite `[T,178,3]` tensors;
  47,254 total frames; archive plus generated mass
  `36,944.5 + 10,309.5 = 47,254`; all 948 recorded joins covered; 192/192
  whole-replay tensors byte-identical. Materialization loads only the frozen
  base, generated bank, ledger, and archive certificate; scoring is separate.

Permissible claim: a frozen, evaluator-independent archive/Sign-JEPA router
implements all-generated fallback when the archive is inadmissible and produces
lower discrete transition-window derivative diagnostics than the defined hard
splice on the frozen hybrid bank. It is not evidence of perceptual seam quality,
human intelligibility, certified hybrid reconstruction, watermark-verified
hybrid attribution, or observed missing-plan fallback frequency.

Reviewer: distinct read-only scientific evidence lane, 2026-09-12.
