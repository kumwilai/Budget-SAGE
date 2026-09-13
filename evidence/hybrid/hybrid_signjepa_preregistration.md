# Frozen protocol: within-sequence Budget-SAGE + Sign-JEPA hybrid

Frozen: 2026-09-12 (Asia/Bangkok), before materializing or scoring the hybrid.

## Claim and scope

This experiment tests a within-sequence hybrid, not a clip-level mixture.
The current nonlearned 40% Budget-SAGE bank remains the archive-backed base.
At every recorded local-assembly join, a temporally aligned segment from the
already frozen, source-text-only Sign-JEPA bank becomes a neural bridge. Whole
replay clips and local clips without a recorded join remain unchanged.

The archive-only and all-generated endpoints remain reported. This experiment
adds one hybrid operating point and does not replace either endpoint. It does
not modify or make a claim about any human-evaluation stimulus, protocol, or
result.

## Frozen inputs

- Archive base: `outputs/revision/clean_rerank_frame40_test.pt`, SHA-256
  `c209fbac4184f3716a32de12a69aecffb2c2e70d614f14657ca085e1d7fad012`.
- Base ledger: `outputs/revision/clean_rerank_frame40_test_ledger.json`, SHA-256
  `a034d55f1dddd1a50514262eac265abe3f29626778cbd638edace1ba6fb46f22`.
- Existing archive certificate is read only to recover the already authenticated
  per-frame source segments; it is not reused as the hybrid certificate:
  `outputs/revision/astra_open_closure_20260906/nonlearned_certificate/nonlearned_manifest.json`.
  SHA-256
  `33cb1c920088bbdf80078d4745975b67f8359ebe16ea6a36778c139e98a30177`.
- Frozen generator bank:
  `outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt`,
  SHA-256
  `2307335ff0ba921034819a4b7a3b7208841a89cb224630083f2c350bfa17c912`.

The generator bank was produced before this experiment from source text via a
train-fitted mT5 gloss predictor, a train-only text-duration policy, and frozen
Sign-JEPA checkpoint. No ground-truth test gloss, duration, pose, signer, or
evaluator output entered that bank.

## Frozen hybridization rule

1. Verify identical 641-ID sets and `178 x 3` joint-coordinate layout.
2. For each base clip, group consecutive `blend_frames` in the committed ledger
   into joins. Merge joins whose eight-frame transition supports overlap.
3. Resample the frozen generated utterance to the base duration by linear
   interpolation. This supplies normalized-progress alignment only; it reads
   no held-out target.
4. For each merged join core `[a,b]`, use an eight-frame flank on each side,
   clipped at the utterance boundary. Align the generated bridge by a single
   3-D translation derived from the mean of body joints 0--7 at the two outer
   endpoints. Interpolate those endpoint translations with a quintic
   minimum-jerk smoothstep. The same translation is applied to every joint, so
   it does not change bone geometry.
5. Blend archive and aligned generated motion with a quintic weight. The weight
   is zero at the outer endpoints with zero first and second continuous-time
   derivatives, one across the join core, and reversed on exit. Thus there is
   no hard archive/generator cut. No post-score parameter adjustment is allowed.
6. Record per-frame generated weights. Archive source mass is multiplied by
   `(1-w)` and generated mass by `w`; their sum must equal every output frame.

The eight-frame flank is fixed as a 0.32-second transition at 25 fps. Every
recorded local join is bridged; there is no quality-based join selection.

## Separation between routing and scoring

The materializer may read only the three frozen inputs above. It must not load
test ground truth, decoded evaluator hypotheses, BLEU/WER/DTW values, or human
evidence. It writes a frozen pose bank, frame-weight ledger, and manifest into
`outputs/reviewer_closure_takeover_20260912/hybrid_signjepa/`.

Only after their hashes are committed may a separate scoring command open the
official test reference and evaluator. Existing endpoint scores were already
known; the bridge rule was not chosen from a hybrid score or sweep.

## Acceptance and reporting

- exactly 641 outputs, no missing or extra IDs;
- all tensors finite and shaped `[T,178,3]`;
- every original ledger join covered by a nonzero generated bridge;
- archive plus generated frame mass conserved to numerical tolerance;
- original whole-replay content unchanged byte-for-byte;
- hard-cut boundary position jumps compared with the unsmoothed archive/gen
  splice are reported, and smoothing must not worsen their aggregate median;
- BLEU-1, BLEU-4, WER, DTW-MJE, duration, hand/body speed, pose deviation, and
  jerk are reported for all 641 items without exclusions;
- failure remains a result; no second flank, checkpoint, selector, or smoothing
  variant will be tried to improve the test score.

The hybrid is not covered by the existing archive-only certificate. Its new
manifest authenticates input/output hashes and mass accounting but is not
described as independent reconstruction or public nonrepudiation. Existing
keyed-watermark evidence remains applicable to the generated endpoint; a
hybrid-specific detector claim requires a separate verified run.
