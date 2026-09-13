# Preregistration: pose-watermark detector v2 and embedding v2

Date: 2026-09-11. Author: automated analysis run (kumwilai@gmail.com).
Interpreter: `/home/kumwilai/research/coopns-slr/.venv/bin/python`.
Working directory: `/home/kumwilai/research/signgen-t2m`.
Device: CPU for everything except a single evaluator pass (Phase 2).

Frozen and hashed BEFORE anything in this run touches the 641-item test split
beyond the diagnostic disclosed in section 1. Everything below is a
commitment, not a description of results.

## 0. Claim under test

The earlier run (`outputs/watermark_generative_route_20260911/`, preregistration
sha256 065bdc4701a77ea0871cb15cbc1b41c3660cf0e0359545ced6b00ba5fcfcf3e1)
recorded a true-positive rate of 0.000 at z>8 under the benchmark evaluator's
own x2 frame decimation, under coordinate quantization at 1e-2, and under blind
resampling at 0.5x. Two of those three are detector defects, not lost signal.
This run builds one detector (v2) that serves both the July banks and a new
embedding (v2), and measures it honestly.

## 1. Disclosure: diagnostics already run on the test marked banks

Before this document was frozen, a seed-0 40-clip subsample of the two existing
TEST marked banks was used to reproduce the diagnosis. This is disclosed here
because it is the only way the diagnosis could be confirmed, and because it
means the test split is not virgin with respect to the detector-defect
question. It selected nothing: no amplitude, no threshold, no configuration.

Recorded in `outputs/watermark_detector_v2_20260911/diagnosis_reproduction.json`
(sha256 95766c481d1ec58785dcc70523e68fcee700f2c7c685c9837c8af05e18bd06c9), produced by
`scripts/watermark_detector_v2_diagnosis.py` (sha256 93d38f0ba5b54f2e2cfe64971dd475136a13956195af778f332160bd1f20735c).
What it showed, on the 40-clip seed-0 subsample:

- **Desynchronisation confirmed.** July's detector on `p[::2]` gives z median
  1.702 at eps=1e-3. Re-indexing the chip stream to `base[t']` instead of
  `base[t'//2]` gives z median 139.590, minimum 70.754 at eps=1e-3 and
  median 51.176, minimum 21.923 at eps=2.5e-4.
- **Dead channels confirmed.** Under quantization at 1e-2, a median of 65 of
  534 channels per clip become exactly constant, hit `sig.clamp_min(1e-5)`
  and receive weight 1e10; July's z sits at median -3.825, and the algebraic
  all-dead floor is -4*sqrt(2/pi)/sqrt(1-2/pi) = -5.294, which is where the
  recorded negative pool sits. Excluding the channels whose std hits the clamp
  floor restores z to median 87.07, minimum 44.56 at eps=1e-3.
  A rule phrased purely as "more than half the residual samples are exactly
  zero" is a no-op here (z median -3.829): it misses about 12 channels per clip
  whose residual is a *constant nonzero* float-rounding value, which have std
  exactly 0 and are therefore clamped. The dead-channel rule in section 2
  therefore takes the union of the two conditions.
- A functional smoke test of detector v2 was also run on an 8-clip seed-0
  subsample of the eps=1e-3 test bank and on 2 dev clips, to confirm the code
  runs. Its numbers select nothing and are superseded by Phase 2.

## 2. Detector v2 (fixed here)

Source: `scripts/pose_watermark_v2.py`, sha256 `60dffbfd9674b0ec6b02e2398f7862b899c1dc8d8244e4d28acb9497ba930b2a`.
`scripts/pose_watermark.py` is NOT modified; the July scheme stays reproducible.

**Front end (unchanged from July).** k=9 moving-average temporal high-pass
residual r; per-channel inverse-noise-power weight w = 1/sigma^2 with
sigma = r.std(dim=0).clamp_min(1e-5), over 178 joints x 3 axes = 534 channels.

**Dead-channel rule (new).** A channel gets weight zero when more than half its
high-pass residual samples are exactly zero, OR its residual std is at or below
the 1e-5 clamp floor. Committed prediction: on unquantised float32 poses this
never fires, so clean behaviour is identical to July's; this is verified and
reported in Phase 1 and Phase 2.

**Blind hypothesis search.** L is the embedded chip length in frames (L=2 for
the July banks, L=4 for embedding v2) and is a declared property of the scheme
version, not fitted. Effective chip length L' = f*L with f on a 3% geometric
grid over [0.4, 2.5], `np.geomspace(0.4, 2.5, 63)`, 63 values; phase
phi in {0, 1/4, 1/2, 3/4} * L', 4 values; 252 hypotheses. Chip index of
received frame t' is floor((t'+phi)/L'). The pattern is `base[chip index]`
from the same SHA-256(key||sid) stream, drawn row-major so that base[:m] is the
identical prefix for any longer draw (verified). The payload block of chip i is
floor(16*i/C), C the chip count under the hypothesis, which is invariant under
uniform resampling.

**Statistic.** Per hypothesis, per-chip keyed correlations are summed into the
16 payload blocks and folded: sum_b |c_b|, standardised by its own analytic
folded-normal null (mu0 = sqrt(2/pi) sum_b sqrt(v_b), var0 = (1-2/pi) sum_b v_b,
v_b = n_b * sum_channels w). S = the maximum of that standardised folded sum
over the 252 hypotheses. Payload bits are the block signs at the argmax.

**Null absorbing the search multiplicity.** The identical maximised statistic is
computed on the same tensor under M=32 wrong keys derived as
SHA-256(key||"null"||i), i = 0..31. z_key = (S_true - mean_i S_i) / std_i S_i
with std over the 32 (ddof=1).

**Primary threshold: z_key > 8, fixed here, not moved afterwards.**

Secondary, reported for the identity condition only as the bridge to the July
numbers: the analytic July z with the dead-channel fix, identity hypothesis
(f=1, phi=0).

Binding is untouched: pattern seeded on key||sid, payload hashing sid.

**Declared limit.** The detector needs the clip identifier. A tensor arriving
with no identifier would require enumerating 641 candidates; that is not
designed, not measured and not claimed here.

## 3. Embedding v2 (fixed here)

Chip grid L = 4 frames at 25 fps. Rectangular chips on face (joints 50:178) and
body (0:8); Hann-COLA shaped chips on the same grid on hands (8:50), the window
being the 2L-long periodic Hann with hop L, centred on the rectangular chip, so
the amplitude envelope overlap-adds to 1 away from the clip edges. Measured
before freezing: RMS third difference per unit amplitude is 0.2416 for
Hann-COLA against 1.7188 rectangular at L=4, a factor 7.11; the jerk metric is
hands-only. Amplitude per channel a = alpha * sigma, sigma the channel's k=9
high-pass std in the UNMARKED clip, with alpha_face = alpha and
alpha_hand = alpha_body = alpha/2. Applied after generation; the route is
untouched. Payload and pattern binding identical to July.

Candidate amplitudes: alpha in {0.35, 0.5, 0.7}. Selected on development only.

## 4. Two tiers

- **Tier 1**: detector v2 applied to the already-existing test bank
  `outputs/watermark_generative_route_20260911/test641_mt5gloss_wm_eps2.5e-4.pt`
  with no re-embedding. Its motion cost is already recorded in
  `marked_eps2.5e-4_motion.json`.
- **Tier 2**: embedding v2 at the three alpha values.

If Tier 1 clears the gate it is the answer, because it is cheaper and better
evidenced.

## 5. Phase 1, development pilot. ALL selection happens here and nowhere else.

Bank: `outputs/revision/generative_route_restore_20260911/dev80_mt5gloss.pt`,
80 clips. Four configurations: Tier 1 (July pattern, eps=2.5e-4, L=2) and
Tier 2 at alpha in {0.35, 0.5, 0.7} (L=4).

Transform list (32), applied to the marked bank and to every pool, detection
blind:

- identity
- additive Gaussian noise, sigma in {1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2}
- temporal moving-average smoothing, k in {3, 5, 7, 9, 11, 15}
- blind temporal resampling, factor in {0.5, 0.75, 0.9, 1.1, 1.25, 1.5}
- certificate-guided resampling (resample then invert to the certified T),
  factor in {0.75, 1.25}
- coordinate quantization, step in {1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2}
- evaluator-style decimation `p[::2]`
- odd-phase decimation `p[1::2]`
- face drop: joints 50:178 set to zero

Pools, each n=80, failures kept in the denominator:

| pool | definition |
|---|---|
| P positives | marked dev clips, own key, own id |
| N1 | unmarked dev clips, own key, own id |
| N2 | marked dev clips, wrong key, own id |
| N3 | marked dev clips, own key, another clip's id (seed-0 derangement) |
| N4 | dev ground truth poses, own key, own id |

Motion ratios (hand/body/face speed, hand jerk, hand pose std) against the dev
ground truth, using the same arithmetic as the July run's `motion_ratios`.

**Selection rule, fixed in advance.** A configuration passes iff all of:

1. 80 of 80 positives above z_key 8 under BOTH decimation phases, with the
   minimum z_key over those 160 detections at least 12;
2. clean (identity) minimum z_key at least 20;
3. hand jerk ratio no worse than 1.02 x the dev80 unmarked baseline, and hand
   speed ratio within +/-0.5% of the dev80 unmarked baseline. (The quoted
   0.924 and 0.5427 are test-bank values; selection is dev-only, so the same
   multiplicative tolerances are applied to the dev80 unmarked baseline
   measured in Phase 1. This is stated before any Phase 1 number is produced.)
4. zero detections above z_key 8 across every pool and every transform, and the
   maximum z_key over pool N2 (wrong key) below 5.

Among passing configurations, choose the one with the smallest face-speed
increase over the dev80 unmarked baseline. If no Tier 2 configuration passes,
take Tier 1. **If Tier 1 fails criterion 1, stop and report; nothing is
relaxed.**

## 6. Phase 2, test split, once.

Embed the frozen configuration on
`outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt`
(unless the frozen configuration is Tier 1, in which case the existing
eps=2.5e-4 bank is used unchanged). Then:

- detection over the four pools of 641 plus the positives;
- binding: matches out of 641 and transfer detections out of 641;
- the full 32-transform blind boundary sweep on positives and all four pools;
- all five motion ratios against the test ground truth;
- one evaluator pass using the same command template as the earlier marked
  pass, differing only in input_path and --tag;
- detector v2 over BOTH existing marked banks, so the detector fix alone can be
  shown on the July artifacts.

Per transform we report: true-positive rate at z_key > 8 with a Clopper-Pearson
95% lower bound, false-positive rate per pool with a Clopper-Pearson 95% upper
bound, AUC against the unmarked pool N1, payload bit-error rate with empty
blocks counted as errors, and all five motion ratios. **Detector exceptions
count as misses for positives and as false positives for negatives.**

## 7. Supersession of the earlier preregistration's evaluator sentence

The frozen preregistration of 2026-09-11 (sha256 065bdc47...) states in section
1: "No third evaluator pass will be spent." **That sentence is explicitly
superseded here, before any Phase 2 number is produced.** The rationale is
unchanged from that document's own section 1: this is a cost measurement of a
post-hoc additive signal on an already-committed route. The route, the
checkpoint, the gloss source, the durations and the clip set are frozen and
identical; the only difference is the perturbation. No selection is made on any
back-translation metric: the configuration is frozen by the Phase 1 dev rule in
section 5, which uses only detection statistics and motion ratios, before the
evaluator is invoked at all. The earlier document forbade a third pass
precisely to prevent amplitude selection on test back-translation; that
prohibition is honoured in substance, because selection happens on dev and the
evaluator pass reports a cost that cannot change the frozen choice.

## 8. Residuals we expect to stay broken, to be reported in the same sentence as any survival claim

- quantization at 1e-2 marginal; at 2e-2 and above broken;
- smoothing at k=9 and above broken (that attack destroys the motion);
- additive noise at 1e-2 and above broken (likewise);
- **dropping the 128 face joints breaks the mark, because the mark is
  face-carried.** New declared residual.
- Non-uniform time warps are not designed for and will not be claimed.

## 9. Seeds and stopping conditions

Seeds: perturbation RNG default_rng(0); N3 derangement default_rng(0); any
subsample default_rng(0).

Stop and report, without relaxing anything, if: Tier 1 fails criterion 1 in
Phase 1; or the dead-channel rule fires on unquantised float32; or the marked
test bank fails to load under the evaluator's weights_only loader; or the
evaluator reproduction pass does not reproduce BLEU-4 12.15141308210366 and
WER 86.45392797641934 essentially exactly.
