# Preregistration: keyed pose watermark on the restored generative route (test-641)

Date: 2026-09-11. Author: automated analysis run (kumwilai@gmail.com).
Interpreter: `/home/kumwilai/research/coopns-slr/.venv/bin/python`.
Working directory: `/home/kumwilai/research/signgen-t2m`.

This document is frozen and hashed BEFORE the watermark is embedded and before
any marked artifact is scored. Everything below is a commitment, not a
description of results.

## 0. Purpose and scope

The IEEE TMM revision withdrew the keyed pose watermark together with the
leaking generated branch. A leak-free generative route now exists and covers
all 641 test clips. This run re-instates the watermark on that clean bank and
measures (a) what it costs and (b) where it breaks. It makes no claim about
adversarial watermark removal.

The certificate and the watermark answer different questions and neither
subsumes the other. The certificate answers "does this tensor match the recipe
its issuer declared" and needs a recipe, an archive and a key. The watermark
answers "did this tensor come from this issuer's generative branch" for an
output that arrives with no recipe at all.

## 1. Protocol status of the evaluator pass (declared explicitly)

The single preregistered test pass for the restored generative route has
already been spent on the UNMARKED bank
`outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt`
(recorded in `.../evaluator_workspace/results/test641_mt5gloss_record.json`).

Scoring a marked copy of that identical bank is a **cost measurement of a
post-hoc signal on an already-committed route**. It is NOT a new operating
point, NOT a model or route selection, and NOT a method change. The route,
the checkpoint, the gloss source, the durations and the clip set are all
frozen and identical to the committed run; the only difference is an additive
perturbation of magnitude eps applied after generation. No number produced by
this run may be used to choose between routes, checkpoints, or decoding
settings, and none will be.

Two evaluator invocations will be made, in this order:

- **Pass A (reproduction).** Re-score the unmarked bank byte-identically to the
  committed run. This produces no new information about any new artifact; its
  only purpose is to establish that the evaluator is deterministic so that the
  marked-minus-unmarked difference is attributable to the mark and not to
  evaluator jitter. If Pass A does not reproduce the recorded BLEU-4 of
  12.15141308210366 and WER of 86.45392797641934 essentially exactly, the run
  stops and the failure is reported.
- **Pass B (cost).** Score the marked bank with the same command template,
  differing only in `input_path` and `--tag`.

No third evaluator pass will be spent. In particular, **no watermark parameter
will be chosen on the basis of any back-translation metric.** A secondary
amplitude (section 2.1) will be characterised on CPU only (detection statistics
and motion ratios) and will explicitly NOT be scored, precisely so that no
amplitude is selected on the test set.

## 2. Watermark parameters (fixed here, not tunable afterwards)

The scheme is used exactly as implemented in `scripts/pose_watermark.py`, with
the functions `_pattern`, `_payload_bits`, `embed`, `_highpass`, `detect`,
`_resample` imported unmodified. That file will not be edited.

| parameter | value | same as July? |
|---|---|---|
| amplitude `eps` | 1e-3 | yes (July's `pose_watermark_audit.json` records `eps: 0.001`) |
| payload length `n_blocks` | 16 bits over 16 equal temporal blocks | yes |
| pattern block `frame_block` | 2 frames, antipodal +/-1, shape [T,178,3] | yes |
| pattern seed | SHA256(key \|\| sid), clip-specific | yes |
| payload | first 16 bits of SHA256(json of `{"sid","route","bank"}`) | yes |
| detector | keyed matched filter on the k=9 temporal high-pass residual, inverse-noise-power weighted, folded-normal z | yes |
| key | the repository development key already hardcoded as the `--key` default in `scripts/pose_watermark.py` | yes |

The payload binds to the clip's own identity through two independent paths: the
pattern is seeded on `key||sid`, and the payload bits are the hash of a row
containing `sid`. A mark lifted from one clip therefore cannot authenticate
another; section 5 tests this rather than assuming it.

**On the key.** The key used here is a development key that is present in the
repository, so its identifier (SHA-256 of the key string, first 16 hex
characters) is recorded but the value is not restated in any output. This means
the audit measures *cooperative* provenance -- the threat model stated in the
script's own docstring, honest-but-unverified deployers -- and not secrecy
against an adversary who can read our source tree. A release deployment would
use a secret key; nothing in the statistics below depends on the key's value,
because the pattern is a PRNG draw seeded from it, so any fixed key gives
statistically identical results.

### 2.1 Declared-in-advance concern about the amplitude on this bank

Measured before freezing this document, on a seed-0 random 40-clip subsample of
the unmarked bank (command and output recorded in the run log): the bank's
median per-joint high-pass standard deviation is 7.939e-4, its mean per-frame
joint displacement is 1.689e-3, and its per-axis pose standard deviation is
0.1015. The real test poses give a mean per-frame displacement of 4.131e-3, so
this generated bank is about 2.4x smoother than real motion.

Consequently eps=1e-3 is 0.98% of the per-axis pose standard deviation, which
matches the ~1% figure in July's docstring, but it is **59.2% of the mean
per-frame joint displacement and 1.26x the bank's own median high-pass
standard deviation**, where July's docstring assumed ~15% of frame
displacement. The mark is therefore larger than the high-frequency content of
the signal it is hiding in. We predict, in advance, that detection will be
very strong and that the hand *jerk* ratio will move materially while the
hand *speed* and *pose standard deviation* ratios move little.

We keep eps=1e-3 as the primary, preregistered condition anyway, because the
instruction is to measure July's scheme at July's parameters on this bank, and
because a cost measurement is only informative if the parameter is not chosen
after seeing the cost. A secondary amplitude eps=2.5e-4, which restores the
~15%-of-frame-displacement operating point, will be characterised on CPU only
(detection + motion ratios, no evaluator pass, no shipping claim).

## 3. Detection threshold (fixed here)

- **Primary operating threshold: z > 8.** This is July's fixed threshold, used
  in `scripts/watermark_breaking_point.py` and in the July binding audit. It is
  fixed before seeing any z value from this bank and will not be moved.
- Secondarily we report the empirical true-positive rate at zero false
  positives, defined as the fraction of positives strictly above the maximum z
  of the stated negative pool, and the area under the ROC curve. Every such
  number will be reported with the negative pool named and its denominator
  given.

## 4. Negative pools (declared, because this bank has no unmarked clips in it)

July's bank was mixed: 192 generated clips were marked and 449 retrieved clips
were left unmarked and served as the negative class. **This bank is 100%
generated, so all 641 clips are marked and there is no in-bank negative.** The
negative pools are therefore stated in advance:

| pool | definition | n |
|---|---|---|
| N1 unmarked source | the 641 pre-mark clips, own key, own clip id | 641 |
| N2 wrong key | the 641 marked clips, wrong key, own clip id | 641 |
| N3 wrong clip id (transfer) | the 641 marked clips, own key, another clip's id (seed-0 derangement) | 641 |
| N4 real motion | the 641 real test poses from the evaluator's ground truth, own key, own id | 641 |

Positives are the 641 marked clips under own key and own clip id, n=641.
Failures stay in the denominator: any clip that errors is counted as a
detection failure, not dropped.

## 5. Binding check

For every marked clip we test whether its own key/id pair authenticates it
(z > 8) and whether the same tensor authenticates under another clip's identity
(pool N3). We report matches out of 641 and transfer detections out of 641, the
own-identity z distribution, and the maximum other-identity z.

## 6. Transform sweep (fixed list, run whatever the outcome)

All applied to the marked bank; detection then run blind, i.e. the detector is
not told the transform.

- additive Gaussian noise, sigma in {1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2}
- temporal moving-average smoothing, kernel k in {3, 5, 7, 9, 11, 15}
- blind temporal resampling, factor in {0.5, 0.75, 0.9, 1.1, 1.25, 1.5}
- certificate-guided resampling (resample then invert to the certified frame
  count), factor in {0.75, 1.25}
- coordinate quantization, step in {1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2}
- evaluator-style decimation by 2 (the frame stride the official evaluator
  itself applies to a 25 fps bank), detected on the decimated stream

For every transform we report, side by side, the detection outcome (AUC against
pool N1, true-positive rate at the fixed z>8 threshold with its Clopper-Pearson
95% lower bound, payload bit-error rate) **and** the damage the transform does
to the motion itself (hand speed ratio, hand jerk ratio, hand pose standard
deviation ratio against the real test poses, plus mean absolute deviation from
the marked bank). A transform that defeats the mark is only a problem if it
leaves the signal usable, so the two columns are reported together and the
failure boundary is stated as the headline of the security claim, not as a
footnote.

## 7. What we commit to report, whatever it is

- The full cost table, every evaluator metric before and after, plus all five
  motion ratios before and after. If the mark costs more than it is worth on
  this bank we will say so plainly and recommend against shipping it.
- The failure boundary as an explicit table, in the report body **and** in any
  summary sentence. July's abstract claimed detection "under tested benign
  transforms" while its own body reported a true-positive rate of 0.72 under
  k=9 low-pass filtering and an AUC of 0.60 under 5e-3 quantization. We commit
  that no summary sentence produced from this run will claim robustness beyond
  the operating region the table establishes, and that the first failing
  transform will be named in the same sentence as any survival claim.
- Denominators everywhere, with failures counted in.
- Full provenance: commands, input paths with SHA-256, key identifier (never
  key material), interpreter, seeds.

## 8. Seeds

- perturbation RNG: numpy `default_rng(0)`
- derangement for pool N3: numpy `default_rng(0)`
- the amplitude pre-check subsample: numpy `default_rng(0)`

## 9. Stopping conditions

- Pass A fails to reproduce the recorded unmarked metrics essentially exactly:
  stop and report.
- The marked bank fails to load under the evaluator's `weights_only=True`
  loader: stop and report.
