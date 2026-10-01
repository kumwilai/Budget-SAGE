# Our routes under the Sign-IDD / FGDM authors' SLT: STOPPED at Gate 0, no bank scored

Brief: outputs/baseline_retrain_design_2026-09-28/their_evaluator_brief.md, plus the coordinator's conversion checks
1-4 of 2026-09-29. Code: scripts/recent_baselines/their_evaluator.py, test scripts/recent_baselines/test_their_evaluator.py.
Environment: /home/kumwilai/research/coopns-slr/.venv/bin/python (s2_common.PY), the interpreter R0 ran under. None of
.venv-nslt, .venv_how2sign or .venv_pose has the TensorFlow the SLT's CTC decoder needs.
Lock: another session's training job held /tmp/signgen_heavy.lock the whole time, so every step ran through the
nolock_L.sh `L` guard (>= 9 GB RAM available, >= 7168 MiB GPU free, memcap 9G), one job at a time. Peak 2.9 GB.

## Recorded value reproduced first
The scorer run on R0's saved 642 hypotheses gives BLEU-4 11.9294 and WER 71.9418. Against the recorded 11.93 and 71.94
the difference is 0.00. The evaluator is pinned to recognition beam 10, translation beam 2, alpha -1, with the authors'
ground-truth dev skels as dev. With those settings both Gate 0 runs gave dev 12.13 / 74.17, and their dev .gls and .txt
files are byte-identical to R0's.

## Gate 0 and conversion checks
| step | result | verdict |
|---|---|---|
| check 1, keys | 641/641 SLRTP keys matched by name. The PT key with no SLRTP counterpart is 13December_2010_Monday_tagesschau-6940. The input order equals the authors' test.files order, and the references match test.text and test.gloss on 642/642. | PASS |
| check 2, agreement with the authors' PT ground truth (slrtp178_to_pt50) | Frame counts are equal on 641/641. Max abs difference x/y/z 0.800/1.023/0.996, median 0.188/0.171/0.266, against a tolerance of 1e-4. | FAIL |
| check 3, round trip on manual joints | max abs error 0.0 | PASS |
| check 4, unit test (6 tests, one on a real clip) | rc 0 | PASS |
| Gate 0 SLT, slrtp178_to_pt50 | 0.0% of hypotheses identical to R0. BLEU-4 0.60 against R0's 11.88 on the same 641 keys. WER 100.00 against 71.92 (every gloss deleted). | FAIL |
| Gate 0 SLT, fix1 candidate | 0.2% identical. BLEU-4 9.28 against 11.88. WER 85.69 against 71.92. | FAIL |

The two Gate 0 SLT runs took place before checks 1-4 were requested. Under the new order they would not have been run,
because check 2 fails.

## What check 2 found (checks.json, gate0_diag*.json)
- Side labels are swapped. The authors' PT joints 2-4 and hand 29-49 track SLRTP 3-5 and hand 29-49, and PT 5-7 and
  hand 8-28 track SLRTP 0-2 and hand 8-28 (within-clip correlation up to 0.84 on x and y). slrtp178_to_pt50 follows the
  Sign-IDD helpers.py layout, which the authors' data files do not follow.
- Normalisation differs. In the authors' data the neck is exactly (0, 0, 0) in every frame. The SLRTP neck is not
  (mean y 0.160, z 0.113). Both have nearly the same shoulder width, 0.3345 in the authors' data and 0.3307 in SLRTP.
- The depth axis has the opposite sign and a weak relation: the dev-fit R-squared for z is 0.19, against 0.85 for x
  and for y.
- The nose has no source: the SLRTP face is centred per frame.

After every deterministic fix was applied, the residual stays far above 1e-4:

| mapping | median x/y/z | max |
|---|---|---|
| side swap + per-frame neck centring, no fitted parameters (joints 1-49) | 0.048/0.038/0.382 | 1.11 |
| fix1: side swap + one affine map fitted on dev only + nose fill | 0.036/0.034/0.061 | 0.64 |

The SLRTP release and the authors' PT-150 files are two different pose estimates of the same frames. They are not one
estimate under different normalisation, so no deterministic mapping makes Gate 0 pass. Per the brief, no bank was
scored and no bootstrap was run.

## Commands (S=scripts/recent_baselines, PY=/home/kumwilai/research/coopns-slr/.venv/bin/python, `source $S/nolock_L.sh`)
```
L $PY $S/their_evaluator.py build gate0_slrtp_gt          # inputs/gate0_slrtp_gt.test.gz + config
L $PY $S/their_evaluator.py diag
L $PY $S/their_evaluator.py slt gate0_slrtp_gt            # logs/gate0_slrtp_gt.log
L $PY $S/their_evaluator.py gate0                         # gate0.json (also the R0 reproduction)
L $PY $S/their_evaluator.py fitfix                        # fix1_params.npz, fix1_fit.json (dev only)
L $PY $S/their_evaluator.py diag --fix fix1
L $PY $S/their_evaluator.py build gate0_slrtp_gt --fix fix1
L $PY $S/their_evaluator.py slt gate0_slrtp_gt --fix fix1
L $PY $S/their_evaluator.py gate0 --fix fix1              # gate0__fix1.json
L $PY $S/their_evaluator.py checks                        # checks.json, checks 1-4
# not run (Gate 0 failed): build/slt NAME for ours_fixed ours_learned ours_allgen ustcmoe darslp signidd, then score
```
