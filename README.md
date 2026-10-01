# Budget-SAGE

Repository: <https://github.com/kumwilai/-Budget-SAGE>

Official code and reproducibility record for **Verifiable Source Routing for
Provenance-Governed Text-to-Sign Generation** by Wuttipong Kumwilaisak and
C.-C. Jay Kuo.

Budget-SAGE combines provenance-aware archive routing with a Sign-JEPA
generated-motion escape. The evaluated within-sequence hybrid uses generated
motion at recorded archive joins, applies a fixed quintic bridge, and records
fractional archive/generated source mass. The repository also contains the
cooperative HMAC certificate verifier, source-withdrawal checks, and the keyed
generated-motion watermark implementation.

## What this release supports

- deterministic routing and exact frame-budget admission;
- archive-source ledgers and cooperative reconstruction certificates;
- source-withdrawal re-issuance and tamper rejection;
- the frozen Budget-SAGE/Sign-JEPA join-hybrid implementation;
- a generated-motion watermark and its frozen detector records;
- compact PHOENIX-2014T, CSL-Daily, recent-method and raw-RGB audit records;
- the recognizer-free hand-motion ratio audit behind the Table IV speed, jerk
  and variation columns;
- the learned-route join hybrid and its paired-bootstrap intervals;
- the retimed crossfade study behind the Section VI-C bridge comparison;
- the paired-bootstrap intervals behind the Section VI-E main comparison;
- the Round 3 human-rating analysis code behind the Table V panel, with no
  participant data.
- the recent-method comparison behind the grouped Table IV rows (USTC-MoE,
  DARSLP and Sign-IDD), their Section VI-F margins and the Sign-IDD retrain
  row of Table VIII;
- the aggregate-only CSL-Daily governance counts behind the Section VI-G
  replay budget, join and source-mass sentence.

The command below verifies every released byte against `MANIFEST.sha256` and
checks the headline machine-readable evidence:

```bash
python -B verify_release.py
```

Expected final line:

```text
PASS: release integrity and declared evidence checks
```

For the focused source-only regression gate:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-verification.txt
python -m pytest -q \
  tests/test_budget_sage_signjepa_hybrid.py \
  tests/test_clean_route_certificate.py \
  tests/test_route_certificate_manifest.py \
  tests/test_motion_ratios.py
```

## Layout

- `budget_sage/`: reusable routing, generation and certificate modules.
- `scripts/`: exact research scripts retained for the reported routes and
  audits.
- `scripts/jerk_study/`: the retimed crossfade study behind Section VI-C, kept
  as its own subfolder because it is a self-contained multi-step pipeline.
- `tests/`: focused deterministic and failure-case tests.
- `evidence/hybrid/`: preregistration, materialization, seam audit, score and
  independent review for the frozen join hybrid.
- `evidence/takedown/`: independent eight-arm withdrawal re-verification.
- `evidence/watermark/`: frozen watermark and detector records.
- `evidence/signbase/`: controlled Sign-Base scoring audit.
- `evidence/verification_bundle/`: compact archived verification records.
- `evidence/motion_ratios/`: hand-motion kinematic ratio JSONs for every route
  reported in Table IV.
- `evidence/evaluator_results/`: SLRTP evaluator result records, including the
  `avg_duration` figures reported in Table IV.
- `evidence/hybrid_learned/`: the learned-route join hybrid results, its
  materialization ledger, and its paired-bootstrap intervals.
- `evidence/jerk_study/`: the retimed crossfade study results that back the
  Section VI-C bridge comparison.
- `evidence/paired_intervals/`: the paired-bootstrap intervals behind the
  Section VI-E main comparison.
- `evidence/recent_methods/`: scores, paired intervals, motion ratios, frame
  audit, decision record and reproduction records of the recent-method
  comparison, with its own README mapping each file to the paper.
- `scripts/recent_methods/`: the scripts that produced them.

## Evidence for the 26 September 2026 revision

This revision adds the code and evidence behind five parts of the paper that
the prior release did not yet cover.

The Table IV motion and duration columns come from two places. The speed,
jerk and variation ratios are produced by `scripts/exp_revision_motion_ratios.py`
and stored as JSON files under `evidence/motion_ratios/`. The duration column
is the evaluator's own `avg_duration` field, stored in the evaluator result
records under `evidence/evaluator_results/`.

The learned-hybrid row in Table IV and its confidence intervals in Section
VI-C come from `evidence/hybrid_learned/`. That folder holds the route's
scored results, its paired-bootstrap interval file, its materialization
ledger, and its per-clip provenance ledger.

The Section VI-C crossfade comparison, the values 1.061, 1.048, 0.996 and
0.991, and the variation and DTW evidence behind them, come from the retimed
crossfade study. Its code is `scripts/jerk_study/`, and its results are
`evidence/jerk_study/rcx_results_test641.json` and
`evidence/jerk_study/rcx_summary.json`.

The Section VI-E main-comparison paired intervals come from
`evidence/paired_intervals/`. Both files there were produced by
`scripts/bootstrap_clean_route_comparisons.py`, which is already part of this
release, as recorded in each file's own embedded `meta.implementation` field
and its matching sha256.

The Table V Round 3 rating panel is produced by
`scripts/analyse_signer_round.py`. Only the analysis code is published here.
No participant response, token, or identifier is included. The two
participant codes that appear in the script, P288 and P289, are the research
team's own test accounts, excluded from the analysis by id, and were already
documented that way in the prior release's protocol.


## Evidence added on 27 September 2026 (v1.1.1)

Table IX and Section VI-G report the CSL-Daily development study on 1,077
requests. Its full analysis record holds material derived from the licensed
CSL-Daily corpus, so this release carries an aggregate-only summary,
`evidence/csl_daily_dev/transfer_summary_public_v1.json`, written by
`scripts/summarize_csl_dev1077_transfer.py`. The summary gives the WER and edit
counts of every route and recognizer head, the paired bootstrap intervals, the
preregistered gate outcome, and the SHA-256 of the full record it was read
from. The native-motion calibration of Section VI-G is
`evidence/csl_daily_dev/calibration_analysis_v1.json`.

## Evidence added on 1 October 2026 (v1.2.0)

The 30 September 2026 revision scores three recent systems, USTC-MoE, DARSLP
and Sign-IDD, on the same 641 PHOENIX-2014T requests as our routes. Their
Table IV rows, the Section VI-F margins, the Sign-IDD retrain row of Table
VIII and the margins table of the response letter come from
`evidence/recent_methods/`. Its README maps every file to the sentence or
table cell it supports, and `verify_release.py` checks every printed value.
The code is in `scripts/recent_methods/`. Text predictions, pose banks and
checkpoints are not included.

The Section VI-G governance sentence is backed by
`evidence/csl_daily_dev/governance_public_v1.json`, written by
`scripts/summarize_csl_dev1077_governance.py`. It holds counts only: the fixed
route's 35,284 whole-replay frames of 88,240 (39.986%, within the 2/5
budget), the hybrid's 2,151 recorded and bridged joins, its generated mass of
25.679%, and a mass-conservation check recomputed over all 1,077 ledger rows.
No request identifier or per-request row from the licensed corpus is
included.

## Data and checkpoints

PHOENIX-2014T, CSL-Daily, the SLRTP evaluator, pretrained recognizers, pose
archives and model checkpoints are not redistributed. Obtain them from their
respective custodians and follow their licenses. Released manifests preserve
hashes and input contracts so authorized holders can check local copies.

## Human evaluation instrument

The web instrument used for the human evaluation is published separately, so
that the study materials stay distinct from this code release:

- instrument source: <https://github.com/kumwilai/sign-eval>
- live instrument: <https://kumwilai.github.io/sign-eval/>

That repository contains the participant-facing evaluation page only. No
participant response, identifier, access token, or collected result is
published there or here. Human-evaluation results are not reported in this
release; automatic recognition and trajectory metrics in this repository are
not evidence of human sign-language intelligibility.

## Claim boundaries

This release reproduces the current artifact-level routing, reconstruction,
verification and reported-evidence checks. It does not recreate missing
historical pose producers, continuous historical posterior arrays, or old
recognizer training. HMAC certificates provide authorized cooperative
integrity, not public nonrepudiation. Automatic recognition and trajectory
metrics are not evidence of human sign-language intelligibility.

## Version and citation

Prepared release: `v1.0.0` (2026-09-13). See `CITATION.cff`.

Released under the MIT License; see `LICENSE`. Dataset annotations, checkpoints,
third-party evaluators and third-party code remain governed by their original
licenses and are not relicensed here.
