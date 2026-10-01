# Recent-method comparison evidence (v1.2.0 and v1.2.1, 1 October 2026 revision)

This folder holds the records behind the three recent systems that the
revision scores on the same 641 PHOENIX-2014T test requests as our routes:
USTC-MoE, DARSLP and Sign-IDD. They support

- the grouped Table IV rows for USTC-MoE, DARSLP and Sign-IDD, and the motion
  columns of the Sign-Base row beside them;
- the Section VI-F margins of the fixed route, the all-generated route and
  the retrieval-against-retrieval comparison;
- the "our retrain" Sign-IDD row of Table VIII;
- the margins table and the statements of Comment 3.5 in the response letter.

Every file is a byte-identical copy of a record in the authors' working tree
(`outputs/recent_baselines_2026-09-28/`), except `LEDGER.md`, which is that
tree's `RESULTS.md` copied unchanged under a new name, and the files in
`aggregates/`, which the summary script writes directly. `verify_release.py`
(`verify_recent_methods_evidence`) checks every number below that the
manuscript or the letter prints, at its printed rounding.

All scores come from the SLRTP evaluator (`main.py --fps 25`) on 641 test
requests. All paired intervals are 10,000 paired bootstrap resamples with seed
30373, reported as resample means and 95% percentile intervals, the numbers
the manuscript prints. Files marked `LETTERONLY` are diagnostics that use
ground-truth glosses or lengths. They appear in no table, and the response
letter does not cite them.

Text predictions (`*_text_preds.pt`), generated pose banks, checkpoints and
the per-item tag files of USTC-MoE are not released. The text predictions and
tag files hold back-translated or predicted German text, so this per-item
corpus text is withheld. The counts taken from the tag files and from two
other withheld per-item records are released as aggregates in `aggregates/`.

## scores/

Each run has three files: the evaluator result `<run>.json`, the identity row
`identity_test_before_<run>.json`, scored on the ground truth just before the
run, and `score_<run>_summary.json`, which records the item coverage, the
prediction and ground-truth SHA-256 and a zero identity difference.

| run | supports |
|---|---|
| `ustcmoe_mt5_natlen_test_fps25` | Table IV USTC-MoE row (36.15, 12.73, 89.02, 0.04492, duration 1.323) |
| `darslp_released_periodfree_test_fps25` | Table IV DARSLP row (32.53, 10.63, 93.32, 0.03912, duration 1.167). The letter's 10.63 and 0.0391 |
| `signidd_gate4best_test_seed11_fps25` | Table IV Sign-IDD row (25.56, 6.54, 96.77, 0.03844, duration 1.092) |
| `ustcmoe_mt5_policylen_test_fps25` | Letter. The pre-registered USTC-MoE run with text-predicted lengths, 11.67 BLEU-4 and 85.53 WER, and the 1.06 and 3.49 change to the length-free row |
| `LETTERONLY_ustcmoe_gtgloss_gtlen_test_fps25` | Diagnostic, not cited. The G3 diagnostic with ground-truth glosses and lengths |
| `LETTERONLY_ustcmoe_gtgloss_natlen_test_fps25` | Diagnostic, not cited. The falsifier run with ground-truth glosses and natural length |
| `ustcmoe_mt5_len_fixed_test_fps25`, `ustcmoe_mt5_len_learned_test_fps25` | Letter. USTC-MoE at the fixed-route and learned-route output lengths |
| `darslp_len_fixed_test_fps25`, `darslp_len_learned_test_fps25` | Ledger. DARSLP (with the input period) resampled to the route lengths |
| `darslp_pf_len_fixed_test_fps25`, `darslp_pf_len_learned_test_fps25` | Letter. Period-free DARSLP resampled to the route lengths |
| `signidd_len_fixed_test_fps25`, `signidd_len_learned_test_fps25` | Letter. Sign-IDD at the route lengths |

Three further files are kept here.
- `ustcmoe_mt5_policylen_test_replay_cell.json` and
  `LETTERONLY_ustcmoe_gtgloss_gtlen_test_replay_cell.json` are the replay-share
  cells of those two runs.
- `darslp_withperiod_test_fps25.json` is the DARSLP run with the input period
  kept (BLEU-4 9.03). The letter cites it as the run scored before the
  period-free row was adopted. It is `darslp/results/darslp_test_fps25.json` in
  the working tree, renamed here so that its role is clear.

## bootstrap/

Fixed-route margins over each system, overall and in the two strata of the
fixed route's ledger: whole-clip replay (192 requests) and assembled (449).
These support Section VI-F and the letter's "Fixed route margin" column.

| file | margin printed |
|---|---|
| `ustcmoe_natlen_vs_fixed_paired_bootstrap_test.json` | 2.35 [0.97, 3.78] BLEU-4, 3.42 [1.02, 5.85] WER, and USTC-MoE's BLEU-1 lead of 2.80 [0.67, 4.87] |
| `ustcmoe_vs_fixed_paired_bootstrap_test.json` | the text-predicted-length WER tie, -0.06 [-2.24, 2.10] |
| `darslp_released_periodfree_test_vs_fixed_paired_bootstrap_test.json` | 4.48 [3.07, 5.92] and 7.73 [5.47, 10.05] |
| `signidd_vs_fixed_paired_bootstrap_test.json` | 8.54 [7.15, 9.99] and 11.19 [8.92, 13.48] |
| `*_vs_fixed_strata_paired_bootstrap_test.json` (4 files) | the same comparisons within each stratum (ledger) |

The other comparisons are as follows.

| file | supports |
|---|---|
| `ustcmoe_vs_learned_paired_bootstrap_test.json` | letter, learned-route margin over USTC-MoE, 4.02 and 5.38 |
| `ustcmoe_vs_local_paired_bootstrap_test.json` | letter, the text-predicted-length run ahead of local assembly by 2.48 and 5.11 |
| `retrieval_vs_ustcmoe_paired_bootstrap_test.json` | Section VI-F and letter, USTC-MoE over local assembly by 3.55 [2.40, 4.68] BLEU-4 with a WER tie, and behind unrestricted retrieval by 11.85 [10.51, 13.23] and 9.82 [7.06, 12.58] |
| `allgen_vs_baselines_paired_bootstrap_test.json` | Section VI-F and letter, all-generated route margins, 1.54 and 6.87 over DARSLP, 5.60 and 10.33 over Sign-IDD, -0.59 and 2.56 against USTC-MoE |
| `ustcmoe_lenmatch_paired_bootstrap_test.json` | letter, fixed-route margin at its own output lengths over USTC-MoE, 5.41 and 4.11 |
| `darslp_periodfree_checks_paired_bootstrap_test.json` | letter, DARSLP learned-route margin 6.15 and 9.69, and own-length margin 6.47 and 6.32 |
| `signidd_lenmatch_paired_bootstrap_test.json` | letter, Sign-IDD own-length margin 9.13 and 11.17 |
| `baselines_lenmatch_paired_bootstrap_test.json` | letter, Sign-IDD learned-route margin 10.21 and 13.15, plus the with-period DARSLP length checks (ledger) |

`bootstrap/drivers/` holds the short driver scripts that wrote the second
group. Each is that folder's `run.py` in the working tree, renamed after its
folder. They call `scripts/bootstrap_strict_cac.py` and
`scripts/recent_methods/s2_common.py`. The first group was written by
`scripts/recent_methods/s2_post.py bootstrap`, which calls
`scripts/bootstrap_clean_route_comparisons.py`. Each overall file records that
script's SHA-256 in `meta`.

## motion/

The recognizer-free hand-motion ratios behind the Table IV speed, jerk and
variation columns, made by `scripts/recent_methods/s2_post.py ratios`.
- The three Table IV rows: `ustcmoe_mt5_natlen_test_K_ratios.json` (1.087,
  2.324, 1.090), `darslp_released_periodfree_test_K_ratios.json` (0.760, 0.904,
  0.680) and `signidd_gate4best_test_seed11_K_ratios.json` (1.267, 4.746,
  0.621).
- `ustcmoe_mt5_policylen_test_K_ratios.json` is the text-predicted-length run.
- Before each run the Sign-Base row was reproduced as a control. Those files
  are the `repro_signbase_K_ratios_before_*.json` files (1.212, 4.739, 0.598,
  the Table IV Sign-Base row). Sign-Base's evaluator scores are in
  `evidence/signbase/results/`.

## frame_audit/

A frame-level near-copy audit, made with
`scripts/recent_methods/frame_replay_audit.py` and `audit_ctl.py`. It counts
the generated frames that lie within 0.005 of some training frame (window 1,
stride 1), with secondary 16-frame windows.
- `controls_*` holds the positive control (a retrieval bank, 100% flagged) and
  two negative controls (the ground truth and DARSLP, 0% flagged).
- The rows are the fixed route (96.08%), the learned route (96.22%), USTC-MoE
  (100% of non-padded frames, both runs), DARSLP, Sign-IDD and all-generated
  (all 0%).

These shares are recorded in the ledger. The manuscript does not print them,
so `verify_release.py` does not check them.

## decisions/

`darslp_row_choice.json` is the dev-only rule that chose between the released
DARSLP weights and a retrained variant. It was written before any test number
of the period-free or retrained candidates existed, and its `checked_absent`
field lists the test outputs it confirmed were absent. Its `.sha256` was
recorded at the same time. The rule chose the released weights. The earlier
with-period test score (`scores/darslp_withperiod_test_fps25.json`) already
existed and is disclosed in the letter.

## signidd_reproduction/

The Sign-IDD reproduction under its authors' own protocol, as numbers only.
No SLT output is included.
- `R0_check.json` shows that the authors' SLT reproduces their saved
  ground-truth test output (642 of 642 identical, BLEU-4 11.93).
- `R2_inputs.json` gives the size of the reproduction inputs.
- `R2_check.json` holds the retrain's score: 20.92 BLEU-1, 7.15 BLEU-4 and
  79.08 WER against the published 9.08 and 76.66. These are the Table VIII
  "our retrain" row and the 1.93-point shortfall in Section VI-F.

## cross_evaluator/

The attempt to score our outputs with the Sign-IDD and FGDM authors'
back-translator: `gate0.json`, `gate0__fix1.json`, `checks.json` and the
working `README.md`. Even the converted reference poses scored 9.28 BLEU-4
against 11.88 for the authors' own pose files of the same 641 requests, as the
letter states. That gate failed, so no route was scored this way. The code is
`scripts/recent_methods/their_evaluator.py` and `slrtp_pt_mapping.py`.

## aggregates/

Aggregate-only records of three counts whose per-item sources hold licensed
PHOENIX-2014T text or clip identifiers, written by
`scripts/recent_methods/summarize_exception_records.py`. Each holds the
counts, the denominator, the share, conservation checks recomputed over
every source row, the SHA-256 of each source record and the SHA-256 of the
script. The per-item sources are withheld.

| file | supports |
|---|---|
| `ustcmoe_whole_replay_aggregate.json` | Table IV USTC-MoE whole replay, 32 clips and 1,538 of 81,467 frames (1.888%). The denominator equals the released `frame_audit/ustcmoe_natlen_win1_stride1_thr0.005.json` |
| `unrestricted_retrieval_whole_replay_aggregate.json` | Table IV unrestricted retrieval whole replay and Sections VI-B and VII-A, 632 clips and 62,966 of 63,224 frames (99.592%) |
| `pt_bank_duration_aggregate.json` | Letter, Comment 3.5. 630 of 641 clips of the released Progressive Transformer test bank are exactly half the reference length, rounded up, and all 641 are within one frame of half |

## LEDGER.md

The complete step ledger of this comparison: every step's criterion, number,
denominator, status, artifact and time, in the order they ran. It was checked
for dataset text, hypothesis strings and credentials before release. The
absolute paths in it refer to the authors' machine.

## Notes

- The DARSLP rows in `frame_audit/` (`darslp_*`, and `neg_darslp` in
  `controls_*`) audit the with-period DARSLP bank. That was the bank on hand
  when the audit ran on 28 September. The Table IV row is the period-free run
  of the same released weights, made on 29 September. Its 0% replay comes
  from being a generator, not from a separate audit.
- The with-period score file is released as
  `scores/darslp_withperiod_test_fps25.json`, because the letter cites its 9.03.
- The Table IV USTC-MoE replay columns (32 clips, 1,538 frames, 1.888%) are
  counted from USTC-MoE's per-item tag file. That file holds predicted German
  glosses, so it is withheld. Its counts are released in
  `aggregates/ustcmoe_whole_replay_aggregate.json`.
- `scripts/recent_methods/` holds byte-identical copies of the working-tree
  scripts. The shell drivers name the authors' paths and lock helpers, and
  `memcap.sh` is the memory cap used for every heavy step.
