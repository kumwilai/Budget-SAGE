#!/usr/bin/env bash
# Frozen SLRTP evaluator guard for RCX. Same harness/cwd convention as the earlier jerk-study runs.
set -uo pipefail
R=/home/kumwilai/research/signgen-t2m
P=$R/external/SLRTP-Sign-Production-Evaluation
D=$P/pretrained/SLRTP-Sign-Production-Evaluation-Data
W=$R/outputs/jerk_study/evaluator_workspace
PY=/home/kumwilai/research/coopns-slr/.venv/bin/python
mkdir -p "$W/results"
run () {  # run <pred.pt> <split> <tag>
  cd "$W" || exit 1
  PYTHONPATH="$P" $PY -u "$P/main.py" "$1" "$D/data/$2.pt" "$D/backTranslation_PHIX_model" --tag "$3" 2>&1 | tail -8
}
run "$R/outputs/revision/revision_clean_phrase_lattice_test.pt" test rcx_repro_frozen_local_test
run "$R/outputs/jerk_study/jerkstudy_local_rcx_h2_test.pt"      test jerkstudy_local_rcx_h2_test
run "$R/outputs/jerk_study/jerkstudy_cacroute_rcx_h2_test.pt"   test jerkstudy_cacroute_rcx_h2_test
