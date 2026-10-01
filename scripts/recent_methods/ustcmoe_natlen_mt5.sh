#!/usr/bin/env bash
# USTC-MoE with the shared mT5 glosses at natural length: segments concatenated, no cut, no zero pad, no length
# input of any kind. Two purposes (2026-09-28):
#   - the cheapest falsifier in fable's ustcmoe_row_placement_ruling.md (prediction: total distance 1.40 to 1.55,
#     duration 1.22 to 1.36, per-frame travel 1.12 to 1.18, BLEU-4 11.0 to 12.5, WER 90 to 97);
#   - the length-free variant the user asked for ("our method does not have length information as USTC"): the d4
#     table run takes the pinned text-only lengths, while our retrieval routes take no length input.
# Which USTC-MoE row enters Table VIII is fable's call after this lands. Every step runs behind the heavy lock.
# usage: nohup setsid ustcmoe_natlen_mt5.sh > .../ustc_moe/logs/natlen_mt5.log 2>&1 < /dev/null &
set -u
source "$(dirname "$(readlink -f "$0")")/stage2_lib_snapshot_20260928_1545.sh" light
# USTC_NOLOCK=1 (user, 2026-09-28 16:15, chose option 2 "freeze R1, run USTC-MoE now"): R1 training holds the heavy
# lock and is FROZEN via cgroup.freeze by ustcmoe_now.sh for the length of this job, so nothing else computes.
# Steps skip the lock but keep the memory cap.
if [ "${USTC_NOLOCK:-0}" = 1 ]; then L() { "$S/memcap.sh" "$@"; }; fi
T=ustcmoe_mt5_natlen_test
NAT=$U/inputs/mt5_test_naturallengths.txt
O=$U/natlen
mkdir -p $O
note "USTC-MoE mT5 natural-length run start (PID $$)"
$PY -c "
import json
t = json.load(open('$U/preds/ustcmoe_mt5_policylen_test_tags.json'))
keys = open('$INP/test_keys.txt').read().split()
nat = [t[k]['natural_len'] for k in keys]
open('$NAT', 'w').write('\n'.join(map(str, nat)) + '\n')
print('mean natural length', sum(nat) / len(nat), 'items', len(nat))
" || { stop "d USTC-MoE mT5 natural-length lengths" $? $NAT; exit 1; }
ASM="$PY $U/code/main_641.py --gloss2pose $ROOT/outputs/baselines/ustc_moe/phoenix_gloss2pose_results.pkl \
    --train_label $ROOT/external/baselines/CVPRW-SLP-2025/external/baselines/CVPRW-SLP-2025/hf_data/phoenix-2014t.train --train_pt $DATA/train.pt"
L bash -c "$ASM --keys $INP/test_keys.txt --texts $INP/test_text.txt --lengths $NAT \
           --glosses $U/inputs/mt5_test_glosses.json --out $U/preds/${T}_raw.pt --tags $U/preds/${T}_tags.json && \
           $PY $S/pred_invariants.py --pred $U/preds/${T}_raw.pt --out $U/preds/${T}.pt --keys $INP/test_keys.txt --report $U/preds/${T}_invariants.json" \
  || { stop "d USTC-MoE mT5 natural-length assembly" $? $U/preds; exit 1; }
L $PY $S/s2_inv.py --report $U/preds/${T}_invariants.json --clean $U/preds/${T}.pt --keys $INP/test_keys.txt --lengths $NAT \
    --step "d USTC-MoE mT5 natural-length invariants" --length_source "natural length of the released assembly, no length input" \
  || { stop "d USTC-MoE mT5 natural-length invariants" $? $U/preds; exit 1; }
L $PY $S/s2_score.py --split test --pred $U/preds/${T}.pt --tag ${T}_fps25 --workdir $U \
    --step "d USTC-MoE mT5 natural length, test (641), no length input" \
  || { stop "d USTC-MoE mT5 natural-length score" $? $U; exit 1; }
$PY -c "
import json, subprocess, sys
d = json.load(open('$U/results/${T}_fps25.json'))
td, du = d['total_distance'], d['avg_duration']; pf = td / du
ok = 1.40 <= td <= 1.55 and 1.22 <= du <= 1.36 and 1.12 <= pf <= 1.18
msg = f'total distance {td:.4f}, duration {du:.4f}, per-frame travel {pf:.4f}'
subprocess.run(['$PY', '$S/s2_common.py', 'record', 'd USTC-MoE mT5 natural-length falsifier (fable placement ruling)',
                'total distance 1.40 to 1.55, duration 1.22 to 1.36, per-frame travel 1.12 to 1.18 (BLEU-4 11.0 to 12.5, WER 90 to 97 in the score line)',
                msg, 'PASS' if ok else 'FAIL', '$U/results/${T}_fps25.json'])
"
L $PY $S/s2_post.py bootstrap --name ustcmoe_natlen --text_preds $U/results/${T}_fps25_text_preds.pt --outdir $O \
    --step "d USTC-MoE mT5 natural length vs fixed route" || stop "d USTC-MoE natural-length bootstrap" $? $O
L $PY $S/s2_post.py ratios --name $T --pred $U/preds/${T}.pt --outdir $O --step "d USTC-MoE mT5 natural-length motion" \
  || stop "d USTC-MoE natural-length motion" $? $O
note "USTC-MoE mT5 natural-length run finished"
