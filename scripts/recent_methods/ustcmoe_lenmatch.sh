#!/usr/bin/env bash
# USTC-MoE at OUR routes' output lengths, item by item (user, 2026-09-28 16:30, "to compare, the lengths must be
# equal"). For each route R in {fixed, learned}, the released code (shared mT5 glosses, the authors' own cut-or-pad
# rule) is given the frame count our route R emitted for that request (frame audit per_clip T), so both systems emit
# exactly the same number of frames on every item and the brevity advantage of longer output is removed. Scored under
# main.py --fps 25, then a paired bootstrap against route R on all 641 and the two ledger strata (seed 30373, 10,000).
# Diagnostic until fable rules on placement. Behind the heavy lock through light.sh; USTC_NOLOCK=1 only while R1 is
# frozen by a user-approved pause.
# usage: nohup setsid ustcmoe_lenmatch.sh > .../ustc_moe/logs/lenmatch.log 2>&1 < /dev/null &
set -u
source "$(dirname "$(readlink -f "$0")")/stage2_lib_snapshot_20260928_1545.sh" light
if [ "${USTC_NOLOCK:-0}" = 1 ]; then L() { "$S/memcap.sh" "$@"; }; fi
O=$U/lenmatch; mkdir -p $O
ASM="$PY $U/code/main_641.py --gloss2pose $ROOT/outputs/baselines/ustc_moe/phoenix_gloss2pose_results.pkl \
    --train_label $ROOT/external/baselines/CVPRW-SLP-2025/external/baselines/CVPRW-SLP-2025/hf_data/phoenix-2014t.train --train_pt $DATA/train.pt"
for R in fixed learned; do
  T=ustcmoe_mt5_len_${R}_test; LEN=$U/inputs/lengths_of_${R}_route_test.txt
  $PY -c "
import json
key = {'fixed': 'fixed_route', 'learned': 'learned_route'}['$R']
pc = json.load(open('$RB/frame_audit/{}_win1_stride1_thr0.005.json'.format(key)))['banks'][key]['per_clip']
keys = open('$INP/test_keys.txt').read().split()
L = [pc[k]['T'] for k in keys]
open('$LEN', 'w').write('\n'.join(map(str, L)) + '\n')
print('$R route lengths: mean', sum(L) / len(L), 'items', len(L))
" || { stop "USTC-MoE at $R-route lengths, lengths" $? $LEN; continue; }
  L bash -c "$ASM --keys $INP/test_keys.txt --texts $INP/test_text.txt --lengths $LEN \
             --glosses $U/inputs/mt5_test_glosses.json --out $U/preds/${T}_raw.pt --tags $U/preds/${T}_tags.json && \
             $PY $S/pred_invariants.py --pred $U/preds/${T}_raw.pt --out $U/preds/${T}.pt --keys $INP/test_keys.txt --report $U/preds/${T}_invariants.json" \
    || { stop "USTC-MoE at $R-route lengths, assembly" $? $U/preds; continue; }
  L $PY $S/s2_inv.py --report $U/preds/${T}_invariants.json --clean $U/preds/${T}.pt --keys $INP/test_keys.txt --lengths $LEN \
      --step "USTC-MoE at $R-route lengths, invariants" --length_source "our $R route's emitted frame count per item" \
    || { stop "USTC-MoE at $R-route lengths, invariants" $? $U/preds; continue; }
  L $PY $S/s2_score.py --split test --pred $U/preds/${T}.pt --tag ${T}_fps25 --workdir $U \
      --step "USTC-MoE at $R-route lengths, test (641)" || { stop "USTC-MoE at $R-route lengths, score" $? $U; continue; }
done
cat > $O/run.py <<'EOF'
import json, subprocess, sys
ROOT = "/home/kumwilai/research/signgen-t2m"; sys.path.insert(0, ROOT)
from scripts.bootstrap_strict_cac import load_predictions, paired_bootstrap
RB = f"{ROOT}/outputs/recent_baselines_2026-09-28"; W = f"{ROOT}/outputs/revision/evaluator_workspace/results"
PY = "/home/kumwilai/research/coopns-slr/.venv/bin/python"; S = f"{ROOT}/scripts/recent_baselines"
rec = lambda *a: subprocess.run([PY, f"{S}/s2_common.py", "record", *a])
led = json.load(open(f"{ROOT}/outputs/revision/clean_rerank_frame40_test_ledger.json"))["clips"]
ids = open(f"{ROOT}/outputs/signbase_slrtp_2026-09-11/inputs/test_keys.txt").read().split()
refs = open(f"{ROOT}/outputs/signbase_slrtp_2026-09-11/inputs/test_text.txt", encoding="utf-8").read().splitlines()
assert [c["id"] for c in led] == ids and len(refs) == 641
route = {"fixed": (f"{W}/clean_rerank_frame40_test_text_preds.pt", 15.1032, 85.5825),
         "learned": (f"{W}/clean_strict_cac_frame40_test_text_preds.pt", 16.7618, 83.6217)}
out = {}
for R, (path, b4, wer) in route.items():
    tp = f"{RB}/ustc_moe/results/ustcmoe_mt5_len_{R}_test_fps25_text_preds.pt"
    try: hyp = {R: load_predictions(path), "ustcmoe": load_predictions(tp)}
    except FileNotFoundError: print("missing", tp); continue
    pairs = ((f"ustcmoe_vs_{R}", "ustcmoe", R),)
    full = paired_bootstrap(hyp, refs, pairs=pairs, n_boot=1, seed=30373)["methods"][R]
    ok = round(full["bleu4"], 4) == b4 and round(full["corpus_wer"], 4) == wer
    rec(f"USTC-MoE at {R}-route lengths: {R} route reproduces", f"{b4} / {wer}", f"{full['bleu4']:.4f} / {full['corpus_wer']:.4f}",
        "PASS" if ok else "FAIL", tp)
    if not ok: continue
    for lab, idx in (("all 641", list(range(641))),
                     ("whole-replay 192", [i for i, c in enumerate(led) if c["mode"] == "whole_clip_replay"]),
                     ("assembled 449", [i for i, c in enumerate(led) if c["mode"] == "compositional_local_reuse"])):
        sub = {k: [v[i] for i in idx] for k, v in hyp.items()}
        r = paired_bootstrap(sub, [refs[i] for i in idx], pairs=pairs, n_boot=10000, seed=30373)
        out[f"{R}|{lab}"] = r
        c = r["comparisons"][pairs[0][0]]; d, i = c["point_delta"], c["bootstrap_delta"]; me, le = r["methods"]["ustcmoe"], r["methods"][R]
        rec(f"USTC-MoE at {R}-route lengths vs {R} route, {lab}", "paired bootstrap, seed 30373, 10,000 resamples (delta = USTC-MoE minus ours), equal frame count per item",
            f"{me['bleu4']:.2f}/{me['corpus_wer']:.2f} vs {R} {le['bleu4']:.2f}/{le['corpus_wer']:.2f}, "
            f"dBLEU-4 {d['bleu4']:+.2f} [{i['bleu4']['ci95'][0]:+.2f}, {i['bleu4']['ci95'][1]:+.2f}], "
            f"dWER {d['corpus_wer']:+.2f} [{i['corpus_wer']['ci95'][0]:+.2f}, {i['corpus_wer']['ci95'][1]:+.2f}]", "recorded", tp)
json.dump(out, open(f"{RB}/ustc_moe/lenmatch/ustcmoe_lenmatch_paired_bootstrap_test.json", "w"), indent=1, sort_keys=True, default=str)
EOF
L $PY $O/run.py || stop "USTC-MoE length-matched bootstrap" $? $O
note "USTC-MoE length-matched runs finished"
