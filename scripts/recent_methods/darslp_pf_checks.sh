#!/usr/bin/env bash
# DARSLP diagnostics redone on the FINAL row bank: released weights, period-free input (darslp/test_after_choice),
# chosen by the pre-registered dev rule after B' failed its smoke. Fable remaining_baselines_ruling.md (2026-09-29).
#   A. The final bank against our LEARNED route (all 641 and the two fixed-ledger strata).
#   B. Length-matched: the released output resampled in time (linear, keeps every sign, changes speed) to our fixed and
#      learned routes' frame count per item, scored, and bootstrapped against that route.
# Deltas are the baseline minus ours; paired bootstrap, seed 30373, 10,000 resamples. Derived from baselines_lenmatch.sh
# (which used the with-period stage-1 bank). Every step is queued under the heavy lock through light.sh.
# usage: nohup setsid darslp_pf_checks.sh > .../baselines_lenmatch/darslp_periodfree/logs/run.log 2>&1 < /dev/null &
set -u
source "$(dirname "$(readlink -f "$0")")/stage2_lib_snapshot_20260928_1545.sh" light
[ "${NOLOCK:-0}" = 1 ] && source "$(dirname "$(readlink -f "$0")")/nolock_L.sh"   # user exception 2026-09-29 16:30
if [ "${USTC_NOLOCK:-0}" = 1 ]; then L() { "$S/memcap.sh" "$@"; }; fi
B=$RB/baselines_lenmatch/darslp_periodfree; mkdir -p $B/preds $B/logs
DAR_BANK=$RB/darslp/test_after_choice/preds/darslp_released_periodfree_test.pt
for R in fixed learned; do
  LEN=$U/inputs/lengths_of_${R}_route_test.txt
  [ -s $LEN ] || { stop "baseline length match: $R lengths missing" 1 $LEN; exit 1; }
  # DARSLP resampled in time to our route's lengths (no native length input)
  T=darslp_pf_len_${R}_test
  L $PY -c "
import torch
d = torch.load('$DAR_BANK', weights_only=False)
keys = open('$INP/test_keys.txt').read().split()
L = [int(x) for x in open('$LEN').read().split()]
out = {}
for k, n in zip(keys, L):
    x = d[k].float()                                   # [T, 178, 3]
    y = torch.nn.functional.interpolate(x.permute(1, 2, 0).reshape(1, -1, x.shape[0]), size=n, mode='linear', align_corners=True)
    out[k] = y.reshape(x.shape[1], x.shape[2], n).permute(2, 0, 1).contiguous()
torch.save(out, '$B/preds/${T}_raw.pt')
print('resampled', len(out), 'mean source length', sum(d[k].shape[0] for k in keys) / len(keys), 'mean target', sum(L) / len(L))
" > $B/logs/${T}.log 2>&1 \
    && L $PY $S/pred_invariants.py --pred $B/preds/${T}_raw.pt --out $B/preds/${T}.pt --keys $INP/test_keys.txt \
         --report $B/preds/${T}_invariants.json >> $B/logs/${T}.log 2>&1 \
    && L $PY $S/s2_inv.py --report $B/preds/${T}_invariants.json --clean $B/preds/${T}.pt --keys $INP/test_keys.txt --lengths $LEN \
         --step "DARSLP (period-free) resampled to $R-route lengths, invariants" --length_source "our $R route's emitted frame count per item (linear time resampling)" \
    && L $PY $S/s2_score.py --split test --pred $B/preds/${T}.pt --tag ${T}_fps25 --workdir $B --step "DARSLP (period-free) resampled to $R-route lengths, test (641)" \
    || stop "DARSLP (period-free) resampled to $R-route lengths" $? $B/logs/${T}.log
done
cat > $B/run.py <<'EOF'
import json, subprocess, sys, os
ROOT = "/home/kumwilai/research/signgen-t2m"; sys.path.insert(0, ROOT)
from scripts.bootstrap_strict_cac import load_predictions, paired_bootstrap
RB = f"{ROOT}/outputs/recent_baselines_2026-09-28"; W = f"{ROOT}/outputs/revision/evaluator_workspace/results"; B = f"{RB}/baselines_lenmatch/darslp_periodfree"
PY = "/home/kumwilai/research/coopns-slr/.venv/bin/python"; S = f"{ROOT}/scripts/recent_baselines"
rec = lambda *a: subprocess.run([PY, f"{S}/s2_common.py", "record", *a])
led = json.load(open(f"{ROOT}/outputs/revision/clean_rerank_frame40_test_ledger.json"))["clips"]
ids = open(f"{ROOT}/outputs/signbase_slrtp_2026-09-11/inputs/test_keys.txt").read().split()
refs = open(f"{ROOT}/outputs/signbase_slrtp_2026-09-11/inputs/test_text.txt", encoding="utf-8").read().splitlines()
assert [c["id"] for c in led] == ids and len(refs) == 641
ours = {"fixed": (f"{W}/clean_rerank_frame40_test_text_preds.pt", 15.1032, 85.5825),
        "learned": (f"{W}/clean_strict_cac_frame40_test_text_preds.pt", 16.7618, 83.6217)}
jobs = [("DARSLP final bank (released, period-free, own stop)", f"{RB}/darslp/test_after_choice/results/darslp_released_periodfree_test_fps25_text_preds.pt", "learned")]
for R in ("fixed", "learned"):
    jobs += [(f"DARSLP (period-free) resampled to {R}-route lengths", f"{B}/results/darslp_pf_len_{R}_test_fps25_text_preds.pt", R)]
strata = (("all 641", list(range(641))),
          ("whole-replay 192", [i for i, c in enumerate(led) if c["mode"] == "whole_clip_replay"]),
          ("assembled 449", [i for i, c in enumerate(led) if c["mode"] == "compositional_local_reuse"]))
out, checked = {}, {}
for name, tp, R in jobs:
    if not os.path.exists(tp): rec(f"{name} vs {R} route", "text predictions exist", "missing", "FAIL", tp); continue
    path, b4, wer = ours[R]
    hyp = {R: load_predictions(path), "base": load_predictions(tp)}
    pairs = ((f"base_vs_{R}", "base", R),)
    if R not in checked:
        full = paired_bootstrap(hyp, refs, pairs=pairs, n_boot=1, seed=30373)["methods"][R]
        checked[R] = round(full["bleu4"], 4) == b4 and round(full["corpus_wer"], 4) == wer
        rec(f"baseline length checks: {R} route reproduces", f"{b4} / {wer}", f"{full['bleu4']:.4f} / {full['corpus_wer']:.4f}",
            "PASS" if checked[R] else "FAIL", path)
    if not checked[R]: continue
    for lab, idx in strata:
        sub = {k: [v[i] for i in idx] for k, v in hyp.items()}
        r = paired_bootstrap(sub, [refs[i] for i in idx], pairs=pairs, n_boot=10000, seed=30373)
        out[f"{name}|{R}|{lab}"] = r
        c = r["comparisons"][pairs[0][0]]; d, i = c["point_delta"], c["bootstrap_delta"]; me, le = r["methods"]["base"], r["methods"][R]
        rec(f"{name} vs {R} route, {lab}", "paired bootstrap, seed 30373, 10,000 resamples (delta = baseline minus ours)",
            f"{me['bleu4']:.2f}/{me['corpus_wer']:.2f} vs {R} {le['bleu4']:.2f}/{le['corpus_wer']:.2f}, "
            f"dBLEU-4 {d['bleu4']:+.2f} [{i['bleu4']['ci95'][0]:+.2f}, {i['bleu4']['ci95'][1]:+.2f}], "
            f"dWER {d['corpus_wer']:+.2f} [{i['corpus_wer']['ci95'][0]:+.2f}, {i['corpus_wer']['ci95'][1]:+.2f}]", "recorded", tp)
json.dump(out, open(f"{B}/darslp_periodfree_checks_paired_bootstrap_test.json", "w"), indent=1, sort_keys=True, default=str)
EOF
L $PY $B/run.py || stop "baseline length checks bootstrap" $? $B
note "DARSLP period-free checks finished"
