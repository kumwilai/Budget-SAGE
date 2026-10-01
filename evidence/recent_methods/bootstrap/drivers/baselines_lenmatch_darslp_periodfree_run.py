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
