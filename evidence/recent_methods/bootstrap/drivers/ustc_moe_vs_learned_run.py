import json, subprocess, sys
ROOT = "/home/kumwilai/research/signgen-t2m"; sys.path.insert(0, ROOT)
from scripts.bootstrap_strict_cac import load_predictions, paired_bootstrap
RB = f"{ROOT}/outputs/recent_baselines_2026-09-28"; W = f"{ROOT}/outputs/revision/evaluator_workspace/results"
PY = "/home/kumwilai/research/coopns-slr/.venv/bin/python"; S = f"{ROOT}/scripts/recent_baselines"
OUT = f"{RB}/ustc_moe/vs_learned/ustcmoe_vs_learned_paired_bootstrap_test.json"
rec = lambda *a: subprocess.run([PY, f"{S}/s2_common.py", "record", *a])
led = json.load(open(f"{ROOT}/outputs/revision/clean_rerank_frame40_test_ledger.json"))["clips"]
ids = open(f"{ROOT}/outputs/signbase_slrtp_2026-09-11/inputs/test_keys.txt").read().split()
refs = open(f"{ROOT}/outputs/signbase_slrtp_2026-09-11/inputs/test_text.txt", encoding="utf-8").read().splitlines()
assert [c["id"] for c in led] == ids and len(refs) == 641
hyp = {"learned": load_predictions(f"{W}/clean_strict_cac_frame40_test_text_preds.pt"),
       "ustcmoe_d4": load_predictions(f"{RB}/ustc_moe/results/ustcmoe_mt5_policylen_test_fps25_text_preds.pt"),
       "ustcmoe_natlen": load_predictions(f"{RB}/ustc_moe/results/ustcmoe_mt5_natlen_test_fps25_text_preds.pt")}
pairs = (("d4_vs_learned", "ustcmoe_d4", "learned"), ("natlen_vs_learned", "ustcmoe_natlen", "learned"))
full = paired_bootstrap(hyp, refs, pairs=pairs, n_boot=1, seed=30373)["methods"]["learned"]
ok = round(full["bleu4"], 4) == 16.7618 and round(full["corpus_wer"], 4) == 83.6217
rec("USTC-MoE vs learned route: learned reproduces", "16.7618 / 83.6217 (clean_strict_cac_frame40_test.json)",
    f"{full['bleu4']:.4f} / {full['corpus_wer']:.4f}", "PASS" if ok else "FAIL", OUT)
if not ok: sys.exit(2)
res = {}
for lab, idx in (("all 641", list(range(641))),
                 ("whole-replay 192", [i for i, c in enumerate(led) if c["mode"] == "whole_clip_replay"]),
                 ("assembled 449", [i for i, c in enumerate(led) if c["mode"] == "compositional_local_reuse"])):
    sub = {k: [v[i] for i in idx] for k, v in hyp.items()}
    res[lab] = r = paired_bootstrap(sub, [refs[i] for i in idx], pairs=pairs, n_boot=10000, seed=30373)
    for p, name in (("d4_vs_learned", "ustcmoe_d4"), ("natlen_vs_learned", "ustcmoe_natlen")):
        c = r["comparisons"][p]; d, i = c["point_delta"], c["bootstrap_delta"]; me, le = r["methods"][name], r["methods"]["learned"]
        rec(f"USTC-MoE {name[8:]} vs learned route, {lab}", "paired bootstrap, seed 30373, 10,000 resamples (delta = USTC-MoE minus learned)",
            f"{me['bleu4']:.2f}/{me['corpus_wer']:.2f} vs learned {le['bleu4']:.2f}/{le['corpus_wer']:.2f}, "
            f"dBLEU-4 {d['bleu4']:+.2f} [{i['bleu4']['ci95'][0]:+.2f}, {i['bleu4']['ci95'][1]:+.2f}], "
            f"dWER {d['corpus_wer']:+.2f} [{i['corpus_wer']['ci95'][0]:+.2f}, {i['corpus_wer']['ci95'][1]:+.2f}]", "recorded", OUT)
json.dump(res, open(OUT, "w"), indent=1, sort_keys=True, default=str)
