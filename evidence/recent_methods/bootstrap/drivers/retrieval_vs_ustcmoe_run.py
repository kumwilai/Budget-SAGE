# Retrieval-only comparison (user, 2026-09-29 18:15): our retrieval routes in Table IV against USTC-MoE, the only
# retrieval baseline (length-free table run). All 641, paired bootstrap, seed 30373, 10,000 resamples.
# Delta = USTC-MoE minus ours. Each route must reproduce its recorded value first.
import json, subprocess, sys
ROOT = "/home/kumwilai/research/signgen-t2m"; sys.path.insert(0, ROOT)
from scripts.bootstrap_strict_cac import load_predictions, paired_bootstrap
RB = f"{ROOT}/outputs/recent_baselines_2026-09-28"; W = f"{ROOT}/outputs/revision/evaluator_workspace/results"
PY = "/home/kumwilai/research/coopns-slr/.venv/bin/python"
rec = lambda *a: subprocess.run([PY, f"{ROOT}/scripts/recent_baselines/s2_common.py", "record", *a])
refs = open(f"{ROOT}/outputs/signbase_slrtp_2026-09-11/inputs/test_text.txt", encoding="utf-8").read().splitlines()
routes = {"unrestricted": (f"{W}/revision_clean_topk_test_text_preds.pt", 24.6255, 79.1875, "Unrestricted retrieval, 99.592% whole-clip frames"),
          "fixed": (f"{W}/clean_rerank_frame40_test_text_preds.pt", 15.1032, 85.5825, "Fixed 40%"),
          "learned": (f"{W}/clean_strict_cac_frame40_test_text_preds.pt", 16.7618, 83.6217, "Learned 40%"),
          "local": (f"{W}/revision_clean_phrase_lattice_test_text_preds.pt", 9.1893, 90.6446, "Local assembly, 0% whole-clip frames")}
hyp = {k: load_predictions(v[0]) for k, v in routes.items()}
hyp["ustcmoe"] = load_predictions(f"{RB}/ustc_moe/results/ustcmoe_mt5_natlen_test_fps25_text_preds.pt")
pairs = tuple((f"ustcmoe_vs_{k}", "ustcmoe", k) for k in routes)
r = paired_bootstrap(hyp, refs, pairs=pairs, n_boot=10000, seed=30373)
for k, (path, b4, wer, lab) in routes.items():
    m = r["methods"][k]; ok = round(m["bleu4"], 4) == b4 and round(m["corpus_wer"], 4) == wer
    c = r["comparisons"][f"ustcmoe_vs_{k}"]; d, i = c["point_delta"], c["bootstrap_delta"]; u = r["methods"]["ustcmoe"]
    rec(f"USTC-MoE vs our {lab} (retrieval-only), all 641", f"route reproduces {b4} / {wer}; delta = USTC-MoE minus ours, paired bootstrap seed 30373, 10,000",
        f"route {m['bleu4']:.4f}/{m['corpus_wer']:.4f} ({'reproduced' if ok else 'NOT reproduced'}); USTC-MoE {u['bleu4']:.2f}/{u['corpus_wer']:.2f}; dBLEU-4 {d['bleu4']:+.2f} [{i['bleu4']['ci95'][0]:+.2f}, {i['bleu4']['ci95'][1]:+.2f}], "
        f"dWER {d['corpus_wer']:+.2f} [{i['corpus_wer']['ci95'][0]:+.2f}, {i['corpus_wer']['ci95'][1]:+.2f}]", "recorded" if ok else "FAIL", path)
json.dump(r, open(f"{RB}/retrieval_vs_ustcmoe/retrieval_vs_ustcmoe_paired_bootstrap_test.json", "w"), indent=1, sort_keys=True, default=str)
