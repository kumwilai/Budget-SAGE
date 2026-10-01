# Generation-only comparison (user, 2026-09-29 18:05): our all-generated Sign-JEPA route (Table IV "All-generated",
# outputs/revision/generative_route_restore_20260911, text-only duration policy, mT5 glosses, no held-out input) against
# the three measured baselines. Paired bootstrap, seed 30373, 10,000 resamples, all 641. Delta = baseline minus ours.
import json, subprocess, sys
ROOT = "/home/kumwilai/research/signgen-t2m"; sys.path.insert(0, ROOT)
from scripts.bootstrap_strict_cac import load_predictions, paired_bootstrap
RB = f"{ROOT}/outputs/recent_baselines_2026-09-28"; PY = "/home/kumwilai/research/coopns-slr/.venv/bin/python"
rec = lambda *a: subprocess.run([PY, f"{ROOT}/scripts/recent_baselines/s2_common.py", "record", *a])
refs = open(f"{ROOT}/outputs/signbase_slrtp_2026-09-11/inputs/test_text.txt", encoding="utf-8").read().splitlines()
AG = f"{ROOT}/outputs/revision/generative_route_restore_20260911/evaluator_workspace/results/test641_mt5gloss_text_preds.pt"
hyp = {"allgen": load_predictions(AG),
       "darslp": load_predictions(f"{RB}/darslp/test_after_choice/results/darslp_released_periodfree_test_fps25_text_preds.pt"),
       "signidd": load_predictions(f"{RB}/signidd/gate5/results/signidd_gate4best_test_seed11_fps25_text_preds.pt"),
       "ustcmoe": load_predictions(f"{RB}/ustc_moe/results/ustcmoe_mt5_natlen_test_fps25_text_preds.pt")}
pairs = tuple((f"{b}_vs_allgen", b, "allgen") for b in ("darslp", "signidd", "ustcmoe"))
r = paired_bootstrap(hyp, refs, pairs=pairs, n_boot=10000, seed=30373)
ag = r["methods"]["allgen"]; ok = round(ag["bleu4"], 4) == 12.1514 and round(ag["corpus_wer"], 4) == 86.4539
rec("all-generated Sign-JEPA reproduces", "12.1514 / 86.4539 (test641_mt5gloss.json, Table IV)", f"{ag['bleu4']:.4f} / {ag['corpus_wer']:.4f}", "PASS" if ok else "FAIL", AG)
if ok:
    for p, b in zip([x[0] for x in pairs], ("darslp", "signidd", "ustcmoe")):
        c = r["comparisons"][p]; d, i = c["point_delta"], c["bootstrap_delta"]; m = r["methods"][b]
        rec(f"{b} vs all-generated Sign-JEPA (generation-only), all 641", "paired bootstrap, seed 30373, 10,000 resamples (delta = baseline minus ours)",
            f"{m['bleu4']:.2f}/{m['corpus_wer']:.2f} vs all-generated {ag['bleu4']:.2f}/{ag['corpus_wer']:.2f}, dBLEU-4 {d['bleu4']:+.2f} [{i['bleu4']['ci95'][0]:+.2f}, {i['bleu4']['ci95'][1]:+.2f}], "
            f"dWER {d['corpus_wer']:+.2f} [{i['corpus_wer']['ci95'][0]:+.2f}, {i['corpus_wer']['ci95'][1]:+.2f}]", "recorded", AG)
json.dump(r, open(f"{RB}/allgen_vs_baselines/allgen_vs_baselines_paired_bootstrap_test.json", "w"), indent=1, sort_keys=True, default=str)
