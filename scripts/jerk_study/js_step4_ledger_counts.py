import sys, json, glob
sys.path.insert(0, "/tmp/claude-1000/-home-kumwilai-research-signgen-t2m/ef04ff5e-1c47-4276-b131-90e6a9dbc109/scratchpad")
from js_common import *; import js_repairs as R
raw = torch.load(OUT / "frozen_local_test_raw_segments.pt", weights_only=False); ids = list(raw)
led = json.load(open(ROOT / "outputs/revision/clean_strict_cac_frame40_test_ledger.json")); cac_local = [c["id"] for c in led["clips"] if c["mode"] == "compositional_local_reuse"]
def counts(fn, subset):
    win = frac = exact = tot = 0
    for s in subset:
        segs = raw[s]; out = fn(segs); base = R.plain_concat(segs); tot += out.shape[0]
        diff = (np.abs(out - base) > 0).any(1)
        # window positions: frames inside the declared touched window (2h per join for bridges, w per join for ramps)
        exact += int((~diff).sum()); frac += int(diff.sum())
    return dict(frames=tot, exact_archive_rows=exact, non_archive_positions=frac, pct_exact=round(100*exact/tot, 2))
variants = {"frozen cosine w=4": lambda s: R.frozen_blend(s)[0], "ramp_hold_w4": lambda s: R.ramp_exclusive(s, 4, "hold"), "hermite3_h2": lambda s: R.hermite_bridge(s, 2, 3), "hermite5_h2": lambda s: R.hermite_bridge(s, 2, 5), "hermite5_h3": lambda s: R.hermite_bridge(s, 3, 5)}
res = {}
for name, fn in variants.items():
    res[name] = {"cac_route_432_local_clips": counts(fn, cac_local), "local_route_641_clips": counts(fn, ids)}
    a = res[name]["cac_route_432_local_clips"]; b = res[name]["local_route_641_clips"]
    print(f"{name:20s} CAC-local: frames {a['frames']} exact {a['exact_archive_rows']} ({a['pct_exact']}%) non-archive {a['non_archive_positions']} | local641: frames {b['frames']} exact {b['exact_archive_rows']} ({b['pct_exact']}%) non-archive {b['non_archive_positions']}")
print("ledger frozen: blend-window positions 3763, strictly fractional 1861, compositional_local_frames", led["compositional_local_frames"])
# joins and mass-split rule check: H0+H3 == 1 for cubic and quintic
s = np.linspace(0.01, 0.99, 9); print("cubic H0+H3 max dev", np.abs((2*s**3-3*s**2+1)+(-2*s**3+3*s**2)-1).max(), "quintic H0+H3 max dev", np.abs((1-10*s**3+15*s**4-6*s**5)+(10*s**3-15*s**4+6*s**5)-1).max())
n_joins = sum(len(raw[s]) - 1 for s in ids); n_joins_cac = sum(len(raw[s]) - 1 for s in cac_local); print("joins local641", n_joins, "joins in CAC-local clips", n_joins_cac)
# consolidated evaluator results
ev = {}
for f in sorted(glob.glob(str(OUT / "evaluator_workspace/results/*.json"))):
    d = json.load(open(f)); ev[Path(f).stem] = {"bleu4": d["bleu"]["bleu4"], "bleu1": d["bleu"]["bleu1"], "wer": d["wer"], "dtw_mje": d["dtw_mje"], "chrf": d["chrf"], "rouge": d["rouge"]}
json.dump({"ledger_counts": res, "evaluator": ev, "frozen_ledger": {"blend_window_positions": 3763, "strictly_fractional": 1861}}, open(OUT / "priority3_4_quality_and_ledger.json", "w"), indent=1)
