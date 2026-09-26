"""RCX (retimed convex crossfade) on test-641: reproduction gate, metrics, ledger counts,
convexity assertion, provenance audit, and the two .pt variants for the frozen evaluator."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from js_common import *; import js_repairs as R

raw = torch.load(OUT / "frozen_local_test_raw_segments.pt", weights_only=False); ids = list(raw); gt = load_gt()
local = torch.load(LOCAL_PT, map_location="cpu", weights_only=False)
route = torch.load(ROOT / "outputs/revision/clean_strict_cac_frame40_test.pt", map_location="cpu", weights_only=False)
led = json.load(open(ROOT / "outputs/revision/clean_strict_cac_frame40_test_ledger.json"))
mode = {c["id"]: c["mode"] for c in led["clips"]}
cac_local = [s for s in ids if mode[s] == "compositional_local_reuse"]
print(f"test clips {len(ids)}  CAC-local clips {len(cac_local)}")

res = {"reproduction": {}, "test641": {}, "ledger": {}, "convexity": {}, "provenance": {}}

# ---- reproduction gate -------------------------------------------------------------------
frozen = {s: R.frozen_blend(raw[s])[0] for s in ids}
bitexact = all(np.array_equal(frozen[s], local[s].reshape(local[s].shape[0], -1).numpy()) for s in ids)
print("frozen reconstruction bit-exact vs shipped .pt:", bitexact)
REC = {"hand_jerk": 0.9887, "hand_speed": 0.9884, "hand_posestd": 0.9078}
herm = {s: R.hermite_bridge(raw[s], 2, 3) for s in ids}
rh = ratios(herm, gt, ids)
print("REPRO hermite3 h=2 : " + "  ".join(f"{k} {rh[k]:.4f} (recorded {REC[k]:.4f}, diff {rh[k]-REC[k]:+.4f})" for k in REC))
res["reproduction"] = {"frozen_bit_exact": bool(bitexact), "hermite3_h2": rh,
                       "recorded": REC, "max_abs_diff_vs_recorded": max(abs(rh[k] - REC[k]) for k in REC)}

# ---- RCX ---------------------------------------------------------------------------------
sink = []
t0 = time.time(); rcx = {s: R.rcx_bridge(raw[s], 2, coeff_sink=sink) for s in ids}
C = np.array(sink)
res["convexity"] = {"emitted_rows": int(C.shape[0]), "min_coefficient": float(C.min()),
                    "max_abs_sum_minus_one": float(np.abs(C.sum(1) - 1).max()),
                    "assertion_passed": bool(C.min() >= 0.0 and np.abs(C.sum(1) - 1).max() <= 1e-6),
                    "per_source_a_weight_total": float((C[:, 0] + C[:, 1]).sum()),
                    "per_source_b_weight_total": float((C[:, 2] + C[:, 3]).sum())}
print(f"convexity: rows {C.shape[0]} min coeff {C.min():.6e} max|sum-1| {np.abs(C.sum(1)-1).max():.3e} "
      f"PASSED={res['convexity']['assertion_passed']}  ({time.time()-t0:.0f}s)")

# ---- test-641 metrics, local and CAC-route ------------------------------------------------
route_np = {s: route[s].float().reshape(route[s].shape[0], -1).numpy() for s in ids}
def cacroute(loc):
    d = dict(route_np)
    for s in cac_local: d[s] = loc[s]
    return d
def counts(pred, subset):
    ex = fr = tot = 0
    for s in subset:
        base = R.plain_concat(raw[s]); diff = (np.abs(pred[s] - base) > 0).any(1)
        tot += pred[s].shape[0]; fr += int(diff.sum()); ex += int((~diff).sum())
    return dict(frames=tot, exact_archive_rows=ex, non_archive_positions=fr, pct_exact=round(100 * ex / tot, 2))
for name, loc in (("rcx_h2", rcx), ("hermite3_h2", herm), ("frozen_cosine_w4", frozen)):
    rl = ratios(loc, gt, ids); rr = ratios(cacroute(loc), gt, ids)
    c641 = counts(loc, ids); c432 = counts(loc, cac_local)
    res["test641"][name] = {"local": rl, "cac_route": rr}
    res["ledger"][name] = {"local_route_641_clips": c641, "cac_route_432_local_clips": c432,
                           "frac_non_archive_641": c641["non_archive_positions"] / c641["frames"]}
    print(f"{name:18s} LOCAL jerk {rl['hand_jerk']:.4f} speed {rl['hand_speed']:.4f} std {rl['hand_posestd']:.4f} "
          f"obj {rl['joint_obj']:.4f} xyj {rl['xy_jerk_vs_xyGT']:.4f} | CAC-ROUTE jerk {rr['hand_jerk']:.4f} "
          f"speed {rr['hand_speed']:.4f} std {rr['hand_posestd']:.4f} obj {rr['joint_obj']:.4f} | "
          f"non-arch641 {c641['non_archive_positions']} ({c641['non_archive_positions']/c641['frames']:.4f}) "
          f"| 432: exact {c432['exact_archive_rows']} frac {c432['non_archive_positions']}", flush=True)

# ---- provenance audit: do bridge rows ever read a row a previous join already overwrote? ---
def prov(segs, h=2):
    joins = rows = impure_rows = impure_joins = 0; head = 0
    for i in range(1, len(segs)):
        prev, cur = segs[i - 1], segs[i]; hh = R.eff_h(h, prev, cur, need=2)
        if hh < 1: head = 0; continue
        joins += 1; rows += 2 * hh; na = prev.shape[0]; lo = na - hh - 1; q = (hh + 1.0) / hh + 1.0; bad = 0
        for k in range(1, 2 * hh + 1):
            s = k / (2 * hh + 1); ta = hh * (1.0 - (1.0 - s) ** q); ia = min(int(ta), hh - 1)
            if (lo + ia) < head or (lo + ia + 1) < head: bad += 1
        impure_rows += bad; impure_joins += int(bad > 0); head = hh
    return joins, rows, impure_rows, impure_joins
tot = np.array([prov(raw[s]) for s in ids]).sum(0)
res["provenance"] = {"joins": int(tot[0]), "bridge_rows": int(tot[1]),
                     "rows_reading_a_previously_overwritten_row": int(tot[2]), "joins_affected": int(tot[3])}
print("provenance:", res["provenance"])

# ---- write variants ----------------------------------------------------------------------
def to_pt(d): return {s: torch.from_numpy(np.ascontiguousarray(v.reshape(v.shape[0], 178, 3))).float() for s, v in d.items()}
torch.save(to_pt(rcx), OUT / "jerkstudy_local_rcx_h2_test.pt")
torch.save(to_pt(cacroute(rcx)), OUT / "jerkstudy_cacroute_rcx_h2_test.pt")
json.dump(res, open(OUT / "rcx_results_test641.json", "w"), indent=1)
print("wrote", OUT / "rcx_results_test641.json")
