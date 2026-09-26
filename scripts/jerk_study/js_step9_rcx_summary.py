"""Consolidate the RCX result against the pre-declared acceptance criteria."""
import json, hashlib, ast, pathlib
O = pathlib.Path("/home/kumwilai/research/signgen-t2m/outputs/jerk_study")
pre = json.load(open(O / "rcx_predeclaration.json"))
t = json.load(open(O / "rcx_results_test641.json")); d = json.load(open(O / "rcx_results_dev515.json"))
ev = lambda tag: json.load(open(O / f"evaluator_workspace/results/{tag}.json"))
src = (O / "scripts/js_repairs.py").read_text()
fn = ast.get_source_segment(src, [n for n in ast.parse(src).body if getattr(n, "name", "") == "rcx_bridge"][0])

print("=" * 112)
print(f"{'':34s} {'jerk':>7s} {'speed':>7s} {'posestd':>8s} {'obj':>7s} {'xyjerk':>7s} {'nonarch':>8s} {'frac':>7s}")
rows = []
for split, blob, keyset in (("test641", t["test641"], ("local", "cac_route")), ("dev515", d["variants"], ("local", "cac_route"))):
    for name in ("frozen_cosine_w4" if split == "test641" else "frozen", "hermite3_h2", "rcx_h2"):
        for v in keyset:
            r = blob[name][v]
            na = t["ledger"].get(name, {}).get("local_route_641_clips", {}).get("non_archive_positions", "-") if split == "test641" else "-"
            fr = round(t["ledger"][name]["frac_non_archive_641"], 4) if split == "test641" else "-"
            print(f"{split+' '+name+' '+v:34s} {r['hand_jerk']:7.4f} {r['hand_speed']:7.4f} {r['hand_posestd']:8.4f} "
                  f"{r['joint_obj']:7.4f} {r['xy_jerk_vs_xyGT']:7.4f} {str(na):>8s} {str(fr):>7s}")
            rows.append(dict(split=split, variant=name, route=v, **{k: r[k] for k in
                        ("hand_jerk", "hand_speed", "hand_posestd", "joint_obj", "xy_jerk_vs_xyGT", "xy_speed_vs_xyGT", "n")}))
print("=" * 112)

A = pre["acceptance_criteria"]; chk = {}
for route, key in (("local", "test641_local"), ("cac_route", "test641_cac_route")):
    r = t["test641"]["rcx_h2"][route]
    for m in ("hand_jerk", "hand_speed"):
        lo, hi = A[key][m]; chk[f"test641_{route}_{m}_in_{lo}_{hi}"] = {"value": r[m], "pass": bool(lo <= r[m] <= hi)}
for route in ("local", "cac_route"):
    delta = abs(d["variants"]["rcx_h2"][route]["hand_jerk"] - t["test641"]["rcx_h2"][route]["hand_jerk"])
    ds = abs(d["variants"]["rcx_h2"][route]["hand_speed"] - t["test641"]["rcx_h2"][route]["hand_speed"])
    chk[f"dev515_{route}_jerk_within_0.05_of_test"] = {"value": delta, "pass": bool(delta <= 0.05)}
    chk[f"dev515_{route}_speed_within_0.05_of_test"] = {"value": ds, "pass": bool(ds <= 0.05)}
lp = pre["predicted_ledger_counts_432_cac_route_clips"]; got = t["ledger"]["rcx_h2"]["cac_route_432_local_clips"]
chk["ledger_exact_25757"] = {"value": got["exact_archive_rows"], "pass": got["exact_archive_rows"] == lp["exact_archive_rows"]}
chk["ledger_fractional_3804"] = {"value": got["non_archive_positions"], "pass": got["non_archive_positions"] == lp["strictly_fractional"]}
chk["convexity_assertion"] = {"value": t["convexity"]["min_coefficient"], "pass": t["convexity"]["assertion_passed"]}
chk["refutation_threshold_jerk_le_1.15"] = {"value": t["test641"]["rcx_h2"]["local"]["hand_jerk"],
                                            "pass": t["test641"]["rcx_h2"]["local"]["hand_jerk"] <= 1.15}
G = pre["frozen_evaluator_guard"]; guard = {}
for route, tag, ref in (("local", "jerkstudy_local_rcx_h2_test", G["frozen_reference_local_test"]),
                        ("cac_route", "jerkstudy_cacroute_rcx_h2_test", G["frozen_reference_cacroute_test"])):
    e = ev(tag); cur = {"wer": e["wer"], "bleu4": e["bleu"]["bleu4"], "dtw_mje": e["dtw_mje"]}
    guard[route] = {"rcx": cur, "frozen": ref,
                    "delta": {k: cur[k] - ref[k] for k in cur},
                    "within_tolerance": {k: bool(abs(cur[k] - ref[k]) <= G["tolerance"][k]) for k in cur}}
rp = ev("rcx_repro_frozen_local_test")
repro_ev = {"wer": rp["wer"] - G["frozen_reference_local_test"]["wer"],
            "bleu4": rp["bleu"]["bleu4"] - G["frozen_reference_local_test"]["bleu4"],
            "dtw_mje": rp["dtw_mje"] - G["frozen_reference_local_test"]["dtw_mje"]}
print("evaluator determinism re-run of the frozen local test, difference vs recorded:", {k: f"{v:+.3e}" for k, v in repro_ev.items()})
for route in guard:
    g = guard[route]
    print(f"guard {route:9s} WER {g['rcx']['wer']:.4f} ({g['delta']['wer']:+.4f}, tol 1.0 -> {g['within_tolerance']['wer']}) | "
          f"BLEU4 {g['rcx']['bleu4']:.4f} ({g['delta']['bleu4']:+.4f}, tol 0.3 -> {g['within_tolerance']['bleu4']}) | "
          f"DTW {g['rcx']['dtw_mje']:.6f} ({g['delta']['dtw_mje']:+.6f}, tol 0.0005 -> {g['within_tolerance']['dtw_mje']})")
for k, v in chk.items(): print(f"{'PASS' if v['pass'] else 'FAIL'}  {k:46s} = {v['value']}")
out = {"rows": rows, "acceptance": chk, "evaluator_guard": guard, "evaluator_determinism_delta": repro_ev,
       "hermite3_h2_evaluator": {"local": ev("jerkstudy_local_hermite3_h2_test")["wer"],
                                 "cac_route": ev("jerkstudy_cacroute_hermite3_h2_test")["wer"]},
       "provenance": t["provenance"], "convexity": t["convexity"],
       "sha256": {"js_repairs.py": hashlib.sha256(src.encode()).hexdigest(),
                  "rcx_bridge_source": hashlib.sha256(fn.encode()).hexdigest(),
                  "rcx_predeclaration.json": hashlib.sha256((O / "rcx_predeclaration.json").read_bytes()).hexdigest()},
       "all_acceptance_passed": all(v["pass"] for v in chk.values())}
json.dump(out, open(O / "rcx_summary.json", "w"), indent=1)
print("ALL ACCEPTANCE PASSED:", out["all_acceptance_passed"])
print("sha256:", json.dumps(out["sha256"], indent=1))
