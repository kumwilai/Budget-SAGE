"""RCX dev-515 confirmation: reconstruct dev raw segments, reproduce the recorded dev numbers,
then measure RCX local and CAC-route."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from js_common import *; import js_repairs as R

trace = json.load(open(ROOT / "outputs/sota_chase/phase43_phrase_lattice/trace_revision_clean_phrase_lattice_dev.json"))
local = torch.load(ROOT / "outputs/revision/revision_clean_phrase_lattice_dev.pt", map_location="cpu", weights_only=False)
ids = list(local)
gt = paper_metric.load_gt(ROOT / "external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt")
bank = torch.load(BANK, map_location="cpu", weights_only=False); pool = bank["exemplar_poses"]
raw = {s: [pool[g["source"]][g["start"]:g["end"]].astype(np.float32) for g in trace[s]["segments"]] for s in ids}
del bank, pool
mism = sum(int(not np.array_equal(R.frozen_blend(raw[s])[0], local[s].reshape(local[s].shape[0], -1).numpy())) for s in ids)
print(f"dev clips {len(ids)} reconstruction mismatches {mism}")

route = torch.load(ROOT / "outputs/revision/clean_strict_cac_frame40_dev.pt", map_location="cpu", weights_only=False)
led = json.load(open(ROOT / "outputs/revision/clean_strict_cac_frame40_dev_ledger.json"))
mode = {c["id"]: c["mode"] for c in led["clips"]}
cac_local = [s for s in ids if mode.get(s) == "compositional_local_reuse"]
route_np = {s: route[s].float().reshape(route[s].shape[0], -1).numpy() for s in ids}
print("dev CAC-local clips", len(cac_local), "of", len(ids))

REC = {"frozen": {"hand_jerk": 1.8090, "hand_speed": 1.1058, "hand_posestd": 0.9088},
       "hermite3_h2": {"hand_jerk": 0.9830, "hand_speed": 0.9973, "hand_posestd": 0.9254}}
res = {"n": len(ids), "n_cac_local": len(cac_local), "reconstruction_mismatches": mism, "variants": {}}
for name, fn in (("frozen", lambda s: R.frozen_blend(s)[0]), ("hermite3_h2", lambda s: R.hermite_bridge(s, 2, 3)),
                 ("rcx_h2", lambda s: R.rcx_bridge(s, 2))):
    loc = {s: fn(raw[s]) for s in ids}
    rt = dict(route_np)
    for s in cac_local: rt[s] = loc[s]
    rl = ratios(loc, gt, ids); rr = ratios(rt, gt, ids)
    res["variants"][name] = {"local": rl, "cac_route": rr}
    tag = ""
    if name in REC:
        d = max(abs(rl[k] - REC[name][k]) for k in REC[name]); res["variants"][name]["repro_max_abs_diff"] = d
        tag = f" | REPRO max|diff| vs recorded {d:+.4f}"
    print(f"DEV {name:12s} LOCAL jerk {rl['hand_jerk']:.4f} speed {rl['hand_speed']:.4f} std {rl['hand_posestd']:.4f} "
          f"obj {rl['joint_obj']:.4f} | CAC-ROUTE jerk {rr['hand_jerk']:.4f} speed {rr['hand_speed']:.4f} "
          f"std {rr['hand_posestd']:.4f} obj {rr['joint_obj']:.4f}{tag}", flush=True)
    if name == "rcx_h2":
        to_pt = lambda d: {s: torch.from_numpy(np.ascontiguousarray(v.reshape(v.shape[0], 178, 3))).float() for s, v in d.items()}
        torch.save(to_pt(loc), OUT / "jerkstudy_local_rcx_h2_dev.pt"); torch.save(to_pt(rt), OUT / "jerkstudy_cacroute_rcx_h2_dev.pt")
json.dump(res, open(OUT / "rcx_results_dev515.json", "w"), indent=1)
print("wrote", OUT / "rcx_results_dev515.json")
