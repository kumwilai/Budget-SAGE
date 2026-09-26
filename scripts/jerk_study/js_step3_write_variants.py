import sys, json
sys.path.insert(0, "/tmp/claude-1000/-home-kumwilai-research-signgen-t2m/ef04ff5e-1c47-4276-b131-90e6a9dbc109/scratchpad")
from js_common import *; import js_repairs as R
raw = torch.load(OUT / "frozen_local_test_raw_segments.pt", weights_only=False); ids = list(raw); gt = load_gt()
local = torch.load(LOCAL_PT, map_location="cpu", weights_only=False)
route = torch.load(ROOT / "outputs/revision/clean_strict_cac_frame40_test.pt", map_location="cpu", weights_only=False)
led = json.load(open(ROOT / "outputs/revision/clean_strict_cac_frame40_test_ledger.json"))
mode = {c["id"]: c["mode"] for c in led["clips"]}; cac_local = [s for s in ids if mode[s] == "compositional_local_reuse"]
same = all(torch.equal(route[s].float(), local[s].float()) for s in cac_local); print("route local clips identical to local .pt:", same, "n local", len(cac_local), "whole", len(ids) - len(cac_local))
variants = {"hermite3_h2": lambda s: R.hermite_bridge(s, 2, 3), "hermite5_h2": lambda s: R.hermite_bridge(s, 2, 5), "hermite5_h3": lambda s: R.hermite_bridge(s, 3, 5), "ramp_hold_w4": lambda s: R.ramp_exclusive(s, 4, "hold")}
def to_pt(d): return {s: torch.from_numpy(np.ascontiguousarray(v.reshape(v.shape[0], 178, 3))).float() for s, v in d.items()}
route_np = {s: route[s].float().reshape(route[s].shape[0], -1).numpy() for s in ids}
print("route baseline paper metric:", {k: round(v, 3) for k, v in ratios(route_np, gt, ids).items() if k in ("hand_jerk", "hand_speed", "hand_posestd")}, "(paper: 1.483 / 1.071 / 0.932)")
summary = {}
for name, fn in variants.items():
    loc = {s: fn(raw[s]) for s in ids}
    torch.save(to_pt(loc), OUT / f"jerkstudy_local_{name}_test.pt")
    rt = dict(route_np); 
    for s in cac_local: rt[s] = loc[s]
    torch.save(to_pt(rt), OUT / f"jerkstudy_cacroute_{name}_test.pt")
    rl = ratios(loc, gt, ids); rr = ratios(rt, gt, ids)
    summary[name] = {"local": rl, "cac_route": rr}
    print(f"{name:14s} LOCAL jerk {rl['hand_jerk']:.3f} speed {rl['hand_speed']:.3f} var {rl['hand_posestd']:.3f} | xy-consistent jerk {rl['xy_jerk_vs_xyGT']:.3f} speed {rl['xy_speed_vs_xyGT']:.3f} || CAC ROUTE jerk {rr['hand_jerk']:.3f} speed {rr['hand_speed']:.3f} var {rr['hand_posestd']:.3f}")
    # overshoot diagnostic: hand speed at replaced frames vs archive frames, and worst single-frame hand displacement
    sp_b = []; sp_a = []
    for s in ids:
        p = loc[s].reshape(-1, 178, 3)[:, 8:50]; base = R.plain_concat(raw[s]).reshape(-1, 178, 3)[:, 8:50]
        touched = (np.abs(loc[s] - R.plain_concat(raw[s])) > 0).any(1); v = np.linalg.norm(np.diff(p, axis=0), axis=2).mean(1)
        tm = touched[1:] | touched[:-1]; sp_b += v[tm].tolist(); sp_a += v[~tm].tolist()
    print(f"    per-frame hand speed: bridge frames median {np.median(sp_b):.4f} p99 {np.percentile(sp_b,99):.4f} max {np.max(sp_b):.4f} | archive frames median {np.median(sp_a):.4f} p99 {np.percentile(sp_a,99):.4f} max {np.max(sp_a):.4f}")
json.dump(summary, open(OUT / "priority2_variants_local_and_cacroute.json", "w"), indent=1)
