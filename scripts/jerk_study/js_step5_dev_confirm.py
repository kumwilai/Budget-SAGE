import sys, json
sys.path.insert(0, "/tmp/claude-1000/-home-kumwilai-research-signgen-t2m/ef04ff5e-1c47-4276-b131-90e6a9dbc109/scratchpad")
from js_common import *; import js_repairs as R
bank = torch.load(BANK, map_location="cpu", weights_only=False); pool = bank["exemplar_poses"]
trace = json.load(open(ROOT / "outputs/sota_chase/phase43_phrase_lattice/trace_revision_clean_phrase_lattice_dev.json"))
local = torch.load(ROOT / "outputs/revision/revision_clean_phrase_lattice_dev.pt", map_location="cpu", weights_only=False); ids = list(local)
gt = paper_metric.load_gt(ROOT / "external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt")
raw = {}; mism = 0
for s in ids:
    raw[s] = [pool[g["source"]][g["start"]:g["end"]].astype(np.float32) for g in trace[s]["segments"]]
    mism += int(not np.array_equal(R.frozen_blend(raw[s])[0], local[s].reshape(local[s].shape[0], -1).numpy()))
print("dev clips", len(ids), "reconstruction mismatches", mism)
variants = {"frozen": lambda s: R.frozen_blend(s)[0], "ramp_hold_w4": lambda s: R.ramp_exclusive(s, 4, "hold"), "hermite3_h2": lambda s: R.hermite_bridge(s, 2, 3), "hermite5_h2": lambda s: R.hermite_bridge(s, 2, 5), "hermite5_h3": lambda s: R.hermite_bridge(s, 3, 5), "hermite3_h3": lambda s: R.hermite_bridge(s, 3, 3)}
res = {}
for name, fn in variants.items():
    p = {s: fn(raw[s]) for s in ids}; r = ratios(p, gt, ids); res[name] = r
    print(f"DEV {name:14s} jerk {r['hand_jerk']:.3f} speed {r['hand_speed']:.3f} var {r['hand_posestd']:.3f} OBJ {r['joint_obj']:.3f} | xy-consistent jerk {r['xy_jerk_vs_xyGT']:.3f} speed {r['xy_speed_vs_xyGT']:.3f}")
    if name in ("hermite3_h2", "hermite5_h3"):
        torch.save({s: torch.from_numpy(np.ascontiguousarray(v.reshape(v.shape[0], 178, 3))).float() for s, v in p.items()}, OUT / f"jerkstudy_local_{name}_dev.pt")
json.dump(res, open(OUT / "priority2_dev_confirmation.json", "w"), indent=1)
