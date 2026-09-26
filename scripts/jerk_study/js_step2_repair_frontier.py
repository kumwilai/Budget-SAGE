import sys, json, time
sys.path.insert(0, "/tmp/claude-1000/-home-kumwilai-research-signgen-t2m/ef04ff5e-1c47-4276-b131-90e6a9dbc109/scratchpad")
from js_common import *; import js_repairs as R
raw = torch.load(OUT / "frozen_local_test_raw_segments.pt", weights_only=False); ids = list(raw); gt = load_gt()
local = torch.load(LOCAL_PT, map_location="cpu", weights_only=False)
led = json.load(open(ROOT / "outputs/revision/clean_strict_cac_frame40_test_ledger.json"))
cac_local = {c["id"] for c in led["clips"] if c["mode"] == "compositional_local_reuse"}
variants = {
 "frozen cosine blend w=4": lambda s: R.frozen_blend(s)[0],
 "plain concatenation (no blend)": R.plain_concat,
 "ramp exclusive, shifted target w=4": lambda s: R.ramp_exclusive(s, 4, "shifted"),
 "ramp exclusive, hold target w=4": lambda s: R.ramp_exclusive(s, 4, "hold"),
 "xfade extrap h=2": lambda s: R.xfade_extrap(s, 2), "xfade extrap h=3": lambda s: R.xfade_extrap(s, 3), "xfade extrap h=4": lambda s: R.xfade_extrap(s, 4),
 "hermite3 (C1) h=2": lambda s: R.hermite_bridge(s, 2, 3), "hermite3 (C1) h=3": lambda s: R.hermite_bridge(s, 3, 3),
 "hermite5 (C2) h=1": lambda s: R.hermite_bridge(s, 1, 5), "hermite5 (C2) h=2": lambda s: R.hermite_bridge(s, 2, 5),
 "hermite5 (C2) h=3": lambda s: R.hermite_bridge(s, 3, 5), "hermite5 (C2) h=4": lambda s: R.hermite_bridge(s, 4, 5), "hermite5 (C2) h=6": lambda s: R.hermite_bridge(s, 6, 5),
 "hermite5 pos+vel only (aA=aB=0) h=3": lambda s: R.hermite_bridge(s, 3, 5, accel=False),
 "hermite5 h=2 on top of frozen blend": lambda s: R.hermite_bridge(s, 2, 5, base="frozen"), "hermite5 h=3 on top of frozen blend": lambda s: R.hermite_bridge(s, 3, 5, base="frozen"),
 "binomial r=2 lam=1.0": lambda s: R.binomial_local(s, 2, 1.0), "binomial r=3 lam=0.7": lambda s: R.binomial_local(s, 3, 0.7),
 "binomial r=3 lam=1.0": lambda s: R.binomial_local(s, 3, 1.0), "binomial r=5 lam=0.7": lambda s: R.binomial_local(s, 5, 0.7), "binomial r=5 lam=1.0": lambda s: R.binomial_local(s, 5, 1.0),
}
rows = []
for name, fn in variants.items():
    preds = {}; touched = 0; touched_cac = 0; frames = 0
    for sid in ids:
        p = fn(raw[sid]); preds[sid] = p; base = R.plain_concat(raw[sid])
        n_t = int((np.abs(p - base) > 0).any(1).sum()); touched += n_t; frames += p.shape[0]
        if sid in cac_local: touched_cac += n_t
    if name.startswith("frozen"):
        assert all(np.array_equal(preds[s], local[s].reshape(local[s].shape[0], -1).numpy()) for s in ids), "frozen mismatch"
    r = ratios(preds, gt, ids)
    row = dict(name=name, **r, non_archive_frames_641=touched, frac_non_archive=touched / frames, non_archive_frames_in_432_cac_local_clips=touched_cac); rows.append(row)
    print(f"{name:40s} jerk {r['hand_speed'] and r['hand_jerk']:.3f} speed {r['hand_speed']:.3f} var {r['hand_posestd']:.3f} OBJ {r['joint_obj']:.3f} | xy(lead) jerk {r['xy_jerk_lead_scale']:.3f} speed {r['xy_speed_lead_scale']:.3f} | xy vs xyGT jerk {r['xy_jerk_vs_xyGT']:.3f} | non-archive frames {touched} ({touched/frames:.3%}), in CAC-local clips {touched_cac}", flush=True)
json.dump(rows, open(OUT / "priority2_repair_frontier_test641.json", "w"), indent=1)
