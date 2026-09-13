"""Apply the frozen Phase-1 selection rule. No thresholds are computed here that
are not written in the frozen preregistration."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

OUT = Path("/home/kumwilai/research/signgen-t2m/outputs/watermark_detector_v2_20260911")
S = json.loads((OUT / "phase1_summary.json").read_text())
base = S["unmarked_motion_vs_gt"]
JERK_MAX = base["hand_jerk_ratio"] * 1.02
SPD_LO, SPD_HI = base["hand_speed_ratio"] * 0.995, base["hand_speed_ratio"] * 1.005

rows = []
for name, cfg in S["configs"].items():
    C = cfg["conditions"]
    idn, de, do = C["identity"], C["evaluator_decimate2"], C["decimate2_odd"]
    c1 = (de["tpr_at_zkey8"] == 1.0 and do["tpr_at_zkey8"] == 1.0
          and min(de["z_pos_min"], do["z_pos_min"]) >= 12.0)
    c2 = idn["z_pos_min"] >= 20.0
    m = idn["motion_vs_gt"]
    c3 = (m["hand_jerk_ratio"] <= JERK_MAX and SPD_LO <= m["hand_speed_ratio"] <= SPD_HI)
    fp = max(C[t][f"{p}_fpr_at_zkey8"] for t in C for p in ("N1", "N2", "N3", "N4"))
    n2max = max(C[t]["N2_z_max"] for t in C)
    worst = max(((C[t][f"{p}_z_max"], t, p) for t in C for p in ("N1", "N2", "N3", "N4")))
    c4 = (fp == 0.0 and n2max < 5.0)
    rows.append(dict(name=name, c1=c1, c2=c2, c3=c3, c4=c4, pass_all=all([c1, c2, c3, c4]),
                     clean_zmin=idn["z_pos_min"], clean_zmed=idn["z_pos_median"],
                     dec_tpr=(de["tpr_at_zkey8"], do["tpr_at_zkey8"]),
                     dec_zmin=min(de["z_pos_min"], do["z_pos_min"]),
                     jerk=m["hand_jerk_ratio"], hspd=m["hand_speed_ratio"],
                     face=m["face_speed_ratio"],
                     face_incr=m["face_speed_ratio"] / base["face_speed_ratio"] - 1.0,
                     max_fpr=fp, n2_zmax=n2max, worst_neg=worst,
                     ndead_identity_max=idn["n_dead_channels_max"]))

print("dev80 unmarked baseline:", json.dumps(base))
print(f"gate: jerk<= {JERK_MAX:.5f}  hand speed in [{SPD_LO:.5f},{SPD_HI:.5f}]  (x1.02 / +-0.5%)")
hdr = f"{'config':18s} {'c1dec':6s} {'c2cln':6s} {'c3mot':6s} {'c4neg':6s} {'PASS':5s} {'cleanZmin':10s} {'decZmin':8s} {'jerk':8s} {'handspd':8s} {'face':7s} {'faceΔ%':8s} {'N2zmax':7s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['name']:18s} {str(r['c1']):6s} {str(r['c2']):6s} {str(r['c3']):6s} {str(r['c4']):6s} "
          f"{str(r['pass_all']):5s} {r['clean_zmin']:10.2f} {r['dec_zmin']:8.2f} {r['jerk']:8.4f} "
          f"{r['hspd']:8.4f} {r['face']:7.4f} {100*r['face_incr']:8.2f} {r['n2_zmax']:7.2f}")
print()
for r in rows:
    print(f"{r['name']:18s} worst negative across all pools/transforms: z_key={r['worst_neg'][0]:.2f} "
          f"at {r['worst_neg'][1]}/{r['worst_neg'][2]}; max FPR@z8={r['max_fpr']:.4f}; "
          f"dead channels on identity max={r['ndead_identity_max']:.0f}")

t2 = [r for r in rows if r["name"].startswith("tier2") and r["pass_all"]]
t1 = [r for r in rows if r["name"].startswith("tier1")][0]
if t2:
    sel = min(t2, key=lambda r: r["face_incr"])
    print(f"\nSELECTED (Tier 2, smallest face-speed increase among passers): {sel['name']}")
elif t1["c1"]:
    sel = t1
    print(f"\nNo Tier 2 configuration passes -> SELECTED Tier 1: {t1['name']} "
          f"(pass_all={t1['pass_all']})")
else:
    sel = None
    print("\nTier 1 fails the decimation criterion -> STOP AND REPORT")
(OUT / "phase1_selection.json").write_text(json.dumps(
    {"baseline": base, "jerk_max": JERK_MAX, "speed_window": [SPD_LO, SPD_HI],
     "rows": rows, "selected": sel["name"] if sel else None}, indent=2, default=str))
print("wrote", OUT / "phase1_selection.json")
