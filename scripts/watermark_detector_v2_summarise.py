"""Print the Phase-2 tables and the residual table from the recorded JSON."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

OUT = Path("/home/kumwilai/research/signgen-t2m/outputs/watermark_detector_v2_20260911")
name = sys.argv[1] if len(sys.argv) > 1 else "selected_tier1_eps2.5e-4"
R = json.loads((OUT / f"phase2_{name}.json").read_text())
C = R["conditions"]; n = R["n"]
print(f"bank={name}  n_positives={n}  L={R['L']}  key_id={R.get('key_id_sha256_16')}")
print(f"marked bank sha256 {R['marked_bank']['sha256']}")
hdr = (f"{'transform':26s} {'TPR@z8':>7s} {'CP95lo':>7s} {'zmed':>8s} {'zmin':>8s} {'AUC':>6s} "
       f"{'BER':>6s} {'N1fpr':>6s} {'N2fpr':>6s} {'N3fpr':>6s} {'N4fpr':>6s} {'negZmax':>8s} "
       f"{'hspd':>6s} {'bspd':>6s} {'fspd':>6s} {'jerk':>6s} {'hstd':>6s}")
print(hdr); print("-" * len(hdr))
for t, r in C.items():
    m = r["motion_vs_gt"]
    negmax = max(r[f"{p}_z_max"] for p in ("N1", "N2", "N3", "N4"))
    print(f"{t:26s} {r['tpr_at_zkey8']:7.4f} {r['tpr_cp95_lower']:7.4f} {r['z_pos_median']:8.2f} "
          f"{r['z_pos_min']:8.2f} {r['auc_vs_N1']:6.4f} {r['payload_ber_mean']:6.4f} "
          + " ".join(f"{r[f'{p}_fpr_at_zkey8']:6.4f}" for p in ("N1", "N2", "N3", "N4"))
          + f" {negmax:8.2f} {m['hand_speed_ratio']:6.4f} {m['body_speed_ratio']:6.4f} "
            f"{m['face_speed_ratio']:6.4f} {m['hand_jerk_ratio']:6.4f} {m['hand_posestd_ratio']:6.4f}")
print()
i = C["identity"]
print(f"identity: analytic z with dead-channel fix  median {i['z_analytic_identity_median']:.2f} "
      f"min {i['z_analytic_identity_min']:.2f}")
print(f"dead channels on unquantised float32 (identity): median {i['n_dead_channels_median']:.0f} "
      f"max {i['n_dead_channels_max']:.0f} of 534")
print("binding:", json.dumps(R["binding"]))
errs = {t: {p: C[t][f"{p}_detector_errors"] for p in ("N1", "N2", "N3", "N4")}
        | {"P": C[t]["detector_errors_pos"]} for t in C}
tot = sum(sum(v.values()) for v in errs.values())
print(f"detector exceptions across all transforms and pools: {tot} "
      f"(denominator {n} x {len(C)} transforms x 5 pools = {n*len(C)*5})")
print(f"FPR Clopper-Pearson 95% upper bound at 0/{n}: {C['identity']['N1_fpr_cp95_upper']:.5f}")
