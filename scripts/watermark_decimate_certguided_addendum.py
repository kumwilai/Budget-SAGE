"""POST-HOC addendum (declared as such) to the preregistered watermark sweep.

The preregistered sweep found that evaluator-style decimation by 2 -- the frame
stride the official SLRTP evaluator itself applies to a 25 fps bank -- destroys
detection (TPR@z8 = 0.000).  This addendum tests whether the certificate-guided
inversion that was ALREADY part of July's audit and of the preregistered sweep
(resample back to the certified emitted frame count before detecting) recovers
that case.  It introduces no new detector and no new scheme; it applies an
existing verifier step to one more transform.

It is reported separately from the preregistered records and is explicitly
labelled post-hoc, because it was added after seeing the decimation failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/kumwilai/research/signgen-t2m")
sys.path.insert(0, str(ROOT))

from scripts.pose_watermark import _resample  # noqa: E402
from scripts.watermark_generative_route_20260911 import (  # noqa: E402
    OUT, SRC_BANK, auc_and_tpr, dev_key, key_id, load_bank, true_bits, z_of,
)

BANKS = {"1e-3": "test641_mt5gloss_wm.pt", "2.5e-4": "test641_mt5gloss_wm_eps2.5e-4.pt"}


def main() -> int:
    key = dev_key()
    src = load_bank(SRC_BANK)
    rec = {"stage": "post_hoc_addendum", "declared_post_hoc": True,
           "preregistration_sha256": "065bdc4701a77ea0871cb15cbc1b41c3660cf0e0359545ced6b00ba5fcfcf3e1",
           "interpreter": sys.executable, "key_id_sha256_16": key_id(key),
           "conditions": {}}
    for eps_tag, name in BANKS.items():
        wm = load_bank(OUT / name)
        for cond, fn in (
            ("evaluator_decimate2_certguided",
             lambda p: _resample(p[::2].contiguous(), p.shape[0])),
        ):
            zp, zn, ber = [], [], []
            for sid in wm:
                z, bits = z_of(fn(wm[sid]), key, sid)
                tb = true_bits(sid, "test641_mt5gloss.pt")
                ber.append(float((np.asarray(bits[:len(tb)]) != tb).mean()))
                zp.append(z)
                zn.append(z_of(fn(src[sid]), key, sid)[0])
            row = auc_and_tpr(np.asarray(zp), np.asarray(zn))
            row["payload_ber_mean"] = float(np.mean(ber))
            rec["conditions"][f"eps{eps_tag}::{cond}"] = row
            print(f"[addendum] eps={eps_tag} {cond}: AUC={row['auc']:.4f} "
                  f"TPR@z8={row['tpr_at_z8']:.3f} (CP95 lb {row['tpr_at_z8_cp95_lower']:.3f}) "
                  f"BER={row['payload_ber_mean']:.4f} z_pos_med={row['z_pos_median']:.1f} "
                  f"n_pos={row['n_pos']} n_neg={row['n_neg']}", flush=True)
    out = OUT / "posthoc_decimate_certguided.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
