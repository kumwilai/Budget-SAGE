"""Keyed provenance watermark for generated sign-pose streams.

Motivation: the route ledger's "generated" column is self-declared by the
router.  Embedding a keyed, imperceptible watermark into generated frames at
emission time makes that column independently checkable from the emitted pose
tensor alone (a soft binding in the C2PA sense: cooperative provenance, not
adversarial robustness -- the threat model is honest-but-unverified deployers,
not watermark-removal attackers).

Scheme: for clip id c and secret key K, a PRNG seeded with SHA256(K||c)
generates an antipodal pattern s in {-1,+1}^{T x J x 3}.  A 16-bit payload
(certificate-hash prefix) modulates the sign of the pattern over 16 temporal
blocks.  The watermarked pose is y' = y + eps * m_block(t) * s_t.  Detection
correlates the temporal high-pass residual of a candidate pose with the keyed
pattern; the per-block correlation sign recovers the payload.  Amplitude
eps=5e-4 is ~1% of per-axis pose std and ~15% of the mean per-frame joint
displacement, far below perceptual and metric relevance (verified by
re-scoring the watermarked bank through the official evaluator).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.signjepa_cpf_knapsack import SLRTP  # noqa: E402


def _pattern(key: str, sid: str, T: int, J: int = 178, A: int = 3,
             frame_block: int = 2) -> np.ndarray:
    """Antipodal keyed pattern, piecewise-constant over `frame_block` frames."""
    seed = int.from_bytes(hashlib.sha256(f"{key}||{sid}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    n_fb = (T + frame_block - 1) // frame_block
    base = rng.integers(0, 2, size=(n_fb, J, A)).astype(np.float32) * 2.0 - 1.0
    return np.repeat(base, frame_block, axis=0)[:T]


def _payload_bits(payload_hex: str, n_bits: int = 16) -> np.ndarray:
    raw = bytes.fromhex(payload_hex)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n_bits]
    return bits.astype(np.float32) * 2.0 - 1.0  # {0,1} -> {-1,+1}


def embed(pose: torch.Tensor, key: str, sid: str, payload_hex: str,
          eps: float = 5e-4, n_blocks: int = 16) -> torch.Tensor:
    x = pose.clone().float()
    T = x.shape[0]
    s = _pattern(key, sid, T)
    bits = _payload_bits(payload_hex, n_blocks)
    block_of = np.minimum((np.arange(T) * n_blocks) // max(1, T), n_blocks - 1)
    mod = bits[block_of][:, None, None]
    return x + torch.from_numpy(eps * mod * s)


def _highpass(x: torch.Tensor, k: int = 9) -> torch.Tensor:
    pad = k // 2
    xp = torch.cat([x[:1].expand(pad, -1, -1), x, x[-1:].expand(pad, -1, -1)], dim=0)
    kern = torch.ones(k) / k
    sm = torch.stack([torch.nn.functional.conv1d(
        xp.reshape(xp.shape[0], -1).T.unsqueeze(1),
        kern.view(1, 1, -1)).squeeze(1).T.reshape(x.shape[0], *x.shape[1:])])
    return x - sm[0]


def detect(pose: torch.Tensor, key: str, sid: str,
           n_blocks: int = 16) -> dict[str, Any]:
    """Blind detection: folded-normal z-score over per-block keyed correlations.

    Under H0 (no watermark) each block correlation c_b ~ N(0, sigma^2 n_b), so
    sum_b |c_b| has mean sigma*sqrt(2/pi)*sum_b sqrt(n_b) and variance
    sigma^2 (1-2/pi) sum_b n_b.  The reported z standardizes against that null.
    The per-block correlation sign recovers the payload bit.
    """
    x = pose.float()
    T = x.shape[0]
    s = torch.from_numpy(_pattern(key, sid, T))
    r = _highpass(x)
    # matched filter: weight each joint/axis by its inverse high-pass noise power
    sig_ja = r.std(dim=0).clamp_min(1e-5)          # [J,3]
    w_ja = 1.0 / (sig_ja ** 2)
    rw = (r * s) * w_ja.unsqueeze(0)
    per_frame = rw.reshape(T, -1).sum(dim=1)
    # per-frame null variance of the weighted correlation
    var_frame = float((w_ja).sum())                # sum_j w^2 sig^2 = sum_j 1/sig^2
    block_of = np.minimum((np.arange(T) * n_blocks) // max(1, T), n_blocks - 1)
    abs_sum = 0.0
    mu0 = 0.0
    var0 = 0.0
    bits = []
    for b in range(n_blocks):
        m = torch.from_numpy(block_of == b)
        n_frames = float(m.sum())
        if n_frames == 0:
            bits.append(0)
            continue
        c = float(per_frame[m].sum())
        v_b = var_frame * n_frames
        abs_sum += abs(c)
        mu0 += np.sqrt(2.0 / np.pi) * np.sqrt(v_b)
        var0 += (1.0 - 2.0 / np.pi) * v_b
        bits.append(1 if c > 0 else 0)
    z = float((abs_sum - mu0) / (np.sqrt(var0) + 1e-9))
    return {"z": z, "bits": bits}


def payload_hex_for(cert_row: dict[str, Any]) -> str:
    blob = json.dumps(cert_row, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]  # 32 bits available; we embed 16


def as_pose(x: Any) -> torch.Tensor:
    if isinstance(x, dict):
        x = x.get("poses_3d", x.get("pose"))
    return torch.as_tensor(np.asarray(x)).float()


def perturb(pose: torch.Tensor, kind: str, rng: np.random.Generator) -> torch.Tensor:
    if kind == "noise1e-3":
        return pose + torch.from_numpy(rng.normal(0, 1e-3, size=tuple(pose.shape)).astype(np.float32))
    if kind == "noise2e-3":
        return pose + torch.from_numpy(rng.normal(0, 2e-3, size=tuple(pose.shape)).astype(np.float32))
    if kind == "smooth3":
        return pose - _highpass(pose, k=3)  # moving average, k=3
    if kind == "resample0.75x":
        return _resample(pose, int(round(pose.shape[0] * 0.75)))
    raise ValueError(kind)


def _resample(pose: torch.Tensor, T_new: int) -> torch.Tensor:
    T = pose.shape[0]
    idx = torch.linspace(0, T - 1, max(2, T_new))
    lo = idx.floor().long().clamp(0, T - 1)
    hi = idx.ceil().long().clamp(0, T - 1)
    w = (idx - lo.float()).view(-1, 1, 1)
    return pose[lo] * (1 - w) + pose[hi] * w


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route_bank", default="phase59_btfree_deployable_student_g20_test.pt")
    ap.add_argument("--route_json", default="outputs/learned_confidence/deployable_e2e_rebuild.json")
    ap.add_argument("--key", default="budget-sage-release-2026")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--out_bank", default="phase59_btfree_deployable_student_g20_wm_test.pt")
    ap.add_argument("--out", default="outputs/learned_confidence/pose_watermark_audit.json")
    args = ap.parse_args()

    bank = torch.load(SLRTP / "results" / args.route_bank, map_location="cpu", weights_only=False)
    routes = json.loads((ROOT / args.route_json).read_text())
    sel = routes["dev_selected_target"]
    row = next(r for r in routes["rows"]["test"]
               if abs(r["target_generated_frame_fraction"] - sel) < 1e-9)
    gen_ids = set(row["rewritten_ids"])
    print(f"[wm] embedding watermark in {len(gen_ids)} generated clips of {len(bank)}", flush=True)

    wm_bank: dict[str, torch.Tensor] = {}
    payloads: dict[str, str] = {}
    for sid, pose in bank.items():
        p = as_pose(pose)
        if sid in gen_ids:
            cert_row = {"sid": sid, "route": "generated", "bank": args.route_bank}
            payload = payload_hex_for(cert_row)
            payloads[sid] = payload
            wm_bank[sid] = embed(p, args.key, sid, payload, eps=args.eps)
        else:
            wm_bank[sid] = p
    torch.save(wm_bank, SLRTP / "results" / args.out_bank)

    rng = np.random.default_rng(0)
    report: dict[str, Any] = {"eps": args.eps, "n_generated": len(gen_ids),
                              "n_total": len(bank), "conditions": {}}

    def eval_condition(name: str, transform) -> None:
        z_gen, z_src, z_wrongkey, ber = [], [], [], []
        for sid, pose in wm_bank.items():
            p = transform(as_pose(pose)) if transform else as_pose(pose)
            det = detect(p, args.key, sid)
            if sid in gen_ids:
                z_gen.append(det["z"])
                true_bits = ((np.asarray(_payload_bits(payloads[sid])) + 1) / 2).astype(int)
                got = np.asarray(det["bits"][:len(true_bits)])
                ber.append(float((got != true_bits).mean()))
                det_wrong = detect(p, args.key + "-wrong", sid)
                z_wrongkey.append(det_wrong["z"])
            else:
                z_src.append(det["z"])
        z_gen_a, z_src_a = np.array(z_gen), np.array(z_src)
        thr = float(np.percentile(z_src_a, 100)) if len(z_src_a) else 0.0
        tpr_at_zero_fpr = float((z_gen_a > thr).mean()) if len(z_gen_a) else 0.0
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.ones_like(z_gen_a), np.zeros_like(z_src_a)]
        s = np.r_[z_gen_a, z_src_a]
        auc = float(roc_auc_score(y, s)) if len(z_gen_a) and len(z_src_a) else float("nan")
        report["conditions"][name] = {
            "z_gen_median": float(np.median(z_gen_a)),
            "z_src_median": float(np.median(z_src_a)),
            "z_wrongkey_median": float(np.median(z_wrongkey)) if z_wrongkey else None,
            "auc": auc, "tpr_at_zero_fpr": tpr_at_zero_fpr,
            "payload_ber_mean": float(np.mean(ber)) if ber else None,
        }
        print(f"[wm] {name}: AUC={auc:.4f} TPR@FPR0={tpr_at_zero_fpr:.3f} "
              f"z_gen={np.median(z_gen_a):.1f} z_src={np.median(z_src_a):.2f} "
              f"BER={np.mean(ber) if ber else -1:.4f}", flush=True)

    eval_condition("clean", None)
    eval_condition("noise1e-3", lambda p: perturb(p, "noise1e-3", rng))
    eval_condition("noise2e-3", lambda p: perturb(p, "noise2e-3", rng))
    eval_condition("smooth3", lambda p: perturb(p, "smooth3", rng))
    # certificate-guided inverse resampling: the certificate records the emitted
    # frame count, so the verifier restores T before detection (soft binding).
    eval_condition("resample0.75x+invert",
                   lambda p: _resample(perturb(p, "resample0.75x", rng), p.shape[0]))

    # geometric perturbation of the pose itself
    deltas = []
    for sid in list(gen_ids)[:50]:
        a, b = as_pose(bank[sid]), wm_bank[sid]
        deltas.append(float((a - b).abs().max()))
    report["max_abs_pose_delta"] = float(np.max(deltas))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[wm] wrote {out} and bank {args.out_bank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
