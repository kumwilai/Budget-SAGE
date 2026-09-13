"""Assemble the provenance record for the detector-v2 run."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT = Path("/home/kumwilai/research/signgen-t2m")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/watermark_detector_v2_20260911"
from scripts.watermark_generative_route_20260911 import dev_key, key_id, sha256_file  # noqa

def h(p: Path) -> str:
    return sha256_file(p) if p.exists() else "MISSING"

scripts = ["scripts/pose_watermark_v2.py", "scripts/watermark_detector_v2_diagnosis.py",
           "scripts/watermark_detector_v2_run.py", "scripts/watermark_detector_v2_select.py",
           "scripts/watermark_detector_v2_summarise.py",
           "scripts/watermark_detector_v2_provenance.py"]
inputs = {
    "unmarked_test_bank": "outputs/revision/generative_route_restore_20260911/test641_mt5gloss.pt",
    "unmarked_dev_bank": "outputs/revision/generative_route_restore_20260911/dev80_mt5gloss.pt",
    "july_marked_test_eps2.5e-4": "outputs/watermark_generative_route_20260911/test641_mt5gloss_wm_eps2.5e-4.pt",
    "july_marked_test_eps1e-3": "outputs/watermark_generative_route_20260911/test641_mt5gloss_wm.pt",
    "test_ground_truth": "external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/test.pt",
    "dev_ground_truth": "external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt",
    "back_translation_ckpt": "external/SLRTP-Sign-Production-Evaluation/pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model/best.ckpt",
    "july_scheme_source_unmodified": "scripts/pose_watermark.py",
}
rec = {
    "run": "pose-watermark detector v2 (blind chip-grid search + dead-channel rule) and embedding v2",
    "date": "2026-09-11",
    "interpreter": "/home/kumwilai/research/coopns-slr/.venv/bin/python",
    "device": "CPU for all detection, embedding and motion metrics; GPU only for the single evaluator pass",
    "preregistration": {
        "path": str(OUT / "preregistration_detector_v2_20260911_frozen.md"),
        "sha256": h(OUT / "preregistration_detector_v2_20260911_frozen.md"),
        "supersedes_sentence": ("section 1 of preregistration_watermark_20260911_frozen.md "
                                "(sha256 065bdc4701a77ea0871cb15cbc1b41c3660cf0e0359545ced6b00ba5fcfcf3e1), "
                                "'No third evaluator pass will be spent.'"),
    },
    "key_id_sha256_16": key_id(dev_key()),
    "key_material_recorded": False,
    "seeds": {"perturbation_rng": "sha256('0||<transform>||<sid>')",
              "derangement_rng": 0, "subsample_rng": 0},
    "detector_v2": {
        "hypothesis_grid": "np.geomspace(0.4, 2.5, 63) times L; phases {0,1/4,1/2,3/4}*L'; 252 hypotheses",
        "embedded_chip_length_L": {"july_banks": 2, "embedding_v2": 4},
        "dead_channel_rule": ">50% of high-pass residual samples exactly zero, OR residual std <= 1e-5 clamp floor",
        "null_keys": "SHA-256(key||'null'||i), i=0..31, M=32",
        "primary_threshold": "z_key > 8, fixed in the preregistration",
        "declared_limit": "the detector needs the clip identifier; identifier-free detection is not designed or claimed",
    },
    "scripts": {s: h(ROOT / s) for s in scripts},
    "inputs": {k: {"path": str(ROOT / v), "sha256": h(ROOT / v)} for k, v in inputs.items()},
    "artifacts": {p.name: h(p) for p in sorted(OUT.glob("*.json")) if p.name != "provenance_detector_v2_20260911.json"},
    "commands": [
        "/home/kumwilai/research/coopns-slr/.venv/bin/python scripts/watermark_detector_v2_diagnosis.py",
        "/home/kumwilai/research/coopns-slr/.venv/bin/python scripts/watermark_detector_v2_run.py phase1",
        "/home/kumwilai/research/coopns-slr/.venv/bin/python scripts/watermark_detector_v2_select.py",
        "/home/kumwilai/research/coopns-slr/.venv/bin/python scripts/watermark_detector_v2_run.py phase2",
        ("/home/kumwilai/research/coopns-slr/.venv/bin/python scripts/score_generative_route.py "
         "--bank_pt outputs/watermark_generative_route_20260911/test641_mt5gloss_wm_eps2.5e-4.pt "
         "--split test --tag marked_eps2.5e-4_test641 "
         "--workspace outputs/watermark_detector_v2_20260911/evaluator_workspace"),
    ],
}
ws = OUT / "evaluator_workspace/results"
if ws.exists():
    rec["evaluator_results"] = {p.name: h(p) for p in sorted(ws.glob("*.json"))}
(OUT / "provenance_detector_v2_20260911.json").write_text(json.dumps(rec, indent=2))
print(json.dumps({k: v for k, v in rec.items() if k not in ("inputs", "artifacts")}, indent=2))
print("wrote", OUT / "provenance_detector_v2_20260911.json")
