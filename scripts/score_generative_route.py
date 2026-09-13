"""Score a generated bank with the frozen SLRTP evaluator plus the motion audit.

The evaluator is run unmodified from external/SLRTP-Sign-Production-Evaluation
with the working directory set to a local workspace, so nothing is written into
the external tree.  A ground-truth subset file is built when the bank covers
fewer clips than the released split; it is used only for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/kumwilai/research/signgen-t2m")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

EVAL_DIR = ROOT / "external/SLRTP-Sign-Production-Evaluation"
BT_MODEL = EVAL_DIR / "pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_pt", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workspace",
                    default="outputs/revision/generative_route_restore_20260911/evaluator_workspace")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    bank_path = ROOT / args.bank_pt
    release_gt = EVAL_DIR / f"pretrained/SLRTP-Sign-Production-Evaluation-Data/data/{args.split}.pt"
    workspace = ROOT / args.workspace
    (workspace / "results").mkdir(parents=True, exist_ok=True)

    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    gt = torch.load(release_gt, map_location="cpu", weights_only=True)
    missing = [sid for sid in bank if sid not in gt]
    if missing:
        raise KeyError(f"{len(missing)} generated ids are not in the release, first={missing[:3]}")

    if len(bank) == len(gt):
        gt_path = release_gt
    else:
        subset = {sid: gt[sid] for sid in gt if sid in bank}
        gt_path = workspace / f"gt_subset_{args.split}_{len(subset)}.pt"
        torch.save(subset, gt_path)
    del gt, bank

    cmd = [sys.executable, str(EVAL_DIR / "main.py"), str(bank_path), str(gt_path),
           str(BT_MODEL), "--tag", args.tag, "--fps", "25"]
    proc = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise RuntimeError("evaluator failed")
    metrics = json.loads((workspace / "results" / f"{args.tag}.json").read_text())

    motion_json = workspace / "results" / f"{args.tag}_motion.json"
    motion_cmd = [sys.executable, str(ROOT / "scripts/sign_jepa_source_locked_fusion.py"),
                  "motion_eval", "--pred_pt", str(bank_path.relative_to(ROOT)),
                  "--reference_pt", str(gt_path.relative_to(ROOT)),
                  "--out_json", str(motion_json.relative_to(ROOT))]
    mproc = subprocess.run(motion_cmd, cwd=ROOT, capture_output=True, text=True)
    if mproc.returncode != 0:
        sys.stderr.write(mproc.stdout[-4000:] + mproc.stderr[-4000:])
        raise RuntimeError("motion_eval failed")
    motion = json.loads(motion_json.read_text())

    record = {
        "tag": args.tag,
        "split": args.split,
        "interpreter": sys.executable,
        "evaluator_command": " ".join(cmd),
        "evaluator_cwd": str(workspace),
        "motion_command": " ".join(motion_cmd),
        "inputs": {
            "bank": {"path": str(bank_path), "sha256": sha256(bank_path)},
            "ground_truth": {"path": str(gt_path), "sha256": sha256(gt_path)},
            "back_translation_model_ckpt": {
                "path": str(BT_MODEL / "best.ckpt"), "sha256": sha256(BT_MODEL / "best.ckpt")},
        },
        "metrics": metrics,
        "motion": motion,
    }
    out = ROOT / (args.out_json or f"{args.workspace}/results/{args.tag}_record.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"{args.tag}: n={motion['clips']} BLEU-4={metrics['bleu']['bleu4']:.4f} "
          f"WER={metrics['wer']:.4f} DTW-MJE={metrics['dtw_mje']:.6f} "
          f"avg_duration={metrics['avg_duration']:.4f} "
          f"hand_speed_ratio={motion['hand_speed_ratio']:.4f} "
          f"hand_posestd_ratio={motion['hand_posestd_ratio']:.4f}")
    print(f"record -> {out}")


if __name__ == "__main__":
    main()
