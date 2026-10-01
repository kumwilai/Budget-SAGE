"""Shared helpers for the stage-2 chain (scripts/recent_baselines/stage2_chain.sh).

Paths, recorded values that must reproduce, sha256, and the machine-written summary line that
each step appends to the "Stage 2 (auto)" section at the end of RESULTS.md.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/kumwilai/research/signgen-t2m")
PY = "/home/kumwilai/research/coopns-slr/.venv/bin/python"
EV = ROOT / "external/SLRTP-Sign-Production-Evaluation"
DATA = Path(os.environ.get("S2_DATA", EV / "pretrained/SLRTP-Sign-Production-Evaluation-Data/data"))  # override: CPU tests only
BT_MODEL = EV / "pretrained/SLRTP-Sign-Production-Evaluation-Data/backTranslation_PHIX_model"
RB = ROOT / "outputs/recent_baselines_2026-09-28"
RESULTS_MD = Path(os.environ.get("S2_RESULTS_MD", RB / "RESULTS.md"))  # override: CPU tests only
GATES = Path(os.environ.get("S2_GATES", RB / "stage2/gates"))
INP = ROOT / "outputs/signbase_slrtp_2026-09-11/inputs"
TEST_KEYS = INP / "test_keys.txt"
TEST_TEXT = INP / "test_text.txt"
TEST_LENS = INP / "test_trg_lengths.txt"

PINNED_SHA = {
    "test.pt": "edc66897d5ee72c286f9a55c3a2cb3194eebee7a0443caaa9b52ec9067abe46f",
    "dev.pt": "73acbfbcb63b05eae02071da58dec95177551d90bb3a711a36b78f2eeb73e85c",
    "test_keys.txt": "f4a610be7ceffc9c0870ca8a588f3d8c49c66aeea01d63ad3298d83c34cd7a60",
    "test_text.txt": "758b9eced2cb80a2a24d564f095590d2b3058f2376360b63fe8f201b927d6546",
    "test_trg_lengths.txt": "830b839430f5df5679c181db8ceb351bbaf9144106d4af79a3f89f057582f2ed",
}

# identity rows recorded in stage 1 (identity/results/gt_identity_{split}_fps25.json); difference must be 0.0
IDENTITY = {
    "test": {"bleu4": 12.777148564612425, "wer": 85.77470203767781, "dtw_mje": 0.0, "n": 641},
    "dev": {"bleu4": 13.378651856913775, "wer": 83.37019018133569, "dtw_mje": 0.0, "n": 515},
}

# fixed 40% route back-translations (manuscript Table IV); must reproduce before any bootstrap
FIXED_TEXT_PREDS = ROOT / "outputs/revision/evaluator_workspace/results/clean_rerank_frame40_test_text_preds.pt"
FIXED_TEXT_PREDS_SHA = "49db5307e4803f2564452bd3b5968ef30c9e193fb2b35d0b65e964b711e07003"
FIXED_RECORDED = {"bleu4": 15.1032, "wer": 85.582}
FIXED_LEDGER = ROOT / "outputs/revision/clean_rerank_frame40_test_ledger.json"
# route pose banks, located from the materialization records (output.sha256) and the learned manifest
FIXED_POSE_BANK = ROOT / "outputs/revision/clean_rerank_frame40_test.pt"
FIXED_POSE_BANK_SHA = "c209fbac4184f3716a32de12a69aecffb2c2e70d614f14657ca085e1d7fad012"
LEARNED_POSE_BANK = ROOT / "outputs/revision/clean_strict_cac_frame40_test.pt"
LEARNED_POSE_BANK_SHA = "3e2125b73bfa7591ec0cbab771a617f40bea0de68ec974d3e746676523d56c3d"

# Sign-Base bank and its recorded motion numbers (reproduced in stage 1 with difference 0)
SIGNBASE_BANK = ROOT / "outputs/signbase_slrtp_2026-09-11/eval_20260912/preds/signbase_test_sampleseed11.pt"
SIGNBASE_K = {"hand_speed_ratio": 1.2120929406357495, "hand_jerk_ratio": 4.738737310016611,
              "hand_posestd_ratio": 0.5981033947096596}
SIGNBASE_VAR = {"pose_manual": 0.5588822924094309, "motion_manual": 1.1283555913521435}

# DARSLP paper anchors (paper Table 3 dev "Two-Phase (Ours)", GT (dev) row)
DARSLP_DEV_ANCHOR_B4 = 11.40
DARSLP_DEV_TOL = 0.5
PAPER_GT_DEV = [30.92, 21.34, 16.01, 12.74, 34.71, 30.19]  # B1 B2 B3 B4 chrF ROUGE


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def insert_line(line: str, section: str) -> None:
    """Append `line` at the end of the '## {section}' block of RESULTS.md (before the next '## ' header), or at the
    end of the file if that block is the last one or absent. Added 2026-09-28 for stage 3: once the "Stage 3" sections
    sit below "Stage 2 (auto)", the stage-2 chain's lines must still land in their own table. A file lock serialises
    the two chains' read-modify-write, and the file is replaced atomically."""
    import fcntl
    import tempfile
    lock = open(str(RESULTS_MD) + ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        text = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else ""
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        head = next((i for i, l in enumerate(lines) if l.strip() == f"## {section}"), None)
        nxt = None if head is None else next((i for i in range(head + 1, len(lines)) if lines[i].startswith("## ")), None)
        if nxt is None:
            lines.append(line)
        else:
            j = nxt
            while j - 1 > head and lines[j - 1].strip() == "":
                j -= 1
            lines.insert(j, line)
        fd, tmp = tempfile.mkstemp(dir=str(RESULTS_MD.parent), prefix=".RESULTS.md.")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, RESULTS_MD)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def record(step: str, criterion: str, number: str, passed, artifact, extra: dict | None = None) -> None:
    """Append one machine-written summary line to RESULTS.md and a json copy under stage2/gates/.

    passed: True -> PASS, False -> FAIL, a string -> printed as is (e.g. 'recorded', 'LAUNCHED').
    """
    status = "PASS" if passed is True else "FAIL" if passed is False else str(passed)
    artifact = str(artifact).replace(str(RB) + "/", "")
    line = f"| {step} | {criterion} | {number} | {status} | {artifact} | {now()} |"
    insert_line(line, os.environ.get("S_SECTION", "Stage 2 (auto)"))
    GATES.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in step)[:80]
    (GATES / f"{slug}.json").write_text(json.dumps(
        {"step": step, "criterion": criterion, "number": number, "status": status, "artifact": artifact,
         "time": now(), **(extra or {})}, indent=2))
    print("[record]", line, flush=True)


def run(cmd: list[str], cwd=None, log=None) -> int:
    """Run a subprocess, tee to a log file, return the exit code."""
    print("[run]", " ".join(map(str, cmd)), f"(cwd {cwd})", flush=True)
    if log:
        with open(log, "a") as lf:
            lf.write(f"\n### {now()} {' '.join(map(str, cmd))}\n")
            lf.flush()
            return subprocess.call([str(c) for c in cmd], cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    return subprocess.call([str(c) for c in cmd], cwd=cwd)


def load_json(p):
    return json.loads(Path(p).read_text())


def die(msg: str, code: int = 1):
    print("[fail]", msg, file=sys.stderr, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    # chain-level line for a step that failed before its own program could record:
    # python s2_common.py record STEP CRITERION NUMBER {PASS|FAIL|other} ARTIFACT
    _, _cmd, _step, _crit, _num, _status, _art = sys.argv
    record(_step, _crit, _num, True if _status == "PASS" else False if _status == "FAIL" else _status, _art)
