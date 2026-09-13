"""Issue and verify withdrawn-source certificates for the reported takedown arms.

Each arm is handled in two separate processes so that issuance cannot leak
into verification:

  1. ``clean_route_certificate.py issue`` reads the arm's issuer records
     (pose bank, mask, retrieval trace, fallback trace, ledger) plus the
     withdrawn-source set, and writes an authenticated release manifest.
  2. ``clean_route_certificate.py verify`` is run verbatim under a
     ``sys.addaudithook`` probe that records every file the process opens.
     The recorded list is the evidence that the verifier reads no issuer
     routing record.

Nothing here trains, samples, or touches a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# budget_sage is imported inside main() only for the "run" mode: the
# verify-probe process must not pre-import anything the verifier itself
# opens, or the recorded file list would understate it.

CERTIFICATE_CLI = ROOT / "scripts" / "clean_route_certificate.py"
STUDY = ROOT / "outputs" / "takedown_study"
DEFAULT_ARCHIVE = ROOT / "outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt"
DEFAULT_ARMS = (
    "gate_empty", "top10", "top50", "top250", "top1231",
    "Signer01", "Signer05", "rand250_s1",
)
MIN_AVAILABLE_MB = 11 * 1024


def available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def arm_inputs(arm: str) -> dict[str, Path]:
    run = STUDY / "runs" / arm
    return {
        "pose_bank": run / "route_test.pt",
        "mask": run / f"td_{arm}_learned_frame40_test.npz",
        "retrieval_trace": run / "retrieval_trace.json",
        "fallback_trace": run / "fallback_trace.json",
        "ledger": run / "ledger.json",
        "withdrawn_sources": run / "ban_set.json",
    }


def run_probe(argv: list[str], access_log: Path) -> dict:
    """Run the certificate CLI verbatim while recording every file opened."""
    opened: list[str] = []

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            opened.append(str(args[0]))

    sys.argv = [str(CERTIFICATE_CLI), *argv]
    sys.addaudithook(hook)
    status = 0
    try:
        runpy.run_path(str(CERTIFICATE_CLI), run_name="__main__")
    except SystemExit as exc:  # pragma: no cover - CLI does not exit non-zero
        status = int(exc.code or 0)
    finally:
        seen: list[str] = []
        for path in opened:
            if path not in seen:
                seen.append(path)
        interpreter = str(Path(sys.executable).resolve().parents[1])
        project, external = [], []
        for path in seen:
            resolved = str(Path(path).resolve()) if path and path[0] != "<" else path
            target = project if (resolved.startswith(str(ROOT))
                                 and not resolved.startswith(interpreter)) else external
            target.append(resolved)
        access_log.parent.mkdir(parents=True, exist_ok=True)
        access_log.write_text(json.dumps({
            "argv": argv,
            "project_files_opened": sorted(set(project)),
            "external_files_opened_count": len(set(external)),
            "interpreter_prefix": interpreter,
        }, indent=2, sort_keys=True) + "\n")
    return {"status": status}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS))
    run.add_argument("--out_root", default="outputs/takedown_certificates")
    run.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    run.add_argument("--blend_width", type=int, default=4)
    run.add_argument("--key_env", default="BUDGET_SAGE_MANIFEST_KEY")
    probe = sub.add_parser("verify-probe")
    probe.add_argument("--access_log", required=True)
    probe.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.mode == "verify-probe":
        rest = [a for a in args.rest if a != "--"]
        return run_probe(rest, Path(args.access_log))["status"]

    from budget_sage.audit.certificate import sha256_file

    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for arm in args.arms:
        inputs = arm_inputs(arm)
        missing = [str(p) for p in inputs.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{arm}: missing {missing}")
        out = out_root / arm
        out.mkdir(parents=True, exist_ok=True)
        manifest = out / "manifest.json"
        report = out / "verification.json"
        access_log = out / "verifier_file_access.json"
        free = available_mb()
        if free < MIN_AVAILABLE_MB:
            raise MemoryError(f"{arm}: only {free} MB available, need {MIN_AVAILABLE_MB}")
        issue_cmd = [
            sys.executable, str(CERTIFICATE_CLI), "issue",
            "--pose_bank", str(inputs["pose_bank"]),
            "--mask", str(inputs["mask"]),
            "--retrieval_trace", str(inputs["retrieval_trace"]),
            "--fallback_trace", str(inputs["fallback_trace"]),
            "--ledger", str(inputs["ledger"]),
            "--withdrawn_sources", str(inputs["withdrawn_sources"]),
            "--archive", args.archive,
            "--blend_width", str(args.blend_width),
            "--manifest_out", str(manifest),
            "--key_env", args.key_env,
        ]
        t0 = time.time()
        issued = subprocess.run(issue_cmd, cwd=ROOT, capture_output=True, text=True)
        if issued.returncode != 0:
            raise RuntimeError(f"{arm} issuance failed:\n{issued.stderr[-2000:]}")
        issue_seconds = time.time() - t0
        free = available_mb()
        if free < MIN_AVAILABLE_MB:
            raise MemoryError(f"{arm}: only {free} MB available, need {MIN_AVAILABLE_MB}")
        verify_cmd = [
            sys.executable, str(Path(__file__).resolve()), "verify-probe",
            "--access_log", str(access_log), "--",
            "verify",
            "--pose_bank", str(inputs["pose_bank"]),
            "--manifest", str(manifest),
            "--archive", args.archive,
            "--report_out", str(report),
            "--key_env", args.key_env,
            "--tamper_drills",
        ]
        t0 = time.time()
        verified = subprocess.run(verify_cmd, cwd=ROOT, capture_output=True, text=True)
        if verified.returncode != 0 or not report.exists():
            raise RuntimeError(f"{arm} verification failed:\n{verified.stderr[-4000:]}")
        verify_seconds = time.time() - t0
        manifest_body = json.loads(manifest.read_text())
        summary[arm] = {
            "issue_command": issue_cmd,
            "verify_command": verify_cmd,
            "issue_seconds": round(issue_seconds, 1),
            "verify_seconds": round(verify_seconds, 1),
            "release_schema_version": manifest_body["schema_version"],
            "issuance_inputs": manifest_body["issuance_inputs"],
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(str(manifest)),
            "report": str(report),
            "verifier_file_access": str(access_log),
            "report_body": json.loads(report.read_text()),
        }
        block = summary[arm]["report_body"]["withdrawn_source_set"]
        print(f"[{arm}] {summary[arm]['report_body']['verdict']} "
              f"entries={summary[arm]['report_body']['manifest_entries']} "
              f"withdrawn={block['withdrawn_source_count']} "
              f"exposed={block['emitted_positions_drawing_on_withdrawn_source']} "
              f"drills={len(summary[arm]['report_body']['tamper_drills'])}", flush=True)
    provenance = {
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "cwd": str(ROOT),
        "key_env": args.key_env,
        "seeds": "none; issuance and verification are deterministic",
        "archive": args.archive,
        "archive_sha256": sha256_file(args.archive),
        "blend_width": args.blend_width,
        "scripts": {
            name: sha256_file(str(ROOT / name)) for name in (
                "scripts/clean_route_certificate.py",
                "scripts/certify_takedown_arms.py",
                "budget_sage/audit/certificate.py",
            )
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arms": summary,
    }
    path = out_root / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
