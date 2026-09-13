"""Verify released bytes and headline evidence without private datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IGNORED_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def verify_manifest() -> int:
    manifest = ROOT / "MANIFEST.sha256"
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in expected:
            raise AssertionError(f"duplicate manifest path: {relative}")
        expected[relative] = digest
    actual = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest
        and not (set(path.relative_to(ROOT).parts) & IGNORED_DIRS)
    }
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(path for path in set(expected) & set(actual)
                         if expected[path] != actual[path])
        raise AssertionError(
            f"manifest mismatch: missing={missing}, extra={extra}, changed={changed}"
        )
    return len(actual)


def verify_evidence() -> None:
    hybrid = load_json("evidence/hybrid/materialization_manifest.json")
    summary = hybrid["summary"]
    assert summary["n_clips"] == 641
    assert summary["recorded_joins"] == 948
    assert summary["merged_bridges"] == 695
    assert summary["whole_replay_clips_unchanged"] == 192
    assert abs(summary["generated_mass"] - 10309.5) < 1e-6
    assert abs(summary["archive_mass"] - 36944.5) < 1e-6

    score_record = load_json("evidence/hybrid/score_record.json")
    score = score_record["metrics"]
    assert score_record["motion"]["clips"] == 641
    assert abs(score["bleu"]["bleu4"] - 15.339975745126376) < 1e-12
    assert abs(score["wer"] - 84.68537741894143) < 1e-12
    assert abs(score["dtw_mje"] - 0.04592249542474747) < 1e-14

    takedown = load_json("evidence/takedown/ephemeral_all_arms_v2_comparison.json")
    assert takedown["arms_total"] == 8
    assert takedown["arms_passed"] == 8
    assert len(takedown["rows"]) == 8
    assert all(row["passed"] for row in takedown["rows"])

    signbase = load_json("evidence/signbase/audit.json")
    verdict = signbase["verdict"]
    assert verdict["stored_bank_and_scoring_checks_pass"] is True
    assert verdict["ground_truth_duration_conditioned"] is False


def main() -> None:
    count = verify_manifest()
    verify_evidence()
    print(f"verified {count} files")
    print("PASS: release integrity and declared evidence checks")


if __name__ == "__main__":
    main()
