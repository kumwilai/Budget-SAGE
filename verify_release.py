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


def verify_revision_evidence() -> None:
    """The 26 September 2026 revision: motion ratios, learned hybrid,
    crossfade study, and paired intervals."""
    motion = load_json("evidence/motion_ratios/hybrid_fixed_test_kinematics.json")
    assert motion["n_clips"] == 641
    assert abs(motion["hand_speed_ratio"] - 1.005530949173178) < 1e-9
    assert abs(motion["hand_jerk_ratio"] - 1.2262462545804587) < 1e-9
    assert abs(motion["hand_posestd_ratio"] - 0.8456225659744873) < 1e-9

    evaluator_hybrid = load_json(
        "evidence/evaluator_results/budget_sage_signjepa_hybrid_test.json"
    )
    assert abs(evaluator_hybrid["bleu"]["bleu4"] - 15.339975745126376) < 1e-9
    assert "avg_duration" in evaluator_hybrid

    learned_manifest = load_json(
        "evidence/hybrid_learned/materialization_manifest.json"
    )
    learned_summary = learned_manifest["summary"]
    assert learned_summary["n_clips"] == 641
    assert learned_summary["recorded_joins"] == 951
    assert learned_summary["merged_bridges"] == 706

    learned_bootstrap = load_json("evidence/hybrid_learned/paired_bootstrap_2000.json")
    assert learned_bootstrap["N"] == 641
    assert learned_bootstrap["n_boot"] == 2000

    rcx_summary = load_json("evidence/jerk_study/rcx_summary.json")
    assert rcx_summary["all_acceptance_passed"] is True
    rcx_h2 = {
        (row["route"]): row
        for row in rcx_summary["rows"]
        if row["variant"] == "rcx_h2" and row["split"] == "test641"
    }
    assert abs(rcx_h2["local"]["hand_jerk"] - 1.060807580343339) < 1e-9
    assert abs(rcx_h2["local"]["hand_speed"] - 0.9960648612517052) < 1e-9
    assert abs(rcx_h2["cac_route"]["hand_jerk"] - 1.048360967959011) < 1e-9
    assert abs(rcx_h2["cac_route"]["hand_speed"] - 0.9913116346240357) < 1e-9

    clean_route = load_json(
        "evidence/paired_intervals/clean_route_test_paired_bootstrap.json"
    )
    learned_vs_rerank = clean_route["comparisons"]["learned_vs_rerank"]["bootstrap_delta"]
    assert abs(learned_vs_rerank["bleu4"]["mean"] - 1.6707574071678715) < 1e-9
    assert abs(learned_vs_rerank["corpus_wer"]["mean"] - (-1.966651422947444)) < 1e-9

    exact_excl = load_json(
        "evidence/paired_intervals/exact_excl_sensitivity_test_paired_bootstrap.json"
    )
    learned_vs_nonlearned = exact_excl["comparisons"]["learned_minus_nonlearned"]["bootstrap_delta"]
    assert abs(learned_vs_nonlearned["bleu4"]["mean"] - 1.6169512347747566) < 1e-9
    assert abs(learned_vs_nonlearned["corpus_wer"]["mean"] - (-1.8280776862367958)) < 1e-9


def verify_csl_dev_evidence() -> None:
    """Table IX: the CSL-Daily 1,077-request development study, as aggregates."""
    summary = load_json("evidence/csl_daily_dev/transfer_summary_public_v1.json")
    ensemble = {route: heads["ensemble_last_hyp"]
                for route, heads in summary["recognition"].items()}
    expected = {"local": 91.374, "whole": 91.839, "fixed_h40": 91.558,
                "hybrid": 90.053, "all_generated": 99.229}
    for route, wer in expected.items():
        assert abs(ensemble[route]["wer"] - wer) < 5e-4, route
        assert ensemble[route]["denominator"] == 1077
        assert ensemble[route]["num_ref"] == 8173
    empty = {route: ensemble[route]["empty_hypotheses"] for route in expected}
    assert empty == {"local": 0, "whole": 2, "fixed_h40": 0, "hybrid": 5, "all_generated": 525}
    hybrid_ci = summary["paired_comparisons"]["hybrid_minus_fixed_h40"]["heads"]["ensemble_last_hyp"]["ci95"]
    fixed_ci = summary["paired_comparisons"]["fixed_h40_minus_local"]["heads"]["ensemble_last_hyp"]["ci95"]
    assert abs(hybrid_ci[0] + 2.458) < 5e-4 and abs(hybrid_ci[1] + 0.601) < 5e-4
    assert abs(fixed_ci[0] + 0.502) < 5e-4 and abs(fixed_ci[1] - 0.872) < 5e-4
    assert summary["bootstrap"]["groups"] == 797 and summary["bootstrap"]["draws"] == 10000
    assert summary["preregistered_gate"]["pass"] is False
    assert summary["test_inputs_opened"] == []
    native = load_json("evidence/csl_daily_dev/calibration_analysis_v1.json")
    assert abs(native["native_calibration"]["ensemble_last_hyp"]["point"] - 28.386) < 5e-4


def main() -> None:
    count = verify_manifest()
    verify_evidence()
    verify_revision_evidence()
    verify_csl_dev_evidence()
    print(f"verified {count} files")
    print("PASS: release integrity and declared evidence checks")


if __name__ == "__main__":
    main()
