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


def printed(value: float, shown: float, decimals: int) -> bool:
    """True when value rounds to the number printed with the given decimals."""
    return abs(value - shown) <= 0.5 * 10 ** -decimals + 1e-12


def margin(record: dict, *path: str, flip: bool) -> dict:
    """Bootstrap mean and interval of one comparison, as BLEU-4 and WER margins.

    flip=True turns 'their system minus our route' into 'our route over theirs',
    the orientation the manuscript prints.
    """
    node = record
    for key in path:
        node = node[key]
    delta = node["bootstrap_delta"]
    out = {}
    for metric, sign in (("bleu4", -1 if flip else 1), ("corpus_wer", 1 if flip else -1),
                         ("bleu1", -1 if flip else 1)):
        lo, hi = delta[metric]["ci95"]
        out[metric] = (sign * delta[metric]["mean"],) + tuple(sorted((sign * lo, sign * hi)))
    return out


def check_margin(got: dict, bleu4: tuple, wer: tuple) -> None:
    for metric, shown in (("bleu4", bleu4), ("corpus_wer", wer)):
        if shown is None:
            continue
        for value, number in zip(got[metric], shown):
            assert printed(value, number, 2), (metric, got[metric], shown)


def verify_recent_methods_evidence() -> None:
    """Table IV grouped rows, the Section VI-F margins, Table VIII's Sign-IDD
    retrain row, the Comment 3.5 margins of the response letter, and the
    Section VI-G CSL-Daily governance counts. Every value below is one the
    manuscript or the letter prints, at its printed rounding."""
    base = "evidence/recent_methods/"
    rows = {  # Table IV: BLEU-1, BLEU-4, WER, DTW-MJE, duration
        "scores/ustcmoe_mt5_natlen_test_fps25.json": (36.15, 12.73, 89.02, 0.04492, 1.323),
        "scores/darslp_released_periodfree_test_fps25.json": (32.53, 10.63, 93.32, 0.03912, 1.167),
        "scores/signidd_gate4best_test_seed11_fps25.json": (25.56, 6.54, 96.77, 0.03844, 1.092),
    }
    signbase = load_json("evidence/signbase/results/signbase_seed11_cpu_rescore.json")
    for relative, shown in rows.items():
        record = load_json(base + relative)
        got = (record["bleu"]["bleu1"], record["bleu"]["bleu4"], record["wer"],
               record["dtw_mje"], record["avg_duration"])
        for value, number, decimals in zip(got, shown, (2, 2, 2, 5, 3)):
            assert printed(value, number, decimals), (relative, value, number)
        summary = load_json(base + "scores/score_" + relative.split("/")[1].replace(".json", "_summary.json"))
        assert summary["n"] == 641 and summary["invariants"]["coverage"] == "641/641"
        assert summary["identity"]["difference"] == {"bleu4": 0.0, "wer": 0.0, "dtw_mje": 0.0}
        assert abs(summary["bleu4"] - record["bleu"]["bleu4"]) < 1e-9
    sb = (signbase["bleu"]["bleu1"], signbase["bleu"]["bleu4"], signbase["wer"],
          signbase["dtw_mje"], signbase["avg_duration"])
    for value, number, decimals in zip(sb, (28.02, 8.11, 91.49, 0.03831, 1.092), (2, 2, 2, 5, 3)):
        assert printed(value, number, decimals), ("Sign-Base", value, number)

    ratios = {  # Table IV: speed, jerk, variation
        "motion/ustcmoe_mt5_natlen_test_K_ratios.json": (1.087, 2.324, 1.090),
        "motion/darslp_released_periodfree_test_K_ratios.json": (0.760, 0.904, 0.680),
        "motion/signidd_gate4best_test_seed11_K_ratios.json": (1.267, 4.746, 0.621),
        "motion/repro_signbase_K_ratios_before_ustcmoe_mt5_natlen_test.json": (1.212, 4.739, 0.598),
        "motion/repro_signbase_K_ratios_before_darslp_released_periodfree_test.json": (1.212, 4.739, 0.598),
        "motion/repro_signbase_K_ratios_before_signidd_gate4best_test_seed11.json": (1.212, 4.739, 0.598),
    }
    for relative, shown in ratios.items():
        record = load_json(base + relative)
        assert record["n_clips"] == 641
        got = (record["hand_speed_ratio"], record["hand_jerk_ratio"], record["hand_posestd_ratio"])
        for value, number in zip(got, shown):
            assert printed(value, number, 3), (relative, value, number)

    boot = lambda name: load_json(base + "bootstrap/" + name)  # noqa: E731
    # Section VI-F and the letter: fixed-route margins over each system, all 641 requests
    natlen_fixed = boot("ustcmoe_natlen_vs_fixed_paired_bootstrap_test.json")
    assert natlen_fixed["meta"]["n"] == 641 and natlen_fixed["meta"]["n_boot"] == 10000
    assert natlen_fixed["meta"]["seed"] == 30373
    got = margin(natlen_fixed, "comparisons", "ustcmoe_natlen_vs_fixed", flip=True)
    check_margin(got, (2.35, 0.97, 3.78), (3.42, 1.02, 5.85))
    for value, number in zip((-got["bleu1"][0], -got["bleu1"][2], -got["bleu1"][1]), (2.80, 0.67, 4.87)):
        assert printed(value, number, 2), ("BLEU-1 lead", value, number)
    check_margin(margin(boot("darslp_released_periodfree_test_vs_fixed_paired_bootstrap_test.json"),
                        "comparisons", "darslp_released_periodfree_test_vs_fixed", flip=True),
                 (4.48, 3.07, 5.92), (7.73, 5.47, 10.05))
    check_margin(margin(boot("signidd_vs_fixed_paired_bootstrap_test.json"),
                        "comparisons", "signidd_vs_fixed", flip=True),
                 (8.54, 7.15, 9.99), (11.19, 8.92, 13.48))
    # text-predicted-length USTC-MoE ties the fixed route on WER: -0.06 [-2.24, 2.10]
    policy = boot("ustcmoe_vs_fixed_paired_bootstrap_test.json")
    wer = policy["comparisons"]["ustcmoe_vs_fixed"]["bootstrap_delta"]["corpus_wer"]
    assert printed(wer["mean"], -0.06, 2)
    assert printed(wer["ci95"][0], -2.24, 2) and printed(wer["ci95"][1], 2.10, 2)
    # learned-route margins (letter)
    check_margin(margin(boot("ustcmoe_vs_learned_paired_bootstrap_test.json"),
                        "all 641", "comparisons", "natlen_vs_learned", flip=True),
                 (4.02, 2.58, 5.48), (5.38, 3.03, 7.79))
    check_margin(margin(boot("darslp_periodfree_checks_paired_bootstrap_test.json"),
                        "DARSLP final bank (released, period-free, own stop)|learned|all 641",
                        "comparisons", "base_vs_learned", flip=True),
                 (6.15, 4.70, 7.62), (9.69, 7.43, 12.06))
    check_margin(margin(boot("baselines_lenmatch_paired_bootstrap_test.json"),
                        "Sign-IDD table bank (pinned text-only lengths)|learned|all 641",
                        "comparisons", "base_vs_learned", flip=True),
                 (10.21, 8.76, 11.69), (13.15, 10.88, 15.47))
    # all-generated route margins (Section VI-F and the letter)
    allgen = boot("allgen_vs_baselines_paired_bootstrap_test.json")
    check_margin(margin(allgen, "comparisons", "ustcmoe_vs_allgen", flip=True),
                 (-0.59, -1.73, 0.49), (2.56, 0.56, 4.61))
    check_margin(margin(allgen, "comparisons", "darslp_vs_allgen", flip=True),
                 (1.54, 0.48, 2.58), (6.87, 4.99, 8.80))
    check_margin(margin(allgen, "comparisons", "signidd_vs_allgen", flip=True),
                 (5.60, 4.72, 6.51), (10.33, 8.52, 12.16))
    # fixed-route margins at the fixed route's own output lengths (letter)
    check_margin(margin(boot("ustcmoe_lenmatch_paired_bootstrap_test.json"),
                        "fixed|all 641", "comparisons", "ustcmoe_vs_fixed", flip=True),
                 (5.41, 4.08, 6.81), (4.11, 2.08, 6.23))
    check_margin(margin(boot("darslp_periodfree_checks_paired_bootstrap_test.json"),
                        "DARSLP (period-free) resampled to fixed-route lengths|fixed|all 641",
                        "comparisons", "base_vs_fixed", flip=True),
                 (6.47, 5.19, 7.80), (6.32, 4.28, 8.36))
    check_margin(margin(boot("signidd_lenmatch_paired_bootstrap_test.json"),
                        "Sign-IDD at fixed-route lengths|fixed|all 641", "comparisons", "base_vs_fixed",
                        flip=True),
                 (9.13, 7.80, 10.55), (11.17, 9.10, 13.26))
    # retrieval against retrieval (Section VI-F and the letter)
    retrieval = boot("retrieval_vs_ustcmoe_paired_bootstrap_test.json")
    local = margin(retrieval, "comparisons", "ustcmoe_vs_local", flip=False)
    check_margin(local, (3.55, 2.40, 4.68), None)
    assert local["corpus_wer"][1] < 0 < local["corpus_wer"][2]  # WER tie
    check_margin(margin(retrieval, "comparisons", "ustcmoe_vs_unrestricted", flip=True),
                 (11.85, 10.51, 13.23), (9.82, 7.06, 12.58))
    check_margin(margin(boot("ustcmoe_vs_local_paired_bootstrap_test.json"),
                        "all 641", "comparisons", "d4_vs_local", flip=False),
                 (2.48, 1.31, 3.66), (5.11, 3.24, 7.09))

    # letter: text-predicted-length run, its replacement, the with-period DARSLP run
    policy_score = load_json(base + "scores/ustcmoe_mt5_policylen_test_fps25.json")
    natlen_score = load_json(base + "scores/ustcmoe_mt5_natlen_test_fps25.json")
    assert printed(policy_score["bleu"]["bleu4"], 11.67, 2) and printed(policy_score["wer"], 85.53, 2)
    assert printed(natlen_score["bleu"]["bleu4"] - policy_score["bleu"]["bleu4"], 1.06, 2)
    assert printed(natlen_score["wer"] - policy_score["wer"], 3.49, 2)
    assert printed(load_json(base + "scores/darslp_withperiod_test_fps25.json")["bleu"]["bleu4"], 9.03, 2)
    darslp = load_json(base + "scores/darslp_released_periodfree_test_fps25.json")
    assert printed(darslp["dtw_mje"], 0.0391, 4)
    choice = base + "decisions/darslp_row_choice.json"
    digest = (ROOT / (choice + ".sha256")).read_text(encoding="utf-8").split()[0]
    assert sha256(ROOT / choice) == digest

    # Table VIII and Section VI-F: Sign-IDD retrain under its authors' protocol
    r2 = load_json(base + "signidd_reproduction/R2_check.json")
    assert (r2["test_bleu1"], r2["test_bleu4"], r2["test_wer"]) == (20.92, 7.15, 79.08)
    assert printed(r2["published"]["bleu4"] - r2["test_bleu4"], 1.93, 2)
    r0 = load_json(base + "signidd_reproduction/R0_check.json")
    assert r0["n_items"] == r0["n_identical_hypotheses"] == 642
    # letter: the other back-translator, reference poses converted (9.28) against its own files (11.88)
    cross = load_json(base + "cross_evaluator/gate0__fix1.json")
    assert printed(cross["gate0_641"]["bleu4"], 9.28, 2) and printed(cross["R0_641"]["bleu4"], 11.88, 2)
    assert cross["gate0_641"]["n"] == cross["R0_641"]["n"] == 641 and cross["pass"] is False

    # Section VI-G: CSL-Daily governance, aggregate only
    gov = load_json("evidence/csl_daily_dev/governance_public_v1.json")
    fixed = gov["fixed_route"]
    assert (fixed["replay_frames"], fixed["emitted_frames"]) == (35284, 88240)
    assert printed(100 * fixed["replay_frame_fraction"], 39.986, 3) and fixed["within_budget"] is True
    hybrid = gov["hybrid"]
    assert hybrid["recorded_joins"] == hybrid["bridged_joins"] == hybrid["row_recorded_join_cores_sum"] == 2151
    assert printed(100 * hybrid["generated_fraction"], 25.679, 3)
    assert gov["n_requests"] == gov["ledger_rows"] == 1077
    assert gov["mass_conservation"]["pass"] is True
    assert gov["mass_conservation"]["row_frames_sum"] == hybrid["total_frames"] == 88240
    assert gov["test_inputs_opened"] == []


def main() -> None:
    count = verify_manifest()
    verify_evidence()
    verify_revision_evidence()
    verify_csl_dev_evidence()
    verify_recent_methods_evidence()
    print(f"verified {count} files")
    print("PASS: release integrity and declared evidence checks")


if __name__ == "__main__":
    main()
