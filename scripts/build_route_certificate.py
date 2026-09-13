"""Build a reuse-normalized route certificate for the Budget-SAGE paper.

The certificate is intentionally small and deterministic.  It reads the same
SLRTP evaluator JSON files used by the paper, attaches disclosed whole-clip
reuse counts, and reports lexical gain per unit of whole-clip reuse relative to
the non-retrieval fallback.  This makes the TF-IDF-vs-CAC comparison auditable:
TF-IDF has the higher raw BLEU-4, while CAC is the stronger fixed-budget route.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "external/SLRTP-Sign-Production-Evaluation/results"
OUT_DIR = ROOT / "outputs/route_certificates"
OUT_JSON = OUT_DIR / "budget_sage_tmm_route_certificate.json"
OUT_MD = OUT_DIR / "budget_sage_tmm_route_certificate.md"
LOCAL_REUSE_JSON = OUT_DIR / "local_reuse_certificate.json"
N_TEST = 641

LOCAL_REUSE_METHOD = {
    "Native UPC-guided fallback": "Native fallback",
    "Budget-SAGE budget40 reference": "Budget40 reference",
    "Budget-SAGE budget40 + CAC": "Budget40 CAC",
    "Budget-SAGE budget40 + BT-input-free CAC": "BT-input-free CAC",
    "Budget-SAGE budget40 + BT-feature-free CAC": "BT-input-free CAC",
    "Budget-SAGE budget40 + Pareto-knee WER-priced CAC": "Pareto-knee CAC",
    "Budget-SAGE budget40 + budgeted BT-MBR": "Replay-budgeted BT-MBR",
    "TF-IDF 1-NN retrieval": "TF-IDF 1-NN",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def eval_metrics(filename: str) -> dict[str, float]:
    data = load_json(EVAL_DIR / filename)
    bleu = data.get("bleu", data.get("BLEU", {}))
    return {
        "bleu1": float(bleu.get("bleu1", 0.0)),
        "bleu4": float(bleu.get("bleu4", 0.0)),
        "wer": float(data.get("wer", data.get("WER", 0.0))),
        "dtw_mje": float(data.get("dtw_mje", data.get("DTW MJE", 0.0))),
    }


def cac_block_metrics(name: str) -> dict[str, float | None]:
    data = load_json(ROOT / "outputs/learned_confidence/cac_block_ablation.json")
    for row in data:
        if row.get("name") == name:
            return {
                "bleu1": float(row["test_bleu1"]),
                "bleu4": float(row["test_bleu4"]),
                "wer": float(row["test_wer"]),
                "dtw_mje": None,
            }
    raise KeyError(name)


def cli_lookup() -> dict[str, float]:
    lookup: dict[str, float] = {}
    for path in [
        ROOT / "outputs/cli_audit_test.json",
        ROOT / "outputs/cli_audit_missing_rows.json",
        ROOT / "outputs/cli_audit_wpriced_test.json",
        ROOT / "outputs/cli_audit_wpriced_frontier_test.json",
        ROOT / "outputs/cli_audit_bt_mbr_b40_test.json",
    ]:
        if not path.exists():
            continue
        for row in load_json(path).get("rows", []):
            lookup[str(row.get("tag"))] = float(row["cli_mean"])
    return lookup


def round_or_none(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def local_reuse_lookup() -> dict[str, dict[str, Any]]:
    if not LOCAL_REUSE_JSON.exists():
        return {}
    payload = load_json(LOCAL_REUSE_JSON)
    return {str(row["method"]): row for row in payload.get("rows", [])}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cli = cli_lookup()
    local = local_reuse_lookup()

    source_rows = [
        {
            "method": "Native UPC-guided fallback",
            "role": "non-retrieval anchor",
            "eval_file": "pgrastpp178_mt5raw_clean_native_test.json",
            "cli_tag": None,
            "retrieved_clips": 0,
        },
        {
            "method": "Budget-SAGE budget40 reference",
            "role": "matched-rate local-confidence controller",
            "eval_file": "phase58_budget040_test.json",
            "cli_tag": "budget040_test",
            "retrieved_clips": 256,
        },
        {
            "method": "Budget-SAGE budget40 + CAC",
            "role": "matched-rate corpus-aware controller",
            "eval_file": "phase58_b40_CAC_a0_00_test.json",
            "cli_tag": "CAC_a0_00_test",
            "retrieved_clips": 256,
        },
        {
            "method": "Budget-SAGE budget40 + Pareto-knee WER-priced CAC",
            "role": "matched-rate Pareto-knee WER-priced controller",
            "eval_file": "phase58_b40_CAC_a0_25_test.json",
            "cli_tag": "CAC_w025_test",
            "retrieved_clips": 256,
        },
        {
            "method": "Budget-SAGE budget40 + budgeted BT-MBR",
            "role": "matched-rate SLRTP-stack route-bank selector",
            "eval_file": "bt_mbr_b40_route_selector_test.json",
            "cli_tag": "bt_mbr_b40_test",
            "retrieved_clips": 256,
        },
        {
            "method": "Budget-SAGE budget40 + BT-input-free CAC",
            "role": "matched-rate controller without BT-derived features",
            "eval_file": "outputs/learned_confidence/cac_block_ablation.json",
            "metrics": cac_block_metrics(
                "evaluator-free CAC (drop w_i, cur_bleu4, cur_wer; 20)"
            ),
            "cli_tag": None,
            "retrieved_clips": 256,
        },
        {
            "method": "TF-IDF 1-NN retrieval",
            "role": "retrieval-heavy lexical upper reference",
            "eval_file": "text1nn_tfidf_test.json",
            "cli_tag": "tfidf1nn_test",
            "retrieved_clips": 632,
        },
    ]

    fallback = eval_metrics("pgrastpp178_mt5raw_clean_native_test.json")
    rows: list[dict[str, Any]] = []
    for src in source_rows:
        metrics = src.get("metrics") or eval_metrics(src["eval_file"])
        retrieved = int(src["retrieved_clips"])
        rate = retrieved / N_TEST
        bleu_gain = metrics["bleu4"] - fallback["bleu4"]
        wer_reduction = fallback["wer"] - metrics["wer"]
        bleu_per_rate = None if rate == 0 else bleu_gain / rate
        wer_per_rate = None if rate == 0 else wer_reduction / rate
        cli_tag = src["cli_tag"]
        local_name = LOCAL_REUSE_METHOD.get(src["method"])
        local_row = None if local_name is None else local.get(local_name)
        replay_frame_fraction = (
            None if local_row is None else float(local_row["whole_clip_replay_frame_fraction"])
        )
        bleu_per_frame = (
            None
            if not replay_frame_fraction
            else bleu_gain / replay_frame_fraction
        )
        wer_per_frame = (
            None
            if not replay_frame_fraction
            else wer_reduction / replay_frame_fraction
        )
        rows.append(
            {
                "method": src["method"],
                "role": src["role"],
                "eval_file": src["eval_file"],
                "retrieved_clips": retrieved,
                "total_clips": N_TEST,
                "retrieval_rate": rate,
                "retrieval_rate_pct": 100.0 * rate,
                "cli_mean": None if cli_tag is None else cli.get(cli_tag),
                "bleu1": metrics["bleu1"],
                "bleu4": metrics["bleu4"],
                "wer": metrics["wer"],
                "dtw_mje": metrics["dtw_mje"],
                "bleu4_gain_over_non_retrieval": bleu_gain,
                "wer_reduction_over_non_retrieval": wer_reduction,
                "bleu4_gain_per_retrieval_fraction": bleu_per_rate,
                "wer_reduction_per_retrieval_fraction": wer_per_rate,
                "replay_frame_fraction": replay_frame_fraction,
                "replay_frame_fraction_pct": (
                    None if replay_frame_fraction is None else 100.0 * replay_frame_fraction
                ),
                "bleu4_gain_per_replay_frame_fraction": bleu_per_frame,
                "wer_reduction_per_replay_frame_fraction": wer_per_frame,
            }
        )

    by_method = {row["method"]: row for row in rows}
    cac = by_method["Budget-SAGE budget40 + CAC"]
    wpriced = by_method["Budget-SAGE budget40 + Pareto-knee WER-priced CAC"]
    ef_cac = by_method["Budget-SAGE budget40 + BT-input-free CAC"]
    budgeted_mbr = by_method["Budget-SAGE budget40 + budgeted BT-MBR"]
    ref = by_method["Budget-SAGE budget40 reference"]
    tfidf = by_method["TF-IDF 1-NN retrieval"]
    post_cac_extra_rate = tfidf["retrieval_rate"] - cac["retrieval_rate"]
    post_cac_bleu = tfidf["bleu4"] - cac["bleu4"]
    post_cac_wer = cac["wer"] - tfidf["wer"]
    frame_eff = {
        "cac_bleu4_gain_per_replay_frame_fraction": cac[
            "bleu4_gain_per_replay_frame_fraction"
        ],
        "wpriced_bleu4_gain_per_replay_frame_fraction": wpriced[
            "bleu4_gain_per_replay_frame_fraction"
        ],
        "budgeted_mbr_bleu4_gain_per_replay_frame_fraction": budgeted_mbr[
            "bleu4_gain_per_replay_frame_fraction"
        ],
        "tfidf_bleu4_gain_per_replay_frame_fraction": tfidf[
            "bleu4_gain_per_replay_frame_fraction"
        ],
        "cac_wer_reduction_per_replay_frame_fraction": cac[
            "wer_reduction_per_replay_frame_fraction"
        ],
        "wpriced_wer_reduction_per_replay_frame_fraction": wpriced[
            "wer_reduction_per_replay_frame_fraction"
        ],
        "budgeted_mbr_wer_reduction_per_replay_frame_fraction": budgeted_mbr[
            "wer_reduction_per_replay_frame_fraction"
        ],
        "tfidf_wer_reduction_per_replay_frame_fraction": tfidf[
            "wer_reduction_per_replay_frame_fraction"
        ],
        "bleu4_efficiency_ratio_cac_over_tfidf_by_replay_frame": cac[
            "bleu4_gain_per_replay_frame_fraction"
        ]
        / tfidf["bleu4_gain_per_replay_frame_fraction"],
        "bleu4_efficiency_ratio_wpriced_over_tfidf_by_replay_frame": wpriced[
            "bleu4_gain_per_replay_frame_fraction"
        ]
        / tfidf["bleu4_gain_per_replay_frame_fraction"],
        "bleu4_efficiency_ratio_budgeted_mbr_over_tfidf_by_replay_frame": budgeted_mbr[
            "bleu4_gain_per_replay_frame_fraction"
        ]
        / tfidf["bleu4_gain_per_replay_frame_fraction"],
        "wer_efficiency_ratio_cac_over_tfidf_by_replay_frame": cac[
            "wer_reduction_per_replay_frame_fraction"
        ]
        / tfidf["wer_reduction_per_replay_frame_fraction"],
        "wer_efficiency_ratio_wpriced_over_tfidf_by_replay_frame": wpriced[
            "wer_reduction_per_replay_frame_fraction"
        ]
        / tfidf["wer_reduction_per_replay_frame_fraction"],
        "wer_efficiency_ratio_budgeted_mbr_over_tfidf_by_replay_frame": budgeted_mbr[
            "wer_reduction_per_replay_frame_fraction"
        ]
        / tfidf["wer_reduction_per_replay_frame_fraction"],
    }
    summary = {
        "test_clips": N_TEST,
        "fixed_rate_allocation": {
            "retrieved_clips": cac["retrieved_clips"],
            "retrieval_rate_pct": cac["retrieval_rate_pct"],
            "cac_minus_reference_bleu4": cac["bleu4"] - ref["bleu4"],
            "cac_minus_reference_wer": cac["wer"] - ref["wer"],
        },
        "reuse_normalized_vs_tfidf": {
            "cac_bleu4_gain_per_retrieval_fraction": cac[
                "bleu4_gain_per_retrieval_fraction"
            ],
            "evaluator_free_cac_bleu4_gain_per_retrieval_fraction": ef_cac[
                "bleu4_gain_per_retrieval_fraction"
            ],
            "bt_feature_free_cac_bleu4_gain_per_retrieval_fraction": ef_cac[
                "bleu4_gain_per_retrieval_fraction"
            ],
            "tfidf_bleu4_gain_per_retrieval_fraction": tfidf[
                "bleu4_gain_per_retrieval_fraction"
            ],
            "bleu4_efficiency_ratio_cac_over_tfidf": cac[
                "bleu4_gain_per_retrieval_fraction"
            ]
            / tfidf["bleu4_gain_per_retrieval_fraction"],
            "bleu4_efficiency_ratio_evaluator_free_over_tfidf": ef_cac[
                "bleu4_gain_per_retrieval_fraction"
            ]
            / tfidf["bleu4_gain_per_retrieval_fraction"],
            "bleu4_efficiency_ratio_bt_feature_free_over_tfidf": ef_cac[
                "bleu4_gain_per_retrieval_fraction"
            ]
            / tfidf["bleu4_gain_per_retrieval_fraction"],
            "cac_wer_reduction_per_retrieval_fraction": cac[
                "wer_reduction_per_retrieval_fraction"
            ],
            "evaluator_free_cac_wer_reduction_per_retrieval_fraction": ef_cac[
                "wer_reduction_per_retrieval_fraction"
            ],
            "bt_feature_free_cac_wer_reduction_per_retrieval_fraction": ef_cac[
                "wer_reduction_per_retrieval_fraction"
            ],
            "tfidf_wer_reduction_per_retrieval_fraction": tfidf[
                "wer_reduction_per_retrieval_fraction"
            ],
            "wpriced_wer_reduction_per_retrieval_fraction": wpriced[
                "wer_reduction_per_retrieval_fraction"
            ],
            "budgeted_mbr_bleu4_gain_per_retrieval_fraction": budgeted_mbr[
                "bleu4_gain_per_retrieval_fraction"
            ],
            "budgeted_mbr_wer_reduction_per_retrieval_fraction": budgeted_mbr[
                "wer_reduction_per_retrieval_fraction"
            ],
            "wer_efficiency_ratio_cac_over_tfidf": cac[
                "wer_reduction_per_retrieval_fraction"
            ]
            / tfidf["wer_reduction_per_retrieval_fraction"],
            "wer_efficiency_ratio_wpriced_over_tfidf": wpriced[
                "wer_reduction_per_retrieval_fraction"
            ]
            / tfidf["wer_reduction_per_retrieval_fraction"],
            "wer_efficiency_ratio_evaluator_free_over_tfidf": ef_cac[
                "wer_reduction_per_retrieval_fraction"
            ]
            / tfidf["wer_reduction_per_retrieval_fraction"],
            "wer_efficiency_ratio_bt_feature_free_over_tfidf": ef_cac[
                "wer_reduction_per_retrieval_fraction"
            ]
            / tfidf["wer_reduction_per_retrieval_fraction"],
            "bleu4_efficiency_ratio_budgeted_mbr_over_tfidf": budgeted_mbr[
                "bleu4_gain_per_retrieval_fraction"
            ]
            / tfidf["bleu4_gain_per_retrieval_fraction"],
            "wer_efficiency_ratio_budgeted_mbr_over_tfidf": budgeted_mbr[
                "wer_reduction_per_retrieval_fraction"
            ]
            / tfidf["wer_reduction_per_retrieval_fraction"],
        },
        "frame_normalized_vs_tfidf": frame_eff,
        "wer_priced_vs_tfidf": {
            "retrieved_clip_saving": tfidf["retrieved_clips"] - wpriced["retrieved_clips"],
            "retrieval_rate_saving_pct": tfidf["retrieval_rate_pct"] - wpriced["retrieval_rate_pct"],
            "bleu4_gap": wpriced["bleu4"] - tfidf["bleu4"],
            "wer_delta": wpriced["wer"] - tfidf["wer"],
            "dtw_mje_delta": wpriced["dtw_mje"] - tfidf["dtw_mje"],
            "cli_delta": wpriced["cli_mean"] - tfidf["cli_mean"],
        },
        "post_cac_tfidf_marginal": {
            "additional_retrieved_clips": tfidf["retrieved_clips"] - cac["retrieved_clips"],
            "additional_retrieval_rate_pct": 100.0 * post_cac_extra_rate,
            "bleu4_gain": post_cac_bleu,
            "wer_reduction": post_cac_wer,
            "dtw_mje_cost": tfidf["dtw_mje"] - cac["dtw_mje"],
            "bleu4_gain_per_additional_retrieval_fraction": post_cac_bleu
            / post_cac_extra_rate,
            "wer_reduction_per_additional_retrieval_fraction": post_cac_wer
            / post_cac_extra_rate,
        },
    }

    certificate = {
        "name": "Budget-SAGE TMM route certificate",
        "definition": (
            "Reuse-normalized lexical gain is computed relative to the "
            "non-retrieval fallback as metric gain divided by disclosed "
            "whole-clip retrieval fraction."
        ),
        "summary": summary,
        "rows": rows,
        "sources": {
            "evaluator_dir": str(EVAL_DIR),
            "local_reuse_source": str(LOCAL_REUSE_JSON.relative_to(ROOT)),
            "cli_sources": [
                "outputs/cli_audit_test.json",
                "outputs/cli_audit_missing_rows.json",
                "outputs/cli_audit_wpriced_test.json",
                "outputs/cli_audit_wpriced_frontier_test.json",
                "outputs/cli_audit_bt_mbr_b40_test.json",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(certificate, indent=2) + "\n")

    md_lines = [
        "# Budget-SAGE TMM Route Certificate",
        "",
        "Reuse-normalized lexical gain is computed relative to the "
        "non-retrieval fallback and divided by disclosed whole-clip reuse.",
        "",
        "| Method | Retrieved | Rate (%) | Frame replay (%) | CLI | BLEU-4 | WER | BLEU-4 gain / reuse | WER reduction / reuse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {method} | {retrieved_clips}/{total_clips} | {rate:.2f} | {frame} | {cli} | "
            "{bleu4:.2f} | {wer:.2f} | {bg} | {wr} |".format(
                method=row["method"],
                retrieved_clips=row["retrieved_clips"],
                total_clips=row["total_clips"],
                rate=row["retrieval_rate_pct"],
                frame=(
                    "--"
                    if row["replay_frame_fraction_pct"] is None
                    else f"{row['replay_frame_fraction_pct']:.2f}"
                ),
                cli="--" if row["cli_mean"] is None else f"{row['cli_mean']:.3f}",
                bleu4=row["bleu4"],
                wer=row["wer"],
                bg=(
                    "--"
                    if row["bleu4_gain_per_retrieval_fraction"] is None
                    else f"{row['bleu4_gain_per_retrieval_fraction']:.2f}"
                ),
                wr=(
                    "--"
                    if row["wer_reduction_per_retrieval_fraction"] is None
                    else f"{row['wer_reduction_per_retrieval_fraction']:.2f}"
                ),
            )
        )
    eff = summary["reuse_normalized_vs_tfidf"]
    frame = summary["frame_normalized_vs_tfidf"]
    fixed = summary["fixed_rate_allocation"]
    md_lines.extend(
        [
            "",
            "Fixed-rate CAC allocation retrieves "
            f"{fixed['retrieved_clips']}/{N_TEST} clips "
            f"({fixed['retrieval_rate_pct']:.2f}%) and improves the "
            "budget40 reference by "
            f"{fixed['cac_minus_reference_bleu4']:.2f} BLEU-4 and "
            f"{fixed['cac_minus_reference_wer']:.2f} WER.",
            "",
        "Compared with TF-IDF 1-NN, CAC gives "
        f"{eff['bleu4_efficiency_ratio_cac_over_tfidf']:.2f}x the "
        "BLEU-4 gain per whole-clip reuse fraction and "
        f"{eff['wer_efficiency_ratio_cac_over_tfidf']:.2f}x the "
        "WER reduction per whole-clip reuse fraction.",
        "",
        "The WER-priced CAC operating point uses the same 256 whole-clip "
        "copies and gives "
        f"{eff['wer_efficiency_ratio_wpriced_over_tfidf']:.2f}x the "
        "TF-IDF WER reduction per reuse fraction. It scores "
        f"{wpriced['wer']:.2f} WER, below TF-IDF 1-NN at "
        f"{tfidf['wer']:.2f}, while saving "
        f"{summary['wer_priced_vs_tfidf']['retrieved_clip_saving']} "
        "whole-clip copies.",
        "",
        "Under duration-weighted copied-frame accounting, the same WER-priced "
        "CAC point gives "
        f"{frame['bleu4_efficiency_ratio_wpriced_over_tfidf_by_replay_frame']:.2f}x "
        "the TF-IDF BLEU-4 gain and "
        f"{frame['wer_efficiency_ratio_wpriced_over_tfidf_by_replay_frame']:.2f}x "
        "the TF-IDF WER reduction per replay-frame fraction.",
        "",
        "The BT-input-free CAC row still gives "
        f"{eff['bleu4_efficiency_ratio_bt_feature_free_over_tfidf']:.2f}x "
        "the TF-IDF BLEU-4 gain per reuse fraction. After the CAC point, "
        "moving to TF-IDF 1-NN spends "
        f"{summary['post_cac_tfidf_marginal']['additional_retrieved_clips']} "
        "additional whole-clip copies for only "
        f"{summary['post_cac_tfidf_marginal']['bleu4_gain']:.2f} BLEU-4, "
        f"{summary['post_cac_tfidf_marginal']['wer_reduction']:.2f} WER "
        "reduction, and a "
        f"{summary['post_cac_tfidf_marginal']['dtw_mje_cost']:.5f} "
        "DTW-MJE cost.",
    ]
    )
    OUT_MD.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
