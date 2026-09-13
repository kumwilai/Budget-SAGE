"""Audit the predeclared normalized-exact-caption exclusion sensitivity.

This audit runs before evaluator scoring. It checks input coverage, the frozen
plan, source exclusions, gloss-token accounting, exact frame-budget arithmetic,
and conserved ledgers. It deliberately does not estimate automatic quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PREFIX = "exact_excl_sensitivity"
PREREGISTRATION = ROOT / "outputs/revision/exact_caption_sensitivity_preregistration.json"
PREREGISTRATION_SHA256 = "e28922d1739e957857e8c91ea681bd8a95ab8437bed6500477878418b9d10e48"

from scripts.build_retrieval_gloss_plans import normalize_caption  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def metric_row(path: Path) -> dict[str, float]:
    row = load_json(path)
    return {
        "bleu1": float(row["bleu"]["bleu1"]),
        "bleu4": float(row["bleu"]["bleu4"]),
        "wer": float(row["wer"]),
        "dtw_mje": float(row["dtw_mje"]),
    }


def budget_record(route: dict, split: str, learned: bool) -> dict:
    return route["splits"][split]["budget"] if learned else route["selection"]


def assert_frame_budget(record: dict) -> None:
    replay = int(record["replay_frames"])
    emitted = int(record["emitted_frames"])
    if 5 * replay > 2 * emitted:
        raise AssertionError(f"replay-frame budget failed: {replay}/{emitted}")
    expected = replay / emitted
    if not math.isclose(
        float(record["realized_replay_frame_fraction"]), expected,
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise AssertionError("reported replay-frame fraction is inconsistent")


def audit(split: str, require_unscored: bool) -> dict:
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise AssertionError("sensitivity preregistration changed after freezing")
    base = ROOT / "outputs/revision"
    query_path = base / f"{PREFIX}_{split}_queries.json"
    plans_path = base / f"{PREFIX}_{split}_plans.json"
    donor_path = base / f"{PREFIX}_{split}_donor_trace.json"
    fallback_path = base / f"{PREFIX}_fallback_{split}_trace.json"
    retrieval_path = base / f"{PREFIX}_retrieval_{split}_trace.json"
    train_path = ROOT / "data/phoenix/phoenix_train.json"
    primary_plans_path = base / f"clean_source_{split}_plans.json"

    inputs = [
        query_path, plans_path, donor_path, fallback_path, retrieval_path,
        train_path, primary_plans_path,
    ]
    query_rows = load_json(query_path)
    plans = load_json(plans_path)
    primary_plans = load_json(primary_plans_path)
    donor_rows = load_json(donor_path)
    fallback = load_json(fallback_path)
    retrieval_rows = load_json(retrieval_path)
    train_rows = load_json(train_path)

    if plans != primary_plans:
        raise AssertionError("sensitivity gloss plans differ from the frozen primary plans")
    query_ids = [str(row["id"]) for row in query_rows]
    if len(query_ids) != len(set(query_ids)):
        raise AssertionError("query IDs are not unique")
    expected = set(query_ids)
    if any(set(row) != {"id", "text"} for row in query_rows):
        raise AssertionError("query projection contains fields other than ID and text")
    donor = {str(row["id"]): row for row in donor_rows}
    retrieval = {str(row["id"]): row for row in retrieval_rows}
    if any(set(mapping) != expected for mapping in (plans, donor, fallback, retrieval)):
        raise AssertionError("one or more sensitivity artifacts do not cover all query IDs")

    train = {str(row["id"]): row for row in train_rows}
    query_text = {str(row["id"]): str(row["text"]) for row in query_rows}
    train_norm = {
        source: normalize_caption(row.get("text", ""))
        for source, row in train.items()
    }
    planned_tokens = emitted_tokens = omitted_tokens = 0
    omission_clips = exact_source_links = 0
    forbidden_candidate_rejections = 0
    used_sources: set[str] = set()
    for sid in query_ids:
        qnorm = normalize_caption(query_text[sid])
        exact = {source for source, value in train_norm.items() if value == qnorm}
        row = donor[sid]
        planner = str(row["retrieved_id"])
        forbidden = {str(source) for source in row["forbidden_source_ids"]}
        required = exact | {planner}
        if forbidden != required:
            raise AssertionError(f"forbidden-source set is incomplete for {sid}")
        expected_hash = hashlib.sha256(
            ("\n".join(sorted(forbidden)) + "\n").encode("utf-8")
        ).hexdigest()
        if row["forbidden_source_ids_sha256"] != expected_hash:
            raise AssertionError(f"forbidden-source hash failed for {sid}")
        segments = list(fallback[sid].get("segments", []))
        segment_sources = {str(segment["source"]) for segment in segments}
        if segment_sources.difference(train):
            raise AssertionError(f"non-training segment source for {sid}")
        if segment_sources.intersection(forbidden):
            raise AssertionError(f"forbidden local source was emitted for {sid}")
        selected = retrieval[sid].get("selected") or {}
        if selected:
            replay_source = str(selected["id"])
            if replay_source not in train or replay_source in exact:
                raise AssertionError(f"invalid or exact-caption replay source for {sid}")
        used_sources.update(segment_sources)
        if selected:
            used_sources.add(str(selected["id"]))
        n_plan = int(fallback[sid]["planned_gloss_tokens"])
        n_emit = int(fallback[sid]["emitted_gloss_tokens"])
        n_omit = len(fallback[sid]["omitted_glosses"])
        if n_plan != n_emit + n_omit:
            raise AssertionError(f"gloss-token accounting failed for {sid}")
        planned_tokens += n_plan
        emitted_tokens += n_emit
        omitted_tokens += n_omit
        omission_clips += int(n_omit > 0)
        exact_source_links += len(exact)
        forbidden_candidate_rejections += int(
            fallback[sid]["excluded_source_candidate_rejections"]
        )

    routes = {}
    for name, learned in (("learned", True), ("nonlearned", False)):
        route_path = base / f"{PREFIX}_{name}_frame40_{split}_route.json"
        ledger_path = base / f"{PREFIX}_{name}_frame40_{split}_ledger.json"
        pose_path = base / f"{PREFIX}_{name}_frame40_{split}.pt"
        mask_path = base / f"{PREFIX}_{name}_frame40_{split}.npz"
        inputs.extend([route_path, ledger_path, pose_path, mask_path])
        route = load_json(route_path)
        ledger = load_json(ledger_path)
        record = budget_record(route, split, learned)
        assert_frame_budget(record)
        if not ledger.get("exact_caption_exclusion_enforced"):
            raise AssertionError(f"ledger did not enforce exact-caption exclusion: {name}")
        if ledger.get("normalized_exact_caption_source_links") != 0:
            raise AssertionError(f"ledger reports exact-caption source mass: {name}")
        if float(ledger["unknown_or_untraced_frames"]) != 0.0:
            raise AssertionError(f"ledger contains unknown source mass: {name}")
        if int(ledger["whole_clip_replay_frames"]) != int(record["replay_frames"]):
            raise AssertionError(f"route and ledger replay frames differ: {name}")
        if int(ledger["total_frames"]) != int(record["emitted_frames"]):
            raise AssertionError(f"route and ledger total frames differ: {name}")
        routes[name] = {
            "selected_clips": int(record["selected_clips"]),
            "replay_frames": int(record["replay_frames"]),
            "total_frames": int(record["emitted_frames"]),
            "replay_frame_fraction": float(record["realized_replay_frame_fraction"]),
            "unique_training_sources": int(ledger["unique_training_sources"]),
        }

    evaluator_results = ROOT / "outputs/revision/evaluator_workspace/results"
    score_paths = [
        evaluator_results / f"{PREFIX}_{method}_{split}{suffix}"
        for method in ("fallback", "retrieval", "learned_frame40", "nonlearned_frame40")
        for suffix in (".json", "_text_preds.pt")
    ]
    if require_unscored:
        present = [str(path.relative_to(ROOT)) for path in score_paths if path.exists()]
        if present:
            raise AssertionError(f"sensitivity was already scored: {present}")
        metrics = None
        paired_bootstrap = None
    else:
        missing = [str(path.relative_to(ROOT)) for path in score_paths if not path.exists()]
        if missing:
            raise AssertionError(f"sensitivity score artifacts are missing: {missing}")
        inputs.extend(score_paths)
        metrics = {
            method: metric_row(
                evaluator_results / f"{PREFIX}_{method}_{split}.json"
            )
            for method in (
                "fallback", "retrieval", "learned_frame40", "nonlearned_frame40"
            )
        }
        bootstrap_path = base / f"{PREFIX}_{split}_paired_bootstrap.json"
        if bootstrap_path.exists():
            inputs.append(bootstrap_path)
            paired_bootstrap = load_json(bootstrap_path)["comparisons"]
        else:
            paired_bootstrap = None

    return {
        "verdict": "PASS",
        "analysis_id": "exact-caption-exclusion-sensitivity-v1",
        "split": split,
        "post_test_robustness_analysis": True,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "query_count": len(query_ids),
        "plans_identical_to_frozen_primary": True,
        "exact_caption_training_source_links_excluded": exact_source_links,
        "normalized_exact_caption_source_mass": 0,
        "planned_gloss_tokens": planned_tokens,
        "emitted_gloss_tokens": emitted_tokens,
        "omitted_gloss_tokens": omitted_tokens,
        "clips_with_omissions": omission_clips,
        "forbidden_candidate_rejections": forbidden_candidate_rejections,
        "candidate_training_sources_observed": len(used_sources),
        "routes": routes,
        "metrics": metrics,
        "paired_bootstrap": paired_bootstrap,
        "evaluator_results_present_before_scoring": False if require_unscored else None,
        "inputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(inputs))
        ],
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--require_unscored", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(args.split, args.require_unscored)
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": report["verdict"],
        "split": report["split"],
        "omitted_gloss_tokens": report["omitted_gloss_tokens"],
        "routes": report["routes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
