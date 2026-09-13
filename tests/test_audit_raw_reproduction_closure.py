from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import audit_raw_reproduction_closure as audit


@pytest.fixture(scope="session", autouse=True)
def cpu_runtime():
    audit.configure_runtime("cpu")


@lru_cache(maxsize=1)
def _fixture_source_closure():
    return audit.source_closure()


@pytest.fixture
def owned_root(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "OUT", tmp_path / "revision")
    manifest, info, vocab = metadata_fixture(n=7096)
    for row in manifest:
        row["length"] = 8
    selected = audit.select_pilot(audit.validate_metadata(manifest, info, vocab))
    monkeypatch.setattr(audit, "FROZEN_PILOT_IDS", tuple(row["id"] for row in selected))
    monkeypatch.setattr(audit, "CALIBRATION_ID", selected[0]["id"])
    # Run-report tests isolate their small immutable calibration fixture. The
    # real calibration derivation/validation has separate, unmocked tests below.
    monkeypatch.setattr(audit, "validate_calibration", audit.read_bound)
    return audit.OUT


def png(path, value=42, shape=(260, 210, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full(shape, value, dtype=np.uint8)
    assert cv2.imwrite(str(path), pixels)


def frame_fixture(tmp_path, count=2):
    row = {"id": "train_clip", "length": count, "gloss": "A B", "gloss_indices": [1, 2]}
    raw, cached = tmp_path / "raw", tmp_path / "cached"
    for i in range(1, count + 1):
        png(raw / row["id"] / f"images{i:04d}.png", value=i)
        png(cached / row["id"] / f"images{i:04d}.png", value=i, shape=(256, 256, 3))
    return raw, cached, row


def entry_fixture():
    logits = np.array([[0, 8, 0], [0, 0, 8]], dtype=np.float64)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    entry = {"log_probs": np.log(probabilities).astype(np.float32), "T_out": 2, "gloss_indices": [1, 2]}
    return entry, {"id": "clip", "gloss": "A B", "gloss_indices": [1, 2]}, [[0, 1], [1, 2]]


def metadata_fixture(n=27):
    manifest, info = [], {}
    for i in range(n):
        row = {"id": f"clip{i:05d}", "length": 16 + i, "signer": f"Signer{i % 9 + 1:02d}", "gloss": "A B"}
        manifest.append(row)
        info[i] = {"fileid": row["id"], "signer": row["signer"], "label": row["gloss"], "num_frames": 0}
    return manifest, info, {"A": [1, 1], "B": [2, 1]}


def test_path_confinement_and_wrong_stage(owned_root):
    assert audit.owned(owned_root / "raw_repro_pilot_v1" / "a.json").parent.name == "raw_repro_pilot_v1"
    for path in (owned_root.parent / "outside.json", owned_root / "router_run_v1" / "a.json",
                 owned_root / "raw_repro_pilot_v0" / "a.json", owned_root / "raw_repro_pilot_v1" / ".." / "bad.json"):
        with pytest.raises(audit.AuditError, match="output"):
            audit.owned(path)
    with pytest.raises(audit.AuditError, match="wrong output"):
        audit.stage_dir(owned_root / "raw_repro_full_v1", "pilot")


def test_output_symlink_escape_is_rejected(owned_root):
    owned_root.mkdir()
    outside = owned_root.parent / "outside"
    outside.mkdir()
    (owned_root / "raw_repro_pilot_v1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(audit.AuditError, match="symlink"):
        audit.write_new(owned_root / "raw_repro_pilot_v1" / "x.json", {})
    assert not (outside / "x.json").exists()


def test_exclusive_artifacts_cannot_be_overwritten(owned_root):
    path = owned_root / "raw_repro_pilot_v1" / "x.json"
    audit.write_new(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        audit.write_new(path, {"original": False})
    assert path.read_bytes() == before
    with pytest.raises(audit.AuditError, match="exists"):
        audit.stage_dir(path.parent, "pilot")


@pytest.mark.parametrize("bad", ["..", ".", "a/b", "a\\b", "", "/tmp", "id*", None])
def test_clip_ids_cannot_escape_inputs(bad):
    with pytest.raises(audit.AuditError):
        audit.checked_id(bad)


def test_file_hash_drift_and_missing_input(tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"original")
    record = audit.file_record(path)
    path.write_bytes(b"changed!")
    with pytest.raises(audit.AuditError, match="drift"):
        audit.verify_record(record)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        audit.verify_record(record)


def test_input_mutation_during_hash_is_detected(tmp_path, monkeypatch):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"original")
    def mutating_hash(p):
        p.write_bytes(b"mutated and resized")
        return "a" * 64
    monkeypatch.setattr(audit, "sha256", mutating_hash)
    with pytest.raises(audit.AuditError, match="mutation while hashing"):
        audit.file_record(path)


def test_metadata_selection_is_order_independent_and_covers_strata():
    manifest, info, vocab = metadata_fixture()
    rows = audit.validate_metadata(manifest, info, vocab)
    first = audit.select_pilot(rows)
    assert first == audit.select_pilot(list(reversed(rows)))
    assert len(first) == len({r["id"] for r in first}) == 10
    assert len({r["signer"] for r in first}) == 9
    assert {r["length_tertile"] for r in first} == {0, 1, 2}


@pytest.mark.parametrize("mutation,match", [
    (lambda m, i, v: m.append(dict(m[0])), "duplicate training"),
    (lambda m, i, v: i[1].update(fileid=i[0]["fileid"]), "duplicate CorrNet"),
    (lambda m, i, v: i.pop(1), "metadata index"),
    (lambda m, i, v: m.pop(), "universe mismatch"),
    (lambda m, i, v: i[0].update(signer="wrong"), "metadata mismatch"),
    (lambda m, i, v: v.pop("B"), "token drop"),
    (lambda m, i, v: m[0].update(length=0), "length"),
    (lambda m, i, v: v.update(B=[1, 1]), "vocabulary IDs"),
])
def test_metadata_rejects_duplicate_missing_and_token_drop(mutation, match):
    manifest, info, vocab = metadata_fixture()
    mutation(manifest, info, vocab)
    with pytest.raises(audit.AuditError, match=match):
        audit.validate_metadata(manifest, info, vocab)


def test_pixels_exact_and_all_frame_denominators(tmp_path):
    raw, cached, row = frame_fixture(tmp_path)
    inventory = audit.frame_inventory(raw, cached, row)
    result = audit.verify_pixels(inventory, raw, cached, row)
    assert result == {"frames": 2, "pixel_channels": 2 * 256 * 256 * 3, "exact": True}


@pytest.mark.parametrize("mutation,match", [
    (lambda r, c: (r / "images0001.png").unlink(), "missing/extra"),
    (lambda r, c: (r / "images0002.png").rename(r / "images0003.png"), "missing frame sequence"),
    (lambda r, c: (r / "unexpected.txt").write_text("extra"), "missing/extra"),
    (lambda r, c: (r / "images0002.png").rename(r / "bad0002.png"), "frame name"),
])
def test_missing_or_malformed_frame_inputs_fail(tmp_path, mutation, match):
    raw, cached, row = frame_fixture(tmp_path)
    mutation(raw / row["id"], cached / row["id"])
    with pytest.raises(audit.AuditError, match=match):
        audit.frame_inventory(raw, cached, row)


def test_duplicate_hardlinked_frames_fail(tmp_path):
    raw, cached, row = frame_fixture(tmp_path)
    folder = raw / row["id"]
    (folder / "images0002.png").unlink()
    os.link(folder / "images0001.png", folder / "images0002.png")
    with pytest.raises(audit.AuditError, match="duplicate/hardlinked"):
        audit.frame_inventory(raw, cached, row)


def test_corrupt_frame_fails_without_skip(tmp_path):
    raw, cached, row = frame_fixture(tmp_path)
    (raw / row["id"] / "images0001.png").write_bytes(b"not an image")
    inventory = audit.frame_inventory(raw, cached, row)
    with pytest.raises(audit.AuditError, match="corrupt/undecodable"):
        audit.verify_pixels(inventory, raw, cached, row)


def test_pixel_mismatch_fails_even_with_matching_file_inventory(tmp_path):
    raw, cached, row = frame_fixture(tmp_path)
    png(cached / row["id"] / "images0001.png", value=177, shape=(256, 256, 3))
    inventory = audit.frame_inventory(raw, cached, row)
    with pytest.raises(audit.AuditError, match="pixel mismatch"):
        audit.verify_pixels(inventory, raw, cached, row)


def test_postseal_frame_mutation_fails(tmp_path):
    raw, cached, row = frame_fixture(tmp_path)
    inventory = audit.frame_inventory(raw, cached, row)
    png(raw / row["id"] / "images0001.png", value=177)
    with pytest.raises(audit.AuditError, match="input mutation"):
        audit.verify_pixels(inventory, raw, cached, row)


@pytest.mark.parametrize("mutation,match", [
    (lambda e: e["log_probs"].__setitem__((0, 0), np.nan), "nonfinite"),
    (lambda e: e["log_probs"].__setitem__((0, 0), np.inf), "nonfinite"),
    (lambda e: e.update(T_out=1), "T_out"),
    (lambda e: e.update(gloss_indices=[1]), "token drop"),
    (lambda e: e.update(gloss_indices=[1, 0]), "token drop"),
    (lambda e: e.update(log_probs=e["log_probs"].astype(np.float64)), "dtype"),
    (lambda e: e.update(log_probs=np.zeros((2, 3), np.float32)), "normalized"),
    (lambda e: e.update(name="test/clip"), "name/ID"),
])
def test_posterior_nan_token_drop_and_shape_fail_closed(mutation, match):
    entry, row, spans = entry_fixture()
    mutation(entry)
    with pytest.raises(audit.AuditError, match=match):
        audit.validate_posterior(entry, row, 3)


def test_continuous_equality_and_decision_equivalence_are_separate():
    expected, row, spans = entry_fixture()
    observed = copy.deepcopy(expected)
    observed["log_probs"][0, 0] += np.float32(2e-6)
    comparison = audit.compare_entries(observed, expected, row, 3, spans, spans)
    assert comparison["log_probs_exact"] is False
    assert all(comparison["strict_margin_certified"])
    assert comparison["continuous_posterior_reproduction_claim"] is False
    assert "log_probs_within_tolerance" not in comparison
    assert comparison["elements"] == 6 and comparison["tokens"] == 2


def test_continuous_error_is_diagnostic_only_but_span_drift_fails():
    expected, row, spans = entry_fixture()
    observed = copy.deepcopy(expected)
    observed["log_probs"][0, 0] += np.float32(1e-3)
    result = audit.compare_entries(observed, expected, row, 3, spans, spans)
    assert result["max_abs_error"] > 1e-5 and all(result["strict_margin_certified"])
    with pytest.raises(audit.AuditError, match="span mismatch"):
        audit.compare_entries(expected, expected, row, 3, [[0, 2], [1, 2]], spans)


def test_strict_ctc_disallows_infeasible_and_fallback_states():
    entry, row, spans = entry_fixture()
    with pytest.raises(audit.AuditError, match="infeasible"):
        audit.strict_ctc_spans(entry["log_probs"], [1, 1])
    source = SimpleNamespace(ctc_forced_align=lambda *args, **kwargs: [(0, 2), (0, 2)])
    with pytest.raises(audit.AuditError, match="no-fallback"):
        audit.checked_alignment(source, entry["log_probs"], [1, 2])


def test_strict_ctc_matches_exhaustive_best_path_with_repeated_tokens():
    rng = np.random.default_rng(77)
    for targets in ([1], [1, 2], [1, 1]):
        logits = rng.normal(size=(5, 3))
        lp = logits - np.log(np.exp(logits).sum(axis=1))[:, None]
        best = None
        best_score = -np.inf
        for path in itertools.product(range(3), repeat=5):
            collapsed = [value for i, value in enumerate(path) if value != 0 and (i == 0 or path[i - 1] != value)]
            if collapsed == targets:
                score = sum(lp[t, value] for t, value in enumerate(path))
                if score > best_score:
                    best, best_score = path, score
        spans, start, target = [], None, 0
        for t, value in enumerate(best):
            if value and (t == 0 or best[t - 1] != value):
                start = t
            if value and (t == 4 or best[t + 1] != value):
                spans.append([start, t + 1])
                target += 1
        assert audit.strict_ctc_spans(lp, list(targets)) == spans


def test_runtime_drift_fails():
    expected = {"packages": {"torch": "1"}, "device": "cuda:0"}
    for observed in ({"packages": {"torch": "2"}, "device": "cuda:0"},
                     {"packages": {"torch": "1"}, "device": "cpu"}):
        with pytest.raises(audit.AuditError, match="runtime drift"):
            audit.validate_runtime(expected, observed)


def sealed_fixture(owned_root):
    output = owned_root / "raw_repro_prereg_v1"
    if output.exists():
        return output, audit.sha256(output / "seal.json")
    output = audit.stage_dir(output, "prereg")
    manifest, info, vocab = metadata_fixture(n=7096)
    for row in manifest:
        row["length"] = 8
    rows = audit.validate_metadata(manifest, info, vocab)
    selected, partition = audit.selection_partition(rows, "pilot")
    closure = _fixture_source_closure()
    calibration = fixture_calibration(owned_root, rows)
    gate_path = owned_root / "raw_repro_synthetic_gate_v1" / "gate.json"
    gate = {"schema": audit.SCHEMA, "status": "PASS", "tested_closure": closure, "tested_runtime": {"device": "cpu"},
            "core_ast_sha256": closure["core_ast_sha256"], "calibration": calibration}
    audit.write_new(gate_path, gate)
    clips = []
    for row in selected:
        frames_path = output / "frames" / f"{row['id']}.json"
        audit.write_new(frames_path, [{"name": f"images{i:04d}.png"} for i in range(1, row["length"] + 1)])
        clips.append({"metadata": row, "role": "VALIDATION", "frames": audit.file_record(frames_path)})
    doc = {"schema": audit.SCHEMA, "scope": "pilot", "policy": audit.POLICY, "revision": audit.REVISION,
           "gate": audit.file_record(gate_path), "tested_closure": closure, "sources": closure["sources"],
           "assets": closure["model_assets"], "core_ast_sha256": closure["core_ast_sha256"],
           "runtime": gate["tested_runtime"], "vocab_size": 3,
           "claim_limit": audit.CLAIM_LIMIT, "training_universe": rows, "training_count": 7096,
           "selection_uses_model_outputs": False, "selection_algorithm": audit.SELECTION,
           "calibration": calibration, "partition": partition,
           "historical_commitments": audit.read_bound(calibration)["historical_commitments"],
           "clips": clips, "selected_ids": [row["id"] for row in selected]}
    audit.write_new(output / "preregistration.json", doc)
    audit.write_new(output / "seal.json", {"schema": audit.SCHEMA,
                    "preregistration": audit.file_record(output / "preregistration.json")})
    return output, audit.sha256(output / "seal.json")


def fixture_calibration(owned_root, rows=None):
    path = owned_root / "raw_repro_synthetic_gate_v90" / "calibration.json"
    if not path.exists():
        entry, _, _ = entry_fixture()
        rows = rows or [{"id": sid} for sid in audit.FROZEN_PILOT_IDS]
        audit.write_new(path.parent / "historical_commitments.json",
                        {"entries": {r["id"]: audit.entry_commitment(entry) for r in rows}})
        audit.write_new(path, {"v5_assets": _fixture_source_closure()["model_assets"],
                        "historical_commitments": audit.file_record(path.parent / "historical_commitments.json")})
    return audit.file_record(path)


def test_immutable_prereg_seal_and_selection(owned_root):
    output, digest = sealed_fixture(owned_root)
    assert len(audit.load_prereg(output, digest, check_inputs=False)["clips"]) == 9
    with pytest.raises(audit.AuditError, match="explicit seal"):
        audit.load_prereg(output, "", check_inputs=False)
    doc_path = output / "preregistration.json"
    doc_path.chmod(0o644)
    doc = json.loads(doc_path.read_text())
    doc["selected_ids"].pop()
    doc_path.write_text(json.dumps(doc))
    with pytest.raises(audit.AuditError, match="drift"):
        audit.load_prereg(output, digest, check_inputs=False)
    seal_path = output / "seal.json"
    seal_path.chmod(0o644)
    seal_path.write_text(json.dumps({"schema": audit.SCHEMA, "preregistration": audit.file_record(doc_path)}))
    with pytest.raises(audit.AuditError, match="seal drift"):
        audit.load_prereg(output, digest, check_inputs=False)
    with pytest.raises(audit.AuditError, match="selection drift"):
        audit.load_prereg(output, audit.sha256(seal_path), check_inputs=False)


def report_fixture(owned_root, version=1, kind_override=None):
    prereg, seal_hash = sealed_fixture(owned_root)
    doc = audit.load_prereg(prereg, seal_hash, check_inputs=False)
    kind = kind_override or ("pilot" if version == 1 else "repeat")
    directory = audit.stage_dir(owned_root / f"raw_repro_{kind}_v{version}", kind)
    entry, row, spans = entry_fixture()
    comparison = audit.compare_entries(entry, entry, row, 3, spans, spans)
    run_id = f"{version:032x}"
    audit.write_new(directory / "start.json", {"run_id": run_id, "run_dir": str(directory), "stage": kind,
                    "started_unix_ns": version * 100, "started_monotonic_ns": version * 100,
                    "boot_id": "fixture-boot", "seal_sha256": seal_hash})
    audit.write_new(directory / "finish.json", {"run_id": run_id, "run_dir": str(directory),
                    "finished_unix_ns": version * 100 + 50, "finished_monotonic_ns": version * 100 + 50,
                    "boot_id": "fixture-boot"})
    clips = []
    for row in doc["clips"]:
        sid = row["metadata"]["id"]
        record = audit.save_posterior(directory / "posteriors" / f"{sid}.npz", entry, spans)
        historical = audit.save_posterior(directory / "historical" / f"{sid}.npz", entry, spans)
        audit.write_new(directory / "diagnostics" / f"{sid}.json", comparison)
        clips.append({"id": sid, "role": "VALIDATION", "pixels": {"frames": 8, "pixel_channels": 8 * 256 * 256 * 3, "exact": True},
                      "comparison": dict(comparison), "output": record, "historical": historical,
                      "diagnostics": audit.file_record(directory / "diagnostics" / f"{sid}.json")})
    report = {"schema": audit.SCHEMA, "status": "PASS", "scope": "pilot", "kind": kind,
            "result_kind": "individual_decision_equivalence", "deterministic_repeat_claim": False,
            "validation_closure_verified": False, "partition": doc["partition"],
            "historical_agreement_mode": audit.POLICY["historical_comparison"], "exact_continuous_clips": 9,
            "run_id": run_id, "run_dir": str(directory), "report_path": str(directory / "report.json"),
            "start": audit.file_record(directory / "start.json"), "preregistration_seal": audit.file_record(prereg / "seal.json"),
            "finish": audit.file_record(directory / "finish.json"), "role_counts": {"VALIDATION": 9},
            "selected_ids": [c["id"] for c in clips], "clips": clips, "skips": 0, "fallbacks": 0,
            "completed_clips": 9, "expected_clips": 9, "frames": 72, "posterior_elements": 54,
            "pixel_channels": 72 * 256 * 256 * 3, "tokens": 18, "time_steps": 18, "decoded_tokens": 18,
            "claim_limit": copy.deepcopy(audit.CLAIM_LIMIT), "raw_pose_producers_reproduced": False,
            "generalizes_to_full_training": False, "full_training_census_complete": False,
            "preregistration_seal_sha256": seal_hash,
            "historical_continuous_posteriors_reproduced": False, "historical_backend_reproduced": False,
            "runtime": doc["runtime"], "assets": doc["assets"], "sources": doc["sources"], "core_ast_sha256": doc["core_ast_sha256"]}
    counts = {key: report[key] for key in ("frames", "pixel_channels", "posterior_elements", "tokens", "time_steps", "decoded_tokens")}
    audit.write_new(directory / "scientific.json", audit.scientific_document(doc, seal_hash, clips, counts))
    report["scientific_report"] = audit.file_record(directory / "scientific.json")
    audit.write_new(directory / "report.json", report)
    return report


def report_record(report):
    return audit.file_record(report["report_path"])


def overwrite_fixture_report(report):
    path = Path(report["report_path"])
    path.chmod(0o644)
    path.write_bytes(audit.canonical(report))
    return audit.file_record(path)


def test_repeat_comparator_and_output_hash_integrity(owned_root):
    first = report_fixture(owned_root, 1)
    second = report_fixture(owned_root, 2)
    result = audit.compare_runs(report_record(first), report_record(second))
    assert result["status"] == "PASS" and result["clips"] == result["exact_cpu_posterior_clips"] == 9
    assert result["elements"] == 54 and result["validation_closure_verified"] is True
    changed = copy.deepcopy(second)
    changed["runtime"]["device"] = "cuda:0"
    with pytest.raises(audit.AuditError, match="runtime differs"):
        audit.compare_runs(report_record(first), overwrite_fixture_report(changed))
    overwrite_fixture_report(second)
    path = Path(second["clips"][0]["output"]["path"])
    path.chmod(0o644)
    path.write_bytes(b"broken")
    with pytest.raises(audit.AuditError, match="drift"):
        audit.compare_runs(report_record(first), report_record(second))


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["selected_ids"].__setitem__(1, r["selected_ids"][0]), "duplicate/missing"),
    (lambda r: r["clips"].pop(), "duplicate/missing"),
    (lambda r: r.update(completed_clips=8), "denominator"),
    (lambda r: r.update(frames=19), "denominator"),
    (lambda r: r.update(posterior_elements=59), "denominator"),
    (lambda r: r.update(skips=1), "skip/fallback"),
    (lambda r: r.update(fallbacks=1), "skip/fallback"),
    (lambda r: r["clips"][0]["comparison"].update(ctc_spans_exact=False), "equality"),
    (lambda r: r.update(generalizes_to_full_training=True), "pilot promoted"),
    (lambda r: r.update(raw_pose_producers_reproduced=True), "raw-pose"),
    (lambda r: r["claim_limit"].update(not_reproduced=[]), "claim ceiling"),
    (lambda r: r.update(deterministic_repeat_claim=True), "deterministic-repeat claim"),
])
def test_run_denominators_and_claim_ceiling(owned_root, mutation, match):
    report = report_fixture(owned_root)
    mutation(report)
    with pytest.raises(audit.AuditError, match=match):
        audit.validate_run_report(report)


def test_v7_honest_pilot_census_flag_and_array_derived_counts_pass(owned_root):
    report = report_fixture(owned_root)
    checked = audit.validate_run_report(report)
    assert report["full_training_census_complete"] is False
    assert checked["counts"]["frames"] == 72


def test_v7_coherent_full_run_kind_cannot_promote_a_sealed_pilot(owned_root):
    report = report_fixture(owned_root, kind_override="full")
    # Directory, start record, every artifact path and hash coherently say full,
    # but the frozen preregistration still permits only a pilot or its repeat.
    with pytest.raises(audit.AuditError, match="run kind differs"):
        audit.validate_run_report(report)


@pytest.mark.parametrize("bad", [True, None, 0, 1, 0.0, "false", [], "missing"])
def test_v7_pilot_census_promotion_missing_and_ill_typed_flags_reject_before_claim_acceptance(owned_root, monkeypatch, bad):
    report = report_fixture(owned_root)
    if bad == "missing":
        report.pop("full_training_census_complete")
    else:
        report["full_training_census_complete"] = bad
    # Coherently rehashing the complete report must not create claim authority.
    record = overwrite_fixture_report(report)
    monkeypatch.setattr(audit, "load_prereg", lambda *a, **k: pytest.fail("invalid pilot claim reached preregistration"))
    with pytest.raises(audit.AuditError, match="census"):
        audit.validate_run_report(audit.read_bound(record))


@pytest.mark.parametrize("field", ["deterministic_repeat_verified", "all_item_semantic_census_verified"])
@pytest.mark.parametrize("value", [True, False, 1, "true"])
def test_v7_comparison_only_claim_fields_cannot_be_injected_into_individual_report(owned_root, field, value):
    report = report_fixture(owned_root)
    report[field] = value
    with pytest.raises(audit.AuditError, match="comparison-only"):
        audit.validate_run_report(audit.read_bound(overwrite_fixture_report(report)))


@pytest.mark.parametrize("field", ["validation_closure_verified", "deterministic_repeat_claim",
                                 "historical_continuous_posteriors_reproduced", "historical_backend_reproduced"])
def test_v7_adjacent_individual_claim_promotions_remain_rejected(owned_root, field):
    report = report_fixture(owned_root)
    report[field] = True
    with pytest.raises(audit.AuditError, match="claim"):
        audit.validate_run_report(audit.read_bound(overwrite_fixture_report(report)))


@pytest.mark.parametrize("field", ["deterministic_claim", "continuous_posterior_reproduction_claim"])
def test_v7_coherent_per_clip_comparison_claim_promotion_is_reconstructed(owned_root, field):
    report = report_fixture(owned_root)
    report["clips"][0]["comparison"][field] = True
    diagnostic = Path(report["clips"][0]["diagnostics"]["path"])
    diagnostic.chmod(0o644)
    diagnostic.write_bytes(audit.canonical(report["clips"][0]["comparison"]))
    report["clips"][0]["diagnostics"] = audit.file_record(diagnostic)
    with pytest.raises(audit.AuditError, match="scientific comparison"):
        audit.validate_run_report(audit.read_bound(overwrite_fixture_report(report)))


def full_census_claim_fixture():
    """Small synthetic arrays aggregated over an explicitly enumerated 7096 IDs.

    Tests the final pure census guard; run-report integration separately tests
    the mandatory prior array/pixel/span reconstruction. No production cache.
    """
    entry, row, _ = entry_fixture()
    ids = [audit.CALIBRATION_ID] + [f"synthetic_census_{i:04d}" for i in range(7095)]
    rows = [dict(row, id=sid, length=8) for sid in ids]
    roles = {sid: "CALIBRATION" if i == 0 else "NONCALIBRATION_CENSUS" for i, sid in enumerate(ids)}
    prereg = {"scope": "full", "training_universe": rows, "training_count": 7096, "selected_ids": ids,
              "clips": [{"metadata": r, "role": roles[r["id"]]} for r in rows],
              "partition": {"execution_roles": roles}, "vocab_size": entry["log_probs"].shape[1]}
    report = {"scope": "full", "kind": "full", "full_training_census_complete": True, "selected_ids": list(ids),
              "completed_clips": 7096, "expected_clips": 7096,
              "clips": [{"id": sid, "role": roles[sid]} for sid in ids],
              "role_counts": {"CALIBRATION": 1, "NONCALIBRATION_CENSUS": 7095}}
    counts = {"frames": 7096 * 8, "pixel_channels": 7096 * 8 * 256 * 256 * 3,
              "time_steps": 7096 * int(entry["log_probs"].shape[0]),
              "posterior_elements": 7096 * int(entry["log_probs"].size),
              "tokens": 7096 * len(entry["gloss_indices"]), "decoded_tokens": 7096 * len(audit.decoded_tokens(entry["log_probs"]))}
    return report, prereg, counts


def test_v7_complete_full_census_fixture_accepts_true_only_after_derived_invariants():
    report, prereg, counts = full_census_claim_fixture()
    audit.validate_census_claim(report, prereg, counts)


@pytest.mark.parametrize("bad", [False, None, 0, 1, 1.0, "true", [], "missing"])
def test_v7_full_census_false_missing_and_ill_typed_flags_reject(bad):
    report, prereg, counts = full_census_claim_fixture()
    if bad == "missing":
        report.pop("full_training_census_complete")
    else:
        report["full_training_census_complete"] = bad
    with pytest.raises(audit.AuditError, match="census flag"):
        audit.validate_census_claim(report, prereg, counts)


@pytest.mark.parametrize("mutation", [
    lambda r, p, c: r["clips"].pop(),
    lambda r, p, c: r["selected_ids"].__setitem__(1, r["selected_ids"][0]),
    lambda r, p, c: p.update(training_count=7096.0),
    lambda r, p, c: r.update(completed_clips=7095),
    lambda r, p, c: r.update(expected_clips=7096.0),
    lambda r, p, c: r["clips"][0].update(role="VALIDATION"),
    lambda r, p, c: p["partition"]["execution_roles"].update({audit.CALIBRATION_ID: "NONCALIBRATION_CENSUS"}),
    lambda r, p, c: r.update(role_counts={"NONCALIBRATION_CENSUS": 7096}),
    lambda r, p, c: r["role_counts"].update(CALIBRATION=True),
    lambda r, p, c: r.update(kind="pilot"),
    lambda r, p, c: c.update(frames=c["frames"] - 1),
    lambda r, p, c: c.update(pixel_channels=c["pixel_channels"] - 1),
    lambda r, p, c: c.update(time_steps=c["time_steps"] - 1),
    lambda r, p, c: c.update(posterior_elements=c["posterior_elements"] - 1),
    lambda r, p, c: c.update(tokens=c["tokens"] - 1),
])
def test_v7_full_census_true_cannot_bypass_item_role_or_derived_denominators(mutation):
    report, prereg, counts = full_census_claim_fixture()
    mutation(report, prereg, counts)
    with pytest.raises(audit.AuditError, match="census"):
        audit.validate_census_claim(report, prereg, counts)


def test_v7_census_validation_is_mandatory_after_array_reconstruction(owned_root, monkeypatch):
    report = report_fixture(owned_root)
    def reject_full_guard(report, prereg, counts):
        assert counts["frames"] == 72 and counts["posterior_elements"] == 54
        raise audit.AuditError("synthetic terminal census guard")
    monkeypatch.setattr(audit, "validate_census_claim", reject_full_guard)
    with pytest.raises(audit.AuditError, match="terminal census guard"):
        audit.validate_run_report(report)


def test_v7_reuses_only_exact_pinned_v6_calibration_without_cache_deserialization(monkeypatch):
    monkeypatch.setattr(audit.pickle, "load", lambda *a, **k: pytest.fail("production cache cannot be deserialized"))
    record = audit.REVISION["superseded_evidence"]["v6_calibration"]
    before = audit.file_record(record["path"])
    calibration = audit.validate_calibration(record)
    assert audit.file_record(record["path"]) == before == record
    assert calibration["schema"] == "raw_rgb_corrnet_decision_equivalence_v6"
    assert calibration["measurement"]["role"] == "CALIBRATION"
    assert calibration["measurement"]["historical_v5_continuous_gate"]["passed"] is False
    assert "BLOCKED" in audit.REVISION["v6_review_status"]


def test_v7_legacy_calibration_alias_or_coherent_rewrite_cannot_cross_version_boundary(tmp_path, monkeypatch):
    record = audit.REVISION["superseded_evidence"]["v6_calibration"]
    calibration = audit.read_bound(record)
    path = tmp_path / "aliased_calibration.json"
    path.write_bytes(audit.canonical(calibration))
    with pytest.raises(audit.AuditError, match="schema/status/lineage"):
        audit.validate_calibration(audit.file_record(path))
    calibration["measurement"]["is_validation"] = True
    path.write_bytes(audit.canonical(calibration))
    with pytest.raises(audit.AuditError, match="schema/status/lineage"):
        audit.validate_calibration(audit.file_record(path))
    source = tmp_path / "changed_external.py"
    source.write_text("SAFE_FIXTURE = True\n")
    monkeypatch.setattr(audit, "SOURCE_PATHS", dict(audit.SOURCE_PATHS, tconv=source))
    with pytest.raises(audit.AuditError, match="external source drift"):
        audit.validate_calibration(record)


def test_model_inference_requires_explicit_authorization(owned_root):
    output = owned_root / "raw_repro_pilot_v1"
    with pytest.raises(audit.AuditError, match="not been explicitly authorized"):
        audit.execute("missing", "a" * 64, output, "pilot")
    assert not output.exists()


def test_failure_is_retained_and_cannot_overwrite_prior_run(owned_root, monkeypatch):
    output = owned_root / "raw_repro_pilot_v1"
    lock = owned_root / "raw_repro_prereg_v1" / "heavy.lock"
    audit.write_bytes_new(lock, b"")
    monkeypatch.setattr(audit, "load_prereg", lambda *args, **kwargs: {"lock": audit.file_record(lock), "runtime": {"device": "cpu"}})
    def fail_lock(*args):
        raise audit.AuditError("synthetic resource gate failure")
    monkeypatch.setattr(audit, "heavy_lock", fail_lock)
    with pytest.raises(audit.AuditError, match="resource gate"):
        audit.execute("missing", "a" * 64, output, "pilot", authorize_inference=True)
    report = json.loads((output / "failure.json").read_text())
    assert report["status"] == "FAIL" and report["completed_clips"] == 0
    with pytest.raises(audit.AuditError, match="exists"):
        audit.execute("missing", "a" * 64, output, "pilot", authorize_inference=True)


def test_resource_guard_rejects_high_peak_rss(monkeypatch):
    monkeypatch.setattr(audit.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=7 * 1024**2))
    with pytest.raises(audit.AuditError, match="RSS"):
        audit.resources()


def test_bound_external_source_import_and_tiny_fake_inference(owned_root):
    # A source integration check; never instantiates CorrNet or opens its checkpoint/cache.
    output = owned_root / "raw_repro_synthetic_gate_v1"
    junit = output / "tests.xml"
    audit.write_bytes_new(junit, b'<testsuite tests="1"><testcase name="fixture"/></testsuite>')
    result = audit.synthetic_gate(output, junit, fixture_calibration(owned_root))
    assert result["status"] == "PASS" and result["real_inference"] is False
    gate = json.loads((output / "gate.json").read_text())
    assert gate["pixels"]["frames"] == 8
    assert gate["comparison"]["ctc_spans_exact"] is True
    assert audit.validate_gate(output / "gate.json", result["gate_sha256"])["sha256"] == result["gate_sha256"]


def test_safe_source_loader_cannot_download_model(owned_root):
    sources = {key: audit.file_record(path) for key, path in audit.SOURCE_PATHS.items()}
    resnet_record = audit.file_record(audit.DEFAULTS["resnet_init"])
    source = audit.load_corrnet_source(sources, resnet_record, "cpu")
    # Inspect the actual loader bound inside the imported ResNet definition.
    loader = source.corrnet_resnet18.__globals__["model_zoo"].load_url
    with pytest.raises(audit.AuditError, match="unexpected model URL"):
        loader("https://example.com/unregistered.pt")
    assert "OUTPUT_DIR" not in vars(source) and "process_split" not in vars(source)


@pytest.mark.parametrize("mutation,match", [
    (lambda r, t: r.update(feat_len=t.tensor([3.0], dtype=t.float32)), "silent trimming"),
    (lambda r, t: r["sequence_logits"].__setitem__((1, 0, 0), float("nan")), "nonfinite"),
    (lambda r, t: r.update(feat_len=t.tensor([2], dtype=t.int64)), "dtype"),
    (lambda r, t: r.pop("conv_logits"), "schema"),
])
def test_model_return_is_checked_before_source_trimming(mutation, match):
    import torch
    ret = {"sequence_logits": torch.zeros((2, 1, 3)), "conv_logits": torch.zeros((2, 1, 3)),
           "feat_len": torch.tensor([2.0], dtype=torch.float32)}
    mutation(ret, torch)
    source = SimpleNamespace(torch=torch, infer_core=lambda model, vocab, label, video: model(video, None))
    row = {"length": 8, "gloss": "A B"}
    with pytest.raises(audit.AuditError, match=match):
        audit.checked_inference(source, lambda *_: ret, {"A": [1], "B": [2]}, row, None)


def production_return_fixture(input_frames=8):
    import torch
    output_frames = (input_frames + 3) // 4
    ret = {"sequence_logits": torch.zeros((output_frames, 1, 3), dtype=torch.float32),
           "conv_logits": torch.zeros((output_frames, 1, 3), dtype=torch.float32),
           "feat_len": torch.tensor([float(output_frames)], dtype=torch.float32)}
    source = SimpleNamespace(torch=torch, infer_core=lambda model, vocab, label, video: model(video, None))
    return source, ret, {"length": input_frames, "gloss": "A B"}


@pytest.mark.parametrize("input_frames", [1, 4, 5, 8, 99, 100, 101, 512])
def test_exact_production_float32_length_contract_is_accepted_without_conversion(input_frames):
    source, ret, row = production_return_fixture(input_frames)
    actual = audit.checked_inference(source, lambda *_: ret, {"A": [1], "B": [2]}, row, None)
    assert actual is ret and actual["feat_len"] is ret["feat_len"]
    assert actual["feat_len"].dtype == source.torch.float32
    assert actual["feat_len"].item() == (input_frames + 3) // 4


@pytest.mark.parametrize("dtype", ["float64", "float16", "bfloat16", "int64", "int32", "int16", "bool"])
def test_nonproduction_length_dtypes_are_rejected(dtype):
    source, ret, row = production_return_fixture()
    ret["feat_len"] = ret["feat_len"].to(getattr(source.torch, dtype))
    with pytest.raises(audit.AuditError, match="feat_len shape/dtype"):
        audit.checked_inference(source, lambda *_: ret, {"A": [1], "B": [2]}, row, None)


@pytest.mark.parametrize("value,match", [(1.5, "nonintegral"), (2.000000238418579, "nonintegral"),
    (float("nan"), "nonfinite"), (float("inf"), "nonfinite"), (-float("inf"), "nonfinite"),
    (0.0, "padding contract"), (-1.0, "padding contract"), (1.0, "padding contract"), (3.0, "padding contract")])
def test_float32_length_requires_finite_integral_exact_value(value, match):
    source, ret, row = production_return_fixture()
    ret["feat_len"] = source.torch.tensor([value], dtype=source.torch.float32)
    with pytest.raises(audit.AuditError, match=match):
        audit.checked_inference(source, lambda *_: ret, {"A": [1], "B": [2]}, row, None)


@pytest.mark.parametrize("value", [2.0, [], [2.0, 2.0], [[2.0]]])
def test_production_length_shape_is_exactly_one_vector(value):
    source, ret, row = production_return_fixture()
    ret["feat_len"] = source.torch.tensor(value, dtype=source.torch.float32)
    with pytest.raises(audit.AuditError, match="feat_len shape/dtype"):
        audit.checked_inference(source, lambda *_: ret, {"A": [1], "B": [2]}, row, None)


@pytest.mark.parametrize("field", ["sequence_logits", "conv_logits"])
@pytest.mark.parametrize("shape", [(1, 1, 3), (3, 1, 3), (2, 2, 3), (2, 1, 4), (2, 3), (0, 1, 3)])
def test_both_logits_require_the_exact_production_shape(field, shape):
    source, ret, row = production_return_fixture()
    ret[field] = source.torch.zeros(shape, dtype=source.torch.float32)
    with pytest.raises(audit.AuditError, match=field + " shape"):
        audit.checked_inference(source, lambda *_: ret, {"A": [1], "B": [2]}, row, None)


@pytest.mark.parametrize("field", ["sequence_logits", "conv_logits"])
@pytest.mark.parametrize("dtype", ["float64", "float16", "int64"])
def test_both_logits_require_float32(field, dtype):
    source, ret, row = production_return_fixture()
    ret[field] = ret[field].to(getattr(source.torch, dtype))
    with pytest.raises(audit.AuditError, match=field + " dtype"):
        audit.checked_inference(source, lambda *_: ret, {"A": [1], "B": [2]}, row, None)


@pytest.mark.parametrize("field", ["sequence_logits", "conv_logits"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_both_logits_require_every_element_finite(field, value):
    source, ret, row = production_return_fixture()
    ret[field][-1, 0, -1] = value
    with pytest.raises(audit.AuditError, match="nonfinite " + field):
        audit.checked_inference(source, lambda *_: ret, {"A": [1], "B": [2]}, row, None)


@pytest.mark.parametrize("field", ["feat_len", "sequence_logits", "conv_logits"])
def test_production_model_returns_must_be_tensors(field):
    source, ret, row = production_return_fixture()
    ret[field] = ret[field].numpy()
    with pytest.raises(audit.AuditError, match="shape"):
        audit.checked_inference(source, lambda *_: ret, {"A": [1], "B": [2]}, row, None)


def test_pinned_tconv_division_and_padding_produce_integral_float32_lengths():
    # Execute only the bound length-update/padding functions: no TemporalConv
    # instance, learned weights, convolution, or real CorrNet forward is used.
    closure = _fixture_source_closure()
    assert closure["sources"]["tconv"]["sha256"] == "cd6acd78b9613e65f8ec71638a9c525853302404dd0b4defbc2ff6e220eaba44"
    source = audit.load_corrnet_source(closure["sources"], closure["model_assets"]["resnet_init"], "cpu")
    torch = source.torch
    first = torch.div(torch.tensor([112], dtype=torch.int64), 2)
    second = torch.div(first - 4, 2)
    assert first.dtype == second.dtype == torch.float32
    assert first.tolist() == [56.0] and second.tolist() == [26.0]
    temporal = SimpleNamespace(kernel_size=["K5", "P2", "K5", "P2"])
    for frames in (1, 4, 5, 8, 99, 100, 101, 512):
        tiny_video = torch.zeros((frames, 3, 1, 1), dtype=torch.float32)
        _, lengths = source.pad_video_for_model(tiny_video)
        assert lengths.dtype == torch.int64
        actual = source.TemporalConv.update_lgt(temporal, lengths)
        assert actual.dtype == torch.float32 and tuple(actual.shape) == (1,)
        assert actual.item().is_integer() and actual.item() == (frames + 3) // 4


def test_v5_binds_and_preserves_failed_v4_lineage():
    audit.validate_revision_lineage()
    assert audit.REVISION["version"] == 7 and "BLOCKED" in audit.REVISION["v4_review_status"]
    record = audit.REVISION["superseded_evidence"]["v4_pilot_failure"]
    failure = audit.read_bound(record)
    assert failure["completed_clips"] == 0 and failure["active_output"] is None
    assert failure["error"] == "invalid feat_len shape/dtype"


def test_cpu_contract_masks_cuda_and_binds_single_threads(monkeypatch):
    import torch
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    audit.configure_runtime("cpu")
    runtime = audit.runtime_fingerprint("cpu")
    assert runtime["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert runtime["torch_runtime"]["cuda_available"] is False
    assert runtime["torch_runtime"]["intraop_threads"] == runtime["torch_runtime"]["interop_threads"] == 1
    assert runtime["device"] == "cpu"


def test_workspace_lock_rejects_external_path_and_contention(owned_root):
    lock = owned_root / "raw_repro_prereg_v1" / "heavy.lock"
    audit.write_bytes_new(lock, b"")
    with audit.heavy_lock(lock):
        with pytest.raises(audit.AuditError, match="another memory-heavy"):
            with audit.heavy_lock(lock):
                pass
    with pytest.raises(audit.AuditError, match="outside assigned"):
        with audit.heavy_lock("/tmp/signgen_heavy.lock"):
            pass


def test_gpu_diagnostic_failure_does_not_change_cpu_contract(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("synthetic NVML absence")
    monkeypatch.setattr(audit.subprocess, "run", missing)
    diagnostic = audit.gpu_diagnostic()
    assert "synthetic NVML absence" in diagnostic["current_subprocess_error"]
    assert diagnostic["python_before_mask"]["cuda_available"] is False


def test_repeat_rejects_same_report_and_same_run_id(owned_root):
    first = report_fixture(owned_root, 1)
    second = report_fixture(owned_root, 2)
    with pytest.raises(audit.AuditError, match="same report/run"):
        audit.compare_runs(report_record(first), report_record(first))
    second["run_id"] = first["run_id"]
    with pytest.raises(audit.AuditError, match="same canonical run"):
        audit.compare_runs(report_record(first), overwrite_fixture_report(second))


@pytest.mark.parametrize("kind", ["lexical", "symlink", "hardlink"])
def test_repeat_rejects_path_and_file_aliases(owned_root, kind):
    first = report_fixture(owned_root, 1)
    original = report_record(first)
    alias = dict(original)
    path = Path(original["path"])
    if kind == "lexical":
        alias["path"] = str(path.parent / ".." / path.parent.name / path.name)
    else:
        folder = owned_root / "raw_repro_pilot_v2"
        folder.mkdir()
        alias_path = folder / "report.json"
        if kind == "symlink":
            alias_path.symlink_to(path)
        else:
            os.link(path, alias_path)
        alias["path"] = str(alias_path)
    with pytest.raises(audit.AuditError, match="alias|symlink|same report/run"):
        audit.compare_runs(original, alias)


@pytest.mark.parametrize("field", ["elements", "tokens", "T_out", "decoded_token_count", "frames"])
def test_coherent_declared_count_tampering_is_rejected(owned_root, field):
    first = report_fixture(owned_root, 1)
    second = report_fixture(owned_root, 2)
    mapping = {"elements": "posterior_elements", "tokens": "tokens", "T_out": "time_steps",
               "decoded_token_count": "decoded_tokens", "frames": "frames"}
    for clip in second["clips"]:
        destination = clip["pixels"] if field == "frames" else clip["comparison"]
        destination[field] += 1
        if field == "frames":
            clip["pixels"]["pixel_channels"] += 256 * 256 * 3
    second[mapping[field]] += len(second["clips"])
    if field == "frames":
        second["pixel_channels"] += len(second["clips"]) * 256 * 256 * 3
    with pytest.raises(audit.AuditError, match="denominator"):
        audit.compare_runs(report_record(first), overwrite_fixture_report(second))


def test_tiny_continuous_drift_is_not_a_deterministic_repeat(owned_root):
    first = report_fixture(owned_root, 1)
    second = report_fixture(owned_root, 2)
    path = Path(second["clips"][0]["output"]["path"])
    with np.load(path, allow_pickle=False) as source:
        values = {name: source[name] for name in source.files}
    values["log_probs"][0, 0] += np.float32(2e-6)
    path.chmod(0o644)
    with path.open("wb") as handle:
        np.savez(handle, **values)
    second["clips"][0]["output"] = audit.file_record(path)
    second_record = overwrite_fixture_report(second)
    with pytest.raises(audit.AuditError, match="scientific comparison|deterministic.*drift"):
        audit.compare_runs(report_record(first), second_record)
    with pytest.raises(audit.AuditError, match="only.*deterministic"):
        audit.compare_runs(report_record(first), second_record, mode="historical_numerical")


def test_repeat_rejects_disjoint_names_with_shared_posterior_inode(owned_root):
    first = report_fixture(owned_root, 1)
    second = report_fixture(owned_root, 2)
    a = Path(first["clips"][0]["output"]["path"])
    b = Path(second["clips"][0]["output"]["path"])
    b.unlink()
    os.link(a, b)
    with pytest.raises(audit.AuditError, match="overlapping/aliased"):
        audit.compare_runs(report_record(first), report_record(second))


def test_postgate_external_source_mutation_cannot_be_freshly_preregistered(owned_root, monkeypatch):
    output = owned_root / "raw_repro_synthetic_gate_v1"
    copy_path = output / "source_fixture" / "resnet.py"
    copy_path.parent.mkdir(parents=True)
    copy_path.write_bytes(audit.SOURCE_PATHS["resnet"].read_bytes())
    paths = dict(audit.SOURCE_PATHS, resnet=copy_path)
    monkeypatch.setattr(audit, "SOURCE_PATHS", paths)
    junit = output / "tests.xml"
    audit.write_bytes_new(junit, b'<testsuite tests="1"><testcase name="source-mutation-fixture"/></testsuite>')
    gate = audit.synthetic_gate(output, junit, fixture_calibration(owned_root))
    original_core = json.loads((output / "gate.json").read_text())["core_ast_sha256"]
    copy_path.write_bytes(copy_path.read_bytes() + b"\n# mutation after the synthetic gate\n")
    assert audit.inference_core_hash(paths["corrnet_cache"]) == original_core
    with pytest.raises(audit.AuditError, match="tested source closure drift"):
        audit.validate_gate(output / "gate.json", gate["gate_sha256"])
    prereg = owned_root / "raw_repro_prereg_v1"
    with pytest.raises(audit.AuditError, match="tested source closure drift"):
        audit.preregister(prereg, output / "gate.json", gate["gate_sha256"], device="cpu")
    assert not prereg.exists()


def test_same_version_binary_swap_even_with_restored_mtime_is_detected(tmp_path):
    path = tmp_path / "numerical_extension.so"
    path.write_bytes(b"original")
    before = path.stat()
    first = audit.runtime_file_record(path)
    path.write_bytes(b"replaced")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    second = audit.runtime_file_record(path)
    assert first["bytes"] == second["bytes"] and first["sha256"] != second["sha256"]
    with pytest.raises(audit.AuditError, match="runtime drift"):
        audit.validate_runtime({"version": "same", "binary": first}, {"version": "same", "binary": second})


def test_actual_backend_and_environment_drift_are_not_silently_repaired(monkeypatch):
    import torch
    audit.configure_runtime("cpu")
    torch.use_deterministic_algorithms(False)
    with pytest.raises(audit.AuditError, match="actual backend"):
        audit.runtime_fingerprint("cpu")
    assert torch.are_deterministic_algorithms_enabled() is False
    audit.configure_runtime("cpu")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(audit.AuditError, match="thread environment drift"):
        audit.runtime_fingerprint("cpu")


def test_numerical_closure_contains_real_loaded_extensions_and_hashes():
    closure = audit.numerical_binary_closure()
    paths = {record["path"] for record in closure["files"]}
    assert set(closure["loaded_extension_roots"]) <= paths
    assert set(closure["native_dependencies"]) <= paths
    assert closure["sha256"] == audit.object_hash(closure["files"])
    assert closure["distributions"]["numpy"] and closure["distributions"]["torch"]
    assert all(len(record["sha256"]) == 64 for record in closure["files"])
    assert set(closure["required_extensions"]) == {"_hashlib", "_json", "resource"}
    assert set(closure["required_extensions"].values()) <= paths
    assert set(closure["observations"]["loaded_module_files"]) <= paths
    assert set(closure["observations"]["executable_mappings"]) <= paths
    assert str(Path(sys.executable).resolve()) in paths
    assert closure["standard_library_files"]
    assert closure["inventory_sha256"] == audit.object_hash(closure["aliases"])
    assert all(set(r["identity"]) == {"device", "inode", "mode"} for r in closure["files"])
    assert "not fully hermetic" in closure["policy"]["claim"]
    for mapping in closure["observations"]["mapped_file_identities"].values():
        if mapping["virtualization_discrepancy"]:
            exception = audit.RUNTIME_BOUNDARY["mapping_device_exception"]
            assert mapping["path"] == exception["path"]
            assert mapping["mapped"]["device"] == os.makedev(*exception["mapped_device"])
            assert mapping["filesystem"]["device"] == os.makedev(*exception["filesystem_device"])
            assert mapping["mapped"]["inode"] == mapping["filesystem"]["inode"]


@pytest.fixture
def tiny_runtime_boundary(tmp_path, monkeypatch):
    """Exercise real capture/validation with a bounded fixture dependency inventory.

    The separate integration test above hashes the actual full installation.
    Only selectors/dependency resolution are scoped here; module discovery,
    alias capture, fresh file hashing and runtime comparison stay real.
    """
    audit.prepare_runtime_capture()
    original_modules = audit.module_file_inventory
    required = original_modules()[1]
    required_paths = {Path(p) for p in required.values()}
    def fixture_modules():
        paths, required_now = original_modules()
        return {p for p in paths if p.resolve() in required_paths or p.is_relative_to(tmp_path)}, required_now
    executable = Path(sys.executable).resolve()
    identity = {"device": executable.stat().st_dev, "inode": executable.stat().st_ino}
    maps = {str(executable): {"path": str(executable), "mapped": dict(identity), "filesystem": dict(identity),
                              "virtualization_discrepancy": False}}
    monkeypatch.setattr(audit, "RUNTIME_DISTRIBUTIONS", ())
    monkeypatch.setattr(audit, "SOURCE_PATHS", {})
    monkeypatch.setattr(audit, "standard_library_inventory", lambda: ([], set()))
    monkeypatch.setattr(audit, "numerical_extension_roots", lambda: sorted(required_paths))
    monkeypatch.setattr(audit, "module_file_inventory", fixture_modules)
    monkeypatch.setattr(audit, "executable_mapping_records", lambda: copy.deepcopy(maps))
    monkeypatch.setattr(audit, "native_dependencies", lambda roots: (set(roots), {}, executable))
    return maps


def import_safe_fixture(path, monkeypatch, name="raw_repro_v4_safe_fixture"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_late_safe_fixture_module_is_rejected(tiny_runtime_boundary, tmp_path, monkeypatch):
    before = audit.runtime_fingerprint()
    path = tmp_path / "late_fixture.py"
    path.write_text("VALUE = 1\n")
    assert import_safe_fixture(path, monkeypatch).VALUE == 1
    after = audit.runtime_fingerprint()
    assert str(path) in after["numerical_binary_closure"]["observations"]["loaded_module_files"]
    with pytest.raises(audit.AuditError, match="runtime drift"):
        audit.validate_runtime(before, after)


def test_captured_source_same_size_restored_mtime_is_rejected(tiny_runtime_boundary, tmp_path, monkeypatch):
    path = tmp_path / "captured_fixture.py"
    path.write_text("VALUE = 1\n")
    import_safe_fixture(path, monkeypatch)
    original_stat = path.stat()
    before = audit.runtime_fingerprint()
    path.write_text("VALUE = 2\n")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert path.stat().st_size == original_stat.st_size and path.stat().st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(audit.AuditError, match="runtime drift"):
        audit.validate_runtime(before, audit.runtime_fingerprint())


def test_unregistered_mapped_library_is_rejected(tiny_runtime_boundary, tmp_path):
    before = audit.runtime_fingerprint()
    path = tmp_path / "unregistered_library.so"
    path.write_bytes(b"safe simulated library fixture")
    identity = {"device": path.stat().st_dev, "inode": path.stat().st_ino}
    tiny_runtime_boundary[str(path)] = {"path": str(path), "mapped": dict(identity), "filesystem": dict(identity),
                                       "virtualization_discrepancy": False}
    with pytest.raises(audit.AuditError, match="runtime drift"):
        audit.validate_runtime(before, audit.runtime_fingerprint())
    # A forged observation cannot smuggle a mapping past an unchanged manifest.
    forged = copy.deepcopy(before)
    forged["numerical_binary_closure"]["observations"]["executable_mappings"].append(str(path))
    with pytest.raises(audit.AuditError, match="unregistered"):
        audit.validate_runtime(before, forged)


def test_captured_module_disappearance_is_rejected(tiny_runtime_boundary, tmp_path, monkeypatch):
    path = tmp_path / "disappearing_fixture.py"
    path.write_text("VALUE = 1\n")
    import_safe_fixture(path, monkeypatch)
    before = audit.runtime_fingerprint()
    monkeypatch.delitem(sys.modules, "raw_repro_v4_safe_fixture")
    with pytest.raises(audit.AuditError, match="runtime drift"):
        audit.validate_runtime(before, audit.runtime_fingerprint())


def test_runtime_symlink_target_is_bound(tiny_runtime_boundary, tmp_path, monkeypatch):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text("VALUE = 1\n")
    second.write_text("VALUE = 1\n")
    link = tmp_path / "alias.py"
    link.symlink_to(first)
    import_safe_fixture(link, monkeypatch)
    before = audit.runtime_fingerprint()
    record = next(r for r in before["numerical_binary_closure"]["aliases"] if r["path"] == str(link))
    assert record["canonical_path"] == str(first) and record["symlinks"] == [{"path": str(link), "target": str(first)}]
    link.unlink()
    link.symlink_to(second)
    with pytest.raises(audit.AuditError, match="runtime drift"):
        audit.validate_runtime(before, audit.runtime_fingerprint())


def test_unexplained_mapped_device_identity_mismatch_fails(tmp_path, monkeypatch):
    path = tmp_path / "not_the_pinned_wsl_exception.so"
    path.write_bytes(b"safe fixture")
    disk = path.stat()
    original = Path.read_text
    line = f"1000-2000 r-xp 00000000 ff:ff {disk.st_ino} {path}\n"
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: line if str(self) == "/proc/self/maps" else original(self, *a, **k))
    with pytest.raises(audit.AuditError, match="unregistered mapped/filesystem device"):
        audit.executable_mapping_records()


def test_expected_mapped_identity_cannot_disappear_or_change(tiny_runtime_boundary):
    before = audit.runtime_fingerprint()
    changed = copy.deepcopy(before)
    observations = changed["numerical_binary_closure"]["observations"]
    observations["mapped_file_identities"] = {}
    observations["executable_mappings"] = []
    with pytest.raises(audit.AuditError, match="disappeared/changed"):
        audit.validate_runtime(before, changed)
    changed = copy.deepcopy(before)
    record = next(iter(changed["numerical_binary_closure"]["observations"]["mapped_file_identities"].values()))
    record["mapped"]["inode"] += 1
    with pytest.raises(audit.AuditError, match="identity drift"):
        audit.validate_runtime(before, changed)


def test_predeclared_immutable_module_can_load_without_changing_inventory(tiny_runtime_boundary, tmp_path, monkeypatch):
    path = tmp_path / "predeclared.py"
    path.write_text("VALUE = 1\n")
    monkeypatch.setattr(audit, "standard_library_inventory", lambda: ([tmp_path], {path}))
    before = audit.runtime_fingerprint()
    import_safe_fixture(path, monkeypatch)
    after = audit.runtime_fingerprint()
    audit.validate_runtime(before, after)
    assert before["numerical_binary_closure"]["files"] == after["numerical_binary_closure"]["files"]
    path.write_text("VALUE = 2\n")
    with pytest.raises(audit.AuditError, match="runtime drift"):
        audit.validate_runtime(before, audit.runtime_fingerprint())


def test_runtime_alias_binds_transitive_symlink_targets(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n")
    middle = tmp_path / "middle.py"
    middle.symlink_to(target)
    outer = tmp_path / "outer.py"
    outer.symlink_to(middle)
    before = audit.runtime_alias(outer)
    assert {r["path"] for r in before["symlinks"]} == {str(middle), str(outer)}
    additional = tmp_path / "additional.py"
    additional.symlink_to(target)
    middle.unlink()
    middle.symlink_to(additional)
    after = audit.runtime_alias(outer)
    assert before["canonical_path"] == after["canonical_path"]
    assert before != after


def test_unresolved_elf_dependency_cannot_be_silently_omitted(tmp_path, monkeypatch):
    path = tmp_path / "synthetic_elf.so"
    path.write_bytes(b"\x7fELFfixture")
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="unregistered_dependency.so => not found\n", stderr="", returncode=0))
    with pytest.raises(audit.AuditError, match="unresolved/ambiguous native dependency"):
        audit.native_dependencies([path])


@pytest.mark.parametrize("steps,passes", [(1, True), (2, False), (3, False)])
def test_strict_per_frame_margin_boundary_is_not_a_tolerance(steps, passes):
    # Float32 ULPs around -log(2) make the strict equality boundary representable.
    base = np.float32(-np.log(2))
    historical = {"log_probs": np.array([[base, base + np.float32(2**-22)]], np.float32),
                  "gloss_indices": [1], "T_out": 1}
    current = copy.deepcopy(historical)
    current["log_probs"] += np.float32(steps * 2**-24)
    row, spans = {"id": "boundary", "gloss_indices": [1]}, [[0, 1]]
    diagnostic = audit.decision_diagnostics(current, historical, row, 2, spans, spans)
    assert diagnostic["framewise_argmax_exact"] and diagnostic["decoded_tokens_exact"] and diagnostic["ctc_spans_exact"]
    assert diagnostic["strict_margin_certified"] == [passes]
    if steps == 2:
        assert 2 * diagnostic["per_frame_max_abs_error"][0] == diagnostic["historical_log_margins"][0]
    if passes:
        audit.compare_entries(current, historical, row, 2, spans, spans)
    else:
        with pytest.raises(audit.AuditError, match="strict per-frame margin"):
            audit.compare_entries(current, historical, row, 2, spans, spans)


def test_equal_posteriors_with_a_historical_top_tie_are_rejected():
    entry = {"log_probs": np.full((1, 2), -np.log(2), dtype=np.float32), "T_out": 1, "gloss_indices": [1]}
    with pytest.raises(audit.AuditError, match="ties/equality"):
        audit.compare_entries(entry, entry, {"id": "tie", "gloss_indices": [1]}, 2, [[0, 1]], [[0, 1]])


@pytest.mark.parametrize("field,match", [("framewise_argmax_exact", "argmax"),
                                         ("decoded_tokens_exact", "decoded token"), ("ctc_spans_exact", "span")])
def test_each_exact_decision_gate_is_required_even_with_a_true_certificate(field, match):
    entry, row, spans = entry_fixture()
    diagnostic = audit.decision_diagnostics(entry, entry, row, 3, spans, spans)
    assert all(diagnostic["strict_margin_certified"])
    # The theorem implies argmax equality; contradictory reported decisions must
    # nevertheless never override any of the separate exact-decision gates.
    diagnostic[field] = False
    with pytest.raises(audit.AuditError, match=match):
        audit.require_decision_equivalence(diagnostic)


def test_actual_frame_argmax_mismatch_is_rejected():
    historical, row, spans = entry_fixture()
    current = copy.deepcopy(historical)
    current["log_probs"][0] = current["log_probs"][0, [0, 2, 1]]
    with pytest.raises(audit.AuditError, match="framewise argmax"):
        audit.compare_entries(current, historical, row, 3, spans, spans)


def test_margin_certificate_is_all_frames_not_an_average():
    historical, row, spans = entry_fixture()
    current = copy.deepcopy(historical)
    current["log_probs"][1, 0] -= np.float32(5)
    current["log_probs"][1] -= np.float32(np.log(np.exp(current["log_probs"][1].astype(np.float64)).sum()))
    diagnostic = audit.decision_diagnostics(current, historical, row, 3, spans, spans)
    assert diagnostic["strict_margin_certified"] == [True, False]
    with pytest.raises(audit.AuditError, match="strict per-frame"):
        audit.require_decision_equivalence(diagnostic)


def test_v6_claim_and_policy_exclude_continuous_and_backend_reproduction():
    assert "DECISION-equivalent" in audit.CLAIM_LIMIT["scope"]
    assert audit.CLAIM_LIMIT["historical_continuous_posteriors_reproduced"] is False
    assert audit.CLAIM_LIMIT["historical_backend_reproduced"] is False
    assert audit.CLAIM_LIMIT["calibration_is_validation"] is False
    assert "historical continuous posteriors" in audit.CLAIM_LIMIT["not_reproduced"]
    assert "historical backend" in audit.CLAIM_LIMIT["not_reproduced"]
    assert "log_probs_atol" not in audit.POLICY and "log_probs_rtol" not in audit.POLICY
    entry, row, spans = entry_fixture()
    with pytest.raises(audit.AuditError, match="numeric tolerance cannot accept"):
        audit.compare_entries(entry, entry, row, 3, spans, spans, mode="historical_numerical")


def test_real_frozen_selection_excludes_calibration_and_full_census_separates_it():
    prior = audit.v5_preregistration()  # Metadata/hash reads only; no cache loading.
    selected, partition = audit.selection_partition(prior["training_universe"], "pilot")
    assert [r["id"] for r in selected] == prior["selected_ids"][1:] == list(audit.FROZEN_PILOT_IDS[1:])
    assert len(selected) == 9 and sum(r["length"] for r in selected) == 1037
    assert len({r["signer"] for r in selected}) == 8 and {r["length_tertile"] for r in selected} == {0, 1, 2}
    assert partition["calibration_id"] not in partition["execution_roles"]
    census, full = audit.selection_partition(prior["training_universe"], "full")
    assert len(census) == 7096 and full["execution_roles"][audit.CALIBRATION_ID] == "CALIBRATION"
    assert list(full["execution_roles"].values()).count("NONCALIBRATION_CENSUS") == 7095
    assert "VALIDATION" not in full["execution_roles"].values()


@pytest.mark.parametrize("mutation", [
    lambda d: d["partition"].update(calibration_is_validation=True),
    lambda d: d["partition"].update(replacement_clips_allowed=True),
    lambda d: d["clips"][0].update(role="CALIBRATION"),
    lambda d: d["selected_ids"].__setitem__(0, d["partition"]["calibration_id"]),
    lambda d: d["partition"]["frozen_validation_ids"].reverse(),
])
def test_validation_partition_cannot_be_coherently_resealed(owned_root, mutation):
    path, _ = sealed_fixture(owned_root)
    target = path / "preregistration.json"
    doc = json.loads(target.read_text())
    mutation(doc)
    target.chmod(0o644)
    target.write_bytes(audit.canonical(doc))
    seal = path / "seal.json"
    seal.chmod(0o644)
    seal.write_bytes(audit.canonical({"schema": audit.SCHEMA, "preregistration": audit.file_record(target)}))
    with pytest.raises(audit.AuditError, match="partition drift|selection drift"):
        audit.load_prereg(path, audit.sha256(seal), check_inputs=False)


@pytest.mark.parametrize("clock", ["unix", "monotonic"])
@pytest.mark.parametrize("time_value", [100, 140, 150])
def test_same_overlapping_or_touching_run_times_are_rejected(owned_root, clock, time_value):
    first, second = report_fixture(owned_root, 1), report_fixture(owned_root, 2)
    path = Path(second["start"]["path"])
    start = json.loads(path.read_text())
    start[f"started_{clock}_ns"] = time_value
    path.chmod(0o644)
    path.write_bytes(audit.canonical(start))
    second["start"] = audit.file_record(path)
    with pytest.raises(audit.AuditError, match="same/overlapping run time"):
        audit.compare_runs(report_record(first), overwrite_fixture_report(second))


def test_hardlinked_scientific_reports_cannot_prove_distinct_runs(owned_root):
    first, second = report_fixture(owned_root, 1), report_fixture(owned_root, 2)
    path = Path(second["scientific_report"]["path"])
    path.unlink()
    os.link(first["scientific_report"]["path"], path)
    with pytest.raises(audit.AuditError, match="hardlinked scientific"):
        audit.compare_runs(report_record(first), report_record(second))


def test_coherently_reported_tiny_tensor_drift_still_fails_exact_repeat(owned_root):
    first, second = report_fixture(owned_root, 1), report_fixture(owned_root, 2)
    clip = second["clips"][0]
    current, spans = audit.load_posterior(clip["output"])
    historical, old_spans = audit.load_posterior(clip["historical"])
    current["log_probs"][0, 0] += np.float32(2e-6)
    path = Path(clip["output"]["path"])
    path.unlink()
    clip["output"] = audit.save_posterior(path, current, spans)
    doc = audit.load_prereg(Path(second["preregistration_seal"]["path"]).parent,
                           second["preregistration_seal_sha256"], check_inputs=False)
    clip["comparison"] = audit.compare_entries(current, historical, doc["clips"][0]["metadata"], 3, spans, old_spans)
    diagnostic_path = Path(clip["diagnostics"]["path"])
    diagnostic_path.chmod(0o644)
    diagnostic_path.write_bytes(audit.canonical(clip["comparison"]) + b"\n")
    clip["diagnostics"] = audit.file_record(diagnostic_path)
    second["exact_continuous_clips"] = 8
    counts = {key: second[key] for key in ("frames", "pixel_channels", "posterior_elements", "tokens", "time_steps", "decoded_tokens")}
    science = Path(second["scientific_report"]["path"])
    science.chmod(0o644)
    science.write_bytes(audit.canonical(audit.scientific_document(doc, second["preregistration_seal_sha256"], second["clips"], counts)) + b"\n")
    second["scientific_report"] = audit.file_record(science)
    audit.validate_run_report(second)  # Both individual runs pass semantics alone.
    with pytest.raises(audit.AuditError, match="deterministic scientific report mismatch"):
        audit.compare_runs(report_record(first), overwrite_fixture_report(second))


def test_continuous_report_tampering_is_reconstructed_from_arrays(owned_root):
    report = report_fixture(owned_root)
    report["clips"][0]["comparison"]["absolute_error_distribution"]["mean"] = 0.01
    with pytest.raises(audit.AuditError, match="scientific comparison"):
        audit.validate_run_report(report)


def test_historical_reference_array_cannot_be_replaced(owned_root):
    report = report_fixture(owned_root)
    clip = report["clips"][0]
    historical, spans = audit.load_posterior(clip["historical"])
    historical["log_probs"][0, 0] += np.float32(2e-6)
    path = Path(clip["historical"]["path"])
    path.unlink()
    clip["historical"] = audit.save_posterior(path, historical, spans)
    with pytest.raises(audit.AuditError, match="pinned cache commitment"):
        audit.validate_run_report(report)


def test_full_scope_requires_independent_pilot_and_exact_repeat_review():
    with pytest.raises(audit.AuditError, match="pinned pilot, exact repeat, and independent review"):
        audit.validate_pilot_review(None)


def test_v5_failed_calibration_lineage_is_content_bound():
    audit.validate_revision_lineage()
    records = audit.REVISION["superseded_evidence"]
    failure = audit.read_bound(records["v5_pilot_failure"])
    assert failure["active_output"] == records["v5_pilot_output"] and failure["completed_clips"] == 0
    assert audit.read_bound(records["v5_execution_receipt"])["returncode"] == 1
    assert "BLOCKED" in audit.REVISION["v5_review_status"]


def test_calibration_cache_read_requires_explicit_authorization(tmp_path):
    with pytest.raises(audit.AuditError, match="heavy-slot authorization"):
        audit.derive_calibration(tmp_path)


def test_calibration_derivation_reads_cache_once_recomputes_and_seals_metrics(tmp_path, monkeypatch):
    # Real derivation/validation functions with a tiny deduplicated fixture cache;
    # the production 1.08 GB archive is never loaded by this unit test.
    monkeypatch.setattr(audit, "OUT", tmp_path / "revision")
    raw, cached, row = frame_fixture(tmp_path, count=8)
    monkeypatch.setattr(audit, "CALIBRATION_ID", row["id"])
    output = audit.OUT / "raw_repro_synthetic_gate_v6"
    output.mkdir(parents=True)
    past = audit.OUT / "raw_repro_pilot_v5"
    historical, _, spans = entry_fixture()
    current = copy.deepcopy(historical)
    current["log_probs"][0, 0] += np.float32(1e-3)
    actual = audit.save_posterior(past / "posteriors" / f"{row['id']}.npz", current, spans)
    delta = float(np.abs(current["log_probs"].astype(np.float64) - historical["log_probs"].astype(np.float64)).max())
    audit.write_new(past / "failure.json", {"status": "FAIL", "completed_clips": 0,
                    "active_clip": row["id"], "active_output": actual,
                    "error": "posterior tolerance failure; max_abs=" + str(delta)})
    ids = [row["id"]] + [f"fixture{i}" for i in range(7095)]
    cache = {"train/" + sid: historical for sid in ids}
    audit.write_bytes_new(past / "historical.pkl", audit.pickle.dumps(cache))
    audit.write_new(past / "frames.json", audit.frame_inventory(raw, cached, row))
    audit.write_new(past / "prior_seal.json", {"fixture": True})
    revision = copy.deepcopy(audit.REVISION)
    revision["superseded_evidence"].update(v5_pilot_output=actual,
        v5_pilot_failure=audit.file_record(past / "failure.json"), v5_prereg_seal=audit.file_record(past / "prior_seal.json"))
    monkeypatch.setattr(audit, "REVISION", revision)
    prior = {"clips": [{"metadata": row, "frames": audit.file_record(past / "frames.json")}],
             "assets": {"posterior_cache": audit.file_record(past / "historical.pkl")},
             "training_universe": [{"id": sid} for sid in ids],
             "roots": {"raw_root": str(raw), "cache_root": str(cached)},
             "policy": {"log_probs_atol": 1e-5, "log_probs_rtol": 0.0}}
    monkeypatch.setattr(audit, "v5_preregistration", lambda: prior)
    monkeypatch.setattr(audit, "resources", lambda *a, **k: {"peak_rss_bytes": 0, "available_ram_bytes": 20 * 1024**3})
    monkeypatch.setattr(audit, "load_corrnet_source", lambda *a, **k: pytest.fail("calibration cannot load/forward a model"))
    original_load, calls = audit.pickle.load, []
    def counted_load(handle):
        calls.append(handle.name)
        return original_load(handle)
    monkeypatch.setattr(audit.pickle, "load", counted_load)
    result = audit.derive_calibration(output, authorize_cache_read=True)
    measured = audit.validate_calibration(result["calibration"])
    assert len(calls) == 1 and result["new_inference"] is False
    assert measured["measurement"]["role"] == "CALIBRATION" and measured["measurement"]["is_validation"] is False
    assert measured["measurement"]["historical_v5_continuous_gate"]["passed"] is False
    assert measured["measurement"]["historical_v5_continuous_gate"]["max_abs_error"] == delta
    assert measured["measurement"]["pixels"]["frames"] == 8
    assert len(audit.read_bound(measured["historical_commitments"])["entries"]) == 7096
    changed = copy.deepcopy(measured)
    changed["measurement"]["role"] = "VALIDATION"
    audit.write_new(output / "bad_role.json", changed)
    with pytest.raises(audit.AuditError, match="metric or role drift"):
        audit.validate_calibration(audit.file_record(output / "bad_role.json"))
    changed = copy.deepcopy(measured)
    commits = audit.read_bound(measured["historical_commitments"])
    commits["entries"][row["id"]] = "0" * 64
    audit.write_new(output / "bad_commitments.json", commits)
    changed["historical_commitments"] = audit.file_record(output / "bad_commitments.json")
    audit.write_new(output / "bad_reference.json", changed)
    with pytest.raises(audit.AuditError, match="historical tensor drift"):
        audit.validate_calibration(audit.file_record(output / "bad_reference.json"))
    with pytest.raises(audit.AuditError, match="never overwrite"):
        audit.derive_calibration(output, authorize_cache_read=True)


@pytest.mark.parametrize("mutation", [lambda e: e.update(T_out=2.0), lambda e: e.update(gloss_indices=[1.0, 2])])
def test_cache_tensor_commitment_cannot_hide_discrete_type_coercion(mutation):
    entry, _, _ = entry_fixture()
    mutation(entry)
    with pytest.raises(audit.AuditError, match="cannot coerce"):
        audit.entry_commitment(entry)


def test_nine_clip_fake_execution_and_sequential_exact_repeat_end_to_end(owned_root, monkeypatch):
    prereg, seal_hash = sealed_fixture(owned_root)
    doc = audit.load_prereg(prereg, seal_hash, check_inputs=False)
    entry, _, _ = entry_fixture()
    audit.write_bytes_new(prereg / "heavy.lock", b"")
    doc["lock"] = audit.file_record(prereg / "heavy.lock")
    vocab_buffer = audit.io.BytesIO()
    np.save(vocab_buffer, {"A": [1], "B": [2]}, allow_pickle=True)
    audit.write_bytes_new(prereg / "fixture_vocab.npy", vocab_buffer.getvalue())
    doc["assets"] = dict(doc["assets"], vocab=audit.file_record(prereg / "fixture_vocab.npy"))
    cache = {"train/" + row["id"]: entry for row in doc["training_universe"]}
    audit.write_bytes_new(prereg / "fixture_cache.pkl", audit.pickle.dumps(cache))
    doc["assets"]["posterior_cache"] = audit.file_record(prereg / "fixture_cache.pkl")
    doc["roots"], doc["gpu_diagnostics"] = {"raw_root": "fixture", "cache_root": "fixture"}, None
    monkeypatch.setattr(audit, "load_prereg", lambda *a, **k: doc)
    monkeypatch.setattr(audit, "resources", lambda *a, **k: {"peak_rss_bytes": 0, "available_ram_bytes": 20 * 1024**3})
    monkeypatch.setattr(audit, "verify_pixels", lambda frames, *a: {
        "frames": len(frames), "pixel_channels": len(frames) * 256 * 256 * 3, "exact": True})
    monkeypatch.setattr(audit, "frame_inventory", lambda raw, cached, row: [
        {"name": f"images{i:04d}.png"} for i in range(1, row["length"] + 1)])
    import torch
    class FakeModel:
        training = False
        def __call__(self, *args):
            logits = torch.tensor([[[0., 8., 0.]], [[0., 0., 8.]]], dtype=torch.float32)
            return {"sequence_logits": logits, "conv_logits": logits, "feat_len": torch.tensor([2.], dtype=torch.float32)}
    def infer_core(model, vocab, gloss, video):
        ret = model(video, torch.tensor([8]))
        return {"log_probs": torch.log_softmax(ret["sequence_logits"], dim=-1)[:, 0].numpy(),
                "T_out": int(ret["feat_len"][0].item()), "gloss_indices": [1, 2]}
    source = SimpleNamespace(torch=torch, core_ast_sha256=doc["core_ast_sha256"], infer_core=infer_core,
              build_corrnet_model=lambda *a: FakeModel(),
              load_and_preprocess_video=lambda *a: (torch.zeros((8, 3, 224, 224), dtype=torch.float32), 8),
              ctc_forced_align=lambda array, target, **k: audit.strict_ctc_spans(array, target))
    monkeypatch.setattr(audit, "load_corrnet_source", lambda *a: source)
    first_path, second_path = owned_root / "raw_repro_pilot_v100", owned_root / "raw_repro_repeat_v101"
    first = audit.execute(prereg, seal_hash, first_path, "pilot", authorize_inference=True)
    second = audit.execute(prereg, seal_hash, second_path, "repeat", authorize_inference=True,
                           reference_report=first_path / "report.json", reference_hash=first["report_sha256"])
    assert first["status"] == second["status"] == "PASS"
    comparison = json.loads((second_path / "repeat_comparison.json").read_text())
    assert comparison["validation_closure_verified"] and comparison["clips"] == 9
    assert (first_path / "scientific.json").read_bytes() == (second_path / "scientific.json").read_bytes()
