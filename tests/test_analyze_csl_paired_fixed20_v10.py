import hashlib
from types import SimpleNamespace

import pytest

from scripts import analyze_csl_paired_fixed20_v10 as analysis


def _row(reference: str, hypothesis: str) -> dict:
    return {"ref": reference, **{head: hypothesis for head in analysis.HEADS}}


def test_checked_json_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"same": 1, "same": 2}\n', encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(analysis.ProtocolError, match="duplicate JSON key"):
        analysis.checked_json(path, expected)


def test_extract_definitions_rejects_duplicate_authority(tmp_path):
    source = tmp_path / "duplicate.py"
    source.write_text("def target():\n    return 1\n\ndef target():\n    return 2\n")
    with pytest.raises(analysis.ProtocolError, match="missing or duplicated"):
        analysis.extract_definitions(source, {"target"}, {})


def test_hypothesis_coverage_rejects_missing_head():
    sample_id = "sample"
    complete = _row("a b", "a")
    missing = dict(complete)
    missing.pop("right_hyp")
    with pytest.raises(analysis.ProtocolError, match="result schema differs"):
        analysis.hypothesis_coverage(
            [sample_id], {"local": {sample_id: missing},
                          "fixed40": {sample_id: complete}}
        )


def test_recovered_preregistered_bootstrap_preserves_delta_sign():
    _, core, methods, _, runtime = analysis.load_preregistered_methods()
    assert runtime == analysis.EXPECTED_NUMERICAL_RUNTIME
    ids = ("a", "b")
    local = {sample_id: _row("one two", "wrong tokens") for sample_id in ids}
    fixed = {sample_id: _row("one two", "one two") for sample_id in ids}
    groups = {"a": "group-a", "b": "group-b"}
    result = methods["clustered_uncertainty"](
        local, fixed, groups, replicates=32, seed=30373
    )
    for head in core.HEADS:
        row = result["heads"][head]
        assert row["delta_corpus_wer"] == -100.0
        assert row["ci95"] == [-100.0, -100.0]
        assert row["improvement_gate"] == "PASS"


def test_numerical_runtime_rejects_version_mismatch():
    with pytest.raises(analysis.ProtocolError, match="runtime differs"):
        analysis.numerical_runtime(SimpleNamespace(__version__="0.0.0"))


def test_same_process_repeat_is_rejected():
    identity = analysis.process_identity()
    with pytest.raises(analysis.ProtocolError, match="distinct process"):
        analysis.validate_distinct_process(identity, dict(identity))


def test_repeat_runtime_mismatch_is_rejected():
    first = {"runtime": {"numerical": analysis.EXPECTED_NUMERICAL_RUNTIME}}
    mismatched = {**analysis.EXPECTED_NUMERICAL_RUNTIME, "numpy": "0.0.0"}
    with pytest.raises(analysis.ProtocolError, match="runtime differs"):
        analysis.validate_repeat_runtime(first, mismatched)


def test_geometry_retains_per_id_source_diagnostics(monkeypatch):
    import numpy as np

    ids = ["a", "b"]
    poses = {"a": [0, 1], "b": [0, 1, 2]}

    def fake_load_pose(_np, _native_pose, _path, _file_sha, expected_pose):
        return poses[expected_pose["id"]]

    def fake_geometry(value, _scale):
        frames = value["frames"] if isinstance(value, dict) else len(value)
        return {
            "frames": frames,
            "mska_retained_frames": frames,
            "hand_displacement_px_per_frame": float(frames),
            "hand_temporal_variance_px2": float(frames * 2),
        }

    monkeypatch.setattr(analysis, "load_pose", fake_load_pose)
    commitments = {sid: {"id": sid, "sha256": f"hash-{sid}"}
                   for sid in ids}
    jobs = {
        "local": {"pose_commitments": commitments},
        "fixed40_first20": {"pose_commitments": commitments},
    }
    certificates = {
        sid: {
            route: {"payload": {
                "id": sid,
                "route": route,
                "pose": commitments[sid],
                "context": {
                    "implementation_sha256": analysis.PINS[analysis.METHOD_SOURCE],
                    "prereg_sha256": analysis.PINS[analysis.PREREG],
                },
                "source_quarter_frames": {f"train-{sid}": 4 * len(poses[sid])},
                "quarter_frames_total": 4 * len(poses[sid]),
            }}
            for route in ("local", "fixed40")
        }
        for sid in ids
    }
    stage = {"outputs": {
        f"{folder}/{hashlib.sha256(sid.encode()).hexdigest()}.npy": "file-hash"
        for folder in ("local", "fixed40") for sid in ids
    }}
    result = analysis.source_and_geometry_summary(
        np,
        {"native_pose": object(), "pose_geometry": fake_geometry},
        ids,
        jobs,
        stage,
        certificates,
        {"admission": {"boundary_shoulder_scale_native_pixels": 1.0}},
        {sid: {"name": sid, "frames": len(poses[sid]) + 1} for sid in ids},
    )
    assert list(result["geometry_per_id"]) == ["ground_truth", "local", "fixed40"]
    assert list(result["geometry_per_id"]["local"]) == ids
    assert result["geometry_per_id"]["local"]["a"]["train_source_count"] == 1
    assert result["geometry_per_id"]["fixed40"]["b"][
        "output_gt_length_ratio"
    ] == 3 / 4
