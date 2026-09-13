"""Synthetic adversarial checks; no historical data or checkpoint is loaded."""
import builtins
import copy
import io
import json
import os
import pickle
import shutil
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import audit_recent_control_admission as m

REAL = m.ROOT
REAL_OUT = m.OUT


@pytest.fixture(scope="session", autouse=True)
def bind_sources(record_testsuite_property):
    record_testsuite_property("audit_implementation_sha256", m.digest(Path(m.__file__)))
    record_testsuite_property("audit_tests_sha256", m.digest(Path(__file__)))


@pytest.fixture(autouse=True)
def prevent_real_artifact_reads(monkeypatch):
    def wrap(original):
        def guarded(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, bytes, Path)) and ("r" in mode or "+" in mode):
                path = Path(file).resolve()
                banned = (REAL/"data", REAL/"outputs", REAL/"external")
                if any(path.is_relative_to(p) for p in banned) and not path.is_relative_to(REAL_OUT):
                    raise AssertionError("synthetic test accessed real artifact: "+str(path))
            return original(file, mode, *args, **kwargs)
        return guarded
    monkeypatch.setattr(builtins, "open", wrap(builtins.open))
    monkeypatch.setattr(io, "open", wrap(io.open))
    torch.set_num_threads(1)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "OUT", tmp_path/"owned")
    m.OUT.mkdir()
    return tmp_path


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_bytes(m.canonical(value))
    return path


def bank_fixture():
    expected = {"a": {"id": "a", "gloss": "ONE TWO"}, "b": {"id": "b", "gloss": "THREE"}}
    lengths = {"a": 2, "b": 3}
    bank = {sid: {"name": sid, "gloss": row["gloss"], "num_frames": lengths[sid],
                  "keypoint": torch.full((lengths[sid], 133, 3), .5)} for sid, row in expected.items()}
    return bank, expected, lengths


def checkpoint_fixture():
    ck = {"model": {"weight": torch.zeros(2, 3)}, "mean": torch.zeros(3), "std": torch.ones(3),
          "vocab": {"<pad>": 0, "ONE": 1}}
    log = [{"epoch": 1, "wall": .25, "val_pose": .3}, {"epoch": 2, "wall": .5, "val_pose": .2}]
    return ck, log


def test_bank_exact_ids_shapes_gloss_lengths_and_confidence():
    bank, expected, lengths = bank_fixture()
    result = m.bank_summary(bank, expected, lengths)
    assert result["structure_gloss_and_length_valid"]
    assert result["confidence"]["elements"] == 665
    assert result["confidence"]["valid_probability_channel"]
    assert result["frames"] == {"min": 2, "max": 3, "total": 5}


@pytest.mark.parametrize("defect", ("missing_id", "extra_id", "nan", "shape", "gloss", "name", "num_frames", "length"))
def test_bank_defects_fail_closed(defect):
    bank, expected, lengths = bank_fixture()
    if defect == "missing_id":
        del bank["a"]
    elif defect == "extra_id":
        bank["c"] = copy.deepcopy(bank["a"])
    elif defect == "nan":
        bank["a"]["keypoint"][0, 0, 0] = float("nan")
    elif defect == "shape":
        bank["a"]["keypoint"] = torch.zeros(2, 132, 3)
    elif defect == "gloss":
        bank["a"]["gloss"] = "WRONG"
    elif defect == "name":
        bank["a"]["name"] = "b"
    elif defect == "num_frames":
        bank["a"]["num_frames"] = 3
    else:
        lengths["a"] = 7
    result = m.bank_summary(bank, expected, lengths)
    assert not result["structure_gloss_and_length_valid"]


def test_confidence_uses_every_frame_joint_and_separates_nonfinite():
    bank, expected, lengths = bank_fixture()
    bank["a"]["keypoint"][0, 0, 2] = -.1
    bank["b"]["keypoint"][0, 0, 2] = 1.5
    bank["b"]["keypoint"][0, 1, 2] = float("nan")
    c = m.bank_summary(bank, expected, lengths)["confidence"]
    assert c["below_zero"] == c["above_one"] == c["nonfinite"] == 1
    assert c["outside_unit_interval"] == 2 and c["outside_fraction"] == 2/665
    assert c["finite_range"] == [float(np.float32(-.1)), 1.5]
    assert not c["valid_probability_channel"]


@pytest.mark.parametrize("defect", ("duplicate", "empty", "count", "gloss"))
def test_manifest_invalidity(defect):
    rows = [{"id": "a", "gloss": "ONE"}]
    if defect == "duplicate":
        rows *= 2
    elif defect == "empty":
        rows = []
    elif defect == "gloss":
        rows[0]["gloss"] = ""
    with pytest.raises(m.AuditError):
        m.manifest_rows(rows, 2 if defect == "count" else None)


def test_checkpoint_missing_metadata_is_explicit_and_log_is_not_epoch_binding():
    ck, log = checkpoint_fixture()
    result = m.checkpoint_summary(ck, log, "val_pose")
    assert result["missing_frozen_training_metadata"] == ["seed", "epoch", "training_ids_sha256", "code_sha256", "command", "environment"]
    assert not result["frozen_historical_training_provenance_complete"]
    assert result["training_log"]["best_recorded_epoch"] == 2
    assert result["training_log"]["recorded_epoch_wall_seconds"] == .75
    assert "inferred" in result["training_log"]["checkpoint_epoch_binding"]


@pytest.mark.parametrize("defect", ("weight_nan", "mean_nan", "std_zero", "log_nan", "epoch_gap", "vocab"))
def test_checkpoint_invalidities(defect):
    ck, log = checkpoint_fixture()
    if defect == "weight_nan":
        ck["model"]["weight"][0, 0] = float("nan")
    elif defect == "mean_nan":
        ck["mean"][0] = float("nan")
    elif defect == "std_zero":
        ck["std"][0] = 0
    elif defect == "log_nan":
        log[0]["val_pose"] = float("nan")
    elif defect == "epoch_gap":
        log[1]["epoch"] = 4
    else:
        ck["vocab"]["ONE"] = 0
    with pytest.raises(m.AuditError):
        m.checkpoint_summary(ck, log, "val_pose")


class SyntheticPT:
    def generate(self, tokens, mask, max_steps, stop_threshold):
        assert stop_threshold == 2.0
        x = torch.arange(max_steps).float()[None, :, None].repeat(len(tokens), 1, 3)
        return x, torch.full((len(tokens),), max_steps)


def linear_resample(x, target):
    return np.stack([np.interp(np.linspace(0, 1, target), np.linspace(0, 1, len(x)), x[:, i])
                     for i in range(x.shape[1])], axis=-1)


def test_pt_probe_distinguishes_prefix_order_and_export_horizon():
    tokens = torch.tensor([[1, 2], [2, 1]])
    result = m.pt_probe(SyntheticPT(), tokens, tokens > 0, linear_resample)
    assert result["prefix_at_longer_horizon"]["invariant"]
    assert result["same_horizon_reversed_order"]["invariant"]
    assert result["historical_export_horizon"]["max_abs"] == 8
    assert not result["historical_export_mixed_batch"]["invariant"]


class SyntheticVQ:
    K = 16
    @property
    def vq(self):
        return self
    def codes_to_vec(self, codes):
        return (codes.float()+1)[..., None]
    def decode(self, z):
        # A known finite receptive field spanning one future code.
        x = torch.nn.functional.pad(z.transpose(1, 2), (0, 1))
        return (x[..., :-1]+x[..., 1:]).transpose(1, 2).repeat_interleave(4, dim=1)


def test_vq_probe_detects_padding_without_blame_on_order():
    result = m.vq_probe(SyntheticVQ())
    assert result["prefix_with_code_zero_padding"]["max_abs"] == 1
    assert not result["mixed_batch_prefix"]["invariant"]
    assert result["same_padded_horizon_reversed_order"]["invariant"]


@pytest.mark.parametrize("bad", ("nan", "shape", "empty"))
def test_invariance_fails_on_unmeasurable_inputs(bad):
    x = torch.ones(2)
    y = torch.tensor([float("nan"), 1]) if bad == "nan" else torch.ones(3 if bad == "shape" else 0)
    with pytest.raises(m.AuditError):
        m.difference(x, y, 1e-5)


def test_named_local_admission_cannot_be_obtained_by_clean_arrays_or_metadata():
    banks = {"dev": {"structure_gloss_and_length_valid": True, "confidence": {"valid_probability_channel": True}}}
    ck = {"frozen_historical_training_provenance_complete": True}
    train = {"raw_equals_admitted_train": True}
    fidelity = {"named_method_fidelity": True, "ceiling": "Historical control only."}
    static = {"per_result_best_head": {"proved_in_bound_source": False},
              "unseeded_continuous_diffusion": {"proved_in_bound_local_path": False}}
    probes = {"pt": {"historical_export_horizon": {"invariant": True}},
              "vq": {"prefix_with_code_zero_padding": {"invariant": True}}}
    for tag in m.TAGS:
        row = m.admit_local(tag, banks, ck, train, fidelity, static, probes)
        assert row["status"] == "REJECTED"
        assert row["reasons"] == ["historical_architectural_control_not_named_reproduction"]


def test_all_observed_failure_reasons_are_retained():
    banks = {"test": {"structure_gloss_and_length_valid": False, "confidence": {"valid_probability_channel": False}}}
    row = m.admit_local("signidd", banks, {"frozen_historical_training_provenance_complete": False},
        {"raw_equals_admitted_train": False}, {"named_method_fidelity": False, "ceiling": "control only"},
        {"per_result_best_head": {"proved_in_bound_source": True},
         "unseeded_continuous_diffusion": {"proved_in_bound_local_path": True}}, {})
    assert len(row["reasons"]) == 8
    assert "generation_rng_seed_not_bound" in row["reasons"]


def test_source_hash_and_line_hash_detect_tampering(root):
    path = put(root/"source.py", b"def first():\n    return 1\n")
    rec = m.record(path)
    ev = m.symbol(path, "first")
    assert ev["start_line"] == 1 and ev["end_line"] == 2
    assert ev["lines_sha256"] == rec["sha256"]
    put(path, b"def first():\n    return 2\n")
    with pytest.raises(m.AuditError, match="changed"):
        m.verify_record(rec)


def test_path_confinement_includes_symlinks_and_no_overwrite(root):
    outside = put(root/"outside.json", {"data": 1})
    link = m.OUT/"escape"
    link.symlink_to(root, target_is_directory=True)
    for path in (m.OUT, root/"elsewhere.json", link/"outside.json"):
        with pytest.raises(m.AuditError, match="confined"):
            m.write_new(path, {})
    path = m.OUT/"new.json"
    m.write_new(path, {"good": True})
    with pytest.raises(FileExistsError):
        m.write_new(path, {"good": False})
    assert m.read_json(path) == {"good": True} and m.read_json(outside) == {"data": 1}
    directory = m.new_dir(m.OUT/"attempt")
    with pytest.raises(m.AuditError, match="immutable"):
        m.new_dir(directory)


def synthetic_design(root, monkeypatch):
    implementation = put(root/"implementation.py", b"# synthetic\n")
    payload = put(root/"input.json", {"data": 1})
    monkeypatch.setattr(m, "__file__", str(implementation))
    monkeypatch.setattr(m, "dependencies", lambda: {"implementation": implementation, "input": payload})
    design_dir = m.OUT/"design"
    design_hash = m.seal_design(design_dir)
    return design_dir/"design.json", design_hash, payload


def test_sealed_input_and_design_tampering_fail_before_science(root, monkeypatch):
    design_path, h, data = synthetic_design(root, monkeypatch)
    called = []
    monkeypatch.setattr(m, "science", lambda *a: called.append(True))
    with pytest.raises(m.AuditError, match="trusted design"):
        m.run_audit(design_path, "0"*64, m.OUT/"wrong_digest")
    put(data, {"data": 2})
    with pytest.raises(m.AuditError, match="changed"):
        m.run_audit(design_path, h, m.OUT/"tampered")
    assert not called and not (m.OUT/"tampered").exists()


def fake_science(design, h):
    return {"schema": m.SCHEMA, "design_sha256": h,
            "rows": [{"id": tag, "status": "REJECTED", "reasons": ["fixture"], "claim_ceiling": "synthetic only"}
                     for tag in m.TAGS] + [{"id": "organizer_progressive_transformer_release", "status": "ADMITTED",
                                           "reasons": [], "claim_ceiling": "synthetic release only",
                                           "external_evidence": {"complete": True}}]}


def test_two_runs_have_identical_science_and_detect_output_tamper(root, monkeypatch):
    design_path, h, _ = synthetic_design(root, monkeypatch)
    monkeypatch.setattr(m, "science", fake_science)
    monkeypatch.setattr(m, "available_memory_kib", lambda: m.MIN_AVAILABLE_KIB+1)
    a, b = m.OUT/"run1", m.OUT/"run2"
    m.run_audit(design_path, h, a)
    m.run_audit(design_path, h, b)
    assert (a/"scientific.json").read_bytes() == (b/"scientific.json").read_bytes()
    first, second = m.verify_run(design_path, h, a), m.verify_run(design_path, h, b)
    assert first["scientific_sha256"] == second["scientific_sha256"]
    assert first["execution"]["run_id"] != second["execution"]["run_id"]
    assert first["execution"]["start_unix_ns"] != second["execution"]["start_unix_ns"]
    assert (a/"start.json").read_bytes() != (b/"start.json").read_bytes()
    assert m.compare_runs(design_path, h, a, b)["deterministic_repeat_identical"]
    for field in ("run_id", "start_unix_ns", "output_directory"):
        assert field.encode() not in (a/"scientific.json").read_bytes()
    with pytest.raises(m.AuditError, match="immutable"):
        m.run_audit(design_path, h, a)
    put(a/"scientific.json", {"forged": True})
    with pytest.raises(m.AuditError, match="changed"):
        m.verify_run(design_path, h, a)


def test_failure_is_retained_with_every_row_rejected(root, monkeypatch):
    design_path, h, _ = synthetic_design(root, monkeypatch)
    monkeypatch.setattr(m, "available_memory_kib", lambda: m.MIN_AVAILABLE_KIB+1)
    def failure(*args):
        raise m.AuditError("invariance did not reproduce")
    monkeypatch.setattr(m, "science", failure)
    out = m.OUT/"failed"
    with pytest.raises(m.AuditError, match="did not reproduce"):
        m.run_audit(design_path, h, out)
    report = m.read_json(out/"failure.json")
    assert len(report["rows"]) == 5 and all(r["status"] == "REJECTED" for r in report["rows"])
    assert not (out/"scientific.json").exists()


def test_low_memory_fails_before_science(root, monkeypatch):
    design_path, h, _ = synthetic_design(root, monkeypatch)
    monkeypatch.setattr(m, "available_memory_kib", lambda: m.MIN_AVAILABLE_KIB-1)
    called = []
    monkeypatch.setattr(m, "science", lambda *args: called.append(True))
    with pytest.raises(m.AuditError, match="11 GiB"):
        m.run_audit(design_path, h, m.OUT/"low_memory")
    assert not called


def test_external_cannot_admit_missing_run_or_missing_frozen_bundle(root, monkeypatch):
    for provenance in ({"runs": []}, {"runs": [{"tag": "fake"}]}):
        with pytest.raises(m.AuditError, match="missing/duplicate"):
            m.external_release_audit(provenance)


def test_legacy_checkpoint_hash_and_size_are_both_enforced(root):
    p = put(root/"checkpoint.pt", b"real checkpoint fixture")
    rec = m.record(p)
    assert m.verify_legacy_file(rec) == rec
    for key, value in (("sha256", "0"*64), ("bytes", rec["bytes"]+1)):
        altered = {**rec, key: value}
        with pytest.raises(m.AuditError, match="changed"):
            m.verify_legacy_file(altered)


def test_definition_extraction_does_not_execute_top_level_side_effect(root):
    p = put(root/"fixture.py", b"raise RuntimeError('must not execute')\ndef add(x):\n    return x + 1\n")
    loaded = m.load_definitions(p, ["add"])
    assert loaded.add(2) == 3
    with pytest.raises(m.AuditError, match="missing/ambiguous"):
        m.load_definitions(p, ["absent"])


def test_metadata_only_archive_preserves_ids_order_labels_shapes_without_tensors(root):
    bank, _, _ = bank_fixture()
    bank = {"b": bank["b"], "a": bank["a"]}
    path = put(root/"train.pkl", pickle.dumps(bank))
    result = m.load_archive_metadata(path)
    assert list(result) == ["b", "a"]
    assert result["b"]["gloss"] == "THREE"
    assert result["b"]["keypoint"] == {"tensor_shape": [3, 133, 3],
        "tensor_stride": [399, 3, 1], "storage_materialized": False}
    assert not any(isinstance(row["keypoint"], torch.Tensor) for row in result.values())


def test_metadata_only_archive_rejects_unknown_globals(root):
    path = put(root/"unsafe.pkl", pickle.dumps(Path("should_not_construct")))
    with pytest.raises(m.AuditError, match="unapproved metadata-pickle global"):
        m.load_archive_metadata(path)


def test_torch_distribution_and_declared_runtime_suffix_are_both_bound(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.10.0+cu128")
    original = m.importlib.metadata.version
    monkeypatch.setattr(m.importlib.metadata, "version", lambda p: "2.10.0" if p == "torch" else original(p))
    result = m.verify_runtime_version("torch", "2.10.0+cu128")
    assert result == {"recorded_runtime": "2.10.0+cu128", "current_runtime": "2.10.0+cu128", "distribution": "2.10.0"}
    for expected in ("2.9.0+cu128", "2.10.0+cu126", "2.10.0"):
        with pytest.raises(m.AuditError, match="runtime differs"):
            m.verify_runtime_version("torch", expected)


def test_resource_ceiling_is_exclusive_seven_gib(monkeypatch):
    assert m.MAX_RSS_KIB == 7 * 1024 * 1024
    monkeypatch.setattr(m.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=m.MAX_RSS_KIB-1))
    assert m.resource_check() == m.MAX_RSS_KIB-1
    for peak in (m.MAX_RSS_KIB, m.MAX_RSS_KIB+1):
        monkeypatch.setattr(m.resource, "getrusage", lambda _, value=peak: SimpleNamespace(ru_maxrss=value))
        with pytest.raises(m.AuditError, match="strict <7 GiB"):
            m.resource_check()


@pytest.fixture
def pair(root, monkeypatch):
    design_path, h, _ = synthetic_design(root, monkeypatch)
    monkeypatch.setattr(m, "science", fake_science)
    monkeypatch.setattr(m, "available_memory_kib", lambda: m.MIN_AVAILABLE_KIB+1)
    a, b = m.OUT/"run_a", m.OUT/"run_b"
    m.run_audit(design_path, h, a)
    m.run_audit(design_path, h, b)
    return design_path, h, a, b


def forbid_content_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("alias rejection must happen before reading artifacts or the design")
    monkeypatch.setattr(m, "read_json", forbidden)
    monkeypatch.setattr(m, "load_design", forbidden)
    monkeypatch.setattr(m, "digest", forbidden)


def test_comparator_rejects_identical_arguments_before_any_filesystem_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("identical arguments must be rejected before path resolution or reads")
    monkeypatch.setattr(m, "run_directory", forbidden)
    forbid_content_reads(monkeypatch)
    with pytest.raises(m.AuditError, match="arguments are identical"):
        m.compare_runs("unused", "unused", "/same/missing/directory", "/same/missing/directory")


def test_same_successful_run_is_not_an_independent_repeat(pair, monkeypatch):
    design, h, a, _ = pair
    forbid_content_reads(monkeypatch)
    with pytest.raises(m.AuditError, match="identical"):
        m.compare_runs(design, h, a, a)


def test_proc_self_fd_alias_is_rejected_before_reading(pair, monkeypatch):
    design, h, a, _ = pair
    fd = os.open(a, os.O_RDONLY | os.O_DIRECTORY)
    try:
        alias = Path(f"/proc/self/fd/{fd}")
        assert alias.resolve() == a
        forbid_content_reads(monkeypatch)
        with pytest.raises(m.AuditError, match="proc file-descriptor"):
            m.compare_runs(design, h, a, alias)
    finally:
        os.close(fd)


@pytest.mark.parametrize("parent_alias", (False, True))
def test_symlink_directory_alias_is_rejected_before_reading(pair, monkeypatch, parent_alias):
    design, h, a, _ = pair
    alias = m.OUT/"symlink_alias"
    alias.symlink_to(a.parent if parent_alias else a, target_is_directory=True)
    candidate = alias/a.name if parent_alias else alias
    assert candidate.resolve() == a
    forbid_content_reads(monkeypatch)
    with pytest.raises(m.AuditError, match="symlink"):
        m.compare_runs(design, h, a, candidate)


def test_distinct_spelling_same_canonical_path_rejected_before_reading(pair, monkeypatch):
    design, h, a, _ = pair
    relative = os.path.relpath(a, Path.cwd())
    # The supplied path spells exactly the same canonical directory. Any '..'
    # spelling is also rejected by the explicit canonical-path policy.
    forbid_content_reads(monkeypatch)
    with pytest.raises(m.AuditError, match="canonical|traversal"):
        m.compare_runs(design, h, a, relative)


def test_bind_mount_style_directory_inode_alias_rejected_before_reading(pair, monkeypatch):
    design, h, a, b = pair
    original = m.inode_identity
    real_a = original(a, directory=True)
    def mounted(path, *, directory=False):
        if directory and Path(path) == b:
            return real_a
        return original(path, directory=directory)
    monkeypatch.setattr(m, "inode_identity", mounted)
    forbid_content_reads(monkeypatch)
    with pytest.raises(m.AuditError, match="same directory inode"):
        m.compare_runs(design, h, a, b)


@pytest.mark.parametrize("filename", m.RUN_FILES)
def test_hardlinked_execution_artifacts_rejected_before_reading(pair, monkeypatch, filename):
    design, h, a, b = pair
    (b/filename).unlink()
    os.link(a/filename, b/filename)
    assert (a/filename).stat().st_ino == (b/filename).stat().st_ino
    forbid_content_reads(monkeypatch)
    with pytest.raises(m.AuditError, match="hardlinked"):
        m.compare_runs(design, h, a, b)


def test_bind_mount_style_artifact_inode_alias_rejected_before_reading(pair, monkeypatch):
    design, h, a, b = pair
    original = m.inode_identity
    real_a = original(a/"scientific.json")
    def mounted(path, *, directory=False):
        if Path(path) == b/"scientific.json":
            return real_a
        return original(path, directory=directory)
    monkeypatch.setattr(m, "inode_identity", mounted)
    forbid_content_reads(monkeypatch)
    with pytest.raises(m.AuditError, match="share execution-artifact inodes"):
        m.compare_runs(design, h, a, b)


def test_symlinked_artifact_rejected_before_reading(pair, monkeypatch):
    design, h, a, b = pair
    (b/"scientific.json").unlink()
    (b/"scientific.json").symlink_to(a/"scientific.json")
    forbid_content_reads(monkeypatch)
    with pytest.raises(m.AuditError, match="symlink"):
        m.compare_runs(design, h, a, b)


def refresh_fixture_receipts(directory, seal):
    """Adversary updates plain file hashes; semantic identity must still hold."""
    seal["outputs"] = {name: m.record(directory/name) for name in m.RUN_FILES if name != "output_seal.json"}
    put(directory/"output_seal.json", seal)


@pytest.mark.parametrize("filename", ("start.json", "output_seal.json"))
def test_copied_start_or_seal_cannot_be_an_independent_execution(pair, filename):
    design, h, a, b = pair
    shutil.copyfile(a/filename, b/filename)
    if filename == "start.json":
        refresh_fixture_receipts(b, m.read_json(b/"output_seal.json"))
    with pytest.raises(m.AuditError, match="seal inode|seal/start"):
        m.compare_runs(design, h, a, b)


def test_copied_directory_cannot_be_an_independent_execution(pair):
    design, h, a, _ = pair
    copied = m.OUT/"copied_run"
    shutil.copytree(a, copied)
    assert (a/"scientific.json").read_bytes() == (copied/"scientific.json").read_bytes()
    with pytest.raises(m.AuditError, match="seal inode"):
        m.compare_runs(design, h, a, copied)


@pytest.mark.parametrize("field", ("run_id", "start_unix_ns"))
def test_shared_run_id_or_start_time_rejected_even_after_consistent_rehashing(pair, field):
    design, h, a, b = pair
    first = m.read_json(a/"start.json")["execution"]
    start, seal = m.read_json(b/"start.json"), m.read_json(b/"output_seal.json")
    start["execution"][field] = first[field]
    seal["execution"] = copy.deepcopy(start["execution"])
    put(b/"start.json", start)
    if field == "run_id":
        telemetry = m.read_json(b/"telemetry.json")
        telemetry["run_id"] = first["run_id"]
        put(b/"telemetry.json", telemetry)
    refresh_fixture_receipts(b, seal)
    assert m.verify_run(design, h, b)["verified"]  # internally consistent alone
    with pytest.raises(m.AuditError, match="distinct execution"):
        m.compare_runs(design, h, a, b)


@pytest.mark.parametrize("field,value", (("run_id", "bad"), ("start_unix_ns", True), ("start_unix_ns", -1)))
def test_malformed_identity_rejected_after_rehashing(pair, field, value):
    design, h, _, b = pair
    start, seal = m.read_json(b/"start.json"), m.read_json(b/"output_seal.json")
    start["execution"][field] = value
    seal["execution"] = copy.deepcopy(start["execution"])
    put(b/"start.json", start)
    refresh_fixture_receipts(b, seal)
    with pytest.raises(m.AuditError, match="invalid execution"):
        m.verify_run(design, h, b)


@pytest.mark.parametrize("field", ("canonical_path", "device", "inode"))
def test_tampered_directory_binding_rejected_after_rehashing(pair, field):
    design, h, a, b = pair
    start, seal = m.read_json(b/"start.json"), m.read_json(b/"output_seal.json")
    original = start["execution"]["output_directory"][field]
    start["execution"]["output_directory"][field] = str(a) if field == "canonical_path" else original+1
    seal["execution"] = copy.deepcopy(start["execution"])
    put(b/"start.json", start)
    refresh_fixture_receipts(b, seal)
    with pytest.raises(m.AuditError, match="directory inode binding"):
        m.verify_run(design, h, b)


def test_execution_start_and_seal_refuse_overwrite(pair):
    _, _, a, _ = pair
    for path, writer in ((a/"start.json", m.write_new), (a/"output_seal.json", m.write_output_seal)):
        before = path.read_bytes()
        with pytest.raises(FileExistsError):
            writer(path, {"replacement": True})
        assert path.read_bytes() == before
