import json
from argparse import Namespace

import numpy as np
import pytest

from scripts import strict_route_only_cac as strict


def _toy(tmp_path):
    names = [
        "rerank_score", "coverage", "slot_score", "f1", "length_penalty", "score",
        "hand_score", "len_retr", "len_fall", "len_diff", "len_ratio",
        "has_retrieval", "fill_progress", "total_mix_len", "brevity_arg",
        "mean_len_retr_S", "mean_len_fall_NS", "cur_bleu4", "cur_wer",
        "delta_len_clip", "rel_delta_len", "rel_brevity_impact", "wer_cost_clip",
    ]
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, len(names)))
    y = X[:, 0] - 0.2 * X[:, 1] + 0.05 * rng.normal(size=80)
    traj = tmp_path / "trajectory.npz"
    np.savez(traj, X=X, y=y, feature_names=np.asarray(names))
    rows = []
    for i in range(8):
        rows.append({"id": f"id-{i}", "selected": {
            "rerank_score": float(i), "coverage": float(i % 3),
            "slot_score": 1.0, "f1": 0.5, "length_penalty": 0.1,
            "score": float(i) / 10,
        } if i != 6 else None, "text": "must never be read"})
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(rows))
    return traj, trace


def test_selection_succeeds_when_torch_load_is_forbidden(tmp_path, monkeypatch):
    traj, trace = _toy(tmp_path)
    monkeypatch.setattr("torch.load", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("strict route must not call torch.load")))
    args = Namespace(trajectory=str(traj), dev_trace=str(trace), test_trace=str(trace),
                     budget=0.25, n_estimators=30, max_depth=2, learning_rate=0.05,
                     alpha_wer=0.0, route_name="strict_route_only_cac",
                     out=str(tmp_path / "route.json"), out_dir=str(tmp_path))
    out = strict.run(args)
    assert out["test_evaluator_outputs_loaded_at_route_time"] == []
    assert out["splits"]["dev"]["selected"] == 2
    assert list(np.load(tmp_path / "strict_route_only_cac_dev.npz")["in_S"]).count(1) == 2


def test_forbidden_columns_do_not_change_route_and_ties_are_id_stable(tmp_path):
    traj, trace = _toy(tmp_path)
    X, y, names = strict._trajectory(traj)
    model_a = strict.fit_route_model(traj, n_estimators=30, max_depth=2)
    X2 = X.copy()
    for name in strict.FORBIDDEN_MARKERS:
        X2[:, names.index(name)] += 10_000.0
    altered = tmp_path / "altered.npz"
    np.savez(altered, X=X2, y=y, feature_names=np.asarray(names))
    model_b = strict.fit_route_model(altered, n_estimators=30, max_depth=2)
    ids, feats, eligible = strict.load_trace_features(trace)
    a = strict.select_static_topk(model_a, ids, feats, eligible, 3)
    b = strict.select_static_topk(model_b, ids, feats, eligible, 3)
    np.testing.assert_array_equal(a, b)

    class Constant:
        def predict(self, X):
            return np.zeros(len(X))
    tied = strict.select_static_topk(Constant(), ["z", "a", "m"],
                                     np.zeros((3, 9)), np.ones(3, dtype=bool), 2)
    np.testing.assert_array_equal(tied, np.array([0, 1, 1], dtype=np.int8))


def test_invalid_budget_is_rejected(tmp_path):
    traj, trace = _toy(tmp_path)
    args = Namespace(trajectory=str(traj), dev_trace=str(trace), test_trace=str(trace),
                     budget=1.01, n_estimators=10, max_depth=2, learning_rate=0.05,
                     alpha_wer=0.0, route_name="strict_route_only_cac",
                     out=str(tmp_path / "route.json"), out_dir=str(tmp_path))
    with pytest.raises(ValueError, match=r"budget must lie in \[0, 1\]"):
        strict.run(args)


def test_dev_only_does_not_read_test_trace(tmp_path):
    traj, trace = _toy(tmp_path)
    args = Namespace(
        trajectory=str(traj), dev_trace=str(trace),
        test_trace=str(tmp_path / "must-not-exist.json"), dev_only=True,
        budget=0.25, n_estimators=10, max_depth=2, learning_rate=0.05,
        alpha_wer=0.0, route_name="dev_only_route",
        out=str(tmp_path / "route.json"), out_dir=str(tmp_path),
    )
    out = strict.run(args)
    assert set(out["splits"]) == {"dev"}
    assert not (tmp_path / "dev_only_route_test.npz").exists()


def test_test_only_does_not_read_dev_trace(tmp_path):
    traj, trace = _toy(tmp_path)
    args = Namespace(
        trajectory=str(traj), dev_trace=str(tmp_path / "must-not-exist.json"),
        test_trace=str(trace), test_only=True,
        budget=0.25, n_estimators=10, max_depth=2, learning_rate=0.05,
        alpha_wer=0.0, route_name="test_only_route",
        out=str(tmp_path / "route.json"), out_dir=str(tmp_path),
    )
    out = strict.run(args)
    assert set(out["splits"]) == {"test"}
    assert not (tmp_path / "test_only_route_dev.npz").exists()


def test_replay_frame_exact_capacity_and_infeasible_skip():
    class M:
        def predict(self, X): return np.arange(len(X), dtype=float)[::-1]
    ids = ["a", "b", "c"]
    x = np.zeros((3, 9)); eligible = np.ones(3, dtype=bool)
    mask, info = strict.select_replay_frames(M(), ids, x, eligible,
                                             np.array([4, 4, 100]), np.array([4, 4, 4]), "0.4")
    # capacity is 0.4*12=4.8; exact integer admission permits one 4-frame clip.
    assert int(mask.sum()) == 1 and info["scaled_used_cost"] == 20
    assert info["replay_frames"] == 4
    assert info["emitted_frames"] == 12
    assert info["realized_replay_frame_fraction"] == pytest.approx(1 / 3)
    assert info["optimality_claimed"] is False


def test_replay_frame_ties_use_id_order():
    class Constant:
        def predict(self, X): return np.zeros(len(X))
    ids = ["z", "a", "m"]
    x = np.zeros((3, 9))
    mask, _ = strict.select_replay_frames(
        Constant(), ids, x, np.ones(3, dtype=bool),
        np.full(3, 4), np.full(3, 4), "0.4",
    )
    np.testing.assert_array_equal(mask, np.array([0, 1, 0], dtype=np.int8))


def test_replay_metadata_id_mismatch_rejected(tmp_path):
    (tmp_path / "bad.json").write_text('{"ids": ["b"], "r": {"b": 1}, "f": {"b": 1}}')
    with pytest.raises(ValueError, match="IDs"):
        strict.load_length_metadata(tmp_path / "bad.json", ["a"])


def test_replay_metadata_rejects_extra_length_id(tmp_path):
    (tmp_path / "bad.json").write_text(
        '{"ids":["a"],"r":{"a":1,"extra":2},"f":{"a":1}}'
    )
    with pytest.raises(ValueError, match="IDs"):
        strict.load_length_metadata(tmp_path / "bad.json", ["a"])
