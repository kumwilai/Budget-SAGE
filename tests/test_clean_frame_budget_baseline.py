import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.clean_frame_budget_baseline import build_route

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_uses_declared_column_and_exact_frame_budget(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps([
        {"id": "b", "selected": {
            "rerank_score": 1.0, "coverage": 0.0, "slot_score": 0.0,
            "f1": 0.0, "length_penalty": 0.0, "score": 0.0,
        }},
        {"id": "a", "selected": {
            "rerank_score": 1.0, "coverage": 0.0, "slot_score": 0.0,
            "f1": 0.0, "length_penalty": 0.0, "score": 0.0,
        }},
        {"id": "c", "selected": {
            "rerank_score": 0.0, "coverage": 0.0, "slot_score": 0.0,
            "f1": 0.0, "length_penalty": 0.0, "score": 0.0,
        }},
    ]))
    lengths = tmp_path / "lengths.json"
    lengths.write_text(json.dumps({
        "ids": ["a", "b", "c"],
        "r": {"a": 4, "b": 4, "c": 100},
        "f": {"a": 4, "b": 4, "c": 4},
    }))
    record = build_route(
        trace,
        lengths,
        "0.40",
        "rerank_score",
        "toy",
        tmp_path / "mask.npz",
        tmp_path / "record.json",
    )
    mask = np.load(tmp_path / "mask.npz", allow_pickle=False)
    assert mask["ids"].tolist() == ["b", "a", "c"]
    assert mask["in_S"].tolist() == [0, 1, 0]
    assert record["learned"] is False
    assert record["selection"]["realized_replay_frame_fraction"] == 1 / 3


def test_command_line_entry_point_loads_from_project_root():
    completed = subprocess.run(
        [sys.executable, "scripts/clean_frame_budget_baseline.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
