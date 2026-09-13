import copy

import pytest

from scripts.build_retrieval_gloss_plans import build_plans


def _rows():
    queries = [
        {"id": "q1", "text": "rain north", "gloss": "FORBIDDEN", "length": 999},
        {"id": "q2", "text": "sun south", "pose": [1, 2, 3]},
    ]
    training = [
        {"id": "t1", "text": "rain north", "gloss": "EXACT"},
        {"id": "t2", "text": "rain east", "gloss": "RAIN EAST"},
        {"id": "t3", "text": "sun west", "gloss": "SUN WEST"},
    ]
    return queries, training


def test_complete_plans_exclude_exact_captions_and_query_metadata():
    queries, training = _rows()
    plans, trace = build_plans(queries, training)
    assert list(plans) == ["q1", "q2"]
    assert plans["q1"] != ["EXACT"]
    assert all(not row["exact_caption"] for row in trace)
    assert trace[0]["exact_caption_source_ids"] == ["t1"]
    assert set(trace[0]["forbidden_source_ids"]) == {"t1", trace[0]["retrieved_id"]}
    assert trace[0]["retrieved_id"] not in trace[0]["exact_caption_source_ids"]
    assert all("FORBIDDEN" not in token for plan in plans.values() for token in plan)

    poisoned = copy.deepcopy(queries)
    poisoned[0]["gloss"] = "CHANGED"
    poisoned[0]["length"] = -5
    poisoned[1]["pose"] = None
    assert build_plans(poisoned, training) == (plans, trace)


def test_query_train_id_overlap_is_rejected():
    queries, training = _rows()
    queries[0]["id"] = "t1"
    with pytest.raises(ValueError, match="query/training id overlap"):
        build_plans(queries, training)


def test_exact_caption_exclusion_normalizes_terminal_punctuation():
    queries = [{"id": "q", "text": "guten abend ."}]
    training = [
        {"id": "exact", "text": "guten abend", "gloss": "MUST NOT USE"},
        {"id": "safe", "text": "guten morgen", "gloss": "SAFE PLAN"},
    ]
    plans, trace = build_plans(queries, training)
    assert plans["q"] == ["SAFE", "PLAN"]
    assert trace[0]["retrieved_id"] == "safe"
    assert trace[0]["exact_caption_source_ids"] == ["exact"]
    assert trace[0]["forbidden_source_ids"] == ["exact", "safe"]
