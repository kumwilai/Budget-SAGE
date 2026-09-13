import copy

import numpy as np
import pytest
import torch

from scripts.audit_csl_text_route_transfer import (
    assemble_local, build_bank, generate, reconstruct,
    select_replay_frames, validate_source_split, confined_output_path, ROOT,
)


def fixture():
    training = [{"id": f"t{i}", "text": word, "gloss": "A B"}
                for i, word in enumerate(("alpha", "beta", "gamma"))]
    pose = {r["id"]: np.arange(16*133*3, dtype=np.float32).reshape(16, 133, 3) + i
            for i, r in enumerate(training)}
    spans = {sid: [{"gloss": "A", "start": 1, "end": 7},
                   {"gloss": "B", "start": 8, "end": 14}] for sid in pose}
    return training, pose, spans


def test_end_to_end_query_poisoning_and_recipe_reconstruction():
    training, pose, spans = fixture()
    queries = [{"id": "q", "text": "alpha alpha"}]
    first = generate(queries, training, pose, spans, {"q"})
    poisoned = copy.deepcopy(queries)
    poisoned[0].update(gloss="unavailable", length=999, signer=2, pose="held-out")
    second = generate(poisoned, training, pose, spans, {"q"})
    assert first[0:2] == second[0:2]
    assert first[4:] == second[4:]
    for bank in (2, 3):
        assert torch.equal(first[bank]["q"], second[bank]["q"])
    np.testing.assert_array_equal(first[3]["q"].numpy(), reconstruct(first[4]["q"]["recipe"], pose))


def test_missing_units_are_counted_and_empty_local_refuses():
    _, pose, spans = fixture()
    bank, lengths, _ = build_bank(pose, spans)
    local, recipe, omitted = assemble_local(["A", "absent"], bank, lengths, pose, set())
    assert local is not None and len(recipe) == 1
    assert omitted == [{"index": 1, "gloss": "absent"}]
    local, recipe, omitted = assemble_local(["A"], bank, lengths, pose, set(pose))
    assert local is None and recipe == [] and len(omitted) == 1


def test_source_overlap_forbidden_and_out_of_bounds_rejected():
    training, pose, spans = fixture()
    with pytest.raises(ValueError, match="overlaps"):
        validate_source_split(training, spans, pose, {"t0"})
    unit = {"source_id": "t0", "start": 1, "end": 7, "out_len": 8}
    with pytest.raises(ValueError, match="forbidden"):
        reconstruct([unit], pose, {"t0"})
    with pytest.raises(ValueError, match="invalid"):
        reconstruct([{**unit, "end": 100}], pose)


def test_whole_clip_interval_cannot_enter_local_bank():
    _, pose, _ = fixture()
    spans = {"t0": [{"gloss": "A", "start": 0, "end": 16}]}
    bank, lengths, skipped = build_bank(pose, spans)
    assert bank == {} and lengths == {}
    assert skipped["full_clip_span_not_local"] == 1


@pytest.mark.parametrize("outside", ["/tmp/outside-csl-project", "../outside-csl-project", str(ROOT)])
def test_output_path_is_confined_before_writing(outside):
    with pytest.raises(ValueError, match="inside the project"):
        confined_output_path(outside)


def test_output_path_resolves_inside_project():
    assert confined_output_path("outputs/revision/csl-check") == ROOT/"outputs/revision/csl-check"


def test_unequal_lengths_use_local_capacity_and_replay_cost():
    candidates = [{"id": "a", "similarity": 1., "local_len": 4, "whole_len": 2},
                  {"id": "b", "similarity": 1., "local_len": 6, "whole_len": 10}]
    mask, info = select_replay_frames(candidates)
    assert mask == [True, False]
    assert info["replay_frames"] == 2 and info["emitted_frames"] == 8
    assert info["scaled_capacity"] == 20 and info["scaled_used_cost"] == 14


def test_exact_two_fifths_boundary_and_id_tie_break():
    candidates = [{"id": "b", "similarity": 1., "local_len": 4, "whole_len": 4},
                  {"id": "a", "similarity": 1., "local_len": 4, "whole_len": 4},
                  {"id": "c", "similarity": 1., "local_len": 2, "whole_len": 20}]
    mask, info = select_replay_frames(candidates)
    assert mask == [False, True, False]
    assert 5*info["replay_frames"] == 2*info["emitted_frames"]


def test_highconf_length_selection_matches_native_diagnostic():
    from scripts.build_csl_pg_rast_native_hrnet import choose_candidate
    import random
    _, pose, spans = fixture()
    bank, lengths, _ = build_bank(pose, spans)
    _, recipe, _ = assemble_local(["A"], bank, lengths, pose, set())
    native = choose_candidate([(sid, 1, 7) for sid in pose],
                              {sid: {"keypoint": p} for sid, p in pose.items()},
                              random.Random(0), "highconf_len", 6.)
    assert (recipe[0]["source_id"], recipe[0]["start"], recipe[0]["end"]) == native
