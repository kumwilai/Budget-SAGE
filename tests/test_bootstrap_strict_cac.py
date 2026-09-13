import numpy as np

from scripts.bootstrap_strict_cac import paired_bootstrap


def test_paired_bootstrap_matches_expected_direction_and_is_deterministic():
    refs = ["a b c d", "e f g h", "i j k l", "m n o p"]
    hypotheses = {
        "baseline": ["x", "x", "x", "x"],
        "better": list(refs),
    }
    pairs = (("better_vs_baseline", "better", "baseline"),)
    first = paired_bootstrap(hypotheses, refs, pairs=pairs, n_boot=100, seed=9)
    second = paired_bootstrap(hypotheses, refs, pairs=pairs, n_boot=100, seed=9)
    assert first == second
    row = first["comparisons"]["better_vs_baseline"]
    assert row["point_delta"]["bleu4"] > 0
    assert row["point_delta"]["corpus_wer"] < 0
    assert row["bootstrap_delta"]["bleu4"]["ci95"][0] > 0
    assert row["bootstrap_delta"]["corpus_wer"]["ci95"][1] < 0


def test_bootstrap_rejects_misalignment_and_bad_pair():
    refs = ["a b c d"]
    try:
        paired_bootstrap({"method": []}, refs, pairs=(), n_boot=5)
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("misaligned predictions were accepted")

    try:
        paired_bootstrap(
            {"method": list(refs)}, refs,
            pairs=(("bad", "missing", "method"),), n_boot=5,
        )
    except KeyError as exc:
        assert "unknown comparison pair" in str(exc)
    else:
        raise AssertionError("unknown method pair was accepted")


def test_bootstrap_outputs_finite_metrics_for_empty_hypothesis():
    refs = ["a b c d", "e f g h"]
    result = paired_bootstrap(
        {"empty": ["", ""], "copy": list(refs)},
        refs,
        pairs=(("copy_vs_empty", "copy", "empty"),),
        n_boot=10,
        seed=4,
    )
    values = result["methods"]["empty"]
    assert all(np.isfinite(value) for value in values.values())
    assert values["corpus_wer"] == 100.0
