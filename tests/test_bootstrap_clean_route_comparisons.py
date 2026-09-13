import argparse

import pytest

from scripts.bootstrap_clean_route_comparisons import parse_assignment, parse_pair


def test_parse_method_and_pair_specs():
    assert parse_assignment("a=outputs/a.pt") == ("a", "outputs/a.pt")
    assert parse_pair("a_vs_b=a:b") == ("a_vs_b", "a", "b")


@pytest.mark.parametrize("value", ["a", "=x", "a="])
def test_invalid_method_spec(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_assignment(value)


@pytest.mark.parametrize("value", ["x", "x=a", "=a:b", "x=:b"])
def test_invalid_pair_spec(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_pair(value)
