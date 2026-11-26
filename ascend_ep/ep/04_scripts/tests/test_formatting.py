"""Number, unit and missing-value formatting."""
import pytest

from reporting import formatting as F


@pytest.mark.parametrize(("value", "expected"), [
    (1.011065, "1.01"),
    (126599.180072, "1.27e+05"),
    (0.000123456, "0.000123"),
    (0.0, "0"),
    (-3.5, "-3.5"),
    (1, "1"),
    (1234.5678, "1.23e+03"),
])
def test_three_significant_digits(value, expected):
    assert F.number(value) == expected


def test_none_is_never_zero():
    assert F.number(None) == "-"
    assert F.number(float("nan")) == "-"
    assert F.number(float("inf")) == "-"
    assert F.text(None) == "-"
    assert F.boolean(None) == "-"
    assert F.binary(None) == "-"


def test_counts_keep_their_digits():
    assert F.count(12) == "12"
    assert F.count(0) == "0"
    assert F.count(1234567) == "1.23e+06"


def test_units():
    assert F.percent(0.8734) == "87.3%"
    assert F.percent(1.0) == "100%"
    assert F.percent(None) == "-"
    assert F.binary(1536 * 1024) == "1.5 MiB"
    assert F.binary(3 * (1 << 30)) == "3 GiB"
    assert F.na() == "N/A (not applicable to this case)"
    assert F.na(F.REASON_NOT_COMMUNICATION) == "N/A (not a communication payload)"


def test_table_escapes_pipes_and_empty_values():
    lines = F.table(("A", "B"), [("x|y", None)])
    assert lines[0] == "| A | B |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| x\\|y | - |"


def test_empty_table_is_explicit():
    assert F.table(("A",), [], empty="nothing measured") == ["_nothing measured_", ""]


def test_join_and_status_counts():
    assert F.join(["a", "b"]) == "a, b"
    assert F.join([]) == "-"
    assert F.status_counts({"passed": 2, "unsupported": 1}) == "passed=2, unsupported=1"


def test_missing_marker_is_documented_by_the_module():
    assert F.MISSING == "-"
    assert F.NA == "N/A"
