from ocf_datapipes.validation import (
    CheckGreaterThanOrEqualTo,
    CheckLessThanOrEqualTo,
    CheckNotEqualTo,
    CheckValueEqualToFraction,
)
import pytest


def test_check_not_equal_to(topo_datapipe):
    topo_datapipe = CheckNotEqualTo(topo_datapipe, 100000)
    data = next(iter(topo_datapipe))


def test_check_less_than_or_equal_to(topo_datapipe):
    topo_datapipe = CheckLessThanOrEqualTo(topo_datapipe, 100000)
    data = next(iter(topo_datapipe))


def test_check_greater_than_or_equal_to(topo_datapipe):
    topo_datapipe = CheckGreaterThanOrEqualTo(topo_datapipe, -9999)
    data = next(iter(topo_datapipe))


def test_check_value_equal_to_fraction_no_warning(topo_datapipe):
    topo_datapipe = CheckValueEqualToFraction(topo_datapipe, value=0.0, fraction=0.1)
    data = next(iter(topo_datapipe))


def test_check_value_equal_to_fraction_with_warning(topo_datapipe):
    topo_datapipe = CheckValueEqualToFraction(topo_datapipe, value=0.0, fraction=0.0)
    with pytest.raises(Warning):
        data = next(iter(topo_datapipe))
