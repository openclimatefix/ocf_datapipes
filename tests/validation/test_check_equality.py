from ocf_datapipes.validation import (
    CheckGreaterThanOrEqualTo,
    CheckLessThanOrEqualTo,
    CheckNotEqualTo,
)
import numpy as np


def test_check_not_equal_to(topo_datapipe):
    topo_datapipe = CheckNotEqualTo(topo_datapipe, 100000)
    data = next(iter(topo_datapipe))


def test_check_less_than_or_equal_to(topo_datapipe):
    topo_datapipe = CheckLessThanOrEqualTo(topo_datapipe, 100000)
    data = next(iter(topo_datapipe))


def test_check_greater_than_or_equal_to(topo_datapipe):
    topo_datapipe = CheckGreaterThanOrEqualTo(topo_datapipe, 100000)
    data = next(iter(topo_datapipe))
