import numpy as np
import pytest

from ocf_datapipes.transform.xarray import EnsureNPVSystemsPerExample
from ocf_datapipes.utils.consts import Location


def test_ensure_n_pv_systems_per_example_expand(passiv_datapipe):

    data_before = next(iter(passiv_datapipe))

    passiv_datapipe = EnsureNPVSystemsPerExample(passiv_datapipe, n_pv_systems_per_example=12)
    data_after = next(iter(passiv_datapipe))

    assert len(data_before[0, :]) == 2
    assert len(data_after[0, :]) == 12


def test_ensure_n_pv_systems_per_example_random(passiv_datapipe):

    data_before = next(iter(passiv_datapipe))

    passiv_datapipe = EnsureNPVSystemsPerExample(passiv_datapipe, n_pv_systems_per_example=1)
    data_after = next(iter(passiv_datapipe))

    assert len(data_before[0, :]) == 2
    assert len(data_after[0, :]) == 1


def test_ensure_n_pv_systems_per_example_closest_error(passiv_datapipe):

    with pytest.raises(Exception):
        _ = EnsureNPVSystemsPerExample(
            passiv_datapipe, n_pv_systems_per_example=1, method="closest"
        )


def test_ensure_n_pv_systems_per_example_closest(passiv_datapipe):

    # make fake location datapipe
    location = Location(x=2.687e05, y=6.267e05)
    location_datapipe = iter([location])

    data_before = next(iter(passiv_datapipe))

    passiv_datapipe = EnsureNPVSystemsPerExample(
        passiv_datapipe,
        n_pv_systems_per_example=1,
        method="closest",
        locations_datapipe=location_datapipe,
    )

    data_after = next(iter(passiv_datapipe))

    assert len(data_before[0, :]) == 2
    assert len(data_after[0, :]) == 1
    assert data_after.pv_system_id[0] == 9960
