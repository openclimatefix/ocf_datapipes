from datetime import timedelta

import numpy as np

from ocf_datapipes.select import LocationPicker
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration


def test_location_picker_single_location(gsp_datapipe):
    gsp_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe, sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=1)
    )
    location_datapipe = LocationPicker(gsp_datapipe)
    data = next(iter(location_datapipe))
    assert len(data) == 2


def test_location_picker_all_locations(gsp_datapipe):
    dataset = next(iter(gsp_datapipe))
    gsp_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe, sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=1)
    )
    location_datapipe = LocationPicker(gsp_datapipe, return_all_locations=True)
    loc_iterator = iter(location_datapipe)
    for i in range(len(dataset["x_osgb"])):
        loc_data = next(loc_iterator)
        assert np.isclose(loc_data, (dataset["x_osgb"][i], dataset["y_osgb"][i])).all()
