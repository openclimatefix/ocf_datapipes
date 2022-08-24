import numpy as np

from ocf_datapipes.select import LocationPicker
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration

from datetime import timedelta


def test_location_picker_single_location(gsp_dp):
    gsp_dp = AddT0IdxAndSamplePeriodDuration(gsp_dp, sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=1))
    location_dp = LocationPicker(gsp_dp)
    data = next(iter(location_dp))
    assert len(data) == 2


def test_location_picker_all_locations(gsp_dp):
    dataset = next(iter(gsp_dp))
    gsp_dp = AddT0IdxAndSamplePeriodDuration(gsp_dp, sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=1))
    location_dp = LocationPicker(gsp_dp, return_all_locations=True)
    loc_iterator = iter(location_dp)
    for i in range(len(dataset["x_osgb"])):
        loc_data = next(loc_iterator)
        assert np.isclose(loc_data, (dataset["x_osgb"][i], dataset["y_osgb"][i])).all()
