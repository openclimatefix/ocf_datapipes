from datetime import timedelta

import numpy as np

import ocf_datapipes  # noqa
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenConfiguration, OpenPVFromNetCDF
from ocf_datapipes.select import PickLocations
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration


def test_pick_locations_single_location(gsp_datapipe):
    gsp_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe,
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(hours=1),
    )
    location_datapipe = PickLocations(gsp_datapipe)
    data = next(iter(location_datapipe))
    assert data.x is not None
    assert data.y is not None


def test_pick_locations_all_locations(gsp_datapipe):
    dataset = next(iter(gsp_datapipe))
    gsp_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe,
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(hours=1),
    )
    location_datapipe = PickLocations(gsp_datapipe, return_all=True)
    loc_iterator = iter(location_datapipe)
    for i in range(len(dataset["x_osgb"])):
        loc_data = next(loc_iterator)
        assert np.isclose(
            (loc_data.x, loc_data.y), (dataset["x_osgb"][i], dataset["y_osgb"][i])
        ).all()


def test_pick_locations_with_id(configuration_with_pv_netcdf):
    # load configuration
    config_datapipe = OpenConfiguration(configuration_with_pv_netcdf)
    configuration: Configuration = next(iter(config_datapipe))

    pv_location_datapipe = OpenPVFromNetCDF(pv=configuration.input_data.pv)

    location_datapipe = pv_location_datapipe.pick_locations()

    data = next(iter(location_datapipe))

    assert data.id is not None
