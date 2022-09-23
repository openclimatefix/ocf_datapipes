from datetime import timedelta

import numpy as np

import ocf_datapipes  # noqa
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenConfiguration, OpenPVFromNetCDF
from ocf_datapipes.select import LocationPicker
from ocf_datapipes.transform.xarray import AddT0IdxAndSamplePeriodDuration


def test_location_picker_single_location(gsp_datapipe):
    gsp_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe,
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(hours=1),
    )
    location_datapipe = LocationPicker(gsp_datapipe)
    data = next(iter(location_datapipe))
    assert data.x is not None
    assert data.y is not None


def test_location_picker_all_locations(gsp_datapipe):
    dataset = next(iter(gsp_datapipe))
    gsp_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe,
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(hours=1),
    )
    location_datapipe = LocationPicker(gsp_datapipe, return_all_locations=True)
    loc_iterator = iter(location_datapipe)
    for i in range(len(dataset["x_osgb"])):
        loc_data = next(loc_iterator)
        assert np.isclose((loc_data.x,loc_data.y), (dataset["x_osgb"][i], dataset["y_osgb"][i])).all()


def test_location_picker_with_id(configuration_with_pv_parquet):

    # load configuration
    config_datapipe = OpenConfiguration(configuration_with_pv_parquet)
    configuration: Configuration = next(iter(config_datapipe))

    pv_location_datapipe = OpenPVFromNetCDF(
        pv_power_filename=configuration.input_data.pv.pv_files_groups[0].pv_filename,
        pv_metadata_filename=configuration.input_data.pv.pv_files_groups[0].pv_metadata_filename,
    )

    location_datapipe = pv_location_datapipe.location_picker()

    data = next(iter(location_datapipe))

    assert data.id is not None

