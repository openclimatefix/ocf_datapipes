from datetime import timedelta
from pathlib import Path

import pytest
from torchdata.datapipes.iter import Forker, IterDataPipe

import ocf_datapipes
from ocf_datapipes.batch import MergeNumpyExamplesToBatch, MergeNumpyModalities
from ocf_datapipes.convert import (
    ConvertGSPToNumpyBatch,
    ConvertNWPToNumpyBatch,
    ConvertPVToNumpyBatch,
    ConvertSatelliteToNumpyBatch,
)
from ocf_datapipes.load import OpenGSP, OpenNWP, OpenPVFromNetCDF, OpenSatellite, OpenTopography
from ocf_datapipes.select import (
    LocationPicker,
    SelectLiveT0Time,
    SelectLiveTimeSlice,
    SelectSpatialSliceMeters,
    SelectSpatialSlicePixels,
    SelectTimeSlice,
)
from ocf_datapipes.transform.xarray import (
    AddT0IdxAndSamplePeriodDuration,
    ConvertSatelliteToInt8,
    ConvertToNWPTargetTime,
    Downsample,
    EnsureNPVSystemsPerExample,
    ReprojectTopography,
)


class GSPIterator(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe):
        super().__init__()
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for xr_dataset in self.source_datapipe:
            # Iterate through all locations in dataset
            for location_idx in range(len(xr_dataset["x_osgb"])):
                yield xr_dataset.isel(gsp_id=slice(location_idx, location_idx + 1))


@pytest.fixture()
def all_loc_np_datapipe():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "hrv_sat_data.zarr"
    sat_datapipe = OpenSatellite(zarr_path=filename)
    sat_datapipe = ConvertSatelliteToInt8(sat_datapipe)
    sat_datapipe = AddT0IdxAndSamplePeriodDuration(
        sat_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
    )
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "pv" / "passiv" / "test.nc"
    )
    filename_metadata = (
        Path(ocf_datapipes.__file__).parent.parent
        / "tests"
        / "data"
        / "pv"
        / "passiv"
        / "UK_PV_metadata.csv"
    )
    pv_datapipe = OpenPVFromNetCDF(
        pv_power_filename=filename, pv_metadata_filename=filename_metadata
    )
    pv_datapipe = AddT0IdxAndSamplePeriodDuration(
        pv_datapipe,
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=60),
    )

    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
    gsp_datapipe = OpenGSP(gsp_pv_power_zarr_path=filename)
    gsp_datapipe = AddT0IdxAndSamplePeriodDuration(
        gsp_datapipe,
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(hours=2),
    )
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "nwp_data" / "test.zarr"
    )
    nwp_datapipe = OpenNWP(zarr_path=filename)
    nwp_datapipe = AddT0IdxAndSamplePeriodDuration(
        nwp_datapipe, sample_period_duration=timedelta(hours=1), history_duration=timedelta(hours=2)
    )

    location_datapipe1, location_datapipe2, location_datapipe3 = LocationPicker(
        gsp_datapipe, return_all_locations=True
    ).fork(
        3
    )  # Its in order then
    pv_datapipe = SelectSpatialSliceMeters(
        pv_datapipe,
        location_datapipe=location_datapipe1,
        roi_width_meters=960_000,
        roi_height_meters=960_000,
    )  # Has to be large as test PV systems aren't in first 20 GSPs it seems
    pv_datapipe, pv_t0_datapipe = EnsureNPVSystemsPerExample(
        pv_datapipe, n_pv_systems_per_example=8
    ).fork(2)
    sat_datapipe, sat_t0_datapipe = SelectSpatialSlicePixels(
        sat_datapipe,
        location_datapipe=location_datapipe2,
        roi_width_pixels=256,
        roi_height_pixels=128,
        y_dim_name="y_geostationary",
        x_dim_name="x_geostationary",
    ).fork(2)
    nwp_datapipe, nwp_t0_datapipe = SelectSpatialSlicePixels(
        nwp_datapipe,
        location_datapipe=location_datapipe3,
        roi_width_pixels=64,
        roi_height_pixels=64,
        y_dim_name="y_osgb",
        x_dim_name="x_osgb",
    ).fork(2)
    nwp_datapipe = Downsample(nwp_datapipe, y_coarsen=16, x_coarsen=16)
    nwp_t0_datapipe = SelectLiveT0Time(nwp_t0_datapipe, dim_name="init_time_utc")
    nwp_datapipe = ConvertToNWPTargetTime(
        nwp_datapipe,
        t0_datapipe=nwp_t0_datapipe,
        sample_period_duration=timedelta(hours=1),
        history_duration=timedelta(hours=2),
        forecast_duration=timedelta(hours=3),
    )
    gsp_t0_datapipe = SelectLiveT0Time(gsp_datapipe)
    gsp_datapipe = SelectLiveTimeSlice(
        gsp_datapipe,
        t0_datapipe=gsp_t0_datapipe,
        history_duration=timedelta(hours=2),
    )
    sat_t0_datapipe = SelectLiveT0Time(sat_t0_datapipe)
    sat_datapipe = SelectLiveTimeSlice(
        sat_datapipe,
        t0_datapipe=sat_t0_datapipe,
        history_duration=timedelta(hours=1),
    )
    passiv_t0_datapipe = SelectLiveT0Time(pv_t0_datapipe)
    pv_datapipe = SelectLiveTimeSlice(
        pv_datapipe,
        t0_datapipe=passiv_t0_datapipe,
        history_duration=timedelta(hours=1),
    )
    gsp_datapipe = GSPIterator(gsp_datapipe)
    sat_datapipe = ConvertSatelliteToNumpyBatch(sat_datapipe, is_hrv=True)
    sat_datapipe = MergeNumpyExamplesToBatch(sat_datapipe, n_examples_per_batch=4)
    pv_datapipe = ConvertPVToNumpyBatch(pv_datapipe)
    pv_datapipe = MergeNumpyExamplesToBatch(pv_datapipe, n_examples_per_batch=4)
    gsp_datapipe = ConvertGSPToNumpyBatch(gsp_datapipe)
    gsp_datapipe = MergeNumpyExamplesToBatch(gsp_datapipe, n_examples_per_batch=4)
    nwp_datapipe = ConvertNWPToNumpyBatch(nwp_datapipe)
    nwp_datapipe = MergeNumpyExamplesToBatch(nwp_datapipe, n_examples_per_batch=4)
    combined_datapipe = MergeNumpyModalities(
        [gsp_datapipe, pv_datapipe, sat_datapipe, nwp_datapipe]
    )

    return combined_datapipe


@pytest.fixture()
def sat_hrv_np_datapipe():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "hrv_sat_data.zarr"
    dp = OpenSatellite(zarr_path=filename)
    dp = ConvertSatelliteToInt8(dp)
    dp = AddT0IdxAndSamplePeriodDuration(
        dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    dp = ConvertSatelliteToNumpyBatch(dp, is_hrv=True)
    dp = MergeNumpyExamplesToBatch(dp, n_examples_per_batch=4)
    return dp


@pytest.fixture()
def sat_np_datapipe():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "sat_data.zarr"
    dp = OpenSatellite(zarr_path=filename)
    dp = ConvertSatelliteToInt8(dp)
    dp = AddT0IdxAndSamplePeriodDuration(
        dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    dp = ConvertSatelliteToNumpyBatch(dp, is_hrv=False)
    dp = MergeNumpyExamplesToBatch(dp, n_examples_per_batch=4)
    return dp


@pytest.fixture()
def nwp_np_datapipe():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "nwp_data" / "test.zarr"
    )
    dp = OpenNWP(zarr_path=filename)
    dp = AddT0IdxAndSamplePeriodDuration(
        dp, sample_period_duration=timedelta(hours=1), history_duration=timedelta(hours=2)
    )
    # TODO Need to add t0 DataPipe before can make Numpy NWP
    # dp = MergeNumpyExamplesToBatch(dp, n_examples_per_batch=4)
    return dp


@pytest.fixture()
def passiv_np_datapipe():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "pv" / "passiv" / "test.nc"
    )
    filename_metadata = (
        Path(ocf_datapipes.__file__).parent.parent
        / "tests"
        / "data"
        / "pv"
        / "passiv"
        / "UK_PV_metadata.csv"
    )
    dp = OpenPVFromNetCDF(pv_power_filename=filename, pv_metadata_filename=filename_metadata)
    dp = AddT0IdxAndSamplePeriodDuration(
        dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    dp = ConvertPVToNumpyBatch(dp)
    dp = MergeNumpyExamplesToBatch(dp, n_examples_per_batch=4)
    return dp


@pytest.fixture()
def pvoutput_np_datapipe():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent
        / "tests"
        / "data"
        / "pv"
        / "pvoutput"
        / "test.nc"
    )
    filename_metadata = (
        Path(ocf_datapipes.__file__).parent.parent
        / "tests"
        / "data"
        / "pv"
        / "pvoutput"
        / "UK_PV_metadata.csv"
    )
    dp = OpenPVFromNetCDF(pv_power_filename=filename, pv_metadata_filename=filename_metadata)
    dp = AddT0IdxAndSamplePeriodDuration(
        dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )
    dp = ConvertPVToNumpyBatch(dp)
    dp = MergeNumpyExamplesToBatch(dp, n_examples_per_batch=4)
    return dp


@pytest.fixture()
def gsp_np_datapipe():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
    dp = OpenGSP(gsp_pv_power_zarr_path=filename)
    dp = AddT0IdxAndSamplePeriodDuration(
        dp, sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=2)
    )
    dp = ConvertGSPToNumpyBatch(dp)
    dp = MergeNumpyExamplesToBatch(dp, n_examples_per_batch=4)
    return dp
