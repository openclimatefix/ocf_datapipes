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
from ocf_datapipes.transform.xarray import (
    AddNWPTargetTime,
    AddT0IdxAndSamplePeriodDuration,
    ConvertSatelliteToInt8,
    ReprojectTopography,
EnsureNPVSystemsPerExample
)

from ocf_datapipes.select import SelectTimeSlice, SelectSpatialSlicePixels, SelectSpatialSliceMeters, LocationPicker

class GSPIterator(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self):
        for xr_dataset in self.source_dp:
            # Iterate through all locations in dataset
            for location_idx in range(len(xr_dataset["x_osgb"])):
                yield xr_dataset.isel(gsp_id=slice(location_idx,location_idx+1))

@pytest.fixture()
def all_loc_np_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "hrv_sat_data.zarr"
    dp = OpenSatellite(zarr_path=filename)
    dp = ConvertSatelliteToInt8(dp)
    sat_dp = AddT0IdxAndSamplePeriodDuration(
        dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
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
    pv_dp = OpenPVFromNetCDF(pv_power_filename=filename, pv_metadata_filename=filename_metadata)
    pv_dp = AddT0IdxAndSamplePeriodDuration(
        pv_dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60)
    )

    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
    gsp_dp = OpenGSP(gsp_pv_power_zarr_path=filename)
    gsp_dp = AddT0IdxAndSamplePeriodDuration(
        gsp_dp, sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=2)
    )

    location_dp1, location_dp2 = Forker(LocationPicker(gsp_dp, return_all_locations=True), 2) # Its in order then
    # TODO Add t0 selector
    pv_dp = SelectSpatialSliceMeters(pv_dp, location_datapipe=location_dp1, roi_width_meters=960_000, roi_height_meters=960_000) # Has to be large as test PV systems aren't in first 20 GSPs it seems
    pv_dp = EnsureNPVSystemsPerExample(pv_dp, n_pv_systems_per_example=8)
    sat_dp = SelectSpatialSlicePixels(sat_dp, location_datapipe=location_dp2, roi_width_pixels=256, roi_height_pixels=128, y_dim_name="y_geostationary", x_dim_name="x_geostationary")
    gsp_dp = GSPIterator(gsp_dp)
    sat_dp = ConvertSatelliteToNumpyBatch(sat_dp, is_hrv=True)
    sat_dp = MergeNumpyExamplesToBatch(sat_dp, n_examples_per_batch=4)
    pv_dp = ConvertPVToNumpyBatch(pv_dp)
    pv_dp = MergeNumpyExamplesToBatch(pv_dp, n_examples_per_batch=4)
    gsp_dp = ConvertGSPToNumpyBatch(gsp_dp)
    gsp_dp = MergeNumpyExamplesToBatch(gsp_dp, n_examples_per_batch=4)
    combined_dp = MergeNumpyModalities([gsp_dp, pv_dp, sat_dp])

    return combined_dp






@pytest.fixture()
def sat_hrv_np_dp():
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
def sat_np_dp():
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
def nwp_np_dp():
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
def passiv_np_dp():
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
def pvoutput_np_dp():
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
def gsp_np_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
    dp = OpenGSP(gsp_pv_power_zarr_path=filename)
    dp = AddT0IdxAndSamplePeriodDuration(
        dp, sample_period_duration=timedelta(minutes=30), history_duration=timedelta(hours=2)
    )
    dp = ConvertGSPToNumpyBatch(dp)
    dp = MergeNumpyExamplesToBatch(dp, n_examples_per_batch=4)
    return dp
