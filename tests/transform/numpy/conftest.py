from pathlib import Path

import pytest

import ocf_datapipes
from ocf_datapipes.load import OpenGSP, OpenNWP, OpenPVFromNetCDF, OpenSatellite, OpenTopography
from ocf_datapipes.transform.xarray import AddNWPTargetTime, AddT0IdxAndSamplePeriodDuration, ReprojectTopography, ConvertSatelliteToInt8
from datetime import timedelta
from ocf_datapipes.convert import ConvertSatelliteToNumpyBatch, ConvertGSPToNumpyBatch, ConvertNWPToNumpyBatch, ConvertPVToNumpyBatch

@pytest.fixture()
def sat_hrv_np_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "hrv_sat_data.zarr"
    dp = OpenSatellite(zarr_path=filename)
    dp = ConvertSatelliteToInt8(dp)
    dp = AddT0IdxAndSamplePeriodDuration(dp, sample_period_duration=timedelta(minutes=5), history_duration=timedelta(minutes=60))
    dp = ConvertSatelliteToNumpyBatch(dp, is_hrv=True)
    return dp


@pytest.fixture()
def sat_np_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "sat_data.zarr"
    return OpenSatellite(zarr_path=filename)


@pytest.fixture()
def sat_15_np_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "sat_data_15.zarr"
    return OpenSatellite(zarr_path=filename)


@pytest.fixture()
def topo_np_dp():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "europe_dem_2km_osgb.tif"
    )
    return OpenTopography(topo_filename=filename)


@pytest.fixture()
def nwp_np_dp():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "nwp_data" / "test.zarr"
    )
    return OpenNWP(zarr_path=filename)


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
    return OpenPVFromNetCDF(pv_power_filename=filename, pv_metadata_filename=filename_metadata)


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
    return OpenPVFromNetCDF(pv_power_filename=filename, pv_metadata_filename=filename_metadata)


@pytest.fixture()
def gsp_np_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
    return OpenGSP(gsp_pv_power_zarr_path=filename)
