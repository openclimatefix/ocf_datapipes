import pytest

from ocf_datapipes.load import OpenNWP, OpenSatellite, OpenTopography
from ocf_datapipes.transform.xarray import Downsample


def test_nwp_downsample():
    nwp_dp = OpenNWP(zarr_path="tests/data/nwp_data/test.zarr")
    nwp_dp = Downsample(nwp_dp, y_coarsen=16, x_coarsen=16)
    data = next(iter(nwp_dp))
    # Downsample by 16 from 704x548
    assert data.shape[-1] == 34
    assert data.shape[-2] == 44


@pytest.mark.skip("Unskip once everything converted to Lat/Lon")
def test_sat_downsample():
    sat_dp = OpenSatellite(zarr_path="tests/data/hrv_sat_data.zarr")
    sat_dp = Downsample(sat_dp, y_coarsen=16, x_coarsen=16)
    data = next(iter(sat_dp))
    # TODO Update for actual values
    assert data.shape[-1] == 34
    assert data.shape[-2] == 44


def test_topo_downsample():
    topo_dp = OpenTopography(
        topo_filename="tests/data/europe_dem_2km_osgb.tif"
    )
    topo_dp = Downsample(topo_dp, y_coarsen=16, x_coarsen=16)
    data = next(iter(topo_dp))
    assert data.shape == (176, 272)
