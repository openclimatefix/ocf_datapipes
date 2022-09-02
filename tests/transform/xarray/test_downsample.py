from ocf_datapipes.transform.xarray import Downsample


def test_nwp_downsample(nwp_datapipe):
    nwp_datapipe = Downsample(nwp_datapipe, y_coarsen=16, x_coarsen=16)
    data = next(iter(nwp_datapipe))
    # Downsample by 16 from 704x548
    assert data.shape[-1] == 34
    assert data.shape[-2] == 44


def test_sat_downsample(sat_datapipe):
    sat_datapipe = Downsample(
        sat_datapipe,
        y_coarsen=16,
        x_coarsen=16,
        y_dim_name="y_geostationary",
        x_dim_name="x_geostationary",
    )
    data = next(iter(sat_datapipe))
    assert data.shape[-1] == 38
    assert data.shape[-2] == 18


def test_topo_downsample(topo_datapipe):
    topo_datapipe = Downsample(topo_datapipe, y_coarsen=16, x_coarsen=16)
    data = next(iter(topo_datapipe))
    assert data.shape == (176, 272)
