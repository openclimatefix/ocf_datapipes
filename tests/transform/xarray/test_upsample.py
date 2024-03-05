from ocf_datapipes.transform.xarray import Upsample


def test_nwp_upsample(nwp_datapipe):
    nwp_datapipe = Upsample(
        nwp_datapipe, y_upsample=2, x_upsample=2, x_dim_name="x_osgb", y_dim_name="y_osgb"
    )
    data = next(iter(nwp_datapipe))

    # Upsample by 2 from 704x548
    assert data.shape[-1] == 548 * 2 - 1
    assert data.shape[-2] == 704 * 2 - 1


def test_sat_downsample(sat_datapipe):
    sat_datapipe = Upsample(
        sat_datapipe,
        y_upsample=2,
        x_upsample=2,
        y_dim_name="y_geostationary",
        x_dim_name="x_geostationary",
    )
    data = next(iter(sat_datapipe))
    assert data.shape[-1] == 615 * 2
    assert data.shape[-2] == 298 * 2 - 1


def test_nwp_upsample_keep_same_shape(nwp_datapipe):
    nwp_datapipe_new, nwp_datapipe_old = nwp_datapipe.fork(2)

    nwp_datapipe_new = Upsample(
        nwp_datapipe_new,
        y_upsample=2,
        x_upsample=2,
        x_dim_name="x_osgb",
        y_dim_name="y_osgb",
        keep_same_shape=True,
    )
    data_new = next(iter(nwp_datapipe_new))
    data_old = next(iter(nwp_datapipe_old))

    assert data_new.shape[-1] == 548
    assert data_new.shape[-2] == 704

    # check first values are different
    assert data_new.x_osgb.values[0] != data_old.x_osgb.values[0]
    assert data_new.y_osgb.values[0] != data_old.y_osgb.values[0]

    # check middle valyes are the same
    assert data_new.x_osgb.values[274] == data_old.x_osgb.values[274]
    assert data_new.y_osgb.values[351] == data_old.y_osgb.values[352]
