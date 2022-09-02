from ocf_datapipes.validation import CheckVarsAndDims


def test_check_vars_and_dims_gsp(gsp_datapipe):
    gsp_datapipe = gsp_datapipe.check_vars_and_dims(expected_dimensions=("time_utc", "gsp_id"))
    next(iter(gsp_datapipe))


def test_check_vars_and_dims_sat(sat_datapipe):
    sat_datapipe = sat_datapipe.check_vars_and_dims(
        expected_dimensions=("time_utc", "channel", "x_geostationary", "y_geostationary")
    )
    next(iter(sat_datapipe))


def test_check_vars_and_dim_passiv(passiv_datapipe):
    passiv_datapipe = passiv_datapipe.check_vars_and_dims(
        expected_dimensions=("time_utc", "pv_system_id")
    )
    next(iter(passiv_datapipe))


def test_check_vars_and_dim_nwp(nwp_datapipe):
    nwp_datapipe = nwp_datapipe.check_vars_and_dims(
        expected_dimensions=("init_time_utc", "channel", "step", "x_osgb", "y_osgb")
    )
    next(iter(nwp_datapipe))


def test_check_vars_and_dim_topo(topo_datapipe):
    topo_datapipe = topo_datapipe.check_vars_and_dims(expected_dimensions=("x_osgb", "y_osgb"))
    next(iter(topo_datapipe))
