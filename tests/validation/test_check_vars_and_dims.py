from ocf_datapipes.validation import CheckVarsAndDims


def test_check_vars_and_dims_gsp(gsp_dp):
    gsp_dp = gsp_dp.check_vars_and_dims(expected_dimensions=("time_utc", "gsp_id"))
    next(iter(gsp_dp))


def test_check_vars_and_dims_sat(sat_dp):
    sat_dp = sat_dp.check_vars_and_dims(
        expected_dimensions=("time_utc", "channel", "x_geostationary", "y_geostationary")
    )
    next(iter(sat_dp))


def test_check_vars_and_dim_passiv(passiv_dp):
    passiv_dp = passiv_dp.check_vars_and_dims(expected_dimensions=("time_utc", "pv_system_id"))
    next(iter(passiv_dp))


def test_check_vars_and_dim_nwp(nwp_dp):
    nwp_dp = nwp_dp.check_vars_and_dims(
        expected_dimensions=("init_time_utc", "channel", "step", "x_osgb", "y_osgb")
    )
    next(iter(nwp_dp))


def test_check_vars_and_dim_topo(topo_dp):
    topo_dp = topo_dp.check_vars_and_dims(expected_dimensions=("x_osgb", "y_osgb"))
    next(iter(topo_dp))
