from ocf_datapipes.load import OpenPVFromDB, OpenPVFromNetCDF


def test_open_passiv_from_nc():
    pv_datapipe = OpenPVFromNetCDF(
        pv_power_filename="tests/data/pv/passiv/test.nc",
        pv_metadata_filename="tests/data/pv/passiv/UK_PV_metadata.csv",
    )
    data = next(iter(pv_datapipe))
    assert data is not None


def test_open_pvoutput_from_nc():
    pv_datapipe = OpenPVFromNetCDF(
        pv_power_filename="tests/data/pv/pvoutput/test.nc",
        pv_metadata_filename="tests/data/pv/pvoutput/UK_PV_metadata.csv",
    )
    data = next(iter(pv_datapipe))
    assert data is not None
