from ocf_datapipes.load import OpenNWPID
import xarray as xr


def test_make_fake_data(nwp_data_with_id_filename):
    _ = xr.open_dataset(
        nwp_data_with_id_filename,
        engine="h5netcdf",
        chunks="auto",
    )


def test_load_nwp(nwp_data_with_id_filename):
    nwp_datapipe = OpenNWPID(netcdf_path=nwp_data_with_id_filename)
    nwp = next(iter(nwp_datapipe))
    assert nwp is not None
