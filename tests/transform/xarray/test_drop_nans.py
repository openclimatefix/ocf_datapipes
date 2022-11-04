from ocf_datapipes.config.model import PV, PVFiles
from ocf_datapipes.load import OpenPVFromNetCDF


def test_remove_nans_all_pv_systems(pv_parquet_file):

    pv = PV()
    pv_file = PVFiles(
        pv_filename="tests/data/pv/passiv/test.nc",
        pv_metadata_filename="tests/data/pv/passiv/UK_PV_metadata.csv",
        label="solar_sheffield_passiv",
    )
    pv.pv_files_groups = [pv_file]

    pv_datapipe = OpenPVFromNetCDF(pv=pv)

    pv_datapipe = pv_datapipe.remove_nans()
    pv_datapipe = pv_datapipe.check_nans()

    data = next(iter(pv_datapipe))
    assert data is not None
