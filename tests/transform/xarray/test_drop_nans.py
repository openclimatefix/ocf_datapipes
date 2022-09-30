from ocf_datapipes.load import OpenPVFromNetCDF


def test_remove_nans_all_pv_systems(pv_parquet_file):

    pv_datapipe = OpenPVFromNetCDF(
        pv_power_filename=pv_parquet_file,
        pv_metadata_filename="tests/data/pv/passiv/UK_PV_metadata.csv",
    )

    pv_datapipe = pv_datapipe.remove_nans()
    pv_datapipe = pv_datapipe.check_nans()

    data = next(iter(pv_datapipe))
    assert data is not None
