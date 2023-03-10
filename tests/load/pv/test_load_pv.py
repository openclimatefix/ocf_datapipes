from datetime import datetime, timezone

from ocf_datapipes.config.model import PV, PVFiles
from ocf_datapipes.load import OpenPVFromNetCDF


def test_open_passiv_from_nc():
    pv = PV()
    pv_file = PVFiles(
        pv_filename="tests/data/pv/passiv/test.nc",
        pv_metadata_filename="tests/data/pv/passiv/UK_PV_metadata.csv",
        label="solar_sheffield_passiv",
    )
    pv.pv_files_groups = [pv_file]

    pv_datapipe = OpenPVFromNetCDF(pv=pv)
    data = next(iter(pv_datapipe))
    assert data is not None
    assert len(data.pv_system_id) == 2


def test_open_passiv_and_inferred_metadata_from_nc():
    pv = PV()
    pv_file = PVFiles(
        pv_filename="tests/data/pv/passiv/test.nc",
        pv_metadata_filename="tests/data/pv/passiv/UK_PV_metadata.csv",
        inferred_metadata_filename="tests/data/pv/passiv/meta_inferred.csv",
        label="solar_sheffield_passiv",
    )

    pv.pv_files_groups = [pv_file]

    pv_datapipe = OpenPVFromNetCDF(pv=pv)
    data = next(iter(pv_datapipe))
    assert data is not None
    assert len(data.pv_system_id) == 2


def test_open_pvoutput_from_nc():
    pv = PV()
    pv_file = PVFiles(
        pv_filename="tests/data/pv/pvoutput/test.nc",
        pv_metadata_filename="tests/data/pv/pvoutput/UK_PV_metadata.csv",
    )
    pv.pv_files_groups = [pv_file]

    pv_datapipe = OpenPVFromNetCDF(pv=pv)
    data = next(iter(pv_datapipe))
    assert data is not None
    assert len(data.pv_system_id) == 41


def test_open_both_from_nc():
    pv = PV()
    pv_file = PVFiles(
        pv_filename="tests/data/pv/pvoutput/test.nc",
        pv_metadata_filename="tests/data/pv/pvoutput/UK_PV_metadata.csv",
    )
    pv_file_passiv = PVFiles(
        pv_filename="tests/data/pv/passiv/test.nc",
        pv_metadata_filename="tests/data/pv/passiv/UK_PV_metadata.csv",
    )
    pv.pv_files_groups = [pv_file, pv_file_passiv]

    pv_datapipe = OpenPVFromNetCDF(pv=pv)
    data = next(iter(pv_datapipe))
    assert data is not None
    assert len(data.pv_system_id) == 43


def test_load_parquet_file(pv_parquet_file):
    pv = PV(
        start_datetime=datetime(2018, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
    )
    pv_file = PVFiles(
        pv_filename=pv_parquet_file,
        pv_metadata_filename="tests/data/pv/passiv/UK_PV_metadata.csv",
        label="solar_sheffield_passiv",
    )
    pv.pv_files_groups = [pv_file]

    pv_datapipe = OpenPVFromNetCDF(pv=pv)
    data = next(iter(pv_datapipe))
    assert data is not None
