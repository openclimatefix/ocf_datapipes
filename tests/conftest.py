import os
import tempfile
from pathlib import Path

import pytest
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import Base_Forecast, Base_PV

import ocf_datapipes
from ocf_datapipes.load import OpenGSP, OpenNWP, OpenPVFromNetCDF, OpenSatellite, OpenTopography


@pytest.fixture()
def sat_hrv_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "hrv_sat_data.zarr"
    return OpenSatellite(zarr_path=filename)


@pytest.fixture()
def sat_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "sat_data.zarr"
    return OpenSatellite(zarr_path=filename)


@pytest.fixture()
def sat_15_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "sat_data_15.zarr"
    return OpenSatellite(zarr_path=filename)


@pytest.fixture()
def topo_dp():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "europe_dem_2km_osgb.tif"
    )
    return OpenTopography(topo_filename=filename)


@pytest.fixture()
def nwp_dp():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "nwp_data" / "test.zarr"
    )
    return OpenNWP(zarr_path=filename)


@pytest.fixture()
def passiv_dp():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "pv" / "passiv" / "test.nc"
    )
    filename_metadata = (
        Path(ocf_datapipes.__file__).parent.parent
        / "tests"
        / "data"
        / "pv"
        / "passiv"
        / "UK_PV_metadata.csv"
    )
    return OpenPVFromNetCDF(pv_power_filename=filename, pv_metadata_filename=filename_metadata)


@pytest.fixture()
def pvoutput_dp():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent
        / "tests"
        / "data"
        / "pv"
        / "pvoutput"
        / "test.nc"
    )
    filename_metadata = (
        Path(ocf_datapipes.__file__).parent.parent
        / "tests"
        / "data"
        / "pv"
        / "pvoutput"
        / "UK_PV_metadata.csv"
    )
    return OpenPVFromNetCDF(pv_power_filename=filename, pv_metadata_filename=filename_metadata)


@pytest.fixture()
def gsp_dp():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
    return OpenGSP(gsp_pv_power_zarr_path=filename)


@pytest.fixture
def db_connection():
    """Create data connection"""

    with tempfile.NamedTemporaryFile(suffix=".db") as temp:
        url = f"sqlite:///{temp.name}"
        os.environ["DB_URL_PV"] = url
        os.environ["DB_URL"] = url

        connection = DatabaseConnection(url=url, base=Base_PV)
        Base_PV.metadata.create_all(connection.engine)
        Base_Forecast.metadata.create_all(connection.engine)

        yield connection

        Base_PV.metadata.drop_all(connection.engine)
        Base_Forecast.metadata.create_all(connection.engine)


@pytest.fixture(scope="function", autouse=True)
def db_session(db_connection):
    """Creates a new database session for a test."""

    connection = db_connection.engine.connect()
    t = connection.begin()

    with db_connection.get_session() as s:
        s.begin()
        yield s
        s.rollback()

    t.rollback()
    connection.close()

