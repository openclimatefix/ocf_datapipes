import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import (
    Base_Forecast,
    Base_PV,
    GSPYield,
    Location,
    LocationSQL,
    PVSystem,
    PVSystemSQL,
    PVYield,
    pv_output,
    solar_sheffield_passiv,
)

import ocf_datapipes
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import PV, PVFiles
from ocf_datapipes.config.save import save_yaml_configuration
from ocf_datapipes.load import OpenGSP, OpenNWP, OpenPVFromNetCDF, OpenSatellite, OpenTopography


@pytest.fixture()
def sat_hrv_datapipe():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "hrv_sat_data.zarr"
    return OpenSatellite(zarr_path=filename)


@pytest.fixture()
def sat_datapipe():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "sat_data.zarr"
    return OpenSatellite(zarr_path=filename)


@pytest.fixture()
def sat_15_datapipe():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "sat_data_15.zarr"
    return OpenSatellite(zarr_path=filename)


@pytest.fixture()
def topo_datapipe():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "europe_dem_2km_osgb.tif"
    )
    return OpenTopography(topo_filename=filename)


@pytest.fixture()
def nwp_datapipe():
    filename = (
        Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "nwp_data" / "test.zarr"
    )
    return OpenNWP(zarr_path=filename)


@pytest.fixture()
def passiv_datapipe():
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

    pv = PV(
        start_datetime=datetime(2018, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
    )
    pv_file = PVFiles(
        pv_filename=str(filename),
        pv_metadata_filename=str(filename_metadata),
        label="solar_sheffield_passiv",
    )
    pv.pv_files_groups = [pv_file]

    return OpenPVFromNetCDF(pv=pv)


@pytest.fixture()
def pvoutput_datapipe():
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

    pv = PV(
        start_datetime=datetime(2018, 1, 1, tzinfo=timezone.utc),
        end_datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
    )
    pv_file = PVFiles(
        pv_filename=str(filename),
        pv_metadata_filename=str(filename_metadata),
        label="pvoutput.org",
    )
    pv.pv_files_groups = [pv_file]

    return OpenPVFromNetCDF(pv=pv)


@pytest.fixture()
def gsp_datapipe():
    filename = Path(ocf_datapipes.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
    return OpenGSP(gsp_pv_power_zarr_path=filename)


@pytest.fixture
def db_connection():
    """Create data connection"""

    with tempfile.NamedTemporaryFile(suffix=".db") as temp:
        url = f"sqlite:///{temp.name}"
        os.environ["DB_URL_PV"] = url
        os.environ["DB_URL"] = url

        connection = DatabaseConnection(url=url, base=Base_PV, echo=False)
        from nowcasting_datamodel.models import GSPYieldSQL, LocationSQL, PVSystemSQL, PVYieldSQL

        for table in [PVYieldSQL, PVSystemSQL, GSPYieldSQL, LocationSQL]:
            table.__table__.create(connection.engine)

        yield connection

        for table in [PVYieldSQL, PVSystemSQL, GSPYieldSQL, LocationSQL]:
            table.__table__.drop(connection.engine)


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


@pytest.fixture()
def pv_yields_and_systems(db_session):
    """Create pv yields and systems

    Pv systems: Two systems
    PV yields:
        For system 1, pv yields from 4 to 10 at 5 minutes. Last one at 09.55
        For system 2: 1 pv yield at 04.00
    """

    pv_system_sql_1: PVSystemSQL = PVSystem(
        pv_system_id=1,
        provider="pvoutput.org",
        status_interval_minutes=5,
        longitude=0,
        latitude=55,
        ml_capacity_kw=123,
    ).to_orm()
    pv_system_sql_1_ss: PVSystemSQL = PVSystem(
        pv_system_id=1,
        provider=solar_sheffield_passiv,
        status_interval_minutes=5,
        longitude=0,
        latitude=57,
        ml_capacity_kw=124,
    ).to_orm()
    pv_system_sql_2: PVSystemSQL = PVSystem(
        pv_system_id=2,
        provider="pvoutput.org",
        status_interval_minutes=5,
        longitude=0,
        latitude=56,
        ml_capacity_kw=124,
    ).to_orm()
    pv_system_sql_3: PVSystemSQL = PVSystem(
        pv_system_id=3,
        provider=pv_output,
        status_interval_minutes=5,
        longitude=0,
        latitude=57,
        ml_capacity_kw=124,
    ).to_orm()

    pv_yield_sqls = []
    for hour in range(4, 10):
        for minute in range(0, 60, 5):
            pv_yield_1 = PVYield(
                datetime_utc=datetime(2022, 1, 1, hour, minute, tzinfo=timezone.utc),
                solar_generation_kw=hour + minute / 100,
            ).to_orm()
            pv_yield_1.pv_system = pv_system_sql_1
            pv_yield_sqls.append(pv_yield_1)

            pv_yield_1_ss = PVYield(
                datetime_utc=datetime(2022, 1, 1, hour, minute),
                solar_generation_kw=hour + minute / 100,
            ).to_orm()
            pv_yield_1_ss.pv_system = pv_system_sql_1_ss
            pv_yield_sqls.append(pv_yield_1_ss)

    # pv system with gaps every 5 mins
    for minutes in [0, 10, 20, 30]:

        pv_yield_4 = PVYield(
            datetime_utc=datetime(2022, 1, 1, 4, tzinfo=timezone.utc) + timedelta(minutes=minutes),
            solar_generation_kw=4,
        ).to_orm()
        pv_yield_4.pv_system = pv_system_sql_2
        pv_yield_sqls.append(pv_yield_4)

    # add a system with only on pv yield
    pv_yield_5 = PVYield(
        datetime_utc=datetime(2022, 1, 1, 4, tzinfo=timezone.utc) + timedelta(minutes=minutes),
        solar_generation_kw=4,
    ).to_orm()
    pv_yield_5.pv_system = pv_system_sql_3
    pv_yield_sqls.append(pv_yield_5)

    # add to database
    db_session.add_all(pv_yield_sqls)
    db_session.add(pv_system_sql_1)
    db_session.add(pv_system_sql_2)

    db_session.commit()

    return {
        "pv_yields": pv_yield_sqls,
        "pv_systems": [pv_system_sql_1, pv_system_sql_2],
    }


@pytest.fixture()
def gsp_yields(db_session):
    """Make fake GSP data"""

    gsps = list(range(1, 4)) + [317]

    gsp_yield_sqls = []
    for gsp_id in gsps:
        gsp_sql_1: LocationSQL = Location(
            gsp_id=gsp_id, label="GSP_1", installed_capacity_mw=1
        ).to_orm()

        for hour in range(0, 8):
            for minute in range(0, 60, 30):
                gsp_yield_1 = GSPYield(
                    datetime_utc=datetime(2022, 1, 1, hour, minute, tzinfo=timezone.utc),
                    solar_generation_kw=hour + minute,
                )
                gsp_yield_1_sql = gsp_yield_1.to_orm()
                gsp_yield_1_sql.location = gsp_sql_1
                gsp_yield_sqls.append(gsp_yield_1_sql)

    # add to database
    db_session.add_all(gsp_yield_sqls)
    db_session.commit()

    return {
        "gsp_yields": gsp_yield_sqls,
        "gsp_systems": [gsp_sql_1],
    }


@pytest.fixture()
def pv_parquet_file():
    """Create a file with PV data with the following columns

    Columns
    - timestamp
    - ss_id
    - generation_wh
    """

    date = datetime(2022, 9, 1, tzinfo=timezone.utc)
    ids = range(0, 10)
    days = 7

    data = []
    for id in ids:
        # 288 5 minutes stamps in each day
        for i in range(0, 288 * days):

            datestamp = date + timedelta(minutes=i * 5)
            if datestamp.hour > 21 or datestamp.hour < 3:
                value = 0
            else:
                value = 9.1

            data.append([datestamp, 9905 + id, value])

    data_df = pd.DataFrame(data, columns=["timestamp", "ss_id", "generation_wh"])

    data_df.loc[0:3, "generation_wh"] = np.nan

    with tempfile.TemporaryDirectory() as tmpdir:

        filename = tmpdir + "/data.parquet"
        data_df.to_parquet(filename, engine="fastparquet")
        yield filename


@pytest.fixture()
def gsp_zarr_file():
    """GSP zarr file"""

    date = datetime(2022, 9, 1)
    days = 7
    ids = np.array(range(0, 10))
    datetime_gmt = pd.to_datetime([date + timedelta(minutes=30 * i) for i in range(0, days * 24)])

    coords = (
        ("datetime_gmt", datetime_gmt),
        ("gsp_id", ids),
    )

    generation_mw = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(7 * 24, len(ids)),
            )
        ),
        coords=coords,
        name="generation_mw",
    )  # Fake data for testing!

    installedcapacity_mwp = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(7 * 24, len(ids)),
            )
        ),
        coords=coords,
        name="installedcapacity_mwp",
    )  # Fake data for testing!

    generation_mw = generation_mw.to_dataset(name="generation_mw")
    generation_mw = generation_mw.merge(installedcapacity_mwp)
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/gsp.zarr"
        generation_mw.to_zarr(filename)

        yield filename


@pytest.fixture()
def nwp_data_with_id_filename():
    """Create xarray netcdf file for NWP data

    Variables
    - init_time
    - step
    - variables
    - id
    """

    # middle of the UK
    t0_datetime_utc = datetime(2022, 9, 1)
    time_steps = 10
    days = 7
    ids = np.array(range(0, 10)) + 9905
    init_time = [t0_datetime_utc + timedelta(minutes=60 * i) for i in range(0, days * 24)]

    # time = pd.date_range(start=t0_datetime_utc, freq="30T", periods=10)
    step = [timedelta(minutes=60 * i) for i in range(0, time_steps)]

    coords = (
        ("init_time", init_time),
        ("variable", np.array(["si10", "dswrf", "t", "prate"])),
        ("step", step),
        ("id", ids),
    )

    nwp = xr.DataArray(
        abs(  # to make sure average is about 100
            np.random.uniform(
                0,
                200,
                size=(7 * 24, 4, time_steps, len(ids)),
            )
        ),
        coords=coords,
        name="data",
    )  # Fake data for testing!

    nwp = nwp.to_dataset(name="UKV")
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/nwp.netcdf"

        nwp.to_netcdf(filename, engine="h5netcdf")

        yield filename


@pytest.fixture()
def configuration():

    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")

    return load_yaml_configuration(filename=filename)


@pytest.fixture()
def configuration_with_pv_parquet(pv_parquet_file):

    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")

    configuration = load_yaml_configuration(filename=filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        configuration_filename = tmpdir + "/configuration.yaml"
        configuration.input_data.pv.pv_files_groups = [
            configuration.input_data.pv.pv_files_groups[0]
        ]
        configuration.input_data.pv.pv_files_groups[0].pv_filename = pv_parquet_file
        configuration.output_data.filepath = tmpdir
        save_yaml_configuration(configuration=configuration, filename=configuration_filename)

        yield configuration_filename


@pytest.fixture()
def configuration_with_pv_parquet_and_nwp(pv_parquet_file, nwp_data_with_id_filename):

    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")

    configuration = load_yaml_configuration(filename=filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        configuration_filename = tmpdir + "/configuration.yaml"
        configuration.input_data.pv.pv_files_groups[0].pv_filename = pv_parquet_file
        configuration.input_data.pv.pv_files_groups = [
            configuration.input_data.pv.pv_files_groups[0]
        ]
        configuration.input_data.nwp.nwp_zarr_path = nwp_data_with_id_filename
        configuration.output_data.filepath = tmpdir
        save_yaml_configuration(configuration=configuration, filename=configuration_filename)

        yield configuration_filename


@pytest.fixture()
def configuration_with_gsp_and_nwp(gsp_zarr_file, nwp_data_with_id_filename):

    filename = os.path.join(os.path.dirname(ocf_datapipes.__file__), "../tests/config/test.yaml")

    configuration = load_yaml_configuration(filename=filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        configuration_filename = tmpdir + "/configuration.yaml"
        configuration.input_data.gsp.gsp_zarr_path = gsp_zarr_file
        configuration.input_data.nwp.nwp_zarr_path = nwp_data_with_id_filename
        configuration.output_data.filepath = tmpdir
        save_yaml_configuration(configuration=configuration, filename=configuration_filename)

        yield configuration_filename
