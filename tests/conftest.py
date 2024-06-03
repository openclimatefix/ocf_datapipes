import os
import tempfile
from datetime import datetime, timedelta
import uuid


import numpy as np
import pandas as pd
import pytest
import xarray as xr
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import (
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
from ocf_datapipes.load import (
    OpenGSP,
    OpenNWP,
    OpenPVFromNetCDF,
    OpenSatellite,
    OpenTopography,
)

xr.set_options(keep_attrs=True)

# This path is used both here and in tests in deeper test directories
# Make two ways to easily access it
# Weirdly using pytest.fixture(autouse=True) wasn't working - TODO?
_top_test_directory = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def top_test_directory():
    return _top_test_directory


@pytest.fixture()
def sat_hrv_datapipe():
    filename = _top_test_directory + "/data/hrv_sat_data.zarr"
    return OpenSatellite(zarr_path=filename)


@pytest.fixture()
def sat_datapipe():
    filename = f"{_top_test_directory}/data/sat_data.zarr"
    # The saved data is scaled from 0-1024. Now we use data scaled from 0-1
    # Rescale here for subsequent tests
    return OpenSatellite(zarr_path=filename).map(lambda da: da / 1024)


@pytest.fixture()
def sat_15_datapipe():
    filename = f"{_top_test_directory}/data/sat_data_15.zarr"
    # The saved data is scaled from 0-1024. Now we use data scaled from 0-1
    # Rescale here for subsequent tests
    return OpenSatellite(zarr_path=filename).map(_sat_rescale)


@pytest.fixture()
def topo_datapipe():
    filename = f"{_top_test_directory}/data/europe_dem_2km_osgb.tif"
    return OpenTopography(topo_filename=filename)


@pytest.fixture()
def nwp_datapipe():
    filename = f"{_top_test_directory}/data/nwp_data/test.zarr"
    return OpenNWP(zarr_path=filename)


@pytest.fixture()
def icon_eu_datapipe():
    filename = f"{_top_test_directory}/data/icon_eu.zarr"
    return OpenNWP(zarr_path=filename, provider="icon-eu")


@pytest.fixture()
def icon_global_datapipe():
    filename = f"{_top_test_directory}/data/icon_global.zarr"
    return OpenNWP(zarr_path=filename, provider="icon-global")


@pytest.fixture()
def passiv_datapipe():
    filename = f"{_top_test_directory}/data/pv/passiv/test.nc"
    filename_metadata = f"{_top_test_directory}/data/pv/passiv/UK_PV_metadata.csv"

    pv = PV()
    pv_file = PVFiles(
        pv_filename=str(filename),
        pv_metadata_filename=str(filename_metadata),
        label="solar_sheffield_passiv",
    )
    pv.pv_files_groups = [pv_file]

    return OpenPVFromNetCDF(pv=pv)


@pytest.fixture()
def pvoutput_datapipe():
    filename = f"{_top_test_directory}/data/pv/pvoutput/test.nc"
    filename_metadata = f"{_top_test_directory}/data/pv/pvoutput/UK_PV_metadata.csv"

    pv = PV()
    pv_file = PVFiles(
        pv_filename=str(filename),
        pv_metadata_filename=str(filename_metadata),
        label="pvoutput.org",
    )
    pv.pv_files_groups = [pv_file]

    return OpenPVFromNetCDF(pv=pv)


@pytest.fixture()
def gsp_datapipe():
    filename = f"{_top_test_directory}/data/gsp/test.zarr"
    return OpenGSP(gsp_pv_power_zarr_path=filename)


@pytest.fixture
def pv_system_db_data():
    # Create generation data
    n_systems = 10

    t0 = pd.Timestamp.now().floor("5min")
    datetimes = pd.date_range(t0 - timedelta(minutes=120), t0, freq="5min")
    site_uuids = [str(uuid.uuid4()) for _ in range(n_systems)]

    data = np.zeros((len(datetimes), n_systems))

    # Make data a nice sinusoidal curve
    data[:] = (
        0.5
        * (1 - np.cos((datetimes.hour + datetimes.minute / 60) / 24 * 2 * np.pi).values)[:, None]
    )

    # Chuck in some nan values
    data[:, 1] = np.nan
    data[-5:, 2] = np.nan
    data[::3, 3] = np.nan

    da = xr.DataArray(
        data,
        coords=(
            ("end_utc", datetimes),
            ("site_uuid", site_uuids),
        ),
    )

    # Reshape for tabular database
    df_gen = da.to_dataframe("generation_power_kw").reset_index()
    df_gen["start_utc"] = df_gen["end_utc"] - timedelta(minutes=5)

    # Create metadata
    df_meta = pd.DataFrame(
        dict(
            site_uuid=site_uuids,
            orientation=np.random.uniform(0, 360, n_systems),
            tilt=np.random.uniform(0, 90, n_systems),
            longitude=np.random.uniform(-3.07, 0.59, n_systems),
            latitude=np.random.uniform(51.56, 52.89, n_systems),
            capacity_kw=np.random.uniform(1, 5, n_systems),
            ml_id=np.arange(n_systems),
        )
    )

    return df_gen, df_meta


@pytest.fixture
def db_connection(pv_system_db_data):
    """Create data connection"""

    with tempfile.NamedTemporaryFile(suffix=".db") as temp:
        url = f"sqlite:///{temp.name}"
        os.environ["DB_URL_PV"] = url
        os.environ["DB_URL"] = url

        connection = DatabaseConnection(url=url, base=Base_PV, echo=False)
        from nowcasting_datamodel.models import (
            GSPYieldSQL,
            LocationSQL,
            PVSystemSQL,
            PVYieldSQL,
        )

        for table in [PVYieldSQL, PVSystemSQL, GSPYieldSQL, LocationSQL]:
            table.__table__.create(connection.engine)

        # Create and populate pvsites tables
        df_gen, df_meta = pv_system_db_data
        with connection.engine.connect() as conn:
            df_gen.to_sql(name="generation", con=conn, index=False)
            df_meta.to_sql(name="sites", con=conn, index=False)

        yield connection


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
                datetime_utc=datetime(2022, 1, 1, hour, minute),
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
            datetime_utc=datetime(2022, 1, 1, 4) + timedelta(minutes=minutes),
            solar_generation_kw=4,
        ).to_orm()
        pv_yield_4.pv_system = pv_system_sql_2
        pv_yield_sqls.append(pv_yield_4)

    # add a system with only on pv yield
    pv_yield_5 = PVYield(
        datetime_utc=datetime(2022, 1, 1, 4) + timedelta(minutes=minutes),
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

    gsps = list(range(0, 4)) + [18, 317]

    gsp_yield_sqls = []
    for gsp_id in gsps:
        gsp_sql_1: LocationSQL = Location(
            gsp_id=gsp_id, label=f"GSP_{gsp_id}", installed_capacity_mw=1
        ).to_orm()

        for hour in range(0, 8):
            for minute in range(0, 60, 30):
                gsp_yield_1 = GSPYield(
                    datetime_utc=datetime(2022, 1, 1, hour, minute),
                    solar_generation_kw=hour + minute,
                    capacity_mwp=1,
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
def pv_xarray_data():
    datetimes = pd.date_range("2022-09-01 00:00", "2022-09-08 00:00", freq="5min")
    pv_system_ids = (np.arange(10) + 9905).astype(str)

    data = np.full((len(datetimes), len(pv_system_ids)), fill_value=9.1)
    da = xr.DataArray(
        data,
        coords=(
            ("datetime", datetimes),
            ("pv_system_id", pv_system_ids),
        ),
    )

    da = da.where(da.datetime.dt.hour < 21, other=0)
    da = da.where(da.datetime.dt.hour > 3, other=0)

    da.isel(datetime=slice(0, 3), pv_system_id=0).values[:] = np.nan
    return da


@pytest.fixture()
def pv_netcdf_file(pv_xarray_data):
    """Create a netcdf file with PV data with the following dimensions

    - datetime
    - pv_system_id
    """

    ds = pv_xarray_data.to_dataset(dim="pv_system_id")

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "data.nc"
        ds.to_netcdf(filename, engine="h5netcdf")
        yield filename


@pytest.fixture()
def pv_parquet_file(pv_xarray_data):
    """Create a parquet file with PV data with the following columns

    Columns
    - timestamp
    - ss_id
    - generation_wh
    """

    # Convert to watt-hours energy for each 5-minute step
    da = pv_xarray_data / 12

    # Flatten into DataFrame and rename
    data_df = da.to_dataframe("generation_wh").reset_index(level=[0, 1])
    data_df = data_df.rename(dict(datetime="timestamp", pv_system_id="ss_id"), axis=1)

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
        np.random.uniform(
            0,
            200,
            size=(7 * 24, len(ids)),
        ),
        coords=coords,
        name="generation_mw",
    )

    installedcapacity_mwp = xr.DataArray(
        np.random.uniform(
            0,
            200,
            size=(7 * 24, len(ids)),
        ),
        coords=coords,
        name="installedcapacity_mwp",
    )

    capacity_mwp = xr.DataArray(
        np.random.uniform(
            0,
            200,
            size=(7 * 24, len(ids)),
        ),
        coords=coords,
        name="capacity_mwp",
    )

    generation_mw = generation_mw.to_dataset(name="generation_mw")
    generation_mw = generation_mw.merge(installedcapacity_mwp)
    generation_mw = generation_mw.merge(capacity_mwp)
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

    init_times = pd.date_range(start=datetime(2022, 9, 1), freq="60min", periods=24 * 7)
    steps = [timedelta(minutes=60 * i) for i in range(0, 11)]
    variables = ["si10", "dswrf", "t", "prate"]
    ids = np.array(range(0, 10)) + 9905

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("id", ids),
    )

    nwp_array_shape = (len(init_times), len(variables), len(steps), len(ids))

    nwp_data = xr.DataArray(
        np.random.uniform(0, 200, size=nwp_array_shape),
        coords=coords,
    )

    nwp_data = nwp_data.to_dataset(name="UKV")
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/nwp.zarr"

        nwp_data.to_zarr(filename, engine="h5netcdf")

        yield filename


@pytest.fixture()
def nwp_gfs_data():
    """Create xarray netcdf file for NWP data

    Variables
    - init_time
    - step
    - latitude
    - longitude
    """

    init_times = pd.date_range(start=datetime(2022, 9, 1), freq="60min", periods=24 * 7)
    steps = [timedelta(minutes=60 * i) for i in range(0, 11)]
    x = np.array(range(0, 10))
    y = np.array(range(0, 10))
    variables = ["si10", "dswrf", "t", "prate"]

    coords = (
        ("time", init_times),
        ("step", steps),
        ("longitude", x),
        ("latitude", y),
    )

    nwp_array_shape = (len(init_times), len(steps), len(x), len(y))

    data_arrays = []

    for variable in variables:
        data_arrays += [
            xr.DataArray(
                np.random.uniform(0, 200, size=nwp_array_shape),
                coords=coords,
                name=variable,
            )
        ]

    nwp_data = xr.merge(data_arrays)
    return nwp_data


@pytest.fixture()
def nwp_gfs_data_filename(nwp_gfs_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/nwp.zarr"
        nwp_gfs_data.to_zarr(filename)
        yield filename


@pytest.fixture()
def nwp_ukv_data():
    init_times = pd.date_range(start=datetime(2022, 9, 1), freq="180min", periods=24 * 7)
    steps = [timedelta(minutes=60 * i) for i in range(0, 11)]

    # These are the values from the training data but it takes too long:
    # -> x = np.arange(-239_000, 857_000, 2000) # Shape:  (548,)
    # -> y = np.arange(-183_000, 1225_000, 2000)[::-1] # Shape:  (704,)

    # This is much faster:
    x = np.linspace(-239_000, 857_000, 100)
    y = np.linspace(-183_000, 1225_000, 100)[::-1]  # UKV data must run top to bottom
    variables = ["si10", "dswrf", "t", "prate"]

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("x", x),
        ("y", y),
    )

    nwp_array_shape = (len(init_times), len(variables), len(steps), len(x), len(y))

    nwp_data = xr.DataArray(
        np.random.uniform(0, 200, size=nwp_array_shape),
        coords=coords,
    )
    return nwp_data.to_dataset(name="UKV")


@pytest.fixture()
def nwp_ukv_data_filename(nwp_ukv_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/nwp.zarr"
        nwp_ukv_data.to_zarr(filename)
        yield filename


@pytest.fixture()
def pvnet_config_filename():
    return f"{_top_test_directory}/data/configs/pvnet_test_config.yaml"


@pytest.fixture()
def configuration_filename():
    return f"{_top_test_directory}/data/configs/test.yaml"


@pytest.fixture()
def configuration():
    filename = f"{_top_test_directory}/data/configs/test.yaml"
    return load_yaml_configuration(filename=filename)


@pytest.fixture()
def configuration_no_gsp():
    filename = f"{_top_test_directory}/data/configs/wind_test.yaml"
    return load_yaml_configuration(filename=filename)


@pytest.fixture()
def configuration_with_pv_netcdf(pv_netcdf_file):
    filename = f"{_top_test_directory}/data/configs/test.yaml"

    configuration = load_yaml_configuration(filename=filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        configuration_filename = tmpdir + "/configuration.yaml"
        configuration.input_data.pv.pv_files_groups = [
            configuration.input_data.pv.pv_files_groups[0]
        ]
        configuration.input_data.pv.pv_files_groups[0].pv_filename = pv_netcdf_file
        save_yaml_configuration(configuration=configuration, filename=configuration_filename)

        yield configuration_filename


@pytest.fixture()
def configuration_with_pv_netcdf_and_nwp(pv_netcdf_file, nwp_ukv_data_filename):
    filename = f"{_top_test_directory}/data/configs/test.yaml"

    configuration = load_yaml_configuration(filename=filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        configuration_filename = tmpdir + "/configuration.yaml"
        configuration.input_data.pv.pv_files_groups[0].pv_filename = pv_netcdf_file
        configuration.input_data.pv.pv_files_groups = [
            configuration.input_data.pv.pv_files_groups[0]
        ]
        configuration.input_data.nwp["ukv"].nwp_zarr_path = nwp_ukv_data_filename
        save_yaml_configuration(configuration=configuration, filename=configuration_filename)

        yield configuration_filename


@pytest.fixture()
def configuration_with_gsp_and_nwp(gsp_zarr_file, nwp_ukv_data_filename):
    filename = f"{_top_test_directory}/data/configs/test.yaml"

    configuration = load_yaml_configuration(filename=filename)
    with tempfile.TemporaryDirectory() as tmpdir:
        configuration_filename = tmpdir + "/configuration.yaml"
        configuration.input_data.gsp.gsp_zarr_path = gsp_zarr_file
        configuration.input_data.nwp["ukv"].nwp_zarr_path = nwp_ukv_data_filename
        save_yaml_configuration(configuration=configuration, filename=configuration_filename)

        yield configuration_filename


@pytest.fixture()
def wind_configuration_filename():
    return f"{_top_test_directory}/data/configs/wind_test.yaml"
