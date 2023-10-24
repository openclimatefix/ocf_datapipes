""" Test for loading pv data from database """
from datetime import datetime, timedelta, timezone

import pandas as pd
from freezegun import freeze_time
from nowcasting_datamodel.models import PVSystem, PVSystemSQL, pv_output

from ocf_datapipes.config.model import PV, PVFiles
from ocf_datapipes.load.pv.database import (
    OpenPVFromDBIterDataPipe,
    get_metadata_from_database,
    get_pv_power_from_database,
    OpenPVFromPVSitesDBIterDataPipe,
    get_metadata_from_pvsites_database,
    get_pv_power_from_pvsites_database,
)


def test_get_metadata_from_database(pv_yields_and_systems):
    """Test get meteadata from database"""
    meteadata = get_metadata_from_database()
    assert len(meteadata) == 4


@freeze_time("2022-01-01 08:00")
def test_get_pv_power_from_database_no_pv_yields(db_session):
    """Test that nans are return when there are no pv yields in the database"""

    pv_system_sql_1: PVSystemSQL = PVSystem(
        pv_system_id=1,
        provider=pv_output,
        status_interval_minutes=5,
        longitude=0,
        latitude=55,
        ml_capacity_kw=123,
    ).to_orm()
    db_session.add(pv_system_sql_1)
    pv_system_sql_2: PVSystemSQL = PVSystem(
        pv_system_id=2,
        provider=pv_output,
        status_interval_minutes=5,
        longitude=0,
        latitude=55,
        ml_capacity_kw=123,
    ).to_orm()
    db_session.add(pv_system_sql_2)
    db_session.commit()

    """Get pv power from database"""
    pv_power = get_pv_power_from_database(
        history_duration=timedelta(hours=1),
        load_extra_minutes=30,
        interpolate_minutes=30,
        load_extra_minutes_and_keep=30,
    )

    assert len(pv_power) == 19  # 1.5 hours at 5 mins = 6*5
    assert len(pv_power.columns) == 2
    assert pv_power.columns[0] == 10
    assert (
        pd.to_datetime(pv_power.index[0]).isoformat()
        == datetime(2022, 1, 1, 6, 30, tzinfo=timezone.utc).isoformat()
    )
    # some values have been filled with 0.0
    assert pv_power.isna().sum().sum() == 22


@freeze_time("2022-01-01 05:00")
def test_get_pv_power_from_database(pv_yields_and_systems):
    """Get pv power from database"""
    pv_power = get_pv_power_from_database(
        history_duration=timedelta(hours=1),
        load_extra_minutes=30,
        interpolate_minutes=30,
        load_extra_minutes_and_keep=30,
    )

    assert len(pv_power) == 19  # 1.5 hours at 5 mins = 6*12
    assert len(pv_power.columns) == 2
    assert pv_power.columns[0] == 10
    assert (
        pd.to_datetime(pv_power.index[0]).isoformat()
        == datetime(2022, 1, 1, 3, 30, tzinfo=timezone.utc).isoformat()
    )


@freeze_time("2022-01-01 10:54:59")
def test_get_pv_power_from_database_interpolate(pv_yields_and_systems):
    """Get pv power from database, test out get extra minutes and interpolate"""

    pv_power = get_pv_power_from_database(
        history_duration=timedelta(hours=0.5),
        load_extra_minutes=0,
        interpolate_minutes=0,
        load_extra_minutes_and_keep=0,
    )
    assert len(pv_power) == 7  # last data point is at 09:55, but we now get nans
    assert pv_power.isna().sum().sum() == 7 * 4

    pv_power = get_pv_power_from_database(
        history_duration=timedelta(hours=1),
        load_extra_minutes=60,
        interpolate_minutes=30,
        load_extra_minutes_and_keep=30,
    )
    assert len(pv_power) == 19  # 1.5 hours at 5 mins = 12
    assert pv_power.isna().sum().sum() == 24  # the last 1 hour is still nans, for 2 pv systems


@freeze_time("2022-01-01 05:00")
def test_open_pv_datasource_from_database(pv_yields_and_systems):
    pv_datapipe = OpenPVFromDBIterDataPipe(providers=["pvoutput.org"])
    data = next(iter(pv_datapipe))
    assert data is not None


@freeze_time("2022-01-01 05:00")
def test_open_pv_datasource_from_database_config(pv_yields_and_systems):
    pv_config = PV(
        history_minutes=60, forecast_minutes=60 * 24, pv_files_groups=[PVFiles()], is_live=True
    )
    pv_datapipe = OpenPVFromDBIterDataPipe(pv_config=pv_config)
    data = next(iter(pv_datapipe))
    assert data is not None


def test_get_pv_power_from_pvsites_database():
    df_gen = get_pv_power_from_pvsites_database(timedelta(minutes=30))
    # 30 minutes so 5 five-minutely timestamps, 10 PV systems
    assert df_gen.shape == (6, 10)


def test_get_metadata_from_pvsites_database():
    df_meta = get_metadata_from_pvsites_database()
    assert len(df_meta) == 10
    for column in [
        "orientation",
        "tilt",
        "longitude",
        "latitude",
        "capacity_kw",
        "ml_id",
    ]:
        assert column in df_meta.columns


def test_open_pv_from_pvsites_db():
    dp = OpenPVFromPVSitesDBIterDataPipe(history_minutes=30)
    da = next(iter(dp))
    # 30 minutes so 5 five-minutely timestamps, 10 PV systems
    assert da.shape == (6, 10)
    for variable in [
        "time_utc",
        "pv_system_id",
        "observed_capacity_wp",
        "nominal_capacity_wp",
        "orientation",
        "tilt",
        "longitude",
        "latitude",
        "ml_id",
    ]:
        assert variable in da.coords
