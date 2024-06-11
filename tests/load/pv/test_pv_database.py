""" Test for loading pv data from database """

from datetime import timedelta

from ocf_datapipes.load.pv.database import (
    OpenPVFromPVSitesDBIterDataPipe,
    get_metadata_from_pvsites_database,
    get_pv_power_from_pvsites_database,
)


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
