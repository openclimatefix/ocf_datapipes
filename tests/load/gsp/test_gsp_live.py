"""Test for loading pv data from database"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
from freezegun import freeze_time

from ocf_datapipes.load.gsp.database import (
    OpenGSPFromDatabaseIterDataPipe,
    get_gsp_power_from_database,
)


@freeze_time("2022-01-01 01:00")
def test_get_gsp_power_from_database(gsp_yields, db_session):
    """Get GSP power from database"""

    gsp_power, gsp_nominal_capacity, gsp_effective_capacity = get_gsp_power_from_database(
        history_duration=timedelta(hours=1), interpolate_minutes=30, load_extra_minutes=0
    )

    assert len(gsp_power) == 3  # 1 hours at 30 mins + 1
    assert len(gsp_power.columns) == 5
    assert gsp_power.columns[0] == 1
    assert (
        pd.to_datetime(gsp_power.index[0]).isoformat()
        == datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat()
    )
    assert gsp_power.max().max() < 1
    # this because units have changed from kw to mw


@freeze_time("2022-01-01 03:00")
def test_open_gsp_datasource_from_database(gsp_yields):
    pv_dp = OpenGSPFromDatabaseIterDataPipe()
    data = next(iter(pv_dp))
    assert data is not None


@freeze_time("2022-01-01 03:00")
def test_open_gsp_datasource_from_database_no_data():
    pv_dp = OpenGSPFromDatabaseIterDataPipe()
    data = next(iter(pv_dp))
    assert data is not None
    assert len(data.time_utc.values) == 6
    assert len(data.gsp_id.values) == 317
    assert np.shape(data.values) == (6, 317)
