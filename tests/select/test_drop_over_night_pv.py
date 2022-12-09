import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.select import DropNightPV


def test_drop_night_pv(passiv_datapipe):

    night_drop_pv = DropNightPV(passiv_datapipe)

    data = next(iter(night_drop_pv))
    assert "day" or "night" in data.coords["daynight_status"].values
    assert len(data.time_utc) == 289


def test_drop_night_pv_one_system_power_overnight():

    # Make 3 pv systems
    # 1 and 2, produce no power in the night
    # 3 produces power in the night

    time = pd.date_range(start="2022-01-01", freq="5T", periods=289)
    id = [1, 2, 3]
    ALL_COORDS = {
        "time_utc": time,
        "id": id,
    }

    data = np.zeros((len(time), len(id)))
    data[:, 2] = 1.0

    data_array = xr.DataArray(
        np.ones((len(time), len(id))),
        coords=ALL_COORDS,
    )

    # run the function
    night_drop_pv = DropNightPV([data_array])

    # check output, has dropped system 3
    data = next(iter(night_drop_pv))
    assert len(data.id) == 2
    assert len(data.time_utc) == 289
