from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.transform.xarray import AssignDayNightStatus


def test_assign_status_night(passiv_datapipe):
    night_status = AssignDayNightStatus(passiv_datapipe)
    data = next(iter(night_status))
    coords = data.coords["status_day"].values
    assert np.count_nonzero(coords == "night") == 121.0
    assert "day" and "night" in data.coords["status_day"].values
