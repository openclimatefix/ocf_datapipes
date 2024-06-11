"""
This is a class function that assigns day night status
"""

import logging

import numpy as np
import pvlib
import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)

# {month: [sunrise_hour, sunset_hour]}
uk_daynight_dict = {
    1: [7, 17],
    2: [7, 18],
    3: [9, 19],
    4: [6, 20],
    5: [5, 21],
    6: [4, 22],
    7: [5, 22],
    8: [5, 21],
    9: [6, 20],
    10: [7, 21],
    11: [7, 17],
    12: [7, 17],
}

logger.info(
    f"The day and night standard hours are set by {'https://www.timeanddate.com/sun/uk/london'}"
)


@np.vectorize
def month_to_dawn_hour(month):
    """Function returning the first hour after dawn for the given month.

    Args:
        month: Array-like of ints or int month number in range [1, 12]

    Returns:
        Array of the first hour of the day

    """
    return uk_daynight_dict[month][0]


@np.vectorize
def month_to_dusk_hour(month):
    """Function returning the first hour after dusk for the given month.

    Args:
        month: Array-like of ints or int month number in range [1, 12]

    Returns:
        Array of the first hour of the night

    """
    return uk_daynight_dict[month][1]


@functional_datapipe("assign_daynight_status")
class AssignDayNightStatusIterDataPipe(IterDataPipe):
    """Adds a new dimension of day/night status"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        method: str = "simple",
    ):
        """This method adds extra coordinate of day night status.

        Args:
            source_datapipe: Datapipe emitting Xarray Dataset with time_utc coordinate
            method: Method used to assign the day-night status. Either "simple" or "elevation"

        Notes:
            If method is 'simple', then the month and hour are used to decide which datetimes
            correspond to day and night. These are done via a lookup table compiled for UK-London.
            These will be inappropriate far from London.

            If method is 'elevation', then the sun elevation is calculated for each datetime and PV
            system using its latitude and longitude. This is computationally expensive.
        """

        self.source_datapipe = source_datapipe
        self.method = method

        if method == "simple":
            logger.warning(
                "Calculating the day-night status using method 'simple' is only appropriate for "
                "UK PV datasets"
            )
            self._status_func = self._get_status_by_hour

        elif method == "elevation":
            logger.warning(
                "Calculating the day-night status using method 'elevation' can take a long time"
            )
            self._status_func = self._get_status_by_elevation
        else:
            raise ValueError(f"Method '{method}' not recognised")

    def _get_status_by_hour(self, ds):
        # Getting month and hour values from time_utc and stacking them
        months = ds.time_utc.dt.month.values
        hours = ds.time_utc.dt.hour.values

        # Getting the status of day/night for each timestamp in the dates
        is_after_dawn = hours >= month_to_dawn_hour(months)
        is_before_dusk = hours < month_to_dusk_hour(months)

        status_daynight = np.where(
            np.logical_and(is_after_dawn, is_before_dusk),
            "day",
            "night",
        )

        # Assigning a new coordinates of 'status_daynight' in the DataArray
        ds = ds.assign_coords(status_daynight=(("time_utc"), status_daynight))
        return ds

    def _get_status_by_elevation(self, ds):
        elevation = xr.full_like(ds, fill_value=np.nan).astype(np.float32)

        for system_id in ds.pv_system_id.values[:5]:
            ds_sel = ds.sel(pv_system_id=system_id)

            solpos = pvlib.solarposition.get_solarposition(
                time=ds_sel.time_utc,
                latitude=ds_sel.latitude.item(),
                longitude=ds_sel.longitude.item(),
                # method="nrel_numba",
                # Which `method` to use?
                # pyephem seemed to be a good mix between speed and ease but causes
                # segfaults!
                # nrel_numba doesn't work when using multiple worker processes.
                # nrel_c is probably fastest but requires C code to be manually compiled:
                # https://midcdmz.nrel.gov/spa/
            )
            elevation.sel(pv_system_id=system_id).values[:] = solpos["elevation"]

        status_daynight = np.where(
            elevation > 0.0,
            "day",
            "night",
        )

        # Assigning a new coordinates of 'status_daynight' in the DataArray
        ds = ds.assign_coords(status_daynight=(("time_utc"), status_daynight))
        return ds

    def __iter__(self) -> xr.DataArray():
        """Adds extra coordinate of day-night status

        Returns:
            Dataset with additional coordinate describing the datetime as either 'day' or 'night'
        """

        # Reading the Xarray dataset
        for ds in self.source_datapipe:
            yield self._status_func(ds)
