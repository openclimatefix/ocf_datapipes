"""
This is a class function that drops the pv systems with generates power over night.
"""

import logging

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe

logger = logging.getLogger(__name__)


@functional_datapipe("filter_night_pv")
class FilterPvSysGeneratingOvernightIterDataPipe(IterDataPipe):
    """
    Drop the pv systems which generates power over night date from a timeseries xarray Dataset.
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        threshold=0.2,
        daynight_method: str = "simple",
    ):
        """Drops the PV systems producing output overnight.

        Args:
            source_datapipe: A datapipe that emits xarray Dataset of PV generation
            threshold: Relative threshold for night-time production. Any system that generates more
                power than this in any night-time timestamp will be dropped
            daynight_method: Method used to assign datetimes to either 'night' or 'day'. Either
                "simple" or "elevation". See `AssignDayNightStatusIterDataPipe` for details
        """
        assert daynight_method in ["simple", "elevation"]
        self.source_datapipe = source_datapipe
        self.threshold = threshold
        self.daynight_method = daynight_method

    def __iter__(self) -> xr.DataArray():
        # TODO: Make more general

        for ds in self.source_datapipe.assign_daynight_status(self.daynight_method):
            # Select the night-time values
            ds_night = ds.where(ds.status_daynight == "night", drop=True)

            # Find relative maximum night-time generation for each system
            night_time_max_gen = (ds_night / ds_night.observed_capacity_wp).max(dim="time_utc")

            # Find systems above threshold
            mask = night_time_max_gen > self.threshold

            logger.info(
                f"Dropping {mask.sum().item()} PV systems with IDs:"
                f"{mask.where(mask, drop=True).pv_system_id.values}"
            )
            ds = ds.where(~mask, drop=True)
            yield ds
