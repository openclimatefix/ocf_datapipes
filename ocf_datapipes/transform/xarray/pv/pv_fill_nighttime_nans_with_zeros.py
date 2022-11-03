"""Fill nighttime PV with NaNs"""

import logging
from typing import Union

import numpy as np
import pvlib
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.geospatial import osgb_to_lat_lon

logger = logging.getLogger(__name__)


@functional_datapipe("pv_fill_night_nans")
class PVFillNightNansIterDataPipe(IterDataPipe):
    """Fill nighttime nans with zeros"""

    def __init__(self, source_datapipe: IterDataPipe, elevation_limit: int = 5):
        """
        Fill nighttime nans with zeros

        Args:
            source_datapipe: the main data pipe
            elevation_limit: below this limit, all nans will be filled wiht zero.
                This is defaulted to to 5.
        """
        self.source_datapipe = source_datapipe
        self.elevation_limit = elevation_limit

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        """Run iter"""

        for xr_data in self.source_datapipe:

            logger.info("Going to fill night time nans")

            lats, lons = osgb_to_lat_lon(x=xr_data.x_osgb, y=xr_data.y_osgb)

            elevation = np.full_like(xr_data.data, fill_value=np.NaN).astype(np.float32)
            for example_idx, (lat, lon) in enumerate(zip(lats, lons)):

                logger.debug(
                    f"Getting solar elevation for {lat} {lon} " f"{example_idx} out of {len(lats)}"
                )

                # get mask data for nans
                nan_mask = np.isnan(xr_data.data[:, example_idx])
                time_utc_nans = xr_data.time_utc[nan_mask]

                solpos = pvlib.solarposition.get_solarposition(
                    time=time_utc_nans,
                    latitude=lat,
                    longitude=lon,
                    # Which `method` to use?
                    # pyephem seemed to be a good mix between speed and ease but causes
                    # segfaults!
                    # nrel_numba doesn't work when using multiple worker processes.
                    # nrel_c is probably fastest but requires C code to be manually compiled:
                    # https://midcdmz.nrel.gov/spa/
                )
                elevation[nan_mask, example_idx] = solpos["elevation"]

            # get maks data for nighttime and nans
            night_time_mask = elevation < self.elevation_limit
            nan_mask = np.isnan(xr_data.data)
            total_mask = night_time_mask & nan_mask

            # set value
            logger.debug("Setting night nans to 0")
            xr_data.data[total_mask] = 0.0

            while True:
                yield xr_data
