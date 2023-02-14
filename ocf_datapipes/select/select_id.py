"""Selects time slice"""
import logging
from typing import Union

import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

logger = logging.getLogger(__name__)


@functional_datapipe("select_id")
class SelectIDIterDataPipe(IterDataPipe):
    """Selects time slice"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        location_datapipe: IterDataPipe,
        data_source_name: str = "nwp",
    ):
        """
        Selects id

        Args:
            source_datapipe: Datapipe of Xarray objects
            location_datapipe: Location datapipe
            data_source_name: the name of the datasource.
        """
        self.source_datapipe = source_datapipe
        self.location_datapipe = location_datapipe
        self.data_source_name = data_source_name

    def __iter__(self) -> Union[xr.DataArray, xr.Dataset]:
        for xr_data, location in self.source_datapipe.zip_ocf(self.location_datapipe):

            logger.debug(f"Selecting Data on id {location.id} for {self.data_source_name}")

            if self.data_source_name == "nwp":
                try:
                    xr_data = xr_data.sel(id=location.id)
                except Exception as e:
                    logger.warning(f"Could not find {location.id} in nwp {xr_data.id}")
                    raise e

            if self.data_source_name == "pv":
                try:
                    xr_data = xr_data.sel(pv_system_id=[location.id])
                except Exception as e:
                    logger.warning(
                        f"Failed using [{location.id}], now trying without brackets, error: {e}"
                    )
                    try:
                        # TODO Fix this, simple ones fail without the [], MetNet fails with []
                        xr_data = xr_data.sel(pv_system_id=location.id)
                    except Exception as e:
                        logger.warning(f"Could not find {location.id} in pv {xr_data.pv_system_id}")
                        raise e
            logger.debug("Selected Data on id")
            yield xr_data
