"""Converts Satellite to int8 for Power Perceiver"""
import numpy as np
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from typing import Optional


@functional_datapipe("convert_pressure_levels_to_separate_variables")
class ConvertPressureLevelsToSeparateVariablesIterDataPipe(IterDataPipe):
    """Converts pressure levels to separate variables"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        pressure_level_to_use: Optional[list] = None,
    ):
        """
        Converts pressure levels to separate variables

        Args:
            source_datapipe: Datapipe of NWP data with pressure levels
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.pressure_level_to_use = pressure_level_to_use

    def __iter__(self) -> xr.DataArray:
        for xr_dataset in self.source_datapipe:
            # Select the given pressure levels
            if self.pressure_level_to_use is not None:
                xr_dataset = xr_dataset.sel(isobaricInhPa=self.pressure_level_to_use)
            # Unstack the pressure levels into separate variables
            xr_dataarray = xr_dataset.to_stacked_array(
                "level", sample_dims=["latitude", "longitude", "step", "init_time_utc"]
            )
            # Note: NaN values are surface variables
            yield xr_dataarray
