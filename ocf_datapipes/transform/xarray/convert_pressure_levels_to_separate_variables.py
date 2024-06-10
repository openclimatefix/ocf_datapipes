"""Converts pressure levels to separate variables"""

from typing import Optional

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("convert_pressure_levels_to_separate_variables")
class ConvertPressureLevelsToSeparateVariablesIterDataPipe(IterDataPipe):
    """Converts pressure levels to separate variables"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        pressure_level_name: str = "isobaricInhPa",
        pressure_level_to_use: Optional[list] = None,
    ):
        """
        Converts pressure levels to separate variables

        Args:
            source_datapipe: Datapipe of NWP data with pressure levels
            pressure_level_name: Name of the pressure level variable
            pressure_level_to_use: List of pressure levels to use
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.pressure_level_to_use = pressure_level_to_use
        self.pressure_level_name = pressure_level_name
        # For icosohedral grids, the lat/lon points are in one large array, not separate

    def __iter__(self) -> xr.DataArray:
        for xr_dataset in self.source_datapipe:
            # Select the given pressure levels
            if self.pressure_level_to_use is not None:
                xr_dataset = xr_dataset.sel({self.pressure_level_name: self.pressure_level_to_use})
            # Unstack the pressure levels into separate variables
            xr_dataarray = xr_dataset.to_stacked_array(
                "level",
                sample_dims=[v for v in xr_dataset.dims if v != self.pressure_level_name],
            )
            # Note: NaN values are surface variables
            yield xr_dataarray
