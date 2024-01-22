"""
This is a class function infills PV data via interpolation
"""

import xarray as xr
from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("pv_interpolate_infill")
class PVInterpolateInfillIterDataPipe(IterDataPipe):
    """Infills missing values via interpolation"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        time_resolution_minutes: int = 5,
        fill_n: int = 3,
        allow_add_timestamps: bool = False,
    ):
        """Infills missing values via interpolation

        Args:
            source_datapipe: Datapipe emitting Xarray Dataset with time_utc coordinate
            time_resolution_minutes: Time resolution of output Dataset
            fill_n: Number of steps of size `time_resolution_minutes` forward and backward where
                NaNs can be infilled via interpolation.
            allow_add_timestamps: If True, new timestamps will be created in the output dataset
                which were mising in the input dataset, as long as they can be filled. Else the
                output dataset will have the same timestamps as the input dataset.
        """

        self.source_datapipe = source_datapipe
        self.time_resolution_minutes = time_resolution_minutes
        self.fill_n = fill_n
        self.allow_add_timestamps = allow_add_timestamps

    def __iter__(self) -> xr.DataArray():
        """Infills missing values via interpolation

        Returns:
            Dataset with additional values infilled
        """

        # Reading the Xarray dataset
        for ds in self.source_datapipe:
            # Resample to 5-minutely and interpolate up to 15 minutes ahead.

            # Using pandas gives us more options in the interpolation
            df_interp = (
                ds.to_pandas()
                .resample(f"{self.time_resolution_minutes}min")
                .interpolate(
                    method="time",
                    limit=self.fill_n,
                    limit_direction="both",
                )
            )
            # Create new copy of the DataArray with correct time index shape
            ds_interp = ds.sel(time_utc=df_interp.index, method="nearest")

            # Fill with interpolated values
            ds_interp.values[:] = df_interp.values

            if not self.allow_add_timestamps:
                # Filter to the same timestamps as the input
                # For the timestamps dropped here, all values are infilled values
                ds_interp = ds_interp.sel(time_utc=ds.time_utc)

            yield ds_interp
