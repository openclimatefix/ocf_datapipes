"""Downsample Xarray datasets Datapipe"""
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("partial_temporal_downsample")
class PartialTemporalDownsampleIterDataPipe(IterDataPipe):
    """Downsample Xarray dataset with coarsen"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        time_coarsen: int,
        start_idx: int,
        time_dim_name: str = "time_utc",
    ):
        """
        Downsample xarray dataset/dataarrays in time after a given timestep

        Useful for e.g. taking NWPs only every 2 or 3 hours far into the future

        Args:
            source_datapipe: Datapipe emitting Xarray dataset
            time_coarsen: How much to subsample by
            start_idx: IDX after t0_idx for when to start the subsampling
            time_dim_name: Name of the time dimension
        """
        self.source_datapipe = source_datapipe
        self.time_coarsen = time_coarsen
        self.time_dim_name = time_dim_name
        self.start_idx = start_idx

    def __iter__(self):
        """Coarsen the data on the specified dimensions"""
        for xr_data in self.source_datapipe:

            # Split into ones in the past and ones in the future
            same_data = xr_data.sel(
                {self.time_dim_name: slice(0, xr_data.attrs["t0_idx"] + self.start_idx)}
            )
            interpolated_data = xr_data.sel()
            yield xr_data.coarsen(
                {
                    self.time_dim: self.y_coarsen,
                    self.x_dim_name: self.x_coarsen,
                },
                boundary="trim",
            ).mean()
