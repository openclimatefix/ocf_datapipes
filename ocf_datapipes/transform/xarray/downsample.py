"""Downsample Xarray datasets Datapipe"""

from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("downsample")
class DownsampleIterDataPipe(IterDataPipe):
    """Downsample Xarray dataset with coarsen"""

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        y_coarsen: int,
        x_coarsen: int,
        x_dim_name: str = "x_osgb",
        y_dim_name: str = "y_osgb",
    ):
        """
        Downsample xarray dataset/dataarrays with coarsen

        Args:
            source_datapipe: Datapipe emitting Xarray dataset
            y_coarsen: Coarsen value in the y direction
            x_coarsen: Coarsen value in the x direction
            x_dim_name: X dimension name
            y_dim_name: Y dimension name
        """
        self.source_datapipe = source_datapipe
        self.y_coarsen = y_coarsen
        self.x_coarsen = x_coarsen
        self.x_dim_name = x_dim_name
        self.y_dim_name = y_dim_name

    def __iter__(self):
        """Coarsen the data on the specified dimensions"""
        for xr_data in self.source_datapipe:
            yield xr_data.coarsen(
                {self.y_dim_name: self.y_coarsen, self.x_dim_name: self.x_coarsen},
                boundary="trim",
            ).mean()
