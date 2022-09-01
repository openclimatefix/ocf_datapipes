"""Remove bad PV systems from the dataset"""
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("remove_bad_pv_systems")
class RemoveBadPVSystemsIterDataPipe(IterDataPipe):
    """Remove bad PV systems from the dataset"""

    def __init__(self, source_datapipe: IterDataPipe):
        """
        Remove bad PV systems

        Args:
            source_datapipe: Datapipe of PV data
        """
        self.source_datapipe = source_datapipe

    def __iter__(self) -> xr.DataArray:
        for xr_data in self.source_datapipe:
            yield xr_data
