"""Drop GSP National output from xarray"""
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("drop_national_gsp")
class DropNationalGSPIterDataPipe(IterDataPipe):
    """Drops national GSP"""
    def __init__(self, source_datapipe: IterDataPipe):
        """
        Drop GSP National from the dataarray

        Args:
            source_datapipe: Datapipe emitting GSP xarray
        """
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for xr_data in self.source_datapipe:
            # GSP ID 0 is national
            xr_data.isel(gsp_id=slice(1, len(xr_data.gsp_id)))
            yield xr_data
