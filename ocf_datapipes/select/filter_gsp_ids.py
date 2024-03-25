"""Drop GSP output from xarray"""

from typing import List

from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe("filter_gsp_ids")
class FilterGSPIDsIterDataPipe(IterDataPipe):
    """Select GSPs by ID"""

    def __init__(self, source_datapipe: IterDataPipe, gsps_to_keep: List[int] = range(1, 317)):
        """
        Filter GSPs by ID

        Args:
            source_datapipe: Datapipe emitting GSP xarray
            gsps_to_keep: List of GSP Id's to keep, by default all the 317 non-national ones
        """
        self.source_datapipe = source_datapipe
        self.gsps_to_keep = gsps_to_keep

    def __iter__(self):
        for xr_data in self.source_datapipe:
            xr_data = xr_data.isel(gsp_id=self.gsps_to_keep)
            yield xr_data
