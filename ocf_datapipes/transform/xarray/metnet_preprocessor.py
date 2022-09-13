"""Preprocessing for MetNet-type inputs"""
from typing import List

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper


@functional_datapipe("preprocess_metnet")
class PreProcessMetNetIterDataPipe(IterDataPipe):
    """Preprocess set of Xarray datasets similar to MetNet-1"""

    def __init__(
        self,
        source_datapipes: List[IterDataPipe],
        location_datapipe: IterDataPipe,
        context_width,
        context_height,
        center_width,
        center_height,
    ):
        """

        Processes set of Xarray datasets similar to MetNet

        In terms of taking all available source datapipes:
        1. selecting the same context area of interest
        2. Creating a center crop of the center_height, center_width
        3. Downsampling the context area of interest to the same shape as the center crop
        4. Stacking those context images on the center crop.

        This would be designed originally for NWP+Satellite+Topographic data sources.
        To add the PV power for lots of sites, the PV power would
        need to be able to be on a grid for the context/center
        crops and then for the downsample

        This also appends Lat/Lon coordinates to the stack,
         and returns a new Numpy array with the stacked data

        TODO Could also add the national PV as a set of Layers, so also GSP input

        Args:
            source_datapipes: Datapipes that emit xarray datasets
                with latitude/longitude coordinates included
            location_datapipe: Datapipe emitting location coordinate for center of example
            context_width: Width of the context area
            context_height: Height of the context area
            center_width: Center width of the area of interest
            center_height: Center height of the area of interest
        """
        self.source_datapipes = source_datapipes
        self.location_datapipe = location_datapipe
        self.context_width = context_width
        self.context_height = context_height
        self.center_width = center_width
        self.center_height = center_height

    def __iter__(self):
        for xr_datas, location in Zipper(Zipper(*self.source_datapipes), self.location_datapipe):
            # TODO Use location, find center point in xarray datasets
            # TODO Then select the xarray dataset
            # TODO Then on this selected example, select the center area
            # TODO Use the Lat/Long coordinates of the center array for the lat/lon stuff
            for xr_data in xr_datas:
                # TODO Get context area, center area, and resample
                pass

            pass
