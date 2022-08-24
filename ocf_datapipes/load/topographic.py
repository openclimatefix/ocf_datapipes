from pathlib import Path
from typing import Union

import rioxarray
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("open_topography")
class OpenTopographyIterDataPipe(IterDataPipe):
    def __init__(self, topo_filename: Union[Path, str]):
        self.topo_filename = topo_filename

    def __iter__(self):
        topo = rioxarray.open_rasterio(
            filename=self.topo_filename, parse_coordinates=True, masked=True
        )

        # `band` and `spatial_ref` don't appear to hold any useful info. So get rid of them:
        topo = topo.isel(band=0)
        topo = topo.drop_vars(["spatial_ref", "band"])

        # Use our standard naming:
        topo = topo.rename({"x": "x_osgb", "y": "y_osgb"})

        while True:
            yield topo
