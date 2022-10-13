"""Convert point PV sites to image output"""
from typing import List

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Zipper
import xarray as xr
import numpy as np

@functional_datapipe("create_pv_image")
class CreatePVImage(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, image_datapipe: IterDataPipe):
        """
        Creates 2D image of PV sites 
        """
        self.source_datapipe = source_datapipe
        self.image_datapipe = image_datapipe

    def __iter__(self):
        for pv_systems_xr, image_xr in Zipper(self.source_datapipe, self.image_datapipe):
            # Create empty image to use for the PV Systems, assumes image has x and y coordinates
            pv_image = np.zeros((len(pv_systems_xr["time"]),len(image_xr["x"]), len(image_xr["y"])), dtype=np.float32)

