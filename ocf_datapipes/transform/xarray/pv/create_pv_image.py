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
            # Coordinates should be in order for the image, so just need to do the search sorted thing to get the index to add the PV output to
            # Once all the outputs are added, then normalize? Could also normalize PV first, then normalize the normalized PV data
            # In either case have to iterate through all PV systems in example
            for pv_system in pv_systems_xr["pv_system_id"]:
                # Quick check as search sorted doesn't give an error if it is not in the range
                if pv_system["x"] < image_xr["x"][0] or pv_system["x"] > image_xr["x"][-1]:
                    continue
                if pv_system["y"] < image_xr["y"][0] or pv_system["y"] > image_xr["y"][-1]:
                    continue
                x_idx = np.searchsorted(pv_system["x"], image_xr["x"])
                y_idx = np.searchsorted(pv_system["y"], image_xr["y"])
                # Now go by the timestep to create cube of PV data
                for time in range(len(pv_system.time.values)):
                    pv_image[time][x_idx][y_idx] += pv_system["data"][time]

            # TODO Construct Xarray object to return? Or add to PV data?
            yield pv_image
