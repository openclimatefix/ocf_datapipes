from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import numpy as np

@functional_datapipe("location_picker")
class LocationPickerIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self):
        for xr_dataset in self.source_dp:
            # Assumes all datasets have osgb coordinates for selecting locations
            # Pick 1 random location from the input dataset
            location_idx = np.random.randint(0, len(xr_dataset["x_osgb"]))
            location = (xr_dataset["x_osgb"][location_idx], xr_dataset["y_osgb"][location_idx])
            yield location
