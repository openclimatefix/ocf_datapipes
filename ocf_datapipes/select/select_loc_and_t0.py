"""Select the t0 time and lcoation for training"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.utils.consts import Location

logger = logging.getLogger(__name__)




@functional_datapipe("select_loc_and_t0")
class LocationT0PickerIterDataPipe(IterDataPipe):

    def __init__(
        self,
        source_datapipe: IterDataPipe,
        return_all: bool = False,
        x_dim_name: Optional[str] = "x_osgb",
        y_dim_name: Optional[str] = "y_osgb",
        time_dim_name: Optional[str] = "time_utc"
    ):
        """

        Args:
            source_datapipe: Datapipe emitting Xarray Dataset
            return_all: Whether to return all t0-location pairs,
                if True, also returns them in structured order
            x_dim_name: x dimension name, defaulted to 'x_osgb'
            y_dim_name: y dimension name, defaulted to 'y_osgb'
            time_dim_name: time dimension name, defaulted to 'time_utc'
        """
        super().__init__()
        self.source_datapipe = source_datapipe
        self.return_all = return_all
        self.x_dim_name = x_dim_name
        self.y_dim_name = y_dim_name
        self.time_dim_name = time_dim_name
          
    def __len__(self):
        if self.return_all:
            # Cannot lazily find length
            self._load_source()
            length = len(self.xr_dataset[self.time_dim_name])*len(self.xr_dataset[self.x_dim_name])
            return length
        else:
            return np.inf
        
    def _load_source(self):
        if not hasattr(self, "xr_dataset"):
            self.xr_dataset = next(iter(self.source_datapipe))
            
    def __iter__(self) -> tuple[Location, pd.Timestamp]:
        
        self._load_source()
        xr_dataset = self.xr_dataset
        
        if self.return_all:

            logger.debug("Going to return all locations")

            # Iterate through all locations in dataset
            for t0 in xr_dataset[self.time_dim_name].values:
                
                for location_idx in range(len(xr_dataset[self.x_dim_name])):
                    
                        location = Location(
                            x=xr_dataset[self.x_dim_name][location_idx].values,
                            y=xr_dataset[self.y_dim_name][location_idx].values,
                        )
                        # for pv
                        if "pv_system_id" in xr_dataset.coords.keys():
                            location.id = int(xr_dataset["pv_system_id"][location_idx].values)

                        # for gsp
                        if "gsp_id" in xr_dataset.coords.keys():
                            location.id = int(xr_dataset["gsp_id"][location_idx].values)
                            
                        yield location, t0
                    
            else:
                
                while True:
                    location_idx = np.random.randint(0, len(xr_dataset[self.x_dim_name]))

                    location = Location(
                        x=xr_dataset[self.x_dim_name][location_idx].values,
                        y=xr_dataset[self.y_dim_name][location_idx].values,
                    )
                    if "pv_system_id" in xr_dataset.coords.keys():
                        location.id = int(xr_dataset["pv_system_id"][location_idx].values)

                    # for gsp
                    if "gsp_id" in xr_dataset.coords.keys():
                        location.id = int(xr_dataset["gsp_id"][location_idx].values)

                    t0 = np.random.choice(xr_dataset[self.time_dim_name].values)


                    yield location, t0

