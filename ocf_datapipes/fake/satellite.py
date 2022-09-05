"""Generate fake satellite data based off the configuration"""
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import xarray as xr
import numpy as np
import pandas as pd
from ocf_datapipes.config.model import Configuration

@functional_datapipe("fake_satellite")
class FakeSatelliteIterDataPipe(IterDataPipe):
    def __init__(self, configuration_datapipe: IterDataPipe[Configuration]):
        self.configuration_datapipe = configuration_datapipe

    def __iter__(self) -> xr.DataArray:
        for configuration in self.configuration_datapipe:
            yield create_fake_satellite_data(configuration)


def create_fake_satellite_data(configuration: Configuration):
    batch_size = configuration.process.batch_size
    image_size_pixels_height = configuration.input_data.satellite.satellite_image_size_pixels_height
    image_size_pixels_width = configuration.input_data.satellite.satellite_image_size_pixels_width
    history_seq_length = configuration.input_data.satellite.history_seq_length_5_minutes
    seq_length_5 = configuration.input_data.satellite.seq_length_5_minutes
    satellite_channels = configuration.input_data.satellite.satellite_channels
    