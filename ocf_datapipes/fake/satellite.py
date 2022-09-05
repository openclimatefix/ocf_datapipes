"""Generate fake satellite data based off the configuration"""
import numpy as np
import pandas as pd
import xarray as xr
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.config.model import Configuration


@functional_datapipe("fake_satellite")
class FakeSatelliteIterDataPipe(IterDataPipe):
    """Generates the spatially and temporally cropped dataset, as if after SelectSpatial and Temporal slice"""

    def __init__(self, configuration_datapipe: IterDataPipe[Configuration]):
        self.configuration_datapipe = configuration_datapipe

    def __iter__(self) -> xr.DataArray:
        for configuration in self.configuration_datapipe:
            yield create_fake_satellite_data_cropped(configuration)


@functional_datapipe("fake_satellite_raw")
class FakeSatelliteRawIterDataPipe(IterDataPipe):
    """Generates the raw dataset, as if loaded from disk"""

    def __init__(self, configuration_datapipe: IterDataPipe[Configuration]):
        self.configuration_datapipe = configuration_datapipe

    def __iter__(self) -> xr.DataArray:
        for configuration in self.configuration_datapipe:
            yield create_fake_satellite_data_raw(configuration)


def create_fake_satellite_data_cropped(configuration: Configuration):
    batch_size = configuration.process.batch_size
    image_size_pixels_height = configuration.input_data.satellite.satellite_image_size_pixels_height
    image_size_pixels_width = configuration.input_data.satellite.satellite_image_size_pixels_width
    history_seq_length = configuration.input_data.satellite.history_seq_length_5_minutes
    seq_length_5 = configuration.input_data.satellite.seq_length_5_minutes
    satellite_channels = configuration.input_data.satellite.satellite_channels


def create_fake_satellite_data_raw(
    configuration: Configuration,
    fake_jpeg_xl: bool = False,
    rss_extant: bool = False,
    is_live: bool = False,
):
    """Generate raw dataset, as if loaded directly from disk"""
    batch_size = configuration.process.batch_size
    image_size_pixels_height = configuration.input_data.satellite.satellite_image_size_pixels_height
    image_size_pixels_width = configuration.input_data.satellite.satellite_image_size_pixels_width
    history_seq_length = configuration.input_data.satellite.history_seq_length_5_minutes
    seq_length_5 = configuration.input_data.satellite.seq_length_5_minutes
    satellite_channels = configuration.input_data.satellite.satellite_channels

    # Generate last history duration * 3 of data, and potentially next forecast_duration * 3?
