""" Make fake Satellite data """
from datetime import datetime

import numpy as np

from ocf_datapipes.batch.fake.utils import get_n_time_steps_from_config, make_time_utc
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.utils.consts import BatchKey


def make_fake_satellite_data(
    configuration: Configuration, t0_datetime_utc: datetime, is_hrv: bool = False
):
    """
    Make Fake Satellite data ready for ML model. This makes data across all different data inputs

    Args:
        configuration: configuration object
        t0_datetime_utc: one datetime for when t0 is
        is_hrv: option if its hrv or not

    Returns: dictionary of satellite items
    """

    if is_hrv:
        prefix = "hrv"
    else:
        prefix = ""

    variable = prefix + "satellite"

    satellite_config = getattr(configuration.input_data, variable)

    if satellite_config is None:
        return {}

    batch_size = configuration.process.batch_size
    n_channels = len(getattr(satellite_config, f"{variable}_channels"))
    height = getattr(satellite_config, f"{variable}_image_size_pixels_height")
    width = getattr(satellite_config, f"{variable}_image_size_pixels_width")
    n_fourier_features = 8

    # make time matrix
    time_utc = make_time_utc(
        batch_size=batch_size,
        history_minutes=satellite_config.history_minutes,
        forecast_minutes=satellite_config.forecast_minutes,
        t0_datetime_utc=t0_datetime_utc,
        time_resolution_minutes=satellite_config.time_resolution_minutes,
    )
    n_times = time_utc.shape[1]

    batch = {}
    # shape: (batch_size, time, channels, y, x)
    #
    # Or, if the imagery has been patched,
    # shape: (batch_size, time, channels, y, x, n_pixels_per_patch) where n_pixels_per_patch
    # is the *total* number of pixels, # TODO
    batch[getattr(BatchKey, f"{variable}_actual")] = np.random.random(
        (batch_size, n_times, n_channels, height, width)
    )
    batch[getattr(BatchKey, f"{variable}_predicted")] = np.random.random(
        (batch_size, n_times, n_channels, height, width)
    )

    batch[getattr(BatchKey, f"{variable}_t0_idx")] = get_n_time_steps_from_config(
        satellite_config, include_forecast=False
    )

    # HRV satellite coordinates:
    batch[getattr(BatchKey, f"{variable}_y_osgb")] = np.random.randint(0, 100, (batch_size, height))
    batch[getattr(BatchKey, f"{variable}_x_osgb")] = np.random.randint(0, 100, (batch_size, width))
    batch[getattr(BatchKey, f"{variable}_y_geostationary")] = np.random.randint(
        0, 100, (batch_size, height)
    )
    batch[getattr(BatchKey, f"{variable}_x_geostationary")] = np.random.randint(
        0, 100, (batch_size, width)
    )

    batch[
        getattr(BatchKey, f"{variable}_time_utc")
    ] = time_utc  # Seconds since UNIX epoch (1970-01-01).
    # Added by np_batch_processor.Topography:
    batch[getattr(BatchKey, f"{variable}_surface_height")] = np.random.randint(
        0, 100, (batch_size, height, width)
    )

    # HRV satellite Fourier coordinates:
    # Spatial coordinates. Shape: (batch_size, y, x, n_fourier_features_per_dim)
    batch[getattr(BatchKey, f"{variable}_y_osgb_fourier")] = np.random.random(
        (batch_size, height, n_fourier_features)
    )  # TODO check its just 3dims
    batch[getattr(BatchKey, f"{variable}_x_osgb_fourier")] = np.random.random(
        (batch_size, width, n_fourier_features)
    )

    #: Time shape: (batch_size, n_timesteps, n_fourier_features_per_dim)
    batch[getattr(BatchKey, f"{variable}_time_utc_fourier")] = np.random.random(
        (batch_size, n_times, n_fourier_features)
    )
    batch[getattr(BatchKey, f"{variable}_time_utc_fourier_t0")] = np.random.random(
        (batch_size, n_times, n_fourier_features)
    )

    return batch
