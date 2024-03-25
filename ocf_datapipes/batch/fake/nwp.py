""" Make fake NWP data """

from datetime import datetime

import numpy as np

from ocf_datapipes.batch import BatchKey, NWPBatchKey
from ocf_datapipes.batch.fake.utils import get_n_time_steps_from_config, make_time_utc
from ocf_datapipes.config.model import Configuration


def make_fake_nwp_data(
    configuration: Configuration, t0_datetime_utc: datetime, batch_size: int = 8
) -> dict:
    """
    Make Fake NWP data ready for ML model

    Args:
        configuration: configuration object
        t0_datetime_utc: one datetime for when t0 is
        batch_size: Integer batch size to create

    Returns: dictionary of nwp items
    """

    nwp_config = configuration.input_data.nwp

    if configuration.input_data.nwp is None:
        return {}

    batch = {}

    for nwp_source, nwp_config in configuration.input_data.nwp.items():
        source_batch = {}

        n_channels = len(nwp_config.nwp_channels)
        n_y_osgb = nwp_config.nwp_image_size_pixels_height
        n_x_osgb = nwp_config.nwp_image_size_pixels_width
        n_fourier_features = 8

        # make time matrix
        time_utc = make_time_utc(
            batch_size=batch_size,
            history_minutes=nwp_config.history_minutes,
            forecast_minutes=nwp_config.forecast_minutes,
            t0_datetime_utc=t0_datetime_utc,
            time_resolution_minutes=nwp_config.time_resolution_minutes,
        )
        n_times = time_utc.shape[1]

        # main nwp components

        source_batch[NWPBatchKey.nwp_init_time_utc] = (
            time_utc  # Seconds since UNIX epoch (1970-01-01).
        )
        source_batch[NWPBatchKey.nwp_target_time_utc] = (
            time_utc  # Seconds since UNIX epoch (1970-01-01).
        )
        source_batch[NWPBatchKey.nwp] = np.random.random(
            (batch_size, n_times, n_channels, n_y_osgb, n_x_osgb)
        )
        source_batch[NWPBatchKey.nwp_t0_idx] = get_n_time_steps_from_config(
            input_data_configuration=nwp_config, include_forecast=False
        )

        source_batch[NWPBatchKey.nwp_step] = np.random.randint(0, 100, (batch_size, n_times))
        source_batch[NWPBatchKey.nwp_y_osgb] = np.random.randint(0, 100, (batch_size, n_y_osgb))
        source_batch[NWPBatchKey.nwp_x_osgb] = np.random.randint(0, 100, (batch_size, n_x_osgb))
        source_batch[NWPBatchKey.nwp_channel_names] = np.random.randint(0, 100, (n_channels,))

        # fourier components
        source_batch[NWPBatchKey.nwp_x_osgb_fourier] = np.random.random(
            (batch_size, n_x_osgb, n_fourier_features)
        )
        source_batch[NWPBatchKey.nwp_y_osgb_fourier] = np.random.random(
            (batch_size, n_y_osgb, n_fourier_features)
        )
        source_batch[NWPBatchKey.nwp_target_time_utc] = np.random.random(
            (batch_size, n_times, n_fourier_features)
        )
        source_batch[NWPBatchKey.nwp_init_time_utc] = np.random.random(
            (batch_size, n_times, n_fourier_features)
        )

        batch[nwp_source] = source_batch

    return {BatchKey.nwp: batch}
