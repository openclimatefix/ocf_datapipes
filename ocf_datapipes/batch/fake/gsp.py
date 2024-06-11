""" Make fake GSP data """

from datetime import datetime

import numpy as np

from ocf_datapipes.batch import BatchKey
from ocf_datapipes.batch.fake.utils import get_n_time_steps_from_config, make_time_utc
from ocf_datapipes.config.model import Configuration


def make_fake_gsp_data(
    configuration: Configuration, t0_datetime_utc: datetime, batch_size: int = 8
) -> dict:
    """
    Make Fake GSP data ready for ML model

    Args:
        configuration: configuration object
        t0_datetime_utc: one datetime for when t0 is
        batch_size: Integer batch size to create

    Returns: dictionary of gsp items
    """

    gsp_config = configuration.input_data.gsp

    if gsp_config is None:
        return {}

    n_gsps = gsp_config.n_gsp_per_example
    n_fourier_features = 8

    time_utc = make_time_utc(
        batch_size=batch_size,
        history_minutes=gsp_config.history_minutes,
        forecast_minutes=gsp_config.forecast_minutes,
        t0_datetime_utc=t0_datetime_utc,
        time_resolution_minutes=gsp_config.time_resolution_minutes,
    )
    n_times = time_utc.shape[1]

    batch = {}
    batch[BatchKey.gsp_time_utc] = time_utc  # Seconds since UNIX epoch (1970-01-01).
    batch[BatchKey.gsp] = np.random.random((batch_size, n_times, n_gsps))
    batch[BatchKey.gsp_nominal_capacity_mwp] = np.random.randint(
        0, 1000, (batch_size, n_times, n_gsps)
    )
    batch[BatchKey.gsp_effective_capacity_mwp] = np.random.randint(
        0, 1000, (batch_size, n_times, n_gsps)
    )
    batch[BatchKey.gsp_t0_idx] = get_n_time_steps_from_config(
        input_data_configuration=gsp_config, include_forecast=False
    )
    batch[BatchKey.gsp_id] = np.random.randint(0, 1000, (batch_size, n_gsps))
    batch[BatchKey.gsp_x_osgb] = np.random.randint(0, 10**6, (batch_size, n_gsps))
    batch[BatchKey.gsp_y_osgb] = np.random.randint(0, 10**6, (batch_size, n_gsps))

    batch[BatchKey.gsp_x_osgb_fourier] = np.random.random((batch_size, n_gsps, n_fourier_features))
    batch[BatchKey.gsp_y_osgb_fourier] = np.random.random((batch_size, n_gsps, n_fourier_features))
    batch[BatchKey.gsp_time_utc_fourier] = np.random.random(
        (batch_size, n_times, n_fourier_features)
    )

    return batch
