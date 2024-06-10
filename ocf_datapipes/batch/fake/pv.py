""" Make fake PV data """

from datetime import datetime

import numpy as np

from ocf_datapipes.batch import BatchKey
from ocf_datapipes.batch.fake.utils import get_n_time_steps_from_config, make_time_utc
from ocf_datapipes.config.model import Configuration


def make_fake_pv_data(
    configuration: Configuration, t0_datetime_utc: datetime, batch_size: int = 8
) -> dict:
    """
    Make Fake PV data ready for ML model

    Args:
        configuration: configuration object
        t0_datetime_utc: one datetime for when t0 is
        batch_size: Integer batch size to create

    Returns: dictionary of pv items
    """

    pv_config = configuration.input_data.pv
    if pv_config is None:
        return {}

    n_pv_systems = pv_config.n_pv_systems_per_example
    n_fourier_features = 8

    time_utc = make_time_utc(
        batch_size=batch_size,
        history_minutes=pv_config.history_minutes,
        forecast_minutes=pv_config.forecast_minutes,
        t0_datetime_utc=t0_datetime_utc,
        time_resolution_minutes=pv_config.time_resolution_minutes,
    )
    n_times = time_utc.shape[1]

    batch = {}
    batch[BatchKey.pv_time_utc] = time_utc  # Seconds since UNIX epoch (1970-01-01).
    batch[BatchKey.pv] = np.random.random((batch_size, n_times, n_pv_systems))
    batch[BatchKey.pv_t0_idx] = get_n_time_steps_from_config(
        input_data_configuration=pv_config, include_forecast=False
    )
    batch[BatchKey.pv_ml_id] = np.random.randint(0, 100, (batch_size, n_pv_systems))
    batch[BatchKey.pv_id] = np.random.randint(0, 1000, (batch_size, n_pv_systems))
    batch[BatchKey.pv_mask] = np.random.randint(0, 1, (batch_size, n_pv_systems))
    batch[BatchKey.pv_latitude] = np.random.randint(0, 10**6, (batch_size, n_pv_systems))
    batch[BatchKey.pv_longitude] = np.random.randint(0, 10**6, (batch_size, n_pv_systems))

    batch[BatchKey.pv_latitude_fourier] = np.random.random(
        (batch_size, n_pv_systems, n_fourier_features)
    )
    batch[BatchKey.pv_longitude_fourier] = np.random.random(
        (batch_size, n_pv_systems, n_fourier_features)
    )
    batch[BatchKey.pv_time_utc_fourier] = np.random.random(
        (batch_size, n_times, n_fourier_features)
    )

    return batch
