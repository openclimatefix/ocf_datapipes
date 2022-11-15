import pandas as pd

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.utils.consts import BatchKey
from datetime import datetime, timedelta, timezone

from ocf_datapipes.batch.fake.utils import make_t0_datetimes_utc

import numpy as np

from typing import List


def make_fake_batch(configuration: Configuration) -> dict:
    """
    Make a random fake batch, this is useful for models that use this object

    Args:
        configuration: a configuration file

    Returns: dictionary containing the batch

    """


    t0_datetime_utc = datetime.now(tz=timezone.utc)
    t0_datetime_utc = t0_datetime_utc.replace(minute=t0_datetime_utc.minute // 5 * 5)
    t0_datetime_utc = t0_datetime_utc.replace(second=0)
    t0_datetime_utc = t0_datetime_utc.replace(microsecond=0)

    # make fake PV data
    batch_pv = make_fake_pv_data(configuration=configuration, t0_datetime_utc=t0_datetime_utc)

    # make NWP data
    batch_nwp = make_fake_nwp_data(configuration=configuration, t0_datetime_utc=t0_datetime_utc)

    # make hrv and normal satellite data
    batch_satellite = make_fake_satellite_data(
        configuration=configuration, t0_datetime_utc=t0_datetime_utc
    )
    batch_hrv_satellite = make_fake_satellite_data(
        configuration=configuration, t0_datetime_utc=t0_datetime_utc
    )

    # make sun features
    batch_sun = make_fake_sun_data(configuration=configuration, t0_datetime_utc=t0_datetime_utc)

    batch = {**batch_pv, **batch_nwp, **batch_satellite, **batch_hrv_satellite, **batch_sun}

    return batch


def make_fake_satellite_data(configuration: Configuration, t0_datetime_utc: datetime):
    return {}


def make_fake_sun_data(configuration: Configuration, t0_datetime_utc: datetime):

    return {}


def make_fake_nwp_data(configuration: Configuration, t0_datetime_utc: datetime):

    nwp_config = configuration.input_data.nwp
    batch_size = configuration.process.batch_size
    n_channels = len(nwp_config.nwp_channels)
    n_y_osgb = nwp_config.nwp_image_size_pixels_height
    n_x_osgb = nwp_config.nwp_image_size_pixels_width
    n_fourier_features = 8

    time_utc = make_time_utc(
        batch_size=batch_size,
        history_minutes=nwp_config.history_minutes,
        forecast_minutes=nwp_config.forecast_minutes,
        t0_datetime_utc=t0_datetime_utc,
        time_resolution_minutes=nwp_config.,
    )
    n_times = time_utc.shape[1]

    batch = {}
    batch[BatchKey.nwp_init_time_utc] = time_utc  # Seconds since UNIX epoch (1970-01-01).
    batch[BatchKey.nwp_target_time_utc] = time_utc  # Seconds since UNIX epoch (1970-01-01).
    batch[BatchKey.nwp] = np.random.random((batch_size, n_times, n_channels, n_y_osgb, n_x_osgb))
    batch[BatchKey.nwp_t0_idx] = (
        int(nwp_config.history_minutes / nwp_config.time_resolution_minutes) + 1
    )
    batch[BatchKey.nwp_step] = np.random.randint(0, 100, (batch_size, n_times))
    batch[BatchKey.nwp_y_osgb] = np.random.randint(0, 100, (batch_size, n_y_osgb))
    batch[BatchKey.nwp_x_osgb] = np.random.randint(0, 100, (batch_size, n_x_osgb))
    batch[BatchKey.nwp_channel_names] = np.random.randint(0, 100, (n_channels,))

    batch[BatchKey.nwp_x_osgb_fourier] = np.random.random(
        (batch_size, n_x_osgb, n_fourier_features)
    )
    batch[BatchKey.nwp_y_osgb_fourier] = np.random.random(
        (batch_size, n_y_osgb, n_fourier_features)
    )
    batch[BatchKey.nwp_target_time_utc] = np.random.random(
        (batch_size, n_times, n_fourier_features)
    )
    batch[BatchKey.nwp_init_time_utc] = np.random.random((batch_size, n_times, n_fourier_features))

    return batch


def make_fake_pv_data(configuration: Configuration, t0_datetime_utc: datetime):

    pv_config = configuration.input_data.pv
    batch_size = configuration.process.batch_size
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
    batch[BatchKey.pv_t0_idx] = (
        int(pv_config.history_minutes / pv_config.time_resolution_minutes) + 1
    )
    batch[BatchKey.pv_system_row_number] = np.random.randint(0, 100, (batch_size, n_pv_systems))
    batch[BatchKey.pv_id] = np.random.randint(0, 1000, (batch_size, n_pv_systems))
    batch[BatchKey.pv_mask] = np.random.randint(0, 1, (batch_size, n_pv_systems))
    batch[BatchKey.pv_x_osgb] = np.random.randint(0, 10 ** 6, (batch_size, n_pv_systems))
    batch[BatchKey.pv_y_osgb] = np.random.randint(0, 10 ** 6, (batch_size, n_pv_systems))

    batch[BatchKey.pv_x_osgb_fourier] = np.random.random(
        (batch_size, n_pv_systems, n_fourier_features)
    )
    batch[BatchKey.pv_y_osgb_fourier] = np.random.random(
        (batch_size, n_pv_systems, n_fourier_features)
    )
    batch[BatchKey.pv_time_utc_fourier] = np.random.random(
        (batch_size, n_times, n_fourier_features)
    )
    batch[BatchKey.pv_time_utc_fourier_t0] = np.random.random((batch_size, n_fourier_features))

    return batch


def make_time_utc(
    batch_size, history_minutes, forecast_minutes, t0_datetime_utc, time_resolution_minutes
):
    """
    Make time utc array

    Args:
        batch_size: the batch size
        history_minutes: the history minutes we want
        forecast_minutes: the amount of minutes that the forecast will be fore
        t0_datetime_utc: t0_datetime
        time_resolution_minutes: time resolution e.g 5 for daat in 5 minutes chunks

    Returns: array of time_utc

    """
    start_datetime = t0_datetime_utc - timedelta(minutes=history_minutes)
    end_datetime = t0_datetime_utc + timedelta(minutes=forecast_minutes)
    time_utc = pd.date_range(
        start=start_datetime, end=end_datetime, freq=f"{time_resolution_minutes}T"
    )

    time_utc = time_utc.values.astype("datetime64[s]")
    time_utc = np.tile(time_utc, (batch_size, 1))
    return time_utc
