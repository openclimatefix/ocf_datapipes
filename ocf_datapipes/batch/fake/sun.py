from datetime import datetime

import numpy as np

from ocf_datapipes.batch.fake.utils import get_n_time_steps_from_config
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.utils.consts import BatchKey


def make_fake_sun_data(configuration: Configuration, t0_datetime_utc: datetime):

    batch = {}
    batch_size = configuration.process.batch_size

    # HRV Satellite
    n_hrv_satellite_timesteps = get_n_time_steps_from_config(configuration.input_data.hrvsatellite)
    batch[BatchKey.hrvsatellite_solar_azimuth] = np.random.random(
        (batch_size, n_hrv_satellite_timesteps)
    )
    batch[BatchKey.hrvsatellite_solar_elevation] = np.random.random(
        (batch_size, n_hrv_satellite_timesteps)
    )

    # Satellite
    n_satellite_timesteps = get_n_time_steps_from_config(configuration.input_data.satellite)
    batch[BatchKey.satellite_solar_azimuth] = np.random.random(
        (batch_size, n_satellite_timesteps)
    )
    batch[BatchKey.satellite_solar_elevation] = np.random.random(
        (batch_size, n_satellite_timesteps)
    )

    # GSP
    n_satellite_timesteps = get_n_time_steps_from_config(configuration.input_data.gsp)
    batch[BatchKey.gsp_solar_azimuth] = np.random.random(
        (batch_size, n_satellite_timesteps)
    )
    batch[BatchKey.gsp_solar_elevation] = np.random.random(
        (batch_size, n_satellite_timesteps)
    )

    # PV
    n_pv_timesteps = get_n_time_steps_from_config(configuration.input_data.pv)
    batch[BatchKey.pv_solar_azimuth] = np.random.random(
        (batch_size, n_pv_timesteps)
    )
    batch[BatchKey.pv_solar_elevation] = np.random.random(
        (batch_size, n_pv_timesteps)
    )

    # NWP
    n_nwp_timesteps = get_n_time_steps_from_config(configuration.input_data.nwp)
    batch[BatchKey.nwp_target_time_solar_azimuth] = np.random.random(
        (batch_size, n_nwp_timesteps)
    )
    batch[BatchKey.nwp_target_time_solar_elevation] = np.random.random(
        (batch_size, n_nwp_timesteps)
    )

    return batch