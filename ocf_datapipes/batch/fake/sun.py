""" Make fake Sun data """

import numpy as np

from ocf_datapipes.batch import BatchKey, NWPBatchKey
from ocf_datapipes.batch.fake.utils import get_n_time_steps_from_config
from ocf_datapipes.config.model import Configuration


def make_fake_sun_data(configuration: Configuration, batch_size: int = 8):
    """
    Make Fake Sun data ready for ML model. This makes data across all different data inputs

    Args:
        configuration: configuration object
        t0_datetime_utc: one datetime for when t0 is
        batch_size: Integer batch size to create

    Returns: dictionary of pv items
    """

    batch = {}

    # HRV Satellite
    if configuration.input_data.hrvsatellite is not None:
        n_hrv_satellite_timesteps = get_n_time_steps_from_config(
            configuration.input_data.hrvsatellite
        )
        batch[BatchKey.hrvsatellite_solar_azimuth] = np.random.random(
            (batch_size, n_hrv_satellite_timesteps)
        )
        batch[BatchKey.hrvsatellite_solar_elevation] = np.random.random(
            (batch_size, n_hrv_satellite_timesteps)
        )

    # Satellite
    if configuration.input_data.satellite is not None:
        n_satellite_timesteps = get_n_time_steps_from_config(configuration.input_data.satellite)
        batch[BatchKey.satellite_solar_azimuth] = np.random.random(
            (batch_size, n_satellite_timesteps)
        )
        batch[BatchKey.satellite_solar_elevation] = np.random.random(
            (batch_size, n_satellite_timesteps)
        )

    # GSP
    if configuration.input_data.gsp is not None:
        n_satellite_timesteps = get_n_time_steps_from_config(configuration.input_data.gsp)
        batch[BatchKey.gsp_solar_azimuth] = np.random.random((batch_size, n_satellite_timesteps))
        batch[BatchKey.gsp_solar_elevation] = np.random.random((batch_size, n_satellite_timesteps))

    # PV
    if configuration.input_data.pv is not None:
        n_pv_timesteps = get_n_time_steps_from_config(configuration.input_data.pv)
        batch[BatchKey.pv_solar_azimuth] = np.random.random((batch_size, n_pv_timesteps))
        batch[BatchKey.pv_solar_elevation] = np.random.random((batch_size, n_pv_timesteps))

    # NWP
    if configuration.input_data.nwp is not None:
        batch[BatchKey.nwp] = {}

        for nwp_source, nwp_config in configuration.input_data.nwp.items():
            batch[BatchKey.nwp][nwp_source] = {}

            n_nwp_timesteps = get_n_time_steps_from_config(configuration.input_data.nwp[nwp_source])
            batch[BatchKey.nwp][nwp_source][NWPBatchKey.nwp_target_time_solar_azimuth] = (
                np.random.random((batch_size, n_nwp_timesteps))
            )
            batch[BatchKey.nwp][nwp_source][NWPBatchKey.nwp_target_time_solar_elevation] = (
                np.random.random((batch_size, n_nwp_timesteps))
            )

    return batch
