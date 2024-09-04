"""Numpy and xarray batch"""

from enum import Enum, auto
from typing import Union

import numpy as np
import torch
import xarray as xr


class BatchKey(Enum):
    """The names of the different elements of each batch.

    This is also where we document the exact shape of each element.

    Each `DataSource` may be split into several different BatchKey elements. For example, the
    PV PreparedDataSource yields `pv` and `pv_ml_id` BatchKeys.
    """

    # -------------- HRVSATELLITE -----------------------------------
    # shape: (batch_size, time, channels, y, x)
    #
    # Or, if the imagery has been patched,
    # shape: (batch_size, time, channels, y, x, n_pixels_per_patch) where n_pixels_per_patch
    # is the *total* number of pixels,
    # i.e. n_pixels_per_patch_along_height * n_pixels_per_patch_along_width.
    hrvsatellite_actual = auto()
    hrvsatellite_predicted = auto()  # shape: batch_size, time, y, x
    hrvsatellite_t0_idx = auto()  # shape: scalar

    # HRV satellite coordinates:
    hrvsatellite_y_osgb = auto()  # shape: (batch_size, y, x)
    hrvsatellite_x_osgb = auto()  # shape: (batch_size, y, x)
    hrvsatellite_y_geostationary = auto()  # shape: (batch_size, y)
    hrvsatellite_x_geostationary = auto()  # shape: (batch_size, x)
    #: Time is seconds since UNIX epoch (1970-01-01). Shape: (batch_size, n_timesteps)
    hrvsatellite_time_utc = auto()
    # Added by np_batch_processor.Topography:
    hrvsatellite_surface_height = auto()  # The surface height at each pixel. (batch_size, y, x)

    # HRV satellite Fourier coordinates:
    # Spatial coordinates. Shape: (batch_size, y, x, n_fourier_features_per_dim)
    hrvsatellite_y_osgb_fourier = auto()
    hrvsatellite_x_osgb_fourier = auto()
    #: Time shape: (batch_size, n_timesteps, n_fourier_features_per_dim)
    hrvsatellite_time_utc_fourier = auto()

    # -------------- NWP --------------------------------------------
    nwp = auto()  # See `MultiNWPNumpyBatch`

    # -------------- PV ---------------------------------------------
    pv = auto()  # shape: (batch_size, time, n_pv_systems)
    pv_t0_idx = auto()  # shape: scalar
    pv_ml_id = auto()  # shape: (batch_size, n_pv_systems)
    pv_id = auto()  # shape: (batch_size, n_pv_systems)
    pv_observed_capacity_wp = auto()  # shape: (batch_size, n_pv_systems)
    pv_nominal_capacity_wp = auto()  # shape: (batch_size, n_pv_systems)
    #: pv_mask is True for good PV systems in each example.
    # The RawPVDataSource doesn't use pv_mask. Instead is sets missing PV systems to NaN
    # across all PV batch keys.
    pv_mask = auto()  # shape: (batch_size, n_pv_systems)

    # PV coordinates:
    # Each has shape: (batch_size, n_pv_systems), will be NaN for missing PV systems.
    pv_latitude = auto()
    pv_longitude = auto()
    pv_time_utc = auto()  # Seconds since UNIX epoch (1970-01-01).

    # PV Fourier coordinates:
    # Each has shape: (batch_size, n_pv_systems, n_fourier_features_per_dim),
    # and will be NaN for missing PV systems.
    pv_latitude_fourier = auto()
    pv_longitude_fourier = auto()
    pv_time_utc_fourier = auto()  # (batch_size, time, n_fourier_features)

    # -------------- Wind ---------------------------------------------
    wind = auto()  # shape: (batch_size, time, n_pv_systems)
    wind_t0_idx = auto()  # shape: scalar
    wind_ml_id = auto()  # shape: (batch_size, n_pv_systems)
    wind_id = auto()  # shape: (batch_size, n_pv_systems)
    wind_observed_capacity_mwp = auto()  # shape: (batch_size, n_pv_systems)
    wind_nominal_capacity_mwp = auto()  # shape: (batch_size, n_pv_systems)
    #: pv_mask is True for good PV systems in each example.
    # The RawPVDataSource doesn't use pv_mask. Instead is sets missing PV systems to NaN
    # across all PV batch keys.
    wind_mask = auto()  # shape: (batch_size, n_pv_systems)

    # PV coordinates:
    # Each has shape: (batch_size, n_pv_systems), will be NaN for missing PV systems.
    wind_latitude = auto()
    wind_longitude = auto()
    wind_time_utc = auto()  # Seconds since UNIX epoch (1970-01-01).

    # PV Fourier coordinates:
    # Each has shape: (batch_size, n_pv_systems, n_fourier_features_per_dim),
    # and will be NaN for missing PV systems.
    wind_latitude_fourier = auto()
    wind_longitude_fourier = auto()
    wind_time_utc_fourier = auto()  # (batch_size, time, n_fourier_features)

    # -------------- GSP --------------------------------------------
    gsp = auto()  # shape: (batch_size, time, 1)
    gsp_t0_idx = auto()  # shape: scalar
    gsp_id = auto()  # shape: (batch_size)

    # GSP coordinates:
    # Each has shape: (batch_size). No NaNs.
    gsp_y_osgb = auto()
    gsp_x_osgb = auto()
    gsp_time_utc = auto()  # Seconds since UNIX epoch (1970-01-01). (batch_size, time)
    gsp_nominal_capacity_mwp = auto()  # (batch_size)
    gsp_effective_capacity_mwp = auto()  # (batch_size)

    # GSP Fourier coordinates:
    # Each has shape: (batch_size, 1, n_fourier_features_per_dim),
    # no NaNs.
    gsp_y_osgb_fourier = auto()
    gsp_x_osgb_fourier = auto()
    gsp_time_utc_fourier = auto()  # (batch_size, time, n_fourier_features)

    # -------------- SUN --------------------------------------------
    # Solar position at every timestep. shape = (batch_size, n_timesteps)
    # The solar position data comes from two alternative sources: either the Sun pre-prepared
    # batches, or the SunPosition np_batch_processor for the RawDataset.
    hrvsatellite_solar_azimuth = auto()
    hrvsatellite_solar_elevation = auto()
    satellite_solar_azimuth = auto()
    satellite_solar_elevation = auto()
    gsp_solar_azimuth = auto()
    gsp_solar_elevation = auto()
    pv_solar_azimuth = auto()
    pv_solar_elevation = auto()

    # Solar position at the centre of the HRV image at t0
    # (from `power_perceiver.np_batch_processor.SunPosition`)
    # shape = (example,)
    # Not used in Raw data pipeline.
    solar_azimuth_at_t0 = auto()
    solar_elevation_at_t0 = auto()

    # ------------- REQUESTED NUMBER OF TIMESTEPS ---------------------
    # Added by `ReduceNumTimesteps`. Gives the indicies of the randomly selected timesteps.
    # Not used in the Raw data pipeline.
    requested_timesteps = auto()  # shape: (n_requested_timesteps)

    # -------------- SATELLITE -----------------------------------
    # shape: (batch_size, time, channels, y, x)
    #
    # Or, if the imagery has been patched,
    # shape: (batch_size, time, channels, y, x, n_pixels_per_patch) where n_pixels_per_patch
    # is the *total* number of pixels,
    # i.e. n_pixels_per_patch_along_height * n_pixels_per_patch_along_width.
    satellite_actual = auto()
    satellite_predicted = auto()  # shape: batch_size, time, y, x
    satellite_t0_idx = auto()  # shape: scalar

    # HRV satellite coordinates:
    satellite_y_osgb = auto()  # shape: (batch_size, y, x)
    satellite_x_osgb = auto()  # shape: (batch_size, y, x)
    satellite_y_geostationary = auto()  # shape: (batch_size, y)
    satellite_x_geostationary = auto()  # shape: (batch_size, x)
    #: Time is seconds since UNIX epoch (1970-01-01). Shape: (batch_size, n_timesteps)
    satellite_time_utc = auto()
    # Added by np_batch_processor.Topography:
    satellite_surface_height = auto()  # The surface height at each pixel. (batch_size, y, x)

    # HRV satellite Fourier coordinates:
    # Spatial coordinates. Shape: (batch_size, y, x, n_fourier_features_per_dim)
    satellite_y_osgb_fourier = auto()
    satellite_x_osgb_fourier = auto()
    #: Time shape: (batch_size, n_timesteps, n_fourier_features_per_dim)
    satellite_time_utc_fourier = auto()

    # -------------- Sensor ---------------------------------------------
    sensor = auto()  # shape: (batch_size, time, n_pv_systems)
    sensor_t0_idx = auto()  # shape: scalar
    sensor_ml_id = auto()  # shape: (batch_size, n_pv_systems)
    sensor_id = auto()  # shape: (batch_size, n_pv_systems)
    sensor_observed_capacity_wp = auto()  # shape: (batch_size, n_pv_systems)
    sensor_nominal_capacity_wp = auto()  # shape: (batch_size, n_pv_systems)
    #: pv_mask is True for good PV systems in each example.
    # The RawPVDataSource doesn't use pv_mask. Instead is sets missing PV systems to NaN
    # across all PV batch keys.
    sensor_mask = auto()  # shape: (batch_size, n_pv_systems)

    # PV coordinates:
    # Each has shape: (batch_size, n_pv_systems), will be NaN for missing PV systems.
    sensor_latitude = auto()
    sensor_longitude = auto()
    sensor_time_utc = auto()  # Seconds since UNIX epoch (1970-01-01).

    # PV Fourier coordinates:
    # Each has shape: (batch_size, n_pv_systems, n_fourier_features_per_dim),
    # and will be NaN for missing PV systems.
    sensor_latitude_fourier = auto()
    sensor_longitude_fourier = auto()
    sensor_time_utc_fourier = auto()  # (batch_size, time, n_fourier_features)

    wind_solar_azimuth = auto()
    wind_solar_elevation = auto()

    # -------------- TIME -------------------------------------------
    # Sine and cosine of date of year and time of day at every timestep.
    # shape = (batch_size, n_timesteps)
    # This is calculated for wind only inside datapipes.
    wind_date_sin = auto()
    wind_date_cos = auto()
    wind_time_sin = auto()
    wind_time_cos = auto()


class NWPBatchKey(Enum):
    """The names of the different elements of each NWP batch.

    This is also where we document the exact shape of each element.
    """

    # -------------- NWP --------------------------------------------
    nwp = auto()  # shape: (batch_size, target_time_utc, channel, y_osgb, x_osgb)
    nwp_t0_idx = auto()  # shape: scalar
    nwp_target_time_utc = auto()  # shape: (batch_size, target_time_utc)
    nwp_init_time_utc = auto()  # shape: (batch_size, target_time_utc)
    nwp_step = auto()  # Int. Number of hours. shape: (batch_size, target_time_utc)
    nwp_y_osgb = auto()  # shape: (batch_size, y_osgb)
    nwp_x_osgb = auto()  # shape: (batch_size, x_osgb)
    nwp_channel_names = auto()  # shape: (channel,)

    # NWP Fourier features:
    nwp_target_time_utc_fourier = auto()
    nwp_init_time_utc_fourier = auto()
    nwp_y_osgb_fourier = auto()
    nwp_x_osgb_fourier = auto()
    nwp_target_time_solar_azimuth = auto()
    nwp_target_time_solar_elevation = auto()


NWPNumpyBatch = dict[NWPBatchKey, np.ndarray]

NumpyBatch = dict[BatchKey, Union[np.ndarray, dict[str, NWPNumpyBatch]]]

XarrayBatch = dict[BatchKey, Union[xr.DataArray, xr.Dataset]]

TensorBatch = dict[BatchKey, Union[torch.Tensor, dict[str, dict[NWPBatchKey, torch.Tensor]]]]
