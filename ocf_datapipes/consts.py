"""Constants and Enums."""

from enum import Enum, auto
from numbers import Number
from typing import NamedTuple

from pathy import Pathy

PV_TIME_AXIS = 1
PV_SYSTEM_AXIS = 2

Y_OSGB_MEAN = 357021.38
Y_OSGB_STD = 612920.2
X_OSGB_MEAN = 187459.94
X_OSGB_STD = 622805.44

SATELLITE_SPACER_LEN = 17  # Patch of 4x4 + 1 for surface height.
PV_SPACER_LEN = 18  # 16 for embedding dim + 1 for marker + 1 for history


class Location(NamedTuple):
    """Represent a spatial location."""

    x: Number
    y: Number


class BatchKey(Enum):
    """The names of the different elements of each batch.

    This is also where we document the exact shape of each element.

    Each `DataSource` may be split into several different BatchKey elements. For example, the
    PV PreparedDataSource yields `pv` and `pv_system_row_number` BatchKeys.
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
    hrvsatellite_time_utc_fourier_t0 = auto()

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

    # -------------- PV ---------------------------------------------
    pv = auto()  # shape: (batch_size, time, n_pv_systems)
    pv_t0_idx = auto()  # shape: scalar
    pv_system_row_number = auto()  # shape: (batch_size, n_pv_systems)
    pv_id = auto()  # shape: (batch_size, n_pv_systems)
    # PV AC system capacity in watts peak.
    # Warning: In v15, pv_capacity_wp is sometimes 0. This will be fixed in
    # https://github.com/openclimatefix/nowcasting_dataset/issues/622
    pv_capacity_wp = auto()  # shape: (batch_size, n_pv_systems)
    #: pv_mask is True for good PV systems in each example.
    # The RawPVDataSource doesn't use pv_mask. Instead is sets missing PV systems to NaN
    # across all PV batch keys.
    pv_mask = auto()  # shape: (batch_size, n_pv_systems)

    # PV coordinates:
    # Each has shape: (batch_size, n_pv_systems), will be NaN for missing PV systems.
    pv_y_osgb = auto()
    pv_x_osgb = auto()
    pv_time_utc = auto()  # Seconds since UNIX epoch (1970-01-01).

    # PV Fourier coordinates:
    # Each has shape: (batch_size, n_pv_systems, n_fourier_features_per_dim),
    # and will be NaN for missing PV systems.
    pv_y_osgb_fourier = auto()
    pv_x_osgb_fourier = auto()
    pv_time_utc_fourier = auto()  # (batch_size, time, n_fourier_features)
    pv_time_utc_fourier_t0 = auto()  # Added by SaveT0Time. Shape: (batch_size, n_fourier_features)

    # -------------- GSP --------------------------------------------
    gsp = auto()  # shape: (batch_size, time, 1)  (the RawGSPDataSource include a '1',
    # not sure if the prepared dataset does!)
    gsp_t0_idx = auto()  # shape: scalar
    gsp_id = auto()  # shape: (batch_size)

    # GSP coordinates:
    # Each has shape: (batch_size). No NaNs.
    gsp_y_osgb = auto()
    gsp_x_osgb = auto()
    gsp_time_utc = auto()  # Seconds since UNIX epoch (1970-01-01). (batch_size, time)
    gsp_capacity_mwp = auto()  # (batch_size)

    # GSP Fourier coordinates:
    # Each has shape: (batch_size, 1, n_fourier_features_per_dim),
    # no NaNs.
    gsp_y_osgb_fourier = auto()
    gsp_x_osgb_fourier = auto()
    gsp_time_utc_fourier = auto()  # (batch_size, time, n_fourier_features)
    gsp_time_utc_fourier_t0 = auto()  # Added by SaveT0Time. Shape: (batch_size, n_fourier_features)

    # -------------- GSP5Min ----------------------------------------
    # Not used by the Raw data pipeline!
    gsp_5_min = auto()  # shape: (batch_size, time)
    gsp_5_min_time_utc = auto()  # shape: (batch_size, time)
    gsp_5_min_time_utc_fourier = auto()  # shape: (batch_size, time, n_fourier_features)
    gsp_5_min_t0_idx = auto()

    # -------------- SUN --------------------------------------------
    # Solar position at every timestep. shape = (batch_size, n_timesteps)
    # The solar position data comes from two alternative sources: either the Sun pre-prepared
    # batches, or the SunPosition np_batch_processor for the RawDataset.
    hrvsatellite_solar_azimuth = auto()
    hrvsatellite_solar_elevation = auto()
    gsp_solar_azimuth = auto()
    gsp_solar_elevation = auto()
    gsp_5_min_solar_azimuth = auto()
    gsp_5_min_solar_elevation = auto()
    pv_solar_azimuth = auto()
    pv_solar_elevation = auto()
    nwp_target_time_solar_azimuth = auto()
    nwp_target_time_solar_elevation = auto()

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


REMOTE_PATH_FOR_DATA_FOR_UNIT_TESTS = Pathy(
    "gs://ocf-public/data_for_unit_tests/prepared_ML_training_data_v15_v2"
)
