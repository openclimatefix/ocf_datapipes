"""Constants and Enums."""

from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import xarray as xr
from pydantic import BaseModel

PV_TIME_AXIS = 1
PV_SYSTEM_AXIS = 2

Y_OSGB_MEAN = 357021.38
Y_OSGB_STD = 612920.2
X_OSGB_MEAN = 187459.94
X_OSGB_STD = 622805.44

SATELLITE_SPACER_LEN = 17  # Patch of 4x4 + 1 for surface height.
PV_SPACER_LEN = 18  # 16 for embedding dim + 1 for marker + 1 for history

PV_SYSTEM_ID: str = "pv_system_id"
PV_SYSTEM_ROW_NUMBER = "pv_system_row_number"
PV_SYSTEM_X_COORDS = "pv_system_x_coords"
PV_SYSTEM_Y_COORDS = "pv_system_y_coords"

SUN_AZIMUTH_ANGLE = "sun_azimuth_angle"
SUN_ELEVATION_ANGLE = "sun_elevation_angle"

PV_YIELD = "pv_yield"
PV_DATETIME_INDEX = "pv_datetime_index"
DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE = 2048
GSP_ID: str = "gsp_id"
GSP_YIELD = "gsp_yield"
GSP_X_COORDS = "gsp_x_coords"
GSP_Y_COORDS = "gsp_y_coords"
GSP_DATETIME_INDEX = "gsp_datetime_index"
N_GSPS = 317

DEFAULT_N_GSP_PER_EXAMPLE = 32
OBJECT_AT_CENTER = "object_at_center"
DATETIME_FEATURE_NAMES = (
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)
SATELLITE_DATA = "sat_data"
SATELLITE_Y_COORDS = "sat_y_coords"
SATELLITE_X_COORDS = "sat_x_coords"
SATELLITE_DATETIME_INDEX = "sat_datetime_index"
NWP_TARGET_TIME = "nwp_target_time"
NWP_DATA = "nwp"
NWP_X_COORDS = "nwp_x_coords"
NWP_Y_COORDS = "nwp_y_coords"
X_CENTERS_OSGB = "x_centers_osgb"
Y_CENTERS_OSGB = "y_centers_osgb"
TOPOGRAPHIC_DATA = "topo_data"
TOPOGRAPHIC_X_COORDS = "topo_x_coords"
TOPOGRAPHIC_Y_COORDS = "topo_y_coords"

# "Safe" default NWP variable names:
NWP_VARIABLE_NAMES = (
    "t",
    "dswrf",
    "prate",
    "r",
    "sde",
    "si10",
    "vis",
    "lcc",
    "mcc",
    "hcc",
)

# A complete set of NWP variable names.  Not all are currently used.
FULL_NWP_VARIABLE_NAMES = (
    "cdcb",
    "lcc",
    "mcc",
    "hcc",
    "sde",
    "hcct",
    "dswrf",
    "dlwrf",
    "h",
    "t",
    "r",
    "dpt",
    "vis",
    "si10",
    "wdir10",
    "prmsl",
    "prate",
)

SAT_VARIABLE_NAMES = (
    "HRV",
    "IR_016",
    "IR_039",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
    "VIS006",
    "VIS008",
    "WV_062",
    "WV_073",
)

DEFAULT_REQUIRED_KEYS = [
    NWP_DATA,
    NWP_X_COORDS,
    NWP_Y_COORDS,
    SATELLITE_DATA,
    SATELLITE_X_COORDS,
    SATELLITE_Y_COORDS,
    PV_YIELD,
    PV_SYSTEM_ID,
    PV_SYSTEM_ROW_NUMBER,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_Y_COORDS,
    X_CENTERS_OSGB,
    Y_CENTERS_OSGB,
    GSP_ID,
    GSP_YIELD,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    GSP_DATETIME_INDEX,
    TOPOGRAPHIC_DATA,
    TOPOGRAPHIC_Y_COORDS,
    TOPOGRAPHIC_X_COORDS,
] + list(DATETIME_FEATURE_NAMES)
T0_DT = "t0_dt"


SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME = (
    "spatial_and_temporal_locations_of_each_example.csv"
)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class Location(BaseModel):
    """Represent a spatial location."""

    x: float
    y: float
    id: Optional[int]


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
    # Warning: In v15, pv_capacity_watt_power is sometimes 0. This will be fixed in
    # https://github.com/openclimatefix/nowcasting_dataset/issues/622
    pv_capacity_watt_power = auto()  # shape: (batch_size, n_pv_systems)
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
    gsp_capacity_megawatt_power = auto()  # (batch_size)

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
    satellite_solar_azimuth = auto()
    satellite_solar_elevation = auto()
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
    satellite_time_utc_fourier_t0 = auto()


NumpyBatch = dict[BatchKey, np.ndarray]

XarrayBatch = dict[BatchKey, Union[xr.DataArray, xr.Dataset]]

# Means and std computed with
# nowcasting_dataset/scripts/compute_stats_from_batches.py
# using v15 training batches on 2021-11-24.
NWP_MEAN = {
    "t": 285.7799539185846,
    "dswrf": 294.6696933986283,
    "prate": 3.6078121378638696e-05,
    "r": 75.57106712435926,
    "sde": 0.0024915961594965614,
    "si10": 4.931356852411006,
    "vis": 22321.762918384553,
    "lcc": 47.90454236572895,
    "mcc": 44.22781694449808,
    "hcc": 32.87577371914454,
}

NWP_STD = {
    "t": 5.017000766747606,
    "dswrf": 233.1834250473355,
    "prate": 0.00021690701537950742,
    "r": 15.705370079694358,
    "sde": 0.07560040052148084,
    "si10": 2.664583614352396,
    "vis": 12963.802514945439,
    "lcc": 40.06675870700349,
    "mcc": 41.927221148316384,
    "hcc": 39.05157559763763,
}

NWP_CHANNEL_NAMES = tuple(NWP_STD.keys())


def _to_data_array(d):
    return xr.DataArray(
        [d[key] for key in NWP_CHANNEL_NAMES], coords={"channel": list(NWP_CHANNEL_NAMES)}
    ).astype(np.float32)


NWP_MEAN = _to_data_array(NWP_MEAN)
NWP_STD = _to_data_array(NWP_STD)

SAT_MEAN = {
    "HRV": 236.13257536395903,
    "IR_016": 291.61620182554185,
    "IR_039": 858.8040610176552,
    "IR_087": 738.3103442750336,
    "IR_097": 773.0910794778366,
    "IR_108": 607.5318145165666,
    "IR_120": 860.6716261423857,
    "IR_134": 925.0477987594331,
    "VIS006": 228.02134593063957,
    "VIS008": 257.56333202381205,
    "WV_062": 633.5975770915588,
    "WV_073": 543.4963868823854,
}

SAT_STD = {
    "HRV": 935.9717382401759,
    "IR_016": 172.01044433112992,
    "IR_039": 96.53756504807913,
    "IR_087": 96.21369354283686,
    "IR_097": 86.72892737648276,
    "IR_108": 156.20651744208888,
    "IR_120": 104.35287930753246,
    "IR_134": 104.36462050405994,
    "VIS006": 150.2399269307514,
    "VIS008": 152.16086321818398,
    "WV_062": 111.8514878214775,
    "WV_073": 106.8855172848904,
}


def _to_data_array(d):
    return xr.DataArray(
        [d[key] for key in SAT_VARIABLE_NAMES], coords={"channel": list(SAT_VARIABLE_NAMES)}
    ).astype(np.float32)


SAT_MEAN_DA = _to_data_array(SAT_MEAN)
SAT_STD_DA = _to_data_array(SAT_STD)
