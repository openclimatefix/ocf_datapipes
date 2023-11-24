"""Constants and Enums."""

from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import xarray as xr
from pydantic import BaseModel, validator

Y_OSGB_MEAN = 357021.38
Y_OSGB_STD = 612920.2
X_OSGB_MEAN = 187459.94
X_OSGB_STD = 622805.44


DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE = 2048
DEFAULT_N_GSP_PER_EXAMPLE = 32

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


class Location(BaseModel):
    """Represent a spatial location."""

    coordinate_system: Optional[str] = "osgb"  # ["osgb", "lat_lon", "geostationary", "idx"]
    x: float
    y: float
    id: Optional[int]

    @validator("coordinate_system", pre=True, always=True)
    def validate_coordinate_system(cls, v):
        """Validate 'coordinate_system'"""
        allowed_coordinate_systen = ["osgb", "lat_lon", "geostationary", "idx"]
        if v not in allowed_coordinate_systen:
            raise ValueError(f"coordinate_system = {v} is not in {allowed_coordinate_systen}")
        return v

    @validator("x")
    def validate_x(cls, v, values):
        """Validate 'x'"""
        min_x: float
        max_x: float
        if "coordinate_system" not in values:
            raise ValueError("coordinate_system is incorrect")
        co = values["coordinate_system"]
        if co == "osgb":
            min_x, max_x = -103976.3, 652897.98
        if co == "lat_lon":
            min_x, max_x = -180, 180
        if co == "geostationary":
            min_x, max_x = -5568748.275756836, 5567248.074173927
        if co == "idx":
            min_x, max_x = 0, np.inf
        if v < min_x or v > max_x:
            raise ValueError(f"x = {v} must be within {[min_x, max_x]} for {co} coordinate system")
        return v

    @validator("y")
    def validate_y(cls, v, values):
        """Validate 'y'"""
        min_y: float
        max_y: float
        if "coordinate_system" not in values:
            raise ValueError("coordinate_system is incorrect")
        co = values["coordinate_system"]
        if co == "osgb":
            min_y, max_y = -16703.87, 1199851.44
        if co == "lat_lon":
            min_y, max_y = -90, 90
        if co == "geostationary":
            min_y, max_y = 1393687.2151494026, 5570748.323202133
        if co == "idx":
            min_y, max_y = 0, np.inf
        if v < min_y or v > max_y:
            raise ValueError(f"y = {v} must be within {[min_y, max_y]} for {co} coordinate system")
        return v


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
    pv_time_utc_fourier_t0 = auto()  # Added by SaveT0Time. Shape: (batch_size, n_fourier_features)

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
    gsp_time_utc_fourier_t0 = auto()  # Added by SaveT0Time. Shape: (batch_size, n_fourier_features)

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
    sensor_time_utc_fourier_t0 = (
        auto()
    )  # Added by SaveT0Time. Shape: (batch_size, n_fourier_features)


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

# Newer version_7 and higher, MetOffice values
NEW_NWP_STD = {
    "cdcb": 2126.99350113,
    "lcc": 39.33210726,
    "mcc": 41.91144559,
    "hcc": 38.07184418,
    "sde": 0.1029753,
    "hcct": 18382.63958991,
    "dswrf": 190.47216887,
    "dlwrf": 39.45988077,
    "h": 1075.77812282,
    "t": 4.38818501,
    "r": 11.45012499,
    "dpt": 4.57250482,
    "vis": 21578.97975625,
    "si10": 3.94718813,
    "wdir10": 94.08407495,
    "prmsl": 1252.71790539,
    "prate": 0.00021497,
}
NEW_NWP_MEAN = {
    "cdcb": 1412.26599062,
    "lcc": 50.08362643,
    "mcc": 40.88984494,
    "hcc": 29.11949682,
    "sde": 0.00289545,
    "hcct": -18345.97478167,
    "dswrf": 111.28265039,
    "dlwrf": 325.03130139,
    "h": 2096.51991356,
    "t": 283.64913206,
    "r": 81.79229501,
    "dpt": 280.54379901,
    "vis": 32262.03285118,
    "si10": 6.88348448,
    "wdir10": 199.41891636,
    "prmsl": 101321.61574029,
    "prate": 3.45793433e-05,
}


NEW_NWP_MAX = {
    "cdcb": 20632.0,
    "lcc": 100.0,
    "mcc": 100.0,
    "hcc": 100.05,
    "sde": 10.0,
    "hcct": 11579.0,
    "dswrf": 1018.4,
    "dlwrf": 492.4,
    "h": 5241.0,
    "t": 315.8,
    "r": 100.05,
    "dpt": 302.535,
    "vis": 99794.0,
    "si10": 46.15,
    "wdir10": 360.0,
    "prmsl": 105440.0,
    "prate": 0.055556,
}
NEW_NWP_MIN = {
    "cdcb": 5.0,
    "lcc": 0.0,
    "mcc": 0.0,
    "hcc": 0.0,
    "sde": 0.0,
    "hcct": -32766.0,
    "dswrf": 0.0,
    "dlwrf": 131.8,
    "h": 0.0,
    "t": 227.15,
    "r": 7.84,
    "dpt": 227.15,
    "vis": 6.0,
    "si10": 0.05,
    "wdir10": 0.0,
    "prmsl": 94160.0,
    "prate": 0.0,
}

NWP_CHANNEL_NAMES = tuple(NEW_NWP_MEAN.keys())

RSS_MAX = {}
RSS_MIN = {}


NWP_GFS_MEAN = {
    "t": 285.7799539185846,
    "dswrf": 294.6696933986283,
    "prate": 3.6078121378638696e-05,
    "dlwrf": 319,
    "u": 0.552,
    "v": -0.477,
}

NWP_GFS_STD = {
    "t": 5.017000766747606,
    "dswrf": 233.1834250473355,
    "prate": 0.00021690701537950742,
    "dlwrf": 46.571,
    "u": 4.165,
    "v": 4.123,
}

NWP_GFS_CHANNEL_NAMES = tuple(NWP_GFS_STD.keys())


def _to_data_array(d):
    return xr.DataArray(
        [d[key] for key in NWP_CHANNEL_NAMES if key in d.keys()],
        coords={"channel": [_channel for _channel in NWP_CHANNEL_NAMES if _channel in d.keys()]},
    ).astype(np.float32)


NWP_MEAN = _to_data_array(NWP_MEAN)
NWP_STD = _to_data_array(NWP_STD)

NEW_NWP_MEAN = _to_data_array(NEW_NWP_MEAN)
NEW_NWP_STD = _to_data_array(NEW_NWP_STD)

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

# RSS Mean and std taken from random 20% of 2020 RSS imagery
RSS_STD = {
    "HRV": 0.11405209,
    "IR_016": 0.21462157,
    "IR_039": 0.04618041,
    "IR_087": 0.06687243,
    "IR_097": 0.0468558,
    "IR_108": 0.17482725,
    "IR_120": 0.06115861,
    "IR_134": 0.04492306,
    "VIS006": 0.12184761,
    "VIS008": 0.13090034,
    "WV_062": 0.16111417,
    "WV_073": 0.12924142,
}
RSS_MEAN = {
    "HRV": 0.09298719,
    "IR_016": 0.17594202,
    "IR_039": 0.86167645,
    "IR_087": 0.7719318,
    "IR_097": 0.8014212,
    "IR_108": 0.71254843,
    "IR_120": 0.89058584,
    "IR_134": 0.944365,
    "VIS006": 0.09633306,
    "VIS008": 0.11426069,
    "WV_062": 0.7359355,
    "WV_073": 0.62479186,
}


def _to_data_array(d):
    return xr.DataArray(
        [d[key] for key in SAT_VARIABLE_NAMES],
        coords={"channel": list(SAT_VARIABLE_NAMES)},
    ).astype(np.float32)


SAT_MEAN_DA = _to_data_array(SAT_MEAN)
SAT_STD_DA = _to_data_array(SAT_STD)

RSS_MEAN = _to_data_array(RSS_MEAN)
RSS_STD = _to_data_array(RSS_STD)


AWOS_VARIABLE_NAMES = [
    "sky_level_4_coverage",
    "weather_codes",
    "sky_level_3_coverage",
    "elevation",
    "temperature_2m",
    "dewpoint_2m",
    "relative_humidity",
    "wind_direction_deg",
    "wind_speed_knots",
    "precipitation_1hr",
    "pressure_altimeter_inch",
    "pressure_sea_level_millibar",
    "visibility_miles",
    "wind_gust_knots",
    "sky_level_1_altitude_feet",
    "sky_level_2_altitude_feet",
    "sky_level_3_altitude_feet",
    "sky_level_4_altitude_feet",
    "ice_accretion_1hr",
    "ice_accretion_3hr",
    "ice_accretion_6hr",
    "peak_wind_gust_knots",
    "peak_wind_direction_deg",
    "apparent_temperature_fahrenheit",
    "snow_depth_inches",
]
