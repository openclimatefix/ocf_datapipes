"""Statistics and variable names."""

import numpy as np
import xarray as xr

# --------------------------- SOLAR COORDS ---------------------------

ELEVATION_MEAN = 37.4
ELEVATION_STD = 12.7
AZIMUTH_MEAN = 177.7
AZIMUTH_STD = 41.7

# --------------------------- FUNCS ----------------------------------


def _to_data_array(d):
    return xr.DataArray(
        [d[k] for k in d.keys()],
        coords={"channel": [k for k in d.keys()]},
    ).astype(np.float32)


class NWPStatDict(dict):
    """Custom dictionary class to hold NWP normalizarion stats"""

    def __getitem__(self, key):
        if key not in NWP_PROVIDERS:
            raise KeyError(f"{key} is not a supported NWP provider - {NWP_PROVIDERS}")
        elif key in self.keys():
            return super().__getitem__(key)
        else:
            raise KeyError(
                f"Values for {key} not yet available in ocf-datapipes {list(self.keys())}"
            )


# --------------------------- NWP ------------------------------------

NWP_PROVIDERS = [
    "ukv",
    "gfs",
    "icon-eu",
    "icon-global",
    "ecmwf",
    "ecmwf_india",
    "excarta",
    "merra2",
    "merra2_uk",
    "mo_global",
]

# ------ UKV
# Means and std computed WITH version_7 and higher, MetOffice values
UKV_STD = {
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
UKV_MEAN = {
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
UKV_MIN = {
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
UKV_MAX = {
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

UKV_VARIABLE_NAMES = tuple(UKV_MEAN.keys())
UKV_STD = _to_data_array(UKV_STD)
UKV_MEAN = _to_data_array(UKV_MEAN)

# These were calculated from 200 random init times (step 0s) from the MO global data
MO_GLOBAL_INDIA_MEAN = {
    "temperature_sl": 298.2,
    "wind_u_component_10m": 0.5732,
    "wind_v_component_10m": -0.2831,
}

MO_GLOBAL_INDIA_STD = {
    "temperature_sl": 8.473,
    "wind_u_component_10m": 2.599,
    "wind_v_component_10m": 2.016,
}


MO_GLOBAL_VARIABLE_NAMES = tuple(MO_GLOBAL_INDIA_MEAN.keys())
MO_GLOBAL_INDIA_STD = _to_data_array(MO_GLOBAL_INDIA_STD)
MO_GLOBAL_INDIA_MEAN = _to_data_array(MO_GLOBAL_INDIA_MEAN)


# ------ GFS
GFS_STD = {
    "dlwrf": 96.305916,
    "dswrf": 246.18533,
    "hcc": 42.525383,
    "lcc": 44.3732,
    "mcc": 43.150745,
    "prate": 0.00010159573,
    "r": 25.440672,
    "sde": 0.43345627,
    "t": 22.825893,
    "tcc": 41.030598,
    "u10": 5.470838,
    "u100": 6.8899174,
    "v10": 4.7401133,
    "v100": 6.076132,
    "vis": 8294.022,
    "u": 10.614556,
    "v": 7.176398,
}
GFS_MEAN = {
    "dlwrf": 298.342,
    "dswrf": 168.12321,
    "hcc": 35.272,
    "lcc": 43.578342,
    "mcc": 33.738823,
    "prate": 2.8190969e-05,
    "r": 18.359747,
    "sde": 0.36937004,
    "t": 278.5223,
    "tcc": 66.841606,
    "u10": -0.0022310058,
    "u100": 0.0823025,
    "v10": 0.06219831,
    "v100": 0.0797807,
    "vis": 19628.32,
    "u": 11.645444,
    "v": 0.12330122,
}


GFS_VARIABLE_NAMES = tuple(GFS_MEAN.keys())
GFS_STD = _to_data_array(GFS_STD)
GFS_MEAN = _to_data_array(GFS_MEAN)


# ------ ECMWF
# These were calculated from 100 random init times of UK data from 2020-2023
ECMWF_STD = {
    "dlwrf": 15855867.0,
    "dswrf": 13025427.0,
    "duvrs": 1445635.25,
    "hcc": 0.42244860529899597,
    "lcc": 0.3791404366493225,
    "mcc": 0.38039860129356384,
    "prate": 9.81039775069803e-05,
    "sde": 0.000913831521756947,
    "sr": 16294988.0,
    "t2m": 3.692270040512085,
    "tcc": 0.37487083673477173,
    "u10": 5.531515598297119,
    "u100": 7.2320556640625,
    "u200": 8.049470901489258,
    "v10": 5.411230564117432,
    "v100": 6.944501876831055,
    "v200": 7.561611652374268,
    "diff_dlwrf": 131942.03125,
    "diff_dswrf": 715366.3125,
    "diff_duvrs": 81605.25,
    "diff_sr": 818950.6875,
}
ECMWF_MEAN = {
    "dlwrf": 27187026.0,
    "dswrf": 11458988.0,
    "duvrs": 1305651.25,
    "hcc": 0.3961029052734375,
    "lcc": 0.44901806116104126,
    "mcc": 0.3288780450820923,
    "prate": 3.108070450252853e-05,
    "sde": 8.107526082312688e-05,
    "sr": 12905302.0,
    "t2m": 283.48333740234375,
    "tcc": 0.7049227356910706,
    "u10": 1.7677178382873535,
    "u100": 2.393547296524048,
    "u200": 2.7963004112243652,
    "v10": 0.985887885093689,
    "v100": 1.4244288206100464,
    "v200": 1.6010299921035767,
    "diff_dlwrf": 1136464.0,
    "diff_dswrf": 420584.6875,
    "diff_duvrs": 48265.4765625,
    "diff_sr": 469169.5,
}

ECMWF_VARIABLE_NAMES = tuple(ECMWF_MEAN.keys())
ECMWF_STD = _to_data_array(ECMWF_STD)
ECMWF_MEAN = _to_data_array(ECMWF_MEAN)

# These were calculated from 200 random init times of ECMWF data from 2020-2024
INDIA_ECMWF_MEAN = {
    "dlwrf": 56931292.0,
    "dswrf": 31114434.0,
    "duvrs": 3363442.75,
    "hcc": 0.2640938460826874,
    "lcc": 0.09548956900835037,
    "mcc": 0.11118797957897186,
    "prate": 3.028022729267832e-05,
    "sde": 0.0015021926956251264,
    "sr": 31403020.0,
    "t2m": 298.24462890625,
    "tcc": 0.33420485258102417,
    "u10": 0.7755107283592224,
    "u100": 1.0605329275131226,
    "u200": 1.2298915386199951,
    "v10": 0.02332865633070469,
    "v100": -0.07577426731586456,
    "v200": -0.1255049854516983,
    "diff_dlwrf": 1340142.4,
    "diff_dswrf": 820569.5,
    "diff_duvrs": 94480.24,
    "diff_sr": 814910.1,
}

INDIA_ECMWF_STD = {
    "dlwrf": 34551808.0,
    "dswrf": 21211150.0,
    "duvrs": 2300205.5,
    "hcc": 0.3942722678184509,
    "lcc": 0.22802403569221497,
    "mcc": 0.2254289835691452,
    "prate": 0.00023954990319907665,
    "sde": 0.09043189883232117,
    "sr": 23481620.0,
    "t2m": 7.574307918548584,
    "tcc": 0.4046371579170227,
    "u10": 2.7440550327301025,
    "u100": 4.084362506866455,
    "u200": 4.770451068878174,
    "v10": 2.401158571243286,
    "v100": 3.5278923511505127,
    "v200": 3.974159002304077,
    "diff_dlwrf": 292804.8,
    "diff_dswrf": 1082344.9,
    "diff_duvrs": 125904.18,
    "diff_sr": 1088536.2,
}


INDIA_ECMWF_VARIABLE_NAMES = tuple(INDIA_ECMWF_MEAN.keys())
INDIA_ECMWF_STD = _to_data_array(INDIA_ECMWF_STD)
INDIA_ECMWF_MEAN = _to_data_array(INDIA_ECMWF_MEAN)

# ------- Excarta
EXCARTA_MEAN = {
    "10m_wind_speed": 6.228208065032959,
    "10m_wind_speed_angle": 175.47128295898438,
    "100m_wind_speed": 7.9128193855285645,
    "100m_wind_speed_angle": 175.80787658691406,
    "2m_dewpoint_temperature": 274.4578857421875,
    "2m_temperature": 278.8953857421875,
    "dhi": 57.628360730489284,
    "dni": 199.1950671565817,
    "ghi": 166.49980409084822,
    "mean_sea_level_pressure": 1009.3062133789062,
    "surface_pressure": 966.7402954101562,
    "total_precipitation_1hr": 0.11734692007303238,
    "10u": -0.223025843501091,
    "10v": 0.022012686356902122,
    "100u": -0.24015851318836212,
    "100v": -0.05406061187386513,
}
EXCARTA_STD = {
    "10m_wind_speed": 3.7157022953033447,
    "10m_wind_speed_angle": 99.94088745117188,
    "100m_wind_speed": 4.652981281280518,
    "100m_wind_speed_angle": 100.15699768066406,
    "2m_dewpoint_temperature": 20.453125,
    "2m_temperature": 20.99526596069336,
    "dhi": 81.87070491961677,
    "dni": 300.6946092717406,
    "ghi": 253.34016941784316,
    "mean_sea_level_pressure": 13.694265365600586,
    "surface_pressure": 94.92646789550781,
    "total_precipitation_1hr": 0.2296331822872162,
    "10u": 4.701057434082031,
    "10v": 5.518700122833252,
    "100u": 6.01924991607666,
    "100v": 6.966071128845215,
}

EXCARTA_VARIABLE_NAMES = tuple(EXCARTA_MEAN.keys())
EXCARTA_STD = _to_data_array(EXCARTA_STD)
EXCARTA_MEAN = _to_data_array(EXCARTA_MEAN)

# ------ MERRA2
# Calculated on data from 2018-01-01 to 2024-02-29
MERRA2_STD = {"AODANA": 0.26992613}
MERRA2_MEAN = {"AODANA": 0.38423285}

MERRA2_VARIABLE_NAMES = tuple(MERRA2_MEAN.keys())
MERRA2_STD = _to_data_array(MERRA2_STD)
MERRA2_MEAN = _to_data_array(MERRA2_MEAN)


UK_MERRA2_STD = {"AODANA": 0.09051198}
UK_MERRA2_MEAN = {"AODANA": 0.13139527}

UK_MERRA2_VARIABLE_NAMES = tuple(UK_MERRA2_MEAN.keys())
UK_MERRA2_STD = _to_data_array(UK_MERRA2_STD)
UK_MERRA2_MEAN = _to_data_array(UK_MERRA2_MEAN)


# ------ ALL NWPS
# These dictionaries are for convenience
NWP_VARIABLE_NAMES = NWPStatDict(
    ukv=UKV_VARIABLE_NAMES,
    gfs=GFS_VARIABLE_NAMES,
    ecmwf=ECMWF_VARIABLE_NAMES,
    ecmwf_india=INDIA_ECMWF_VARIABLE_NAMES,
    excarta=EXCARTA_VARIABLE_NAMES,
    merra2=MERRA2_VARIABLE_NAMES,
    merra2_uk=UK_MERRA2_VARIABLE_NAMES,
    mo_global=MO_GLOBAL_VARIABLE_NAMES,
)
NWP_STDS = NWPStatDict(
    ukv=UKV_STD,
    gfs=GFS_STD,
    ecmwf=ECMWF_STD,
    ecmwf_india=INDIA_ECMWF_STD,
    excarta=EXCARTA_STD,
    merra2=MERRA2_STD,
    merra2_uk=UK_MERRA2_STD,
    mo_global=MO_GLOBAL_INDIA_STD,
)
NWP_MEANS = NWPStatDict(
    ukv=UKV_MEAN,
    gfs=GFS_MEAN,
    ecmwf=ECMWF_MEAN,
    ecmwf_india=INDIA_ECMWF_MEAN,
    excarta=EXCARTA_MEAN,
    merra2=MERRA2_MEAN,
    merra2_uk=UK_MERRA2_MEAN,
    mo_global=MO_GLOBAL_INDIA_MEAN,
)

# --------------------------- SATELLITE ------------------------------


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


RSS_VARIABLE_NAMES = tuple(RSS_MEAN.keys())
RSS_STD = _to_data_array(RSS_STD)
RSS_MEAN = _to_data_array(RSS_MEAN)

# normalizing from raw values

RSS_RAW_MIN = {
    "IR_016": -2.5118103,
    "IR_039": -64.83977,
    "IR_087": 63.404694,
    "IR_097": 2.844452,
    "IR_108": 199.10002,
    "IR_120": -17.254883,
    "IR_134": -26.29155,
    "VIS006": -1.1009827,
    "VIS008": -2.4184198,
    "WV_062": 199.57048,
    "WV_073": 198.95093,
    "HRV": -1.2278595,
}

RSS_RAW_MAX = {
    "IR_016": 69.60857,
    "IR_039": 339.15588,
    "IR_087": 340.26526,
    "IR_097": 317.86752,
    "IR_108": 313.2767,
    "IR_120": 315.99194,
    "IR_134": 274.82297,
    "VIS006": 93.786545,
    "VIS008": 101.34922,
    "WV_062": 249.91806,
    "WV_073": 286.96323,
    "HRV": 103.90016,
}

RSS_RAW_MIN = _to_data_array(RSS_RAW_MIN)
RSS_RAW_MAX = _to_data_array(RSS_RAW_MAX)


# --------------------------- SENSORS --------------------------------

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

METEOMATICS_MEAN = {
    "air_density_100m:kgm3": 1.1133829082116107,
    "air_density_10m:kgm3": 1.1258197248774804,
    "air_density_200m:kgm3": 1.1025331248239914,
    "air_density_25m:kgm3": 1.1233539544600366,
    "cape:Jkg": 253.03451561805372,
    "wind_dir_100m:d": 169.45384588817276,
    "wind_dir_10m:d": 167.41098619295326,
    "wind_dir_200m:d": 172.84169301261238,
    "wind_gusts_100m:ms": 7.5697940411227,
    "wind_gusts_10m:ms": 6.669156757048757,
    "wind_gusts_200m:ms": 7.931425709746494,
    "wind_speed_100m:ms": 5.700552600676936,
    "wind_speed_10m:ms": 3.965335089434524,
    "wind_speed_200m:ms": 6.165206326392172,
    "100u": -1.3123529788956498,
    "100v": -0.8755045035036727,
    "10u": -1.0394979843264418,
    "10v": -0.66336142004311,
    "200u": -1.424415379022279,
    "200v": -1.229537229620669,
    "diffuse_rad:W": 82.88873314973185,
    "direct_rad:W": 133.4593731845195,
    "global_rad:W": 216.3481430642122,
}
METEOMATICS_STDDEV = {
    "air_density_100m:kgm3": 0.03185123687962139,
    "air_density_10m:kgm3": 0.03697884909883139,
    "air_density_200m:kgm3": 0.030173773277697897,
    "air_density_25m:kgm3": 0.03545224924821819,
    "cape:Jkg": 526.2390700702676,
    "wind_dir_100m:d": 93.64265888083348,
    "wind_dir_10m:d": 93.33727424358632,
    "wind_dir_200m:d": 94.36429741102313,
    "wind_gusts_100m:ms": 3.5043788891546646,
    "wind_gusts_10m:ms": 3.2687753968202165,
    "wind_gusts_200m:ms": 3.8968274114850043,
    "wind_speed_100m:ms": 2.7628611366229463,
    "wind_speed_10m:ms": 2.0002702159429586,
    "wind_speed_200m:ms": 3.2343287696357392,
    "100u": 4.420773201146464,
    "100v": 4.254137684713525,
    "10u": 3.169128942791583,
    "10v": 2.856743008957849,
    "200u": 4.686837329830969,
    "200v": 4.79202321673883,
    "diffuse_rad:W": 103.34484637546339,
    "direct_rad:W": 195.3664813556391,
    "global_rad:W": 287.2154712794865,
}

METEOMATICS_VARIABLE_NAMES = tuple(METEOMATICS_MEAN.keys())
METEOMATICS_STDDEV = _to_data_array(METEOMATICS_STDDEV)
METEOMATICS_MEAN = _to_data_array(METEOMATICS_MEAN)
