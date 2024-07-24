"""
Module for defining limits for NWP data.
"""

# limits for NWP data in accordance with https://huggingface.co/openclimatefix/pvnet_uk_region/blob/main/data_config.yaml
NWP_LIMITS = {
    "t2m": (200, 350),  # Temperature in Kelvin (-100°C to 60°C)
    "dswrf": (0, 1500),  # Downward short-wave radiation flux, W/m^2
    "dlwrf": (0, 750),  # Downward long-wave radiation flux, W/m^2
    "hcc": (0, 100),  # High cloud cover, %
    "mcc": (0, 100),  # Medium cloud cover, %
    "lcc": (0, 100),  # Low cloud cover, %
    "tcc": (0, 100),  # Total cloud cover, %
    "sde": (0, 1000),  # Snowfall depth, meters
    "duvrs": (0, 500),  # Direct UV radiation at surface, W/m^2 (positive values only)
    "u10": (-200, 200),  # U component of 10m wind, m/s
    "v10": (-200, 200),  # V component of 10m wind, m/s
    # UKV NWP channels (additional to ECMWF)
    "prate": (0, 2000),  # Precipitation rate, , kg/m^2/s (equivalent to 0-2000 mm/day)
    "r": (0, 100),  # Relative humidity, %
    "si10": (0, 250),  # Wind speed at 10m, m/s
    "t": (200, 350),  # Temperature in Kelvin (-100°C to 60°C)
    "vis": (0, 100000),  # Visibility, meters
    # Satellite channels (no direct mapping to physical limits, using placeholder values)
    "IR_016": (0, 1000),  # Infrared channel
    "IR_039": (0, 1000),  # Infrared channel
    "IR_087": (0, 1000),  # Infrared channel
    "IR_097": (0, 1000),  # Infrared channel
    "IR_108": (0, 1000),  # Infrared channel
    "IR_120": (0, 1000),  # Infrared channel
    "IR_134": (0, 1000),  # Infrared channel
    "VIS006": (0, 1000),  # Visible channel
    "VIS008": (0, 1000),  # Visible channel
    "WV_062": (0, 1000),  # Water vapor channel
    "WV_073": (0, 1000),  # Water vapor channel
}