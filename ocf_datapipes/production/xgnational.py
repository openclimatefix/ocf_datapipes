"""
Pipeline to return live GSP and NWP data
"""

import logging
from pathlib import Path
from typing import Union

import xarray

import ocf_datapipes  # noqa
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenGSPFromDatabase, OpenNWP

logger = logging.getLogger(__name__)
xarray.set_options(keep_attrs=True)


def select_most_recent_init_time(ds):
    """Select the most recent init time from the input Dataset/DataArray"""
    most_recent_init_time = ds.init_time_utc.max()
    ds = ds.sel(init_time_utc=most_recent_init_time)
    logger.debug(f"Selected most recent NWP observation: {most_recent_init_time.values}")
    return ds


def xgnational_production(configuration_filename: Union[Path, str]) -> dict:
    """
    Create the National XG Boost  using a configuration

    Args:
        configuration_filename: Name of the configuration

    Returns:
        dictionary of 'nwp' and 'gsp' containing xarray for both
    """

    configuration: Configuration = load_yaml_configuration(filename=configuration_filename)

    logger.debug("Opening Datasets")
    base_nwp_datapipe = OpenNWP(configuration.input_data.nwp["ukv"].nwp_zarr_path)
    nwp_datapipe = base_nwp_datapipe.map(select_most_recent_init_time)
    gsp_datapipe = OpenGSPFromDatabase(
        history_minutes=configuration.input_data.gsp.history_minutes,
        interpolate_minutes=configuration.input_data.gsp.live_interpolate_minutes,
        load_extra_minutes=configuration.input_data.gsp.live_load_extra_minutes,
        national_only=True,
    )

    nwp_xr = next(iter(nwp_datapipe))
    gsp_xr = next(iter(gsp_datapipe))

    return {"nwp": nwp_xr, "gsp": gsp_xr}
