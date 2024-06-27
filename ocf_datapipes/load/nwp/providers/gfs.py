import logging

import xarray as xr

from ocf_datapipes.load.nwp.providers.utils import open_zarr_paths

# import pandas as pd

_log = logging.getLogger(__name__)


def open_gfs(zarr_path) -> xr.DataArray:
    """
    Opens the GFS data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    _log.info("Loading NWP GFS data")

    # Open data
    gfs: xr.Dataset = open_zarr_paths(zarr_path, time_dim="init_time_utc")

    # _________________EXTRAPOLATION_________________

    # _log.info("Imputing step 0 for radiation variables")
    #
    # flux_vars = ['dswrf', 'dlwrf']
    # for var in flux_vars:
    #     gfs[var] = xr.concat([
    #         gfs[var].sel(step=slice(pd.Timedelta(hours=3), None)).interp(
    #             step=pd.Timedelta(hours=0),
    #             kwargs={"fill_value": "extrapolate"}
    #         ),
    #         gfs[var].sel(step=slice(pd.Timedelta(hours=3), None))
    #     ], dim='step')

    nwp: xr.DataArray = gfs.to_array()

    del gfs

    nwp = nwp.rename({"variable": "channel"})
    nwp = nwp.transpose("init_time_utc", "step", "channel", "latitude", "longitude")

    return nwp
