"""Create the training/validation datapipe for training the national MetNet/-2 Model"""
import datetime
import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.convert import ConvertGSPToNumpy, ConvertPVToNumpy
from ocf_datapipes.load import (
    OpenConfiguration,
    OpenGSP,
    OpenNWP,
    OpenPVFromNetCDF,
    OpenSatellite,
    OpenTopography,
)
from ocf_datapipes.select import DropGSP, LocationPicker
from ocf_datapipes.training.common import (
    add_selected_time_slices_from_datapipes,
    get_and_return_overlapping_time_periods_and_t0,
    open_and_return_datapipes,
)
from ocf_datapipes.transform.xarray import PreProcessMetNet
from ocf_datapipes.utils.consts import (
    NWP_MEAN,
    NWP_STD,
    PV_YIELD,
    SAT_MEAN,
    SAT_MEAN_DA,
    SAT_STD,
    SAT_STD_DA,
)

xarray.set_options(keep_attrs=True)
logger = logging.getLogger("metnet_datapipe")
logger.setLevel(logging.DEBUG)


def normalize_gsp(x):  # So it can be pickled
    """
    Normalize the GSP data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x / x.capacity_megawatt_power


def normalize_pv(x):  # So it can be pickled
    """
    Normalize the GSP data

    Args:
        x: Input DataArray

    Returns:
        Normalized DataArray
    """
    return x / x.capacity_watt_power


def _remove_nans(x):
    return x.fillna(0.0)


def metnet_national_datapipe(
    configuration_filename: Union[Path, str],
    use_sun: bool = True,
    use_nwp: bool = True,
    use_sat: bool = True,
    use_hrv: bool = True,
    use_pv: bool = True,
    use_gsp: bool = True,
    use_topo: bool = True,
    mode: str = "train",
    max_num_pv_systems: int = -1,
    start_time: datetime.datetime = datetime.datetime(2014, 1, 1),
    end_time: datetime.datetime = datetime.datetime(2023, 1, 1),
) -> IterDataPipe:
    """
    Make GSP national data pipe

    Currently only has GSP and NWP's in them

    Args:
        configuration_filename: the configruation filename for the pipe
        use_sun: Whether to add sun features or not
        use_pv: Whether to use PV input or not
        use_hrv: Whether to use HRV Satellite or not
        use_sat: Whether to use non-HRV Satellite or not
        use_nwp: Whether to use NWP or not
        use_topo: Whether to use topographic map or not
        mode: Either 'train', where random times are selected,
            or 'test' or 'val' where times are sequential
        max_num_pv_systems: max number of PV systems to include, <= 0 if no sampling
        start_time: Start time to select on
        end_time: End time to select from

    Returns: datapipe
    """

    # load datasets
    used_datapipes = open_and_return_datapipes(
        configuration_filename=configuration_filename,
        use_nwp=use_nwp,
        use_topo=use_topo,
        use_sat=use_sat,
        use_hrv=use_hrv,
        use_gsp=use_gsp,
        use_pv=use_pv,
    )
    configuration = used_datapipes["config"]
    # Load GSP national data
    used_datapipes["gsp"] = used_datapipes["gsp"].select_train_test_time(start_time, end_time)

    # Now get overlapping time periods
    used_datapipes = get_and_return_overlapping_time_periods_and_t0(used_datapipes)

    # And now get time slices
    used_datapipes = add_selected_time_slices_from_datapipes(used_datapipes)

    # Now do the extra processing
    gsp_history = used_datapipes["gsp"].normalize(normalize_fn=normalize_gsp)
    gsp_datapipe = used_datapipes["gsp_future"].normalize(normalize_fn=normalize_gsp)
    # Split into GSP for target, only national, and one for history
    gsp_datapipe = DropGSP(gsp_datapipe, gsps_to_keep=[0])

    if "nwp" in used_datapipes.keys():
        # take nwp time slices
        logger.debug("Take NWP time slices")
        nwp_datapipe = used_datapipes["nwp"].normalize(mean=NWP_MEAN, std=NWP_STD)

    if "sat" in used_datapipes.keys():
        logger.debug("Take Satellite time slices")
        # take sat time slices
        sat_datapipe = used_datapipes["sat"].normalize(mean=SAT_MEAN_DA, std=SAT_STD_DA)

    if "hrv" in used_datapipes.keys():
        logger.debug("Take HRV Satellite time slices")
        sat_hrv_datapipe = used_datapipes["hrv"].normalize(mean=SAT_MEAN["HRV"], std=SAT_STD["HRV"])

    if "pv" in used_datapipes.keys():
        logger.debug("Take PV Time Slices")
        # take pv time slices
        pv_datapipe = used_datapipes["pv"].normalize(normalize_fn=normalize_pv)

    if "topo" in used_datapipes.keys():
        topo_datapipe = used_datapipes["topo"].map(_remove_nans)

    # Now combine in the MetNet format
    modalities = []
    if "nwp" in used_datapipes.keys():
        modalities.append(nwp_datapipe)
    if "hrv" in used_datapipes.keys():
        modalities.append(sat_hrv_datapipe)
    if "sat" in used_datapipes.keys():
        modalities.append(sat_datapipe)
    if "topo" in used_datapipes.keys():
        modalities.append(topo_datapipe)

    gsp_datapipe, gsp_loc_datapipe = gsp_datapipe.fork(2, buffer_size=5)

    location_datapipe = LocationPicker(gsp_loc_datapipe)

    metnet_datapipe = PreProcessMetNet(
        modalities,
        location_datapipe=location_datapipe,
        center_width=500_000,
        center_height=1_000_000,
        context_height=10_000_000,
        context_width=10_000_000,
        output_width_pixels=256,
        output_height_pixels=256,
        add_sun_features=use_sun,
    )

    # metnet_datapipe = modalities[0].zip(*modalities[1:])
    gsp_datapipe = ConvertGSPToNumpy(gsp_datapipe)
    gsp_history = gsp_history.map(_remove_nans)
    gsp_history = ConvertGSPToNumpy(gsp_history, return_id=True)
    # if use_gsp and use_pv:
    #    return metnet_datapipe.zip(gsp_history, pv_datapipe, gsp_datapipe)
    # if use_gsp:
    return metnet_datapipe.zip(gsp_history, gsp_datapipe)  # Makes (Inputs, Label) tuples
    # if use_pv:
    #    return metnet_datapipe.zip(pv_datapipe, gsp_datapipe)
