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

from .common import open_and_return_datapipes

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
    )
    configuration = used_datapipes["config"]
    # Load GSP national data
    logger.debug("Opening GSP Data")
    gsp_datapipe, gsp_history = (
        used_datapipes["gsp"].select_train_test_time(start_time, end_time).fork(2)
    )

    # Split into GSP for target, only national, and one for history
    gsp_datapipe = DropGSP(gsp_datapipe, gsps_to_keep=[0])

    logger.debug("Add t0 idx and normalize")

    gsp_datapipe, gsp_time_periods_datapipe, gsp_t0_datapipe = (
        gsp_datapipe.normalize(normalize_fn=normalize_gsp)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=0),
        )
        .fork(3)
    )

    gsp_history, gsp_history_time_periods_datapipe = (
        gsp_history.normalize(normalize_fn=normalize_gsp)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        )
        .fork(2)
    )
    # get time periods
    # get contiguous time periods
    logger.debug("Getting contiguous time periods")
    gsp_time_periods_datapipe = gsp_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(minutes=0),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
    )
    gsp_history_time_periods_datapipe = (
        gsp_history_time_periods_datapipe.get_contiguous_time_periods(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
            forecast_duration=timedelta(minutes=0),
        )
    )

    secondary_datapipes = [gsp_history_time_periods_datapipe]

    # Load NWP data
    if "nwp" in used_datapipes.keys():
        nwp_datapipe, nwp_time_periods_datapipe = used_datapipes["nwp"].fork(2)

        nwp_time_periods_datapipe = nwp_time_periods_datapipe.get_contiguous_time_periods(
            sample_period_duration=timedelta(hours=3),  # Init times are 3 hours apart
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
            time_dim="init_time_utc",
        )
        secondary_datapipes.append(nwp_time_periods_datapipe)

    if "sat" in used_datapipes.keys():
        sat_datapipe, sat_time_periods_datapipe = used_datapipes["sat"].fork(2)

        sat_time_periods_datapipe = sat_time_periods_datapipe.get_contiguous_time_periods(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
            forecast_duration=timedelta(minutes=1),
        )
        secondary_datapipes.append(sat_time_periods_datapipe)

    if "hrv" in used_datapipes.keys():
        sat_hrv_datapipe, sat_hrv_time_periods_datapipe = used_datapipes["hrv"].fork(2)

        sat_hrv_time_periods_datapipe = sat_hrv_time_periods_datapipe.get_contiguous_time_periods(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(
                minutes=configuration.input_data.hrvsatellite.history_minutes
            ),
            forecast_duration=timedelta(minutes=1),
        )
        secondary_datapipes.append(sat_hrv_time_periods_datapipe)

    if "pv" in used_datapipes.keys():
        pv_datapipe, pv_time_periods_datapipe = used_datapipes["pv"].fork(2)

        pv_time_periods_datapipe = pv_time_periods_datapipe.get_contiguous_time_periods(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
            forecast_duration=timedelta(minutes=1),
        )
        secondary_datapipes.append(pv_time_periods_datapipe)

    # find joint overlapping timer periods
    logger.debug("Getting joint time periods")
    overlapping_datapipe = gsp_time_periods_datapipe.select_overlapping_time_slice(
        secondary_datapipes=secondary_datapipes,
    )

    # select time periods
    gsp_t0_datapipe = gsp_t0_datapipe.select_time_periods(time_periods=overlapping_datapipe)

    # select t0 periods
    logger.debug("Select t0 joint")
    num_t0_datapipes = (
        1 + len(secondary_datapipes) if mode == "train" else 2 + len(secondary_datapipes)
    )
    t0_datapipes = gsp_t0_datapipe.select_t0_time(
        return_all_times=False  # if mode == "train" else True
    ).fork(num_t0_datapipes)

    # take pv time slices
    logger.debug("Take GSP time slices")
    gsp_datapipe = gsp_datapipe.select_time_slice(
        t0_datapipe=t0_datapipes[0],
        history_duration=timedelta(minutes=0),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
        sample_period_duration=timedelta(minutes=30),
    )

    if "nwp" in used_datapipes.keys():
        # take nwp time slices
        logger.debug("Take NWP time slices")
        nwp_datapipe = nwp_datapipe.convert_to_nwp_target_time(
            t0_datapipe=t0_datapipes[1],
            sample_period_duration=timedelta(hours=1),
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        ).normalize(mean=NWP_MEAN, std=NWP_STD)

    if "sat" in used_datapipes.keys():
        logger.debug("Take Satellite time slices")
        # take sat time slices
        sat_datapipe = sat_datapipe.select_time_slice(
            t0_datapipe=t0_datapipes[sum([use_nwp, use_sat])],
            history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
            forecast_duration=timedelta(minutes=0),
            sample_period_duration=timedelta(minutes=5),
        ).normalize(mean=SAT_MEAN_DA, std=SAT_STD_DA)

    if "hrv" in used_datapipes.keys():
        logger.debug("Take HRV Satellite time slices")
        sat_hrv_datapipe = sat_hrv_datapipe.select_time_slice(
            t0_datapipe=t0_datapipes[sum([use_nwp, use_sat, use_hrv])],
            history_duration=timedelta(
                minutes=configuration.input_data.hrvsatellite.history_minutes
            ),
            forecast_duration=timedelta(minutes=0),
            sample_period_duration=timedelta(minutes=5),
        ).normalize(mean=SAT_MEAN["HRV"], std=SAT_STD["HRV"])

    if "pv" in used_datapipes.keys():
        logger.debug("Take PV Time Slices")
        # take pv time slices
        pv_datapipe = pv_datapipe.normalize(normalize_fn=normalize_pv)
        pv_datapipe = pv_datapipe.select_time_slice(
            t0_datapipe=t0_datapipes[sum([use_nwp, use_sat, use_hrv, use_pv])],
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
            forecast_duration=timedelta(minutes=0),
            sample_period_duration=timedelta(minutes=5),
        )
    if "gsp" in used_datapipes.keys():
        gsp_history = gsp_history.select_time_slice(
            t0_datapipe=t0_datapipes[sum([use_nwp, use_sat, use_hrv, use_pv, use_gsp])],
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
            forecast_duration=timedelta(minutes=0),
            sample_period_duration=timedelta(minutes=30),
        )

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

    gsp_datapipe, gsp_loc_datapipe = gsp_datapipe.fork(2)

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
    if "pv" in used_datapipes.keys():
        pv_datapipe = (
            pv_datapipe.ensure_n_pv_systems_per_example(n_pv_systems_per_example=max_num_pv_systems)
            .map(_remove_nans)
            .convert_pv_to_numpy(return_pv_system_row=True)
        )
    gsp_datapipe = ConvertGSPToNumpy(gsp_datapipe)
    gsp_history = ConvertGSPToNumpy(gsp_history, return_id=True)
    if mode == "train":
        if use_gsp and use_pv:
            return metnet_datapipe.zip(gsp_history, pv_datapipe, gsp_datapipe)
        if use_gsp:
            return metnet_datapipe.zip(gsp_history, gsp_datapipe)  # Makes (Inputs, Label) tuples
        if use_pv:
            return metnet_datapipe.zip(pv_datapipe, gsp_datapipe)
    else:
        start_time_datapipe = t0_datapipes[len(t0_datapipes) - 1]  # The one extra one
        return combined_datapipe.zip(gsp_history, gsp_datapipe, start_time_datapipe)
