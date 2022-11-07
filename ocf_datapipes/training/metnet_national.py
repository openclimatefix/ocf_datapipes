"""Create the training/validation datapipe for training the national MetNet/-2 Model"""
import datetime
import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray
from torchdata.datapipes.iter import IterDataPipe

from ocf_datapipes.config.model import Configuration
from ocf_datapipes.convert import ConvertGSPToNumpy
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
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD, SAT_MEAN, SAT_MEAN_DA, SAT_STD, SAT_STD_DA

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


def _remove_nans(x):
    return x.fillna(0.0)


def metnet_national_datapipe(
    configuration_filename: Union[Path, str],
    use_sun: bool = True,
    use_nwp: bool = True,
    use_sat: bool = True,
    use_hrv: bool = True,
    use_pv: bool = True,
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
        mode: Either 'train', where random times are selected, or 'test' or 'val' where times are sequential
        max_num_pv_systems: max number of PV systems to include, <= 0 if no sampling
        start_time: Start time to select on
        end_time: End time to select from

    Returns: datapipe
    """

    # load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))

    # Check which modalities to use
    if use_nwp:
        use_nwp = True if configuration.input_data.nwp.nwp_zarr_path != "" else False
    if use_pv:
        use_pv = True if configuration.input_data.pv.pv_files_groups[0].pv_filename != "" else False
    if use_sat:
        use_sat = True if configuration.input_data.satellite.satellite_zarr_path != "" else False
    if use_hrv:
        use_hrv = (
            True if configuration.input_data.hrvsatellite.hrvsatellite_zarr_path != "" else False
        )
    if use_topo:
        use_topo = (
            True if configuration.input_data.topographic.topographic_filename != "" else False
        )
    print(f"NWP: {use_nwp} Sat: {use_sat}, HRV: {use_hrv} PV: {use_pv} Sun: {use_sun}")
    # Load GSP national data
    logger.debug("Opening GSP Data")
    gsp_datapipe = OpenGSP(
        gsp_pv_power_zarr_path=configuration.input_data.gsp.gsp_zarr_path
    ).select_train_test_time(start_time, end_time)

    gsp_datapipe = DropGSP(gsp_datapipe, gsps_to_keep=[0])

    logger.debug("Add t0 idx and normalize")

    gsp_datapipe, gsp_time_periods_datapipe, gsp_t0_datapipe = (
        gsp_datapipe.normalize(normalize_fn=normalize_gsp)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        )
        .fork(3)
    )
    # get time periods
    # get contiguous time periods
    logger.debug("Getting contiguous time periods")
    gsp_time_periods_datapipe = gsp_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
    )

    secondary_datapipes = []

    # Load NWP data
    if use_nwp:
        logger.debug("Opening NWP Data")
        nwp_datapipe, nwp_time_periods_datapipe = (
            OpenNWP(configuration.input_data.nwp.nwp_zarr_path)
            .select_channels(configuration.input_data.nwp.nwp_channels)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(hours=1),
                history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            )
            .fork(2)
        )

        nwp_time_periods_datapipe = nwp_time_periods_datapipe.get_contiguous_time_periods(
            sample_period_duration=timedelta(hours=3),  # Init times are 3 hours apart
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
            time_dim="init_time_utc",
        )
        secondary_datapipes.append(nwp_time_periods_datapipe)

    if use_sat:
        logger.debug("Opening Satellite Data")
        sat_datapipe, sat_time_periods_datapipe = (
            OpenSatellite(configuration.input_data.satellite.satellite_zarr_path)
            .select_channels(configuration.input_data.satellite.satellite_channels)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(minutes=5),
                history_duration=timedelta(
                    minutes=configuration.input_data.satellite.history_minutes
                ),
            )
            .fork(2)
        )

        sat_time_periods_datapipe = sat_time_periods_datapipe.get_contiguous_time_periods(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
            forecast_duration=timedelta(minutes=1),
        )
        secondary_datapipes.append(sat_time_periods_datapipe)

    if use_hrv:
        logger.debug("Opening HRV Satellite Data")
        sat_hrv_datapipe, sat_hrv_time_periods_datapipe = (
            OpenSatellite(configuration.input_data.hrvsatellite.hrvsatellite_zarr_path)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(minutes=5),
                history_duration=timedelta(
                    minutes=configuration.input_data.hrvsatellite.history_minutes
                ),
            )
            .fork(2)
        )

        sat_hrv_time_periods_datapipe = sat_hrv_time_periods_datapipe.get_contiguous_time_periods(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(
                minutes=configuration.input_data.hrvsatellite.history_minutes
            ),
            forecast_duration=timedelta(minutes=1),
        )
        secondary_datapipes.append(sat_hrv_time_periods_datapipe)

    if use_pv:
        logger.debug("Opening PV")
        pv_datapipe, pv_time_periods_datapipe = (
            OpenPVFromNetCDF(pv=configuration.input_data.pv)
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(minutes=5),
                history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
            )
            .fork(2)
        )

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
    t0_datapipes = gsp_t0_datapipe.select_t0_time(
        return_all_times=False if mode == "train" else True
    ).fork(1 + len(secondary_datapipes))

    # take pv time slices
    logger.debug("Take GSP time slices")
    gsp_datapipe = gsp_datapipe.select_time_slice(
        t0_datapipe=t0_datapipes[0],
        history_duration=timedelta(minutes=0),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
        sample_period_duration=timedelta(minutes=30),
    )

    if use_nwp:
        # take nwp time slices
        logger.debug("Take NWP time slices")
        nwp_datapipe = nwp_datapipe.convert_to_nwp_target_time(
            t0_datapipe=t0_datapipes[1],
            sample_period_duration=timedelta(hours=1),
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        ).normalize(mean=NWP_MEAN, std=NWP_STD)

    if use_sat:
        logger.debug("Take Satellite time slices")
        # take sat time slices
        sat_datapipe = sat_datapipe.select_time_slice(
            t0_datapipe=t0_datapipes[sum([use_nwp, use_sat])],
            history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
            forecast_duration=timedelta(minutes=0),
            sample_period_duration=timedelta(minutes=5),
        ).normalize(mean=SAT_MEAN_DA, std=SAT_STD_DA)

    if use_hrv:
        logger.debug("Take HRV Satellite time slices")
        sat_hrv_datapipe = sat_hrv_datapipe.select_time_slice(
            t0_datapipe=t0_datapipes[sum([use_nwp, use_sat, use_hrv])],
            history_duration=timedelta(
                minutes=configuration.input_data.hrvsatellite.history_minutes
            ),
            forecast_duration=timedelta(minutes=0),
            sample_period_duration=timedelta(minutes=5),
        ).normalize(mean=SAT_MEAN["HRV"], std=SAT_STD["HRV"])

    if use_pv:
        logger.debug("Take PV Time Slices")
        # take pv time slices
        if use_sat:
            sat_datapipe, image_datapipe = sat_datapipe.fork(2)
        elif use_hrv:
            sat_hrv_datapipe, image_datapipe = sat_hrv_datapipe.fork(2)
        elif use_nwp:
            nwp_datapipe, image_datapipe = nwp_datapipe.fork(2)

        pv_datapipe = pv_datapipe.select_time_slice(
            t0_datapipe=t0_datapipes[sum([use_nwp, use_sat, use_hrv, use_pv])],
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
            forecast_duration=timedelta(minutes=0),
            sample_period_duration=timedelta(minutes=5),
        ).create_pv_image(image_datapipe, normalize=True, max_num_pv_systems=max_num_pv_systems)

    if use_topo:
        topo_datapipe = OpenTopography(
            configuration.input_data.topographic.topographic_filename
        ).map(_remove_nans)

    # Now combine in the MetNet format
    modalities = []
    if use_nwp:
        modalities.append(nwp_datapipe)
    if use_hrv:
        modalities.append(sat_hrv_datapipe)
    if use_sat:
        modalities.append(sat_datapipe)
    if use_pv:
        modalities.append(pv_datapipe)
    if use_topo:
        modalities.append(topo_datapipe)

    gsp_datapipe, gsp_loc_datapipe = gsp_datapipe.fork(2)

    location_datapipe = LocationPicker(gsp_loc_datapipe)

    combined_datapipe = PreProcessMetNet(
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

    gsp_datapipe = ConvertGSPToNumpy(gsp_datapipe)
    return combined_datapipe.zip(gsp_datapipe)  # Makes (Inputs, Label) tuples
