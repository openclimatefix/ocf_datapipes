"""Wrapper for Power Perceiver Production Data Pipeline"""
import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray
from torchdata.datapipes.iter import IterDataPipe

import ocf_datapipes  # noqa
from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenConfiguration, OpenNWPID, OpenPVFromNetCDF
from ocf_datapipes.select import SelectOverlappingTimeSlice
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD

logger = logging.getLogger(__name__)
xarray.set_options(keep_attrs=True)


def nwp_pv_datapipe(configuration_filename: Union[Path, str]) -> IterDataPipe:
    """
    Create the Power Perceiver production pipeline using a configuration

    Args:
        configuration_filename: Name of the configuration

    Returns:
        DataPipe ready to be put in a Dataloader for production
    """
    ####################################
    #
    # main data pipe for loading a simple site level forecast
    #
    #####################################
    # load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))

    logger.debug("Opening Datasets")
    pv_datapipe, pv_location_datapipe = OpenPVFromNetCDF(
        pv_power_filename=configuration.input_data.pv.pv_files_groups[0].pv_filename,
        pv_metadata_filename=configuration.input_data.pv.pv_files_groups[0].pv_metadata_filename,
        start_datetime=configuration.input_data.pv.start_datetime,
        end_datetime=configuration.input_data.pv.end_datetime,
    ).fork(2)

    nwp_datapipe = OpenNWPID(configuration.input_data.nwp.nwp_zarr_path)

    logger.debug("Add t0 idx")
    pv_datapipe, pv_t0_datapipe = pv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
    ).fork(2)

    logger.debug("Getting locations")
    location_datapipe1, location_datapipe2 = pv_location_datapipe.location_picker().fork(2)
    logger.debug("Got locations")

    logger.debug("Making PV space slice")
    pv_datapipe, pv_time_periods_datapipe, pv_t0_datapipe, = (
        pv_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_watt_power)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        )
        .select_spatial_slice_meters(
            location_datapipe=location_datapipe1,
            roi_width_meters=configuration.input_data.pv.pv_image_size_meters_width,
            roi_height_meters=configuration.input_data.pv.pv_image_size_meters_height,
        )
        .ensure_n_pv_systems_per_example(n_pv_systems_per_example=1)
        .remove_nans()
        .fork(3)
    )

    # select id from nwp data
    nwp_datapipe, nwp_time_periods_datapipe, nwp_t0_datapipe = (
        nwp_datapipe.add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(hours=1),
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
        )
        .select_id(
            location_datapipe=location_datapipe2,
        )
        .fork(3)
    )

    # get contiguous time periods
    pv_t0_datapipe = pv_t0_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
    )
    nwp_t0_datapipe = nwp_t0_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        time_dim="init_time_utc",
    )
    # find joint overlapping timer periods
    overlapping_datapipe = SelectOverlappingTimeSlice(
        source_datapipes=[pv_t0_datapipe, nwp_t0_datapipe]
    )
    pv_time_periods, nwp_time_periods = overlapping_datapipe.fork(2)

    # select time periods
    pv_t0_datapipe = pv_time_periods_datapipe.select_time_periods(time_periods=pv_time_periods)
    nwp_t0_datapipe = nwp_time_periods_datapipe.select_time_periods(
        time_periods=nwp_time_periods, dim_name="init_time_utc"
    )
    # select t0 periods
    pv_t0_datapipe = pv_t0_datapipe.select_t0_time()
    nwp_t0_datapipe = nwp_t0_datapipe.select_t0_time(dim_name="init_time_utc")

    # take pv time slices
    pv_datapipe = (
        pv_datapipe.select_time_slice(
            t0_datapipe=pv_t0_datapipe,
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
            sample_period_duration=timedelta(minutes=5),
        )
        .convert_pv_to_numpy_batch()
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )

    # take nwp time slices
    nwp_datapipe = (
        nwp_datapipe.convert_to_nwp_target_time(
            t0_datapipe=nwp_t0_datapipe,
            sample_period_duration=timedelta(hours=1),
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        )
        .normalize(mean=NWP_MEAN, std=NWP_STD)
        .convert_nwp_to_numpy_batch()
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )

    ####################################
    #
    # Join data pipes together, and get extra details
    #
    #####################################
    logger.debug("Combine all the data sources")
    combined_datapipe = (
        MergeNumpyModalities([pv_datapipe, nwp_datapipe])
        .encode_space_time()
        .add_sun_position(modality_name="pv")
    )

    return combined_datapipe
