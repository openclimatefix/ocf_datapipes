"""Wrapper for Power Perceiver Production Data Pipeline"""
import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import xarray
from torchdata.datapipes.iter import IterDataPipe

import ocf_datapipes  # noqa
from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenConfiguration, OpenNWPID, OpenPVFromNetCDF
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD

logger = logging.getLogger(__name__)
xarray.set_options(keep_attrs=True)

# should scale with batch_size #TODO
BUFFER_SIZE = 100


def nwp_pv_datapipe(
    configuration_filename: Union[Path, str], tag: Optional[str] = "train"
) -> IterDataPipe:
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
    pv_datapipe, pv_location_datapipe = OpenPVFromNetCDF(pv=configuration.input_data.pv).fork(
        2, buffer_size=BUFFER_SIZE
    )

    nwp_datapipe = OpenNWPID(configuration.input_data.nwp.nwp_zarr_path)

    logger.debug("Add t0 idx and normalize")
    pv_datapipe = pv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
    ).normalize(normalize_fn=lambda x: x / x.capacity_watt_power)
    nwp_datapipe = nwp_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(hours=1),
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
    )

    if tag == "test":
        return_all = True
    else:
        return_all = False
    logger.debug("Getting locations")
    (
        location_datapipe1,
        location_datapipe2,
        location_datapipe3,
    ) = pv_location_datapipe.location_picker(return_all_locations=return_all).fork(
        3, buffer_size=BUFFER_SIZE
    )
    logger.debug("Got locations")

    logger.debug("Making PV space slice")
    pv_datapipe, pv_time_periods_datapipe, pv_t0_datapipe, = (
        pv_datapipe.select_id(location_datapipe=location_datapipe1, data_source_name="pv")
        .pv_remove_zero_data(
            window=timedelta(
                minutes=configuration.input_data.pv.history_minutes
                + configuration.input_data.pv.forecast_minutes
            )
        )
        .remove_nans()
        .fork(3, buffer_size=BUFFER_SIZE)
    )

    # select id from nwp data
    nwp_datapipe, nwp_time_periods_datapipe = nwp_datapipe.select_id(
        location_datapipe=location_datapipe2, data_source_name="nwp"
    ).fork(2, buffer_size=BUFFER_SIZE)

    # get contiguous time periods
    pv_time_periods_datapipe = pv_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
    )
    nwp_time_periods_datapipe = nwp_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        time_dim="init_time_utc",
    )
    # find joint overlapping timer periods
    overlapping_datapipe = pv_time_periods_datapipe.select_overlapping_time_slice(
        secondary_datapipes=[nwp_time_periods_datapipe],
        location_datapipe=location_datapipe3,
    )

    # select time periods
    pv_t0_datapipe = pv_t0_datapipe.select_time_periods(time_periods=overlapping_datapipe)

    # select t0 periods
    pv_t0_datapipe, nwp_t0_datapipe = pv_t0_datapipe.select_t0_time(
        return_all_times=return_all
    ).fork(2)

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

    combined_datapipe = combined_datapipe.add_length(
        configuration=configuration, train_validation_test=tag
    )

    return combined_datapipe
