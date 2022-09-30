"""Data pipeline for loading GSP national"""
import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray
from torchdata.datapipes.iter import IterDataPipe

import ocf_datapipes  # noqa
from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenConfiguration, OpenGSPNational, OpenNWPID
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD

logger = logging.getLogger(__name__)
xarray.set_options(keep_attrs=True)

# should scale with batch_size #TODO
BUFFER_SIZE = 100


def gsp_national_datapipe(configuration_filename: Union[Path, str]) -> IterDataPipe:
    """
    Make GSP national data pipe

    Currently only has GSP and NWP's in them

    Args:
        configuration_filename: the configruation filename for the pipe

    Returns: datapipe
    """

    # load configuration
    config_datapipe = OpenConfiguration(configuration_filename)
    configuration: Configuration = next(iter(config_datapipe))

    # Load GSP national data
    logger.debug("Load GSP data")
    gsp_datapipe = OpenGSPNational(
        gsp_pv_power_zarr_path=configuration.input_data.gsp.gsp_zarr_path
    )

    # Load NWP data
    logger.debug("Load NWP data")
    nwp_datapipe = OpenNWPID(configuration.input_data.nwp.nwp_zarr_path)

    logger.debug("Add t0 idx and normalize")
    gsp_datapipe, gsp_time_periods_datapipe, gsp_t0_datapipe = (
        gsp_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_megawatt_power)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(minutes=30),
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        )
        .fork(3)
    )

    nwp_datapipe, nwp_time_periods_datapipe = nwp_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(hours=1),
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
    ).fork(2)

    # get time periods
    # get contiguous time periods
    logger.debug("Getting contiguous time periods")
    gsp_time_periods_datapipe = gsp_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
    )
    nwp_time_periods_datapipe = nwp_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=60),
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        time_dim="init_time_utc",
    )
    # find joint overlapping timer periods
    logger.debug("Getting joint time periods")
    overlapping_datapipe = gsp_time_periods_datapipe.select_overlapping_time_slice(
        secondary_datapipes=[nwp_time_periods_datapipe],
    )
    gsp_time_periods, nwp_time_periods = overlapping_datapipe.fork(2, buffer_size=BUFFER_SIZE)
    # select time periods
    gsp_t0_datapipe = gsp_t0_datapipe.select_time_periods(time_periods=gsp_time_periods)

    # select t0 periods
    logger.debug("Select t0 joint")
    gsp_t0_datapipe, nwp_t0_datapipe = gsp_t0_datapipe.select_t0_time().fork(2)

    # take pv time slices
    logger.debug("Take GSP time slices")
    gsp_datapipe = (
        gsp_datapipe.select_time_slice(
            t0_datapipe=gsp_t0_datapipe,
            history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
            sample_period_duration=timedelta(minutes=30),
        )
        .convert_gsp_to_numpy_batch()
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )

    # take nwp time slices
    logger.debug("Take NWP time slices")
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
    # Join data pipes together, and get extra details
    #####################################
    logger.debug("Combine all the data sources")
    combined_datapipe = (
        MergeNumpyModalities([gsp_datapipe, nwp_datapipe])
        # .encode_space_time()
        # .add_sun_position(modality_name="gsp")
    )

    return combined_datapipe
