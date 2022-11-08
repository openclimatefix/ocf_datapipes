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
from ocf_datapipes.load import OpenConfiguration, OpenPVFromNetCDF

logger = logging.getLogger(__name__)
xarray.set_options(keep_attrs=True)

# default is set to 1000
BUFFERSIZE = -1


def simple_pv_datapipe(
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
    pv_datapipe, pv_location_datapipe = (
        OpenPVFromNetCDF(pv=configuration.input_data.pv)
        .pv_fill_night_nans()
        .fork(2, buffer_size=BUFFERSIZE)
    )

    logger.debug("Add t0 idx")
    pv_datapipe = pv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.pv.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
    )

    logger.debug("Getting locations")
    # might have to fork this if we add NWPs
    location_datapipe1 = pv_location_datapipe.location_picker()
    logger.debug("Got locations")

    logger.debug("Making PV space slice")
    pv_datapipe, pv_t0_datapipe, pv_time_periods_datapipe = (
        pv_datapipe.normalize(normalize_fn=lambda x: x / x.capacity_watt_power)
        .add_t0_idx_and_sample_period_duration(
            sample_period_duration=timedelta(
                minutes=configuration.input_data.pv.time_resolution_minutes
            ),
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        )
        .select_spatial_slice_meters(
            location_datapipe=location_datapipe1,
            roi_width_meters=configuration.input_data.pv.pv_image_size_meters_width,
            roi_height_meters=configuration.input_data.pv.pv_image_size_meters_height,
        )
        .ensure_n_pv_systems_per_example(n_pv_systems_per_example=1)
        .remove_nans()
        .fork(3, buffer_size=BUFFERSIZE)
    )

    # get contiguous time periods
    pv_time_periods_datapipe = pv_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.pv.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
    )

    # select time periods
    pv_t0_datapipe = pv_t0_datapipe.select_time_periods(time_periods=pv_time_periods_datapipe)
    pv_t0_datapipe = pv_t0_datapipe.select_t0_time()
    # take time slices
    pv_datapipe = (
        pv_datapipe.select_time_slice(
            t0_datapipe=pv_t0_datapipe,
            history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
            sample_period_duration=timedelta(
                minutes=configuration.input_data.pv.time_resolution_minutes
            ),
        )
        .convert_pv_to_numpy_batch()
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )

    ####################################
    #
    # Join data pipes together, and get extra details
    # TODO add simple NWP data
    #
    #####################################
    logger.debug("Combine all the data sources")
    combined_datapipe = (
        MergeNumpyModalities([pv_datapipe])
        # .align_gsp_to_5_min(batch_key_for_5_min_datetimes=BatchKey.pv_time_utc)
        .encode_space_time().add_sun_position(modality_name="pv")
    )

    combined_datapipe = combined_datapipe.add_length(
        configuration=configuration, train_validation_test=tag
    )

    return combined_datapipe
