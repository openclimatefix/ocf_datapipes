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
from ocf_datapipes.load import OpenConfiguration, OpenGFSForecast, OpenNWPID, OpenPVFromNetCDF
from ocf_datapipes.utils.consts import NWP_GFS_MEAN, NWP_GFS_STD, NWP_MEAN, NWP_STD

logger = logging.getLogger(__name__)
xarray.set_options(keep_attrs=True)

# should scale with batch_size #TODO
BUFFER_SIZE = -1


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
    pv_datapipe, pv_location_datapipe = (
        OpenPVFromNetCDF(pv=configuration.input_data.pv)
        .pv_fill_night_nans()
        .fork(2, buffer_size=BUFFER_SIZE)
    )

    if configuration.input_data.nwp.nwp_provider == "UKMetOffice":
        nwp_datapipe = OpenNWPID(configuration.input_data.nwp.nwp_zarr_path)
    elif configuration.input_data.nwp.nwp_provider == "GFS":
        nwp_datapipe = OpenGFSForecast(configuration.input_data.nwp.nwp_zarr_path)
    else:
        raise Exception(
            f"NWP provider {configuration.input_data.nwp.nwp_provider} "
            f'not in "UKMetOffice" or "GFS"'
        )

    logger.debug("Add t0 idx and normalize")
    pv_datapipe = pv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.pv.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
    ).normalize(normalize_fn=lambda x: x / x.capacity_watt_power)
    nwp_datapipe = nwp_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.nwp.time_resolution_minutes
        ),
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
        location_datapipe4,
    ) = pv_location_datapipe.location_picker(
        return_all_locations=return_all,
        x_dim_name=configuration.input_data.pv.x_dim_name,
        y_dim_name=configuration.input_data.pv.y_dim_name,
    ).fork(
        4, buffer_size=BUFFER_SIZE
    )

    logger.debug("Making PV space slice")
    pv_datapipe = pv_datapipe.select_id(location_datapipe=location_datapipe1, data_source_name="pv")

    # filter out nans and remove zero data, do no this for testing
    if tag != "test":
        pv_datapipe = pv_datapipe.pv_remove_zero_data(
            window=timedelta(
                minutes=configuration.input_data.pv.history_minutes
                + configuration.input_data.pv.forecast_minutes
            )
        ).remove_nans()

    # split into 3 forks
    pv_datapipe, pv_time_periods_datapipe, pv_t0_datapipe = pv_datapipe.fork(
        3, buffer_size=BUFFER_SIZE
    )

    # select square from nwp data
    if configuration.input_data.nwp.index_by_id:
        nwp_datapipe = nwp_datapipe.select_id(
            location_datapipe=location_datapipe2,
        )
    else:
        nwp_datapipe = nwp_datapipe.select_spatial_slice_pixels(
            location_datapipe=location_datapipe2,
            roi_height_pixels=configuration.input_data.nwp.nwp_image_size_pixels_height,
            roi_width_pixels=configuration.input_data.nwp.nwp_image_size_pixels_width,
            y_dim_name=configuration.input_data.nwp.y_dim_name,
            x_dim_name=configuration.input_data.nwp.x_dim_name,
        )

    nwp_datapipe, nwp_time_periods_datapipe = nwp_datapipe.fork(2, buffer_size=BUFFER_SIZE)

    #
    # get contiguous time periods
    pv_time_periods_datapipe = pv_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.pv.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
    )

    nwp_time_periods_datapipe = nwp_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.nwp.time_resolution_minutes
        ),
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
    location_datapipe4 = location_datapipe3.number_of_locations()
    pv_t0_datapipe, nwp_t0_datapipe = pv_t0_datapipe.select_t0_time(
        return_all_times=return_all, number_locations_datapipe=location_datapipe4
    ).fork(2)

    # take pv time slices
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

    # take nwp time slices
    nwp_datapipe = nwp_datapipe.convert_to_nwp_target_time(
        t0_datapipe=nwp_t0_datapipe,
        sample_period_duration=timedelta(
            minutes=configuration.input_data.nwp.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
    )

    if configuration.input_data.nwp.nwp_provider == "UKMetOffice":
        nwp_datapipe = nwp_datapipe.normalize(mean=NWP_MEAN, std=NWP_STD)
    else:
        nwp_datapipe = nwp_datapipe.normalize(mean=NWP_GFS_MEAN, std=NWP_GFS_STD)

    nwp_datapipe = nwp_datapipe.convert_nwp_to_numpy_batch().merge_numpy_examples_to_batch(
        n_examples_per_batch=configuration.process.batch_size
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

    # combined_datapipe = combined_datapipe.add_length(
    #     configuration=configuration, train_validation_test=tag
    # )

    return combined_datapipe
