"""Data pipeline for loading PV, NWP and GSP"""
import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray
from torchdata.datapipes.iter import IterDataPipe

import ocf_datapipes  # noqa
from ocf_datapipes.batch import MergeNumpyModalities
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenConfiguration, OpenNWP, OpenPVFromNetCDF, OpenSatellite
from ocf_datapipes.utils.consts import NWP_MEAN, NWP_STD

logger = logging.getLogger(__name__)
xarray.set_options(keep_attrs=True)

# should scale with batch_size #TODO
BUFFER_SIZE = 100


def pv_nwp_satellite_data_pipeline(configuration: Union[Path, str, Configuration]) -> IterDataPipe:
    """
    Make data pipe with PV, NWP and Satellite

    The location can be made either from GSP or PV

    Args:
        configuration: the configuration filename for the pipe, can also be the actual configuration

    Returns: datapipe
    """

    # load configuration
    if type(configuration) != Configuration:
        config_datapipe = OpenConfiguration(configuration)
        configuration: Configuration = next(iter(config_datapipe))

    # Load NWP data
    logger.debug("Load NWP data")
    nwp_datapipe = OpenNWP(configuration.input_data.nwp.nwp_zarr_path)

    # Load PV data
    logger.debug("Load PV data")
    pv_datapipe, pv_location_datapipe = (
        OpenPVFromNetCDF(pv=configuration.input_data.pv).pv_fill_night_nans().fork(2)
    )

    logger.debug("Load Satellite data")
    satellite_datapipe = OpenSatellite(
        zarr_path=configuration.input_data.satellite.satellite_zarr_path
    )

    # add satellite data
    logger.debug("Add t0 idx")
    pv_datapipe = pv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.pv.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
    )

    logger.debug("Add t0 idx")
    nwp_datapipe = nwp_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(hours=1),
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
    )

    logger.debug("Add t0 idx")
    satellite_datapipe = satellite_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.satellite.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
    )

    # Pick locations
    location_datapipes = pv_location_datapipe.location_picker().fork(4, buffer_size=BUFFER_SIZE)

    # take PV space slice
    pv_datapipe, pv_time_periods_datapipe, pv_t0_datapipe = pv_datapipe.select_spatial_slice_meters(
        location_datapipe=location_datapipes[1],
        roi_height_meters=configuration.input_data.pv.pv_image_size_meters_height,
        roi_width_meters=configuration.input_data.pv.pv_image_size_meters_width,
        y_dim_name="y_osgb",
        x_dim_name="x_osgb",
    ).fork(3)
    # take NWP space slice
    nwp_datapipe, nwp_time_periods_datapipe = nwp_datapipe.select_spatial_slice_pixels(
        location_datapipe=location_datapipes[2],
        roi_height_pixels=configuration.input_data.nwp.nwp_image_size_pixels_height,
        roi_width_pixels=configuration.input_data.nwp.nwp_image_size_pixels_width,
        y_dim_name="y_osgb",
        x_dim_name="x_osgb",
    ).fork(2)

    # take Satellite space slice
    (
        satellite_datapipe,
        satellite_time_periods_datapipe,
    ) = satellite_datapipe.select_spatial_slice_pixels(
        location_datapipe=location_datapipes[3],
        roi_height_pixels=configuration.input_data.satellite.satellite_image_size_pixels_height,
        roi_width_pixels=configuration.input_data.satellite.satellite_image_size_pixels_width,
        y_dim_name="y_geostationary",
        x_dim_name="x_geostationary",
    ).fork(
        2
    )

    # get time periods
    # get contiguous time periods
    logger.debug("Getting contiguous time periods")
    nwp_time_periods_datapipe = nwp_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(hours=3),  # Init times are 3 hours apart
        history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        time_dim="init_time_utc",
    )
    pv_time_periods_datapipe = pv_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(minutes=5),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
    )
    satellite_time_periods_datapipe = satellite_time_periods_datapipe.get_contiguous_time_periods(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.satellite.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.satellite.forecast_minutes),
    )

    # find joint overlapping time periods
    logger.debug("Getting joint time periods")
    overlapping_datapipe = pv_time_periods_datapipe.select_overlapping_time_slice(
        secondary_datapipes=[
            nwp_time_periods_datapipe,
            satellite_time_periods_datapipe,
        ],
    )
    gsp_time_periods, nwp_time_periods, pv_time_periods = overlapping_datapipe.fork(
        3, buffer_size=BUFFER_SIZE
    )
    # select time periods
    pv_t0_datapipe = pv_t0_datapipe.select_time_periods(time_periods=gsp_time_periods)

    # select t0 periods
    logger.debug("Select t0 joint")
    (
        nwp_t0_datapipe,
        pv_t0_datapipe,
        satellite_t0_datapipe,
    ) = pv_t0_datapipe.select_t0_time().fork(3)

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

    # take pv time slices
    logger.debug("Take PV time slices")
    pv_datapipe = (
        pv_datapipe.select_time_slice(
            t0_datapipe=pv_t0_datapipe,
            sample_period_duration=timedelta(minutes=5),
            history_duration=timedelta(minutes=configuration.input_data.nwp.history_minutes),
            forecast_duration=timedelta(minutes=configuration.input_data.nwp.forecast_minutes),
        )
        .convert_pv_to_numpy_batch()
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )

    # take pv time slices
    logger.debug("Take Satellite time slices")
    satellite_datapipe = (
        satellite_datapipe.select_time_slice(
            t0_datapipe=satellite_t0_datapipe,
            sample_period_duration=timedelta(
                minutes=configuration.input_data.satellite.time_resolution_minutes
            ),
            history_duration=timedelta(minutes=configuration.input_data.satellite.history_minutes),
            forecast_duration=timedelta(
                minutes=configuration.input_data.satellite.forecast_minutes
            ),
        )
        .convert_satellite_to_numpy_batch()
        .merge_numpy_examples_to_batch(n_examples_per_batch=configuration.process.batch_size)
    )

    ####################################
    # Join data pipes together, and get extra details
    #####################################
    logger.debug("Combine all the data sources")
    combined_datapipe = (
        MergeNumpyModalities([nwp_datapipe, pv_datapipe, satellite_datapipe])
        # .encode_space_time()
        # .add_sun_position(modality_name="gsp")
    )

    return combined_datapipe
