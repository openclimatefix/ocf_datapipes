"""Wrapper for Power Perceiver Production Data Pipeline"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray
from torch.utils.data.datapipes.datapipe import IterDataPipe

import ocf_datapipes  # noqa
from ocf_datapipes.batch import MergeNumpyModalities, MergeNWPNumpyModalities
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenNWP, OpenPVFromNetCDF
from ocf_datapipes.training.common import normalize_pv
from ocf_datapipes.utils.consts import NWP_MEANS, NWP_STDS

logger = logging.getLogger(__name__)
xarray.set_options(keep_attrs=True)

BUFFER_SIZE = 5


def pv_nwp_datapipe(
    configuration_filename: Union[Path, str],
) -> IterDataPipe:
    """Create datapipe which yields PV and NWP data

    This could be used to predict PV data from NWP inputs

    Args:
        configuration_filename: Name of the configuration

    Returns:
        DataPipe ready to be put in a Dataloader for production
    """

    # Load configuration
    configuration: Configuration = load_yaml_configuration(configuration_filename)

    # Load and normalize PV data, and fill night-time NaNs
    pv_datapipe = (
        OpenPVFromNetCDF(pv=configuration.input_data.pv)
        .pv_fill_night_nans()
        .normalize(normalize_fn=normalize_pv)
    )

    # Load and normalize NWP data - There may be multiple NWP sources
    nwp_datapipe_dict = {}
    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        nwp_datapipe_dict[nwp_source] = OpenNWP(
            nwp_conf.nwp_zarr_path, provider=nwp_conf.nwp_provider
        ).normalize(mean=NWP_MEANS[nwp_conf.nwp_provider], std=NWP_STDS[nwp_conf.nwp_provider])

    # Add t0 idx to PV
    pv_datapipe = pv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.pv.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
    )

    # Add t0 idx to NWP and fork for time periods overlap
    nwp_time_periods_datapipes = {}
    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        nwp_datapipe_dict[nwp_source], nwp_time_periods_datapipes[nwp_source] = (
            nwp_datapipe_dict[nwp_source]
            .add_t0_idx_and_sample_period_duration(
                sample_period_duration=timedelta(hours=3),
                history_duration=timedelta(minutes=nwp_conf.history_minutes),
            )
            .fork(2, buffer_size=BUFFER_SIZE)
        )

    # Fork datapipe for raw data, time slicing, and location slicing uses
    (
        pv_datapipe,
        pv_time_periods_datapipe,
        pv_valid_times_datapipe,
        pv_location_datapipe,
    ) = pv_datapipe.fork(4, buffer_size=BUFFER_SIZE)

    # Get locations - Construct datapipe yielding the locations to be used for each sample
    location_datapipe = pv_location_datapipe.pick_locations()

    # Take NWP spatial slice
    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        loc_dp, location_datapipe = location_datapipe.fork(2, buffer_size=BUFFER_SIZE)

        nwp_datapipe_dict[nwp_source] = nwp_datapipe_dict[nwp_source].select_spatial_slice_pixels(
            loc_dp,
            roi_height_pixels=nwp_conf.nwp_image_size_pixels_height,
            roi_width_pixels=nwp_conf.nwp_image_size_pixels_width,
        )

    # Slice PV systems
    pv_datapipe = pv_datapipe.select_id(location_datapipe=location_datapipe, data_source_name="pv")

    # Get contiguous time periods for each data source
    pv_time_periods_datapipe = pv_time_periods_datapipe.find_contiguous_t0_time_periods(
        sample_period_duration=timedelta(
            minutes=configuration.input_data.pv.time_resolution_minutes
        ),
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
    )

    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        nwp_time_periods_datapipes[nwp_source] = nwp_time_periods_datapipes[
            nwp_source
        ].find_contiguous_t0_time_periods(
            sample_period_duration=timedelta(hours=3),
            history_duration=timedelta(minutes=nwp_conf.history_minutes),
            forecast_duration=timedelta(minutes=nwp_conf.forecast_minutes),
            time_dim="init_time_utc",
        )

    # Find joint overlapping time periods
    overlapping_datapipe = pv_time_periods_datapipe.filter_to_overlapping_time_periods(
        secondary_datapipes=[*nwp_time_periods_datapipes.values()],
    )
    pv_time_periods, nwp_time_periods = overlapping_datapipe.fork(2, buffer_size=BUFFER_SIZE)

    # Filter PVs to times valid for all data sources
    pv_valid_times_datapipe = pv_valid_times_datapipe.filter_time_periods(
        time_periods=overlapping_datapipe
    )

    # Select t0 times
    t0_datapipe = pv_valid_times_datapipe.pick_t0_times()

    # Take NWP time slices and convert to NumpyBatch
    nwp_numpy_modalities = dict()

    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        nwp_t0_datapipe, t0_datapipe = t0_datapipe.fork(2)

        nwp_numpy_modalities[nwp_source] = (
            nwp_datapipe_dict[nwp_source]
            .select_time_slice_nwp(
                t0_datapipe=nwp_t0_datapipe,
                sample_period_duration=timedelta(hours=3),
                history_duration=timedelta(minutes=nwp_conf.history_minutes),
                forecast_duration=timedelta(minutes=nwp_conf.forecast_minutes),
            )
            .convert_nwp_to_numpy_batch()
        )

    # Combine the NWPs into NumpyBatch
    nwps_numpy_datapipe = MergeNWPNumpyModalities(nwp_numpy_modalities)

    # Take PV time slices and convert to NumpyBatch
    pv_numpy_datapipe = pv_datapipe.select_time_slice(
        t0_datapipe=t0_datapipe,
        history_duration=timedelta(minutes=configuration.input_data.pv.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.pv.forecast_minutes),
        sample_period_duration=timedelta(
            minutes=configuration.input_data.pv.time_resolution_minutes
        ),
    ).convert_pv_to_numpy_batch()

    # Join data pipes together, and add solar coords
    combined_datapipe = MergeNumpyModalities([pv_numpy_datapipe, nwps_numpy_datapipe])

    combined_datapipe = combined_datapipe.add_sun_position(modality_name="pv")

    # Batch the data with batch size 4
    combined_datapipe = combined_datapipe.batch(4).merge_numpy_batch()

    return combined_datapipe
