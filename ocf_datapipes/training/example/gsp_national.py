"""Data pipeline for loading GSP national"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray as xr
from torch.utils.data.datapipes.datapipe import IterDataPipe

import ocf_datapipes  # noqa
from ocf_datapipes.batch import MergeNumpyModalities, MergeNWPNumpyModalities
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenGSPNational, OpenNWP
from ocf_datapipes.training.common import normalize_gsp
from ocf_datapipes.utils.consts import NWP_MEANS, NWP_STDS

logger = logging.getLogger(__name__)
xr.set_options(keep_attrs=True)

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

    # Load configuration
    configuration: Configuration = load_yaml_configuration(configuration_filename)

    # Load and normalize GSP national data
    gsp_datapipe = OpenGSPNational(
        gsp_pv_power_zarr_path=configuration.input_data.gsp.gsp_zarr_path
    ).normalize(normalize_fn=normalize_gsp)

    # Load and normalize NWP data - There may be multiple NWP sources
    nwp_datapipe_dict = {}
    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
        nwp_datapipe_dict[nwp_source] = OpenNWP(
            nwp_conf.nwp_zarr_path, provider=nwp_conf.nwp_provider
        ).normalize(mean=NWP_MEANS[nwp_conf.nwp_provider], std=NWP_STDS[nwp_conf.nwp_provider])

    # Add t0 idx to GSP
    gsp_datapipe = gsp_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
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
            .fork(2)
        )

    # Fork the GSP data for different uses
    gsp_datapipe, gsp_time_periods_datapipe, gsp_valid_times_datapipe = gsp_datapipe.fork(3)

    # Get contiguous time periods for each data source
    gsp_time_periods_datapipe = gsp_time_periods_datapipe.find_contiguous_t0_time_periods(
        sample_period_duration=timedelta(minutes=30),
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
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
    overlapping_datapipe = gsp_time_periods_datapipe.filter_to_overlapping_time_periods(
        secondary_datapipes=[*nwp_time_periods_datapipes.values()],
    )

    # Filter GSP to times valid for all data sources
    valid_periods_datapipe = gsp_valid_times_datapipe.filter_time_periods(
        time_periods=overlapping_datapipe
    )

    # Select t0 times
    gsp_t0_datapipe, nwp_t0_datapipe = valid_periods_datapipe.pick_t0_times().fork(2)

    # Take GSP time slices and convert to NumpyBatch
    gsp_numpy_datapipe = gsp_datapipe.select_time_slice(
        t0_datapipe=gsp_t0_datapipe,
        history_duration=timedelta(minutes=configuration.input_data.gsp.history_minutes),
        forecast_duration=timedelta(minutes=configuration.input_data.gsp.forecast_minutes),
        sample_period_duration=timedelta(minutes=30),
    ).convert_gsp_to_numpy_batch()

    # Take NWP time slices and convert to NumpyBatch
    nwp_numpy_modalities = dict()

    for nwp_source, nwp_conf in configuration.input_data.nwp.items():
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

    # Join data sources together
    combined_datapipe = MergeNumpyModalities([gsp_numpy_datapipe, nwps_numpy_datapipe])

    # Now batch the data with batch size 4
    combined_datapipe = combined_datapipe.batch(4).merge_numpy_batch()

    return combined_datapipe
