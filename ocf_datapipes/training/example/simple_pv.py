"""Wrapper for Power Perceiver Production Data Pipeline"""

from datetime import timedelta
from pathlib import Path
from typing import Union

import xarray
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.model import Configuration
from ocf_datapipes.load import OpenPVFromNetCDF
from ocf_datapipes.training.common import normalize_pv

xarray.set_options(keep_attrs=True)

# default is set to 1000
BUFFERSIZE = 1000


def simple_pv_datapipe(
    configuration_filename: Union[Path, str],
) -> IterDataPipe:
    """Create the simple datapipe which loads PV data.

    This could be used for training an autoregressive model for site-level PV output.

    Args:
        configuration_filename: Name of the configuration

    Returns:
        IterDataPipe yielding batches of samples
    """

    # Load configuration
    configuration: Configuration = load_yaml_configuration(configuration_filename)

    # Unpack for convenience
    pv_config = configuration.input_data.pv

    # Open dataset
    pv_datapipe = OpenPVFromNetCDF(pv=pv_config)

    # Fork datapipe for raw data, time slicing, and location slicing uses
    pv_datapipe, pv_t0_datapipe, pv_time_periods_datapipe, pv_location_datapipe = pv_datapipe.fork(
        4, buffer_size=BUFFERSIZE
    )

    # Preprocess PV timeseries
    pv_datapipe = pv_datapipe.normalize(normalize_fn=normalize_pv)

    # Getting locations - Construct datapipe yielding the locations to be used for each sample
    location_datapipe = pv_location_datapipe.pick_locations()

    # Slice systems
    pv_datapipe = pv_datapipe.select_id(location_datapipe=location_datapipe, data_source_name="pv")

    # Add t0 idx
    pv_datapipe = pv_datapipe.add_t0_idx_and_sample_period_duration(
        sample_period_duration=timedelta(minutes=pv_config.time_resolution_minutes),
        history_duration=timedelta(minutes=pv_config.history_minutes),
    )

    # Get contiguous time periods
    pv_time_periods_datapipe = pv_time_periods_datapipe.find_contiguous_t0_time_periods(
        sample_period_duration=timedelta(minutes=pv_config.time_resolution_minutes),
        history_duration=timedelta(minutes=pv_config.history_minutes),
        forecast_duration=timedelta(minutes=pv_config.forecast_minutes),
    )

    # Select time periods
    pv_t0_datapipe = pv_t0_datapipe.filter_time_periods(time_periods=pv_time_periods_datapipe)
    pv_t0_datapipe = pv_t0_datapipe.pick_t0_times()

    # Take time slices
    pv_datapipe = pv_datapipe.select_time_slice(
        t0_datapipe=pv_t0_datapipe,
        history_duration=timedelta(minutes=pv_config.history_minutes),
        forecast_duration=timedelta(minutes=pv_config.forecast_minutes),
        sample_period_duration=timedelta(minutes=pv_config.time_resolution_minutes),
    )

    # Convert to numpybatch object
    pv_datapipe = pv_datapipe.convert_pv_to_numpy_batch()

    # Add solar coordinates
    pv_datapipe = pv_datapipe.add_sun_position(modality_name="pv")

    # Batch the data with batch size 4
    pv_datapipe = pv_datapipe.batch(4).merge_numpy_batch()

    return pv_datapipe
